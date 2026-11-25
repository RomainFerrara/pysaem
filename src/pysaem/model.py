"""Model specification: an ODE system written as expression strings.

Expressions are parsed on a whitelist and never executed. The symbolic form
gives the Jacobian, the time breakpoints, the switching functions on the states and the
source of the compiled kernels.
Models written as Python functions are read into the same form by reader.py, which
runs their code; a specification holding such code is read only on request.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanAtom, BooleanFunction
from sympy.codegen.cfunctions import expm1, log1p
from sympy.printing.pycode import PythonCodePrinter

T = sp.Symbol("t", real=True)


class normcdf(sp.Function):
    """Distribution function of the standard normal, written erfc(-x/sqrt(2))/2 in the kernels
    so that its lower tail keeps its precision."""

    nargs = 1

    @classmethod
    def eval(cls, x):
        if x.is_Number:
            return sp.erfc(-x / sp.sqrt(2)) / 2

    def fdiff(self, argindex=1):
        x = self.args[0]
        return sp.exp(-x**2 / 2) / sp.sqrt(2 * sp.pi)

    def _eval_evalf(self, prec):
        return (sp.erfc(-self.args[0] / sp.sqrt(2)) / 2)._eval_evalf(prec)

FUNCTIONS = {
    "exp": sp.exp,
    "log": sp.log,
    "log10": lambda x: sp.log(x) / sp.log(10),
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "tanh": sp.tanh,
    "sin": sp.sin,
    "cos": sp.cos,
    "min": sp.Min,
    "max": sp.Max,
    "pow": sp.Pow,
    "ln": sp.log,
    "tan": sp.tan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "atan": sp.atan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan2": sp.atan2,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "mod": sp.Mod,
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "gamma": sp.gamma,
    "lgamma": sp.loggamma,
    "log1p": log1p,
    "expm1": expm1,
    "normcdf": normcdf,
}
# names of constants, usable when no symbol of the model takes them
CONSTANTS = {"pi": sp.pi, "inf": sp.oo}
_PREFIX = re.compile(r"\b(?:np|numpy|math)\.")
_SYNTAX = {"if", "else", "and", "or", "not", "where"}
# a function name may also name a symbol of the model: a call is a function, a name a symbol
_RESERVED = _SYNTAX | {"t", "True", "False", "None"}
# signals of the amounts received, read as regressors after the regressors and covariates; before
# the first amount record, the start of the solve and zeros (solve.Problem._with_history)
HISTORY = {
    "t_last": "the time of the most recent amount record",
    "amount_last": "the amount of the most recent amount record",
    "t_first": "the time of the first amount record",
    "n_amounts": "the number of amount records so far",
}
_NAME = re.compile(r"[A-Za-z_]\w*\Z")
_TOKEN = re.compile(r"(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|(?P<name>[A-Za-z_]\w*)")
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: _power(a, b),
}
_CMPOPS = {ast.Lt: sp.Lt, ast.LtE: sp.Le, ast.Gt: sp.Gt, ast.GtE: sp.Ge,
           ast.Eq: lambda a, b: sp.Eq(a, b, evaluate=False),
           ast.NotEq: lambda a, b: sp.Ne(a, b, evaluate=False)}


def _power(a, b):
    # an exact power of a large integer exponent would take SymPy forever (9**9**9, (3*x)**9**9)
    if isinstance(b, sp.Integer) and abs(int(b)) > 64:
        a = a.xreplace({n: sp.Float(n) for n in a.atoms(sp.Rational) if n not in (1, -1)})
    return a**b


_ODE = re.compile(r"(?:d/dt\s*\(\s*(\w+)\s*\)|d(\w+)/dt|ddt_(\w+))\s*=(.*)")
_INIT = re.compile(r"(\w+)\s*\(\s*0\s*\)\s*=(.*)")
_LET = re.compile(r"(\w+)\s*=(.*)")
_IF = re.compile(r"(if|elif|elseif|else if)\s+(.+?):?")


def parse_equations(text: str) -> dict:
    """States, equations and variables from lines written as on paper: d/dt(A) = ... (also
    dA/dt or ddt_A), A(0) = ... (zero when absent), name = ..., # for a comment, ; between
    statements on one line, and if / elif / else / end blocks that assign variables. Returns states, odes and aux."""
    odes, aux, init = {}, {}, {}
    block = None  # list of (condition or None, {name: expr}) while inside an if block
    statements = [(k, part.strip()) for k, raw in enumerate(text.splitlines(), 1)
                  for part in raw.split("#")[0].split(";")]
    for k, line in statements:
        if not line:
            continue
        where = f"equations, line {k}"
        if m := _IF.fullmatch(line):
            if m[1] == "if":
                if block is not None:
                    raise ModelError(f"{where}: nested if blocks are not supported")
                block = []
            elif block is None:
                raise ModelError(f"{where}: {m[1]} without if")
            block.append((m[2].strip(), {}))
        elif line.rstrip(":") == "else":
            if block is None:
                raise ModelError(f"{where}: else without if")
            block.append((None, {}))
        elif line in ("end", "endif", "end if"):
            if block is None:
                raise ModelError(f"{where}: end without if")
            aux.update(_branches(block, where))
            block = None
        elif m := _ODE.fullmatch(line):
            if block is not None:
                raise ModelError(f"{where}: an equation inside an if block; assign a variable "
                                 "there and use it in the equation")
            odes[m[1] or m[2] or m[3]] = m[4].strip()
        elif m := _INIT.fullmatch(line):
            init[m[1]] = m[2].strip()
        elif m := _LET.fullmatch(line):
            (block[-1][1] if block is not None else aux)[m[1]] = m[2].strip()
        else:
            raise ModelError(f"{where}: cannot read {line!r}")
    if block is not None:
        raise ModelError("equations: an if block is not closed by end")
    orphan = sorted(set(init) - set(odes))
    if orphan:
        raise ModelError(f"equations: initial value of {orphan} without an equation")
    return {"states": {x: init.get(x, "0") for x in odes}, "odes": odes, "aux": aux}


def _branches(block, where) -> dict:
    """One where() per variable assigned in an if block; every branch must assign it,
    and the block must end with else."""
    if block[-1][0] is not None:
        raise ModelError(f"{where}: an if block needs an else branch")
    names = list(dict.fromkeys(n for _, a in block for n in a))
    out = {}
    for n in names:
        missing = [c or "else" for c, a in block if n not in a]
        if missing:
            raise ModelError(f"{where}: {n} is not assigned in the branch(es) {missing}")
        expr = f"({block[-1][1][n]})"
        for cond, a in reversed(block[:-1]):
            expr = f"where({cond}, {a[n]}, {expr})"
        out[n] = expr
    return out


def canonical_input(name, v) -> dict:
    """An input as {target, delay, scale, duration}; a list of targets written as one object
    per target ({"target": "A", "fraction": "f"}, ...) or as a map of shares ({"target": {"A":
    "f", "B": "1 - f"}}) becomes parallel lists, fraction read as scale, a missing value 0 for a
    delay or duration and 1 for a scale."""
    if isinstance(v, dict) and isinstance(v.get("target"), dict):
        # shares written as a map: {"target": {"A": 0.7, "B": 0.3}}
        if "scale" in v or "fraction" in v:
            raise ModelError(f"input {name!r}: the shares are given twice, in the map of targets "
                             "and in scale")
        return {**v, "target": list(v["target"]), "scale": list(v["target"].values())}
    if not isinstance(v, list):
        return dict(v) if isinstance(v, dict) else {"target": v}
    if not v or not all(isinstance(x, dict) for x in v):
        raise ModelError(f"input {name!r}: a list of targets is one object per target, "
                         f"as {{'target': 'A', 'scale': 'f'}}; got {v!r}")
    parts = [{("scale" if k == "fraction" else k): x for k, x in part.items()} for part in v]
    out = {"target": [part.get("target") for part in parts]}
    for key, empty in (("delay", 0), ("scale", 1), ("duration", 0)):
        if any(key in part for part in parts):
            out[key] = [part.get(key, empty) for part in parts]
    unknown = sorted({k for part in parts for k in part} - {"target", "delay", "scale", "duration"})
    if unknown:
        raise ModelError(f"input {name!r}: unknown key(s) {unknown}")
    return out


class ModelError(ValueError):
    """The model specification is invalid; the message says where."""


# terminal: at most one event, which ends the follow-up of every output; single: at most one
# event; repeated: any number
HAZARD_KINDS = ("terminal", "single", "repeated")


def hazard_map(hazards) -> dict:
    """The hazards field as a dict from output to kind; a list names no kind and is refused."""
    if not isinstance(hazards or {}, dict):
        raise ModelError("hazards maps each hazard output to its kind "
                         f"({', '.join(HAZARD_KINDS)}), as {{'h': 'terminal'}}")
    return dict(hazards or {})


@dataclass(frozen=True)
class Symbolic:
    """Symbolic form of a model, auxiliaries substituted."""

    rhs: list
    out: list
    init: list
    targets: list
    delays: list
    scales: list
    durations: list
    breakpoints: list                 # times where a function of t alone switches
    jac: sp.Matrix | None
    linear: bool
    autonomous: bool
    # 0 for a breakpoint met once, P for one met at every breakpoint + k P (t % P, floor(t / P))
    periods: list = field(default_factory=list)
    # functions of t, the states and the signals whose sign change is a switch: located by the solver
    switches: list = field(default_factory=list)
    # whether an expression reads a signal of HISTORY
    history: bool = False


class Model:
    """States, parameters, inputs and outputs of an individual-level ODE model.

    states maps each state to its initial value, odes to its derivative, aux to
    intermediate expressions in order. outputs names states or auxiliaries.
    inputs maps an input name to {"target": state, "delay": expr, "scale": expr,
    "duration": expr}; a duration gives every amount of that input at a constant rate over
    that time, whatever the rate of the record.
    regressors are signals read from the data over time, covariates are constants
    of a subject; both can appear in any expression, in any form. So can the signals of
    HISTORY, never declared; a delay, scale or duration reads them at its record, that record
    counted. interpolated names the regressors read linearly between records instead of
    piecewise constant. hazards maps an output that is a hazard to its kind (HAZARD_KINDS);
    the solver integrates it (lowered).
    """

    def __init__(self, *, states, odes, parameters, outputs, aux=None, inputs=None,
                 regressors=None, covariates=None, interpolated=None, t0=None, name="model",
                 hazards=None):
        self.name = str(name)
        self.states = {k: str(v) for k, v in dict(states).items()}
        self.odes = {k: str(v) for k, v in dict(odes).items()}
        self.parameters = list(parameters)
        self.outputs = list(outputs)
        self.aux = {k: str(v) for k, v in dict(aux or {}).items()}
        self.inputs = {k: canonical_input(k, v) for k, v in dict(inputs or {}).items()}
        self.regressors = list(regressors or [])
        self.covariates = list(covariates or [])
        self.interpolated = list(interpolated or [])
        self.t0 = None if t0 is None else float(t0)
        self.hazards = hazard_map(hazards)
        self._check_names()
        # in the order of the outputs: one model, one hash
        self.hazards = {o: self.hazards[o] for o in self.outputs if o in self.hazards}
        self.symbolic  # validates every expression

    # how a model written as Python code was read (reader.py); None for equations
    reading = None
    # its symbolic form is its equations: sensitivity.py can differentiate them
    exact_derivatives = True

    n_states = property(lambda self: len(self.states))
    n_parameters = property(lambda self: len(self.parameters))
    n_outputs = property(lambda self: len(self.outputs))
    n_inputs = property(lambda self: len(self.inputs))
    n_regressors = property(lambda self: len(self.regressors))
    # the signal vector r of the kernels
    signals = property(lambda self: self.regressors + self.covariates + list(HISTORY))

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "states": self.states,
            "parameters": self.parameters,
            "aux": self.aux,
            "odes": self.odes,
            "outputs": self.outputs,
            "inputs": self.inputs,
            "regressors": self.regressors,
            "covariates": self.covariates,
            "interpolated": self.interpolated,
            "t0": self.t0,
        }
        # written only when present: a model without hazards keeps its hash
        if self.hazards:
            d["hazards"] = self.hazards
        return d

    @classmethod
    def from_dict(cls, d: dict, *, python: bool = False) -> "Model":
        """Model from a specification, tolerant on how derivatives are keyed:
        X, dX, dXdt, dX/dt and ddt_X all name the equation of state X. A specification
        holding Python code ("python") runs that code: it is read with python=True only."""
        if not isinstance(d, dict):
            raise ModelError("a model specification is an object with states, odes, "
                             "parameters and outputs")
        d = dict(d)
        if isinstance(d.get("dose"), str):
            d["inputs"] = {**(d.get("inputs") or {}), "dose": {"target": d.pop("dose")}}
        if "output" in d and "outputs" not in d:
            d["outputs"] = d.pop("output")
        for key in ("parameters", "outputs", "regressors", "covariates", "interpolated"):
            if isinstance(d.get(key), str):
                d[key] = [d[key]]
        if "python" in d:
            if not python:
                raise ModelError("this specification holds Python code, which runs when it is "
                                 "read: Model.from_dict(d, python=True) reads it")
            keys = ("states", "parameters", "outputs", "inputs", "regressors", "covariates",
                    "interpolated", "t0", "name", "values", "hazards")
            unknown = [k for k in d if k not in (*keys, "python")]
            if unknown:
                raise ModelError(f"unknown field(s) {unknown} in a model given as Python code")
            return cls.from_python(d["python"], **{k: d[k] for k in keys if d.get(k) is not None})
        if isinstance(d.get("equations"), str):
            parsed = parse_equations(d.pop("equations"))
            for key, part in parsed.items():
                d[key] = {**part, **(d.get(key) or {})} if key != "aux" else {**(d.get(key) or {}), **part}
        states = d.get("states") if isinstance(d.get("states"), dict) else {}
        if isinstance(d.get("odes"), dict):
            alias = {}
            for s in states:
                alias.update({s: s, f"d{s}": s, f"d{s}dt": s, f"d{s}/dt": s, f"d{s}_dt": s,
                              f"ddt_{s}": s})
            d["odes"] = {alias.get(str(k).replace(" ", ""), k): v for k, v in d["odes"].items()}
        fields = ("name", "states", "parameters", "aux", "odes", "outputs", "inputs",
                  "regressors", "covariates", "interpolated", "t0", "hazards")
        unknown = [k for k in d if k not in fields]
        if unknown:
            raise ModelError(f"unknown field(s) {unknown} in the model; the fields are "
                             f"{list(fields[1:])}, or \"python\" with the code of a model "
                             "written in Python")
        missing = [k for k in ("states", "parameters", "odes", "outputs") if k not in d]
        if missing:
            raise ModelError(f"the model lacks the field(s) {missing}")
        # an output written as an expression becomes an auxiliary of its own
        if isinstance(d["outputs"], list) and isinstance(d["states"], dict):
            aux = dict(d.get("aux") or {})
            outputs = []
            for k, out in enumerate(d["outputs"], 1):
                out = str(out)
                if out not in d["states"] and out not in aux and not _NAME.match(out):
                    aux[f"output_{k}"] = out
                    out = f"output_{k}"
                outputs.append(out)
            d["aux"], d["outputs"] = aux, outputs
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def lowered(self) -> "Model":
        """The model solved: each interpolated regressor becomes a state that starts at the
        value of the record and moves at the slope to the next one, both read as regressors.
        Each hazard h becomes a state after all others, _cum_h, its integral from the start of
        the solve (0 there, derivative h), whose column replaces that of h; the columns of the
        hazards themselves follow those of the outputs."""
        if not self.interpolated and not self.hazards:
            return self
        states, odes = dict(self.states), dict(self.odes)
        regs = [r for r in self.regressors if r not in self.interpolated]
        for r in self.interpolated:
            states[r], odes[r] = f"_at_{r}", f"_slope_{r}"
            regs += [f"_at_{r}", f"_slope_{r}"]
        taken = {*states, *self.parameters, *self.aux, *regs, *self.covariates}
        integral = {}
        for h in self.hazards:
            name = f"_cum_{h}"
            while name in taken:
                name = "_" + name
            taken.add(name)
            states[name], odes[name], integral[h] = "0", h, name
        return Model(states=states, odes=odes, parameters=self.parameters,
                     outputs=[integral.get(o, o) for o in self.outputs] + list(self.hazards),
                     aux=self.aux, inputs=self.inputs, regressors=regs, covariates=self.covariates,
                     t0=self.t0, name=self.name)

    @classmethod
    def from_json(cls, text: str, *, python: bool = False) -> "Model":
        return cls.from_dict(json.loads(text), python=python)

    def equations(self) -> str:
        """The model as equations written as on paper, the form parse_equations reads."""
        lines = [f"{n} = {e}" for n, e in self.aux.items()]
        lines += [f"{s}(0) = {v}" for s, v in self.states.items() if v != "0"]
        lines += [f"d/dt({s}) = {self.odes[s]}" for s in self.states]
        return "\n".join(lines)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def _check_names(self):
        groups = {
            "state": list(self.states),
            "parameter": self.parameters,
            "auxiliary": list(self.aux),
            "regressor": self.regressors,
            "covariate": self.covariates,
        }
        seen = {}
        for kind, names in groups.items():
            for n in names:
                if not isinstance(n, str) or not _NAME.match(n):
                    raise ModelError(f"{kind} name {n!r} is not a valid identifier")
                if n in _RESERVED:
                    raise ModelError(f"{kind} name {n!r} is reserved")
                if n in HISTORY:
                    raise ModelError(reserved(kind, n))
                if n in seen:
                    raise ModelError(f"{n!r} is declared as {seen[n]} and as {kind}")
                seen[n] = kind
        unknown = [r for r in self.interpolated if r not in self.regressors]
        if unknown:
            raise ModelError(f"interpolated signal(s) {unknown} are not regressors")
        missing = [s for s in self.states if s not in self.odes]
        extra = [s for s in self.odes if s not in self.states]
        if missing:
            raise ModelError(f"state(s) without equation: {', '.join(missing)}")
        if extra:
            raise ModelError(f"equation(s) for undeclared state(s): {', '.join(extra)}")
        if not self.outputs:
            raise ModelError("the model declares no output")
        for o in self.outputs:
            if o not in self.states and o not in self.aux:
                raise ModelError(f"output {o!r} is neither a state nor an auxiliary")
        for h, kind in self.hazards.items():
            if h not in self.outputs:
                raise ModelError(f"hazard {h!r} is not an output; the outputs are {self.outputs}")
            if kind not in HAZARD_KINDS:
                raise ModelError(f"hazard {h!r}: kind {kind!r} is not one of {list(HAZARD_KINDS)}")
        for name, spec in self.inputs.items():
            unknown = set(spec) - {"target", "delay", "scale", "duration"}
            if unknown:
                raise ModelError(f"input {name!r}: unknown key(s) {sorted(unknown)}")
            targets = spec.get("target")
            targets = targets if isinstance(targets, list) else [targets]
            for tg in targets:
                if not isinstance(tg, str) or tg not in self.states:
                    raise ModelError(f"input {name!r}: target {tg!r} is not a state")
            for key in ("delay", "scale", "duration"):
                v = spec.get(key)
                if isinstance(v, list) and len(v) != len(targets):
                    raise ModelError(f"input {name!r}: {len(v)} values of {key} for "
                                     f"{len(targets)} targets")

    @cached_property
    def delivered(self) -> list:
        """One (input, target, delay, scale, duration) per state an input reaches: an input
        with several targets splits each amount among them, a value per target or one for all."""
        out = []
        for k, spec in enumerate(self.inputs.values()):
            targets = spec["target"] if isinstance(spec["target"], list) else [spec["target"]]
            for j, tg in enumerate(targets):
                pick = [v[j] if isinstance(v, list) else v
                        for v in (spec.get(key) for key in ("delay", "scale", "duration"))]
                out.append((k, tg, *pick))
        return out

    @cached_property
    def _symbols(self) -> dict:
        table = {}
        # covariates travel with the regressors: constant signals of a subject
        for prefix, names in (("_y", self.states), ("_p", self.parameters),
                              ("_r", self.signals), ("_a", self.aux)):
            for i, n in enumerate(names):
                table[n] = sp.Symbol(f"{prefix}{i}", real=True)
        return table

    def _parse(self, expr, where: str, allowed: set) -> sp.Expr:
        """Parse one expression; allowed is the set of user names it may use."""
        # habits of people and language models are accepted: np.exp, math.log, ^, ln
        text = _PREFIX.sub("", str(expr).strip()).replace("^", "**")

        def mangle(m):
            name = m.group("name")
            if name is None or name in _SYNTAX or name == "t":
                return m.group(0)
            if name in FUNCTIONS and (name not in self._symbols
                                      or text[m.end():].lstrip().startswith("(")):
                return m.group(0)
            if name in CONSTANTS and name not in self._symbols:
                return name
            if name not in self._symbols:
                raise ModelError(f"{where}: unknown symbol {name!r}")
            if name not in allowed:
                raise ModelError(f"{where}: {name!r} cannot be used here")
            return self._symbols[name].name

        mangled = _TOKEN.sub(mangle, text)
        try:
            tree = ast.parse(mangled, mode="eval")
        except SyntaxError as e:
            raise ModelError(f"{where}: cannot parse {text!r} ({e.msg})") from None
        by_name = {s.name: s for s in self._symbols.values()}
        try:
            out = _to_sympy(tree.body, by_name)
        except ModelError as e:
            raise ModelError(f"{where}: {e}") from None
        except (TypeError, ValueError, ZeroDivisionError) as e:
            raise ModelError(f"{where}: invalid expression {text!r} ({e})") from None
        if _is_condition(out):
            raise ModelError(f"{where}: a condition is not a value, wrap it in where()")
        return out

    @cached_property
    def symbolic(self) -> Symbolic:
        sy = self._symbols
        params = set(self.parameters)
        constant = params | set(self.covariates)
        history = set(HISTORY)
        dynamic = constant | set(self.states) | set(self.regressors) | history

        aux = {}
        for name, expr in self.aux.items():
            e = self._parse(expr, f"auxiliary {name}", dynamic | set(aux_names(aux, sy)))
            aux[sy[name]] = e.xreplace(aux)
        everything = dynamic | set(self.aux)

        rhs = [self._parse(self.odes[s], f"equation of {s}", everything).xreplace(aux)
               for s in self.states]
        out = [sy[o].xreplace(aux) for o in self.outputs]
        # a regressor in an initial value is read at the start, or at the reset; an auxiliary
        # that does not depend on the states or on t may be used too
        ysyms = [sy[s] for s in self.states]
        free = set(ysyms) | {T}
        fixed_aux = [n for n in self.aux if not aux[sy[n]].free_symbols & free]
        init = []
        for s, expr in self.states.items():
            try:
                e = self._parse(expr, f"initial value of {s}",
                                constant | set(self.regressors) | history
                                | set(fixed_aux)).xreplace(aux)
                if T in e.free_symbols:
                    raise ModelError(f"initial value of {s}: t cannot be used here")
                init.append(e)
            except ModelError as e:
                raise ModelError(f"{e} (an initial value depends on parameters, covariates, "
                                 "regressors and the variables defined from them only)") from None

        targets, delays, scales, durations = [], [], [], []
        states = list(self.states)
        names = list(self.inputs)
        for k, target, delay, scale, duration in self.delivered:
            name = names[k]
            targets.append(states.index(target))
            # the signals of HISTORY are read at the record, the record counted
            delays.append(self._parse(delay or 0, f"delay of input {name}", constant | history))
            scales.append(self._parse(scale or 1, f"scale of input {name}", constant | history))
            # zero: the rate of the record holds
            durations.append(self._parse(duration or 0, f"duration of input {name}",
                                         constant | history))
            if any(T in e.free_symbols for e in (delays[-1], scales[-1], durations[-1])):
                raise ModelError(f"input {name}: t cannot be used in its delay, scale or duration")

        yset = set(ysyms)
        hsyms = {sy[h] for h in HISTORY}
        # covariates are constant and the signals of HISTORY change at records only: both can
        # move a breakpoint; regressors and states cannot
        found = _breakpoints(rhs + out, yset | {sy[r] for r in self.regressors}, hsyms)
        breakpoints, periods = [b for b, _ in found], [q for _, q in found]
        state_conditions = any(
            rel.free_symbols & yset for e in rhs for rel in e.atoms(Relational))
        try:
            jac = sp.Matrix([e.rewrite(sp.Piecewise) for e in rhs]).jacobian(ysyms) \
                if ysyms else sp.Matrix(0, 0, [])
            # d(C**g)/dC comes out as g*C**g/C, NaN at C = 0: written C**(g - 1)
            jac = jac.applyfunc(lambda e: sp.powsimp(e, force=True, combine="exp"))
        except Exception:
            jac = None
        linear = (jac is not None and not state_conditions
                  and not any(e.free_symbols & yset for e in jac))
        reads = any(sp.sympify(e).free_symbols & hsyms
                    for e in (*rhs, *out, *init, *delays, *scales, *durations))
        # the signals of HISTORY move with the amounts: no steady state, no closed form
        autonomous = not any(T in e.free_symbols for e in rhs) and not reads
        return Symbolic(rhs, out, init, targets, delays, scales, durations, breakpoints, jac,
                        linear, autonomous, periods, _switches(rhs, yset), reads)

    def source(self) -> dict:
        """Python source of the kernels, keyed by function name."""
        s = self.symbolic
        n = self.n_states
        rate = set(s.targets)
        src = {
            "rhs": _emit("rhs", "t, y, p, r, u, dy", [f"dy[{i}]" for i in range(n)], s.rhs,
                         suffix=[f" + u[{i}]" if i in rate else "" for i in range(n)]),
            "out": _emit("out", "t, y, p, r, o", [f"o[{i}]" for i in range(len(s.out))], s.out),
            "init": _emit("init", "p, r, y", [f"y[{i}]" for i in range(n)], s.init),
            # b holds the breakpoints, then their periods
            "brk": _emit("brk", "p, r, b", [f"b[{i}]" for i in range(2 * len(s.breakpoints))],
                         s.breakpoints + [sp.Float(q) for q in s.periods]),
            "swf": _emit("swf", "t, y, p, r, g", [f"g[{i}]" for i in range(len(s.switches))],
                         s.switches),
            # d holds the delays, then the durations
            "inp": _emit("inp", "p, r, d, s",
                         [f"d[{i}]" for i in range(2 * len(s.targets))]
                         + [f"s[{i}]" for i in range(len(s.targets))],
                         s.delays + s.durations + s.scales),
        }
        if s.jac is not None:
            try:
                src["jac"] = _emit("jac", "t, y, p, r, J",
                                   [f"J[{i}, {j}]" for i in range(n) for j in range(n)],
                                   list(s.jac))
            except ModelError:
                pass
        return src

    def dependencies(self, output: str) -> list:
        """Parameters an output depends on: those of its expression and, for each state it reads,
        those of that state's equation, initial value and inputs, and so on through the states
        these read."""
        if output not in self.outputs:
            raise ModelError(f"{output!r} is not an output; the outputs are {self.outputs}")
        s = self.symbolic
        ysyms = [self._symbols[x] for x in self.states]
        todo, used, seen = [s.out[self.outputs.index(output)]], set(), set()
        while todo:
            free = sp.sympify(todo.pop()).free_symbols
            used |= free
            for i, y in enumerate(ysyms):
                if y in free and i not in seen:
                    seen.add(i)
                    todo += [s.rhs[i], s.init[i]]
                    todo += [e for m, target in enumerate(s.targets) if target == i
                             for e in (s.delays[m], s.scales[m], s.durations[m])]
        return [n for n in self.parameters if self._symbols[n] in used]

    def unused_parameters(self) -> list:
        """Parameters that no equation, output, initial value or input uses."""
        s = self.symbolic
        used = set()
        for e in [*s.rhs, *s.out, *s.init, *s.delays, *s.scales, *s.durations]:
            used |= sp.sympify(e).free_symbols
        return [n for n in self.parameters if self._symbols[n] not in used]

    def kernel_globals(self, jit: bool = True) -> dict:
        """Names the kernels call besides numpy and math: none for equations; the functions
        of a model compiled as written (native.py)."""
        return {}

    def python_functions(self) -> dict:
        """The kernels as plain Python callables (reference path and tests)."""
        ns, src = {"np": np, "math": math, **self.kernel_globals(jit=False)}, self.source()
        for code in src.values():
            exec(code, ns)  # noqa: S102 - generated source, or code checked by reader.py
        return {k: ns[k] for k in src if callable(ns.get(k))}

    def evaluator(self, exprs: list, module=None):
        """f(t, y, p, r): the values of expressions of the model's symbols, as an array; with
        module in place of np and math, an array of the objects it computes."""
        ns, n = {"np": module or np, "math": module or math}, len(exprs)
        exec(_emit("f", "t, y, p, r, o", [f"o[{i}]" for i in range(n)], exprs), ns)  # noqa: S102
        f = ns["f"]

        def evaluate(t, y, p, r):
            o = np.zeros(n) if module is None else np.empty(n, dtype=object)
            f(t, y, p, r, o)
            return o
        return evaluate


def reserved(kind: str, name: str) -> str:
    """The refusal of a declared name of HISTORY."""
    return (f"{kind} name {name!r} is reserved: it is {HISTORY[name]}, read in any expression "
            "without being declared")


def history_breakpoints(model) -> int:
    """How many breakpoints read a signal of HISTORY; they come last."""
    hsyms = {model._symbols[h] for h in HISTORY}
    return sum(bool(sp.sympify(b).free_symbols & hsyms) for b in model.symbolic.breakpoints)


def reads_inputs_history(model) -> bool:
    """Whether a delay, scale or duration of an input reads a signal of HISTORY."""
    s, hsyms = model.symbolic, {model._symbols[h] for h in HISTORY}
    return any(sp.sympify(e).free_symbols & hsyms for e in (*s.delays, *s.scales, *s.durations))


def aux_names(aux: dict, table: dict) -> list:
    done = {s for s in aux}
    return [n for n, s in table.items() if s in done]


def _to_sympy(node, symbols: dict):
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ModelError(f"constant {v!r} is not a number")
        return sp.Integer(v) if isinstance(v, int) else sp.Float(v)
    if isinstance(node, ast.Name):
        if node.id == "t":
            return T
        if node.id in symbols:
            return symbols[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ModelError(f"unknown symbol {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_to_sympy(node.left, symbols),
                                      _to_sympy(node.right, symbols))
    if isinstance(node, ast.UnaryOp):
        v = _to_sympy(node.operand, symbols)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.Not):
            return sp.Not(v)
    if isinstance(node, ast.Compare):
        terms = [_to_sympy(n, symbols) for n in [node.left, *node.comparators]]
        rels = []
        for op, a, b in zip(node.ops, terms, terms[1:]):
            if type(op) not in _CMPOPS:
                raise ModelError("only <, <=, >, >=, == and != comparisons are allowed")
            rels.append(_CMPOPS[type(op)](a, b))
        return sp.And(*rels)
    if isinstance(node, ast.BoolOp):
        vals = [_to_sympy(v, symbols) for v in node.values]
        return sp.And(*vals) if isinstance(node.op, ast.And) else sp.Or(*vals)
    if isinstance(node, ast.IfExp):
        return _piecewise(_to_sympy(node.test, symbols), _to_sympy(node.body, symbols),
                          _to_sympy(node.orelse, symbols))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        args = [_to_sympy(a, symbols) for a in node.args]
        if node.func.id == "where":
            if len(args) != 3:
                raise ModelError("where() takes a condition and two values")
            return _piecewise(*args)
        if node.func.id in FUNCTIONS:
            return FUNCTIONS[node.func.id](*args)
        raise ModelError(f"function {node.func.id!r} is not allowed")
    raise ModelError(f"syntax not allowed: {type(node).__name__}")


def _is_condition(e) -> bool:
    # a Symbol is also a sympy Boolean, so test the condition classes only
    return isinstance(e, (Relational, BooleanFunction, BooleanAtom))


def _piecewise(cond, a, b):
    if not _is_condition(cond) or _is_condition(a) or _is_condition(b):
        raise ModelError("the first argument of where() must be a comparison")
    return sp.Piecewise((a, cond), (b, True))


def _linear_in_t(u, excluded: set):
    """(a, b) of u = a t + b with a a nonzero number and b free of t and of excluded, or None."""
    if u.free_symbols & excluded:
        return None
    try:
        poly = sp.Poly(u, T)
    except sp.PolynomialError:
        return None
    a, b = poly.coeff_monomial(T), poly.coeff_monomial(1)
    if poly.degree() != 1 or not a.is_number or T in b.free_symbols:
        return None
    return a, b


def _floor_of_mod(d):
    """d with each floor((x - mod(x, P))/P + 1/2), the reading of x // P, written as the integer
    (x - mod(x, P))/P it is: t - 24*(t // 24) is then mod(t, 24)."""
    swap = {}
    for f in d.atoms(sp.floor):
        whole = f.args[0] - sp.Rational(1, 2)
        for m in f.args[0].atoms(sp.Mod):
            x, P = m.args
            if sp.simplify(whole - (x - m) / P) == 0:
                swap[f] = whole
    return sp.simplify(d.xreplace(swap)) if swap else d


def _breakpoints(exprs: list, excluded: set, history: set = frozenset()) -> list:
    """(time, period) where a function of t alone switches or jumps; time is an expression of
    parameters, covariates and the signals history, period 0 for a single time or P for the
    times time + k P. Periodic ones come from mod(a t + b, P) and floor or ceil(a t + b), with a
    and P numbers. Those that read history come last."""
    found = {}

    def add(at, period):
        at = sp.simplify(at)
        found[(str(at), period)] = (at, period)

    for e in exprs:
        for f in e.atoms(sp.Mod, sp.floor, sp.ceiling):
            u, P = f.args if isinstance(f, sp.Mod) else (f.args[0], sp.Integer(1))
            lin = _linear_in_t(u, excluded)
            if lin and P.is_number and P > 0:
                add(-lin[1] / lin[0], float(P / abs(lin[0])))
        for rel in e.atoms(Relational):
            fs = rel.free_symbols
            # an equality holds on no interval: nothing switches there
            if T not in fs or fs & excluded or isinstance(rel, (sp.Eq, sp.Ne)):
                continue
            d = _floor_of_mod(rel.lhs - rel.rhs)
            lin = _linear_in_t(d, excluded)
            if lin:
                add(-lin[1] / lin[0], 0.0)
                continue
            # mod(a t + b, P) compared with c: switches where a t + b = c + k P
            mods = d.atoms(sp.Mod)
            if len(mods) != 1:
                continue
            m = next(iter(mods))
            M = sp.Dummy("m")
            dm = d.subs(m, M)
            lin, P = _linear_in_t(m.args[0], excluded), m.args[1]
            if T in dm.free_symbols or not lin or not P.is_number or not P > 0:
                continue
            try:
                roots = sp.solve(dm, M)
            except NotImplementedError:
                continue
            for c in roots:
                if T not in sp.sympify(c).free_symbols:
                    add((c - lin[1]) / lin[0], float(P / abs(lin[0])))
    last = {k: bool(found[k][0].free_symbols & history) for k in found}
    return [found[k] for k in sorted(found, key=lambda k: (last[k], k))]


def _switches(exprs: list, states: set) -> list:
    """Switching functions of the equations: lhs - rhs of each relation on a state, a - b for each
    pair of arguments of a Min or Max and the argument of an Abs or a sign that read a state. An
    equality holds on no interval and is left out."""
    found = {}

    def add(g):
        g = sp.sympify(g)
        if not g.free_symbols & states or str(-g) in found:
            return
        try:
            _code(g)
        except ModelError:
            return
        found.setdefault(str(g), g)

    for e in exprs:
        for rel in e.atoms(Relational):
            if not isinstance(rel, (sp.Eq, sp.Ne)):
                add(rel.lhs - rel.rhs)
        for f in e.atoms(sp.Min, sp.Max):
            for a in range(len(f.args)):
                for b in range(a + 1, len(f.args)):
                    add(f.args[a] - f.args[b])
        for f in e.atoms(sp.Abs, sp.sign, sp.Heaviside):
            add(f.args[0])
    return [found[k] for k in sorted(found)]


def _emit(name: str, args: str, lhs: list, exprs: list, suffix: list | None = None) -> str:
    """Source of one kernel; inputs unpack from y, p, r by symbol prefix."""
    exprs = [sp.sympify(e) for e in exprs]
    repl, reduced = sp.cse(exprs, symbols=sp.numbered_symbols("_x")) if exprs else ([], [])
    lines = [f"def {name}({args}):"]
    used = set().union(*(e.free_symbols for e in exprs)) if exprs else set()
    access = {"_y": "y", "_p": "p", "_r": "r"}
    for sym in sorted(used, key=lambda s: (s.name[:2], int(s.name[2:] or 0) if s != T else 0)):
        if sym == T:
            continue
        lines.append(f"    {sym.name} = {access[sym.name[:2]]}[{sym.name[2:]}]")
    for s, e in repl:
        lines.append(f"    {s} = {_code(e)}")
    suffix = suffix or [""] * len(lhs)
    for target, e, extra in zip(lhs, reduced, suffix):
        lines.append(f"    {target} = {_code(e)}{extra}")
    if len(lines) == 1:
        lines.append("    pass")
    return "\n".join(lines) + "\n"


class _Printer(PythonCodePrinter):
    def _print_Not(self, expr):
        # the stock printer glues "not" to a bare symbol
        return f"(not ({self._print(expr.args[0])}))"

    def _print_Infinity(self, expr):
        return "np.inf"

    def _print_Float(self, expr):
        # all the digits: 1/168 written with 15 of them makes floor(168*t/168) wrong at t = 168
        return repr(float(expr))

    def _print_normcdf(self, expr):
        return f"(0.5*math.erfc(-({self._print(expr.args[0])})/1.4142135623730951))"

    def _print_Mod(self, expr):
        # % binds as * does: -k*mod(t, 24) must not become -k*t % 24
        return f"(({self._print(expr.args[0])}) % ({self._print(expr.args[1])}))"

    def _print_Pow(self, expr, rational=False):
        # x**-2 raises in a compiled kernel at x = 0, 1/x**2 gives inf as numpy does
        if expr.exp.is_Integer and expr.exp < -1:
            return f"(1/({self._print(sp.Pow(expr.base, -expr.exp, evaluate=False))}))"
        return super()._print_Pow(expr, rational)

    def _print_Integer(self, expr):
        # Numba holds integers in 64 bits
        return str(expr.p) if -2**63 <= expr.p < 2**63 else repr(float(expr.p))

    def _print_NegativeInfinity(self, expr):
        return "(-np.inf)"


# math functions without a NumPy namesake under the same name; erf, erfc and the gammas stay math
_NUMPY_NAMES = {"asin": "arcsin", "acos": "arccos", "atan": "arctan", "atan2": "arctan2",
                "asinh": "arcsinh", "acosh": "arccosh", "atanh": "arctanh", "erf": None,
                "erfc": None, "lgamma": None, "gamma": None}
_MATH = re.compile(r"\bmath\.(\w+)")


def _code(expr) -> str:
    text = _Printer().doprint(expr)
    if "Not supported" in text:
        raise ModelError(f"cannot generate code for {expr}")
    return _MATH.sub(lambda m: f"math.{m[1]}" if _NUMPY_NAMES.get(m[1], m[1]) is None
                     else f"np.{_NUMPY_NAMES.get(m[1], m[1])}", text)
