"""Models compiled from their Python code as written: the safety net of reader.py.

Used when valid code cannot be written as equations. Numba compiles the code as it is, after the
rewrites of _Prepare that keep its meaning, and the solver calls it through generated wrappers.
Such a model has no closed form and no exact derivatives (finite differences instead), finds only
the switching times its code writes with parameters, covariates and numbers, and its kernels are
slower than generated ones. A switch on a state is not located: the integrators step across it
under error control.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import dis
import inspect
import math
import re
import types
from collections import Counter
from functools import cached_property

import numba
import numpy as np

from . import reader
from .model import HISTORY, Model, ModelError

_HEADER = "import math\nimport numpy as np\n"
_ROLES = ("rhs", "observe", "init")


class NativeModel(Model):
    """A model whose rhs, observe and init are its Python code compiled as written. forms says
    whether each returns a list, a dict or a single value (None: not compiled)."""

    exact_derivatives = False

    def __init__(self, *, python, states, parameters, outputs, forms, inputs=None,
                 regressors=None, covariates=None, t0=None, name="model", reason="",
                 values=None, init=None, autonomous=False, switches=()):
        self.python, self.reason = python, str(reason)
        # the code names a signal of HISTORY, quoted or as p.name
        self.reads_history = any(re.search(rf"""['"]{h}['"]|\.{h}\b""", python) for h in HISTORY)
        self.forms = {r: forms.get(r) for r in _ROLES}
        self.values = dict(values or {})
        self.autonomous, self.switches = bool(autonomous), list(switches)
        self._globals = {}
        tree = reader.parse(python)
        states = list(states)
        # a shell of equations carries the names, the inputs and constant initial values
        super().__init__(states={s: (init[i] if init else "0") for i, s in enumerate(states)},
                         odes={s: "0" for s in states}, parameters=parameters,
                         outputs=list(outputs), aux={o: "0" for o in outputs if o not in states},
                         inputs=inputs, regressors=regressors, covariates=covariates, t0=t0,
                         name=name)
        self.prepared, self.array_p = _prepare(
            tree, self.parameters + self.covariates + self.regressors + list(HISTORY),
            [r for r in _ROLES if self.forms[r]])

    @cached_property
    def symbolic(self):
        # the times where a condition of the code on t switches: the solver stops there
        breakpoints = []
        for text in self.switches:
            try:
                allowed = set(self.parameters) | set(self.covariates) | set(HISTORY)
                breakpoints.append(self._parse(text, "a switching time", allowed))
            except ModelError:
                pass
        # those that read HISTORY last (history_breakpoints)
        hsyms = {self._symbols[h] for h in HISTORY}
        breakpoints.sort(key=lambda b: bool(b.free_symbols & hsyms))
        base = Model.symbolic.func(self)
        history = base.history or self.reads_history
        return dataclasses.replace(base, jac=None, linear=False,
                                   autonomous=self.autonomous and not history,
                                   breakpoints=breakpoints, periods=[0.0] * len(breakpoints),
                                   switches=[], history=history)

    def to_dict(self) -> dict:
        d = {"name": self.name, "python": self.python, "states": list(self.states),
             "parameters": self.parameters, "outputs": self.outputs, "inputs": self.inputs,
             "regressors": self.regressors, "covariates": self.covariates, "t0": self.t0}
        if self.values:
            d["values"] = self.values
        return d

    def equations(self) -> str:
        raise ModelError("this model is compiled as written: it has no equations (its code is in .python)")

    def dependencies(self, output: str) -> list:
        # the code is not read as equations: any parameter may reach any output
        Model.dependencies(self, output)  # checks that output is one
        return list(self.parameters)

    def unused_parameters(self) -> list:
        # a parameter the code never names (quoted, or as p.name) and no input uses
        s = self.symbolic
        inputs = set().union(*(e.free_symbols for e in [*s.delays, *s.scales, *s.durations]))
        return [n for n in self.parameters if self._symbols[n] not in inputs and not
                re.search(rf"""['"]{re.escape(n)}['"]|\.{re.escape(n)}\b""", self.python)]

    def kernel_globals(self, jit: bool = True) -> dict:
        if jit not in self._globals:
            ns = reader.namespace(reader.parse(self.prepared), symbolic=False)
            for k, v in list(ns.items()):
                # Numba reads arrays, not lists, from the module
                if isinstance(v, (list, tuple)) and v and _numeric(v):
                    ns[k] = np.asarray(v)
            if jit:
                done = {}
                for k, v in list(ns.items()):
                    if isinstance(v, reader.Module) and reader.real_of(v) is not None:
                        ns[k] = reader.real_of(v)
                    elif inspect.isfunction(v) and v.__globals__ is ns:
                        # an index out of a table raises, as in Python, instead of reading memory
                        if v not in done:
                            done[v] = numba.njit(error_model="numpy", boundscheck=True)(v)
                        ns[k] = done[v]
            fns = reader.roles(ns)
            out = {f"_u_{r}": fns[r] for r in _ROLES if self.forms[r] and fns[r] is not None}
            self._globals[jit] = {**out, "_Dict": numba.typed.Dict, "_nbt": numba.types}
        return self._globals[jit]

    def source(self) -> dict:
        base = Model.source(self)
        n, rate = self.n_states, set(self.symbolic.targets)
        src = {"brk": base["brk"], "inp": base["inp"], "init": base["init"], "swf": base["swf"]}
        src["rhs"] = self._wrapper("rhs(t, y, p, r, u, dy)", "_u_rhs(t, y.copy(), d)", "dy", "rhs",
                                   list(self.states), [f" + u[{i}]" if i in rate else "" for i in range(n)]) \
            if n else "def rhs(t, y, p, r, u, dy):\n    pass\n"
        src["out"] = self._wrapper("out(t, y, p, r, o)", "_u_observe(t, y.copy(), d)", "o", "observe",
                                   self.outputs)
        if self.forms["init"]:
            src["init"] = self._wrapper("init(p, r, y)", "_u_init(d)", "y", "init", list(self.states))
        return src

    def _wrapper(self, head, call, target, role, names, suffix=None):
        form = self.forms[role]
        lines = [f"def {head}:", *self._fill(), "    try:", f"        v = {call}"]
        if form == "list":
            # fewer values than expected on a branch the check did not reach: no reading past them
            lines.append(f"        if len(v) != {len(names)}:\n            raise IndexError()")
        for i, n in enumerate(names):
            v = "v" if form == "scalar" else f"v[{n!r}]" if form == "dict" else f"v[{i}]"
            lines.append(f"        {target}[{i}] = {v}{suffix[i] if suffix else ''}")
        # an error in the code is a failed solve, not stale values
        lines += ["    except Exception:", f"        {target}[:] = np.nan"]
        return "\n".join(lines) + "\n"

    def _fill(self) -> list:
        """d, the p of the code, from the parameters p and the regressors, covariates and, when the
        code reads them, signals of HISTORY in r."""
        n_p, n_c, n_g = len(self.parameters), len(self.covariates), len(self.regressors)
        n_h = len(HISTORY) if self.reads_history else 0
        if self.array_p:
            lines = [f"    d = np.empty({n_p + n_c + n_g + n_h})"]
            lines += [f"    d[:{n_p}] = p[:{n_p}]"] if n_p else []
            lines += [f"    d[{n_p}:{n_p + n_c}] = r[{n_g}:{n_g + n_c}]"] if n_c else []
            lines += [f"    d[{n_p + n_c}:{n_p + n_c + n_g}] = r[:{n_g}]"] if n_g else []
            lines += [f"    d[{n_p + n_c + n_g}:] = r[{n_g + n_c}:{n_g + n_c + n_h}]"] if n_h else []
            return lines
        lines = ["    d = _Dict.empty(key_type=_nbt.unicode_type, value_type=_nbt.float64)"]
        lines += [f"    d[{k!r}] = p[{i}]" for i, k in enumerate(self.parameters)]
        lines += [f"    d[{k!r}] = r[{n_g + j}]" for j, k in enumerate(self.covariates)]
        lines += [f"    d[{k!r}] = r[{j}]" for j, k in enumerate(self.regressors)]
        lines += [f"    d[{k!r}] = r[{n_g + n_c + j}]" for j, k in enumerate(list(HISTORY)[:n_h])]
        return lines


def _numeric(v) -> bool:
    """True for a list of numbers, or a table of them."""
    try:
        np.array(v, dtype=float)
    except (ValueError, TypeError):
        return False
    return True


def native_model(source, tree, spec, values, reason) -> NativeModel:
    """The code compiled as written, checked against Python at points drawn around the values."""
    from .solve import compile_model

    if spec["interpolated"]:
        raise ModelError("interpolated regressors need a model read as equations")
    if spec["hazards"]:
        raise ModelError("hazards need a model read as equations")
    ns = reader.namespace(tree, symbolic=False)
    fns = reader.roles(ns)
    reader.require(fns, spec, ns)
    states = list(spec["states"])
    init = _constant_init(ns, states)
    used = [r for r in _ROLES if fns[r] is not None and (r == "observe" or states)
            and not (r == "init" and init is not None)]
    forms, keys, rows, first = {}, {}, [], None
    with np.errstate(all="ignore"):
        for spread, points in reader.point_rounds(tree, spec, values):
            for point in points:
                row = {}
                for role in used:
                    try:
                        raw = reader.call(fns[role], role, *point)
                        form = _form(raw)
                        if form == "dict":
                            keys.setdefault(role, list(raw))
                        names = (spec["outputs"] or keys.get(role)) if role == "observe" else states
                        v = reader.numbers(raw, names if form == "dict" else None)
                    except ModelError:
                        raise
                    except Exception as e:
                        first = first or (f"{role} raises {type(e).__name__}: {e}{_hint(e, states)} at "
                                          f"{reader.point_text(point)}")
                        row = None
                        break
                    if forms.setdefault(role, form) != form:
                        raise ModelError(f"{role} returns a {form} at some points, a {forms[role]} at others")
                    if not np.all(np.isfinite(v)):
                        row = None
                        break
                    row[role] = v
                if row:
                    rows.append((point, row))
            if len(rows) >= reader.MIN_POINTS:
                break
    if not rows and first:
        raise reader.CodeFails(f"the code fails at every point: {first}")
    if len(rows) < reader.MIN_POINTS:
        raise reader.CodeFails(reader.too_few("the code gives finite values", len(rows), spec, spread)
                               + (f"; {first}" if first else ""))
    n_out = len(rows[0][1]["observe"])
    outputs = spec["outputs"] or keys.get("observe") or [f"out{k + 1}" for k in range(n_out)]
    for role in used:
        want = len(outputs) if role == "observe" else len(states)
        got = sorted({len(r[role]) for _, r in rows})
        if got != [want]:
            what = f"the outputs {outputs}" if role == "observe" else f"{len(states)} states {states}"
            raise reader.CodeFails(f"{role} returns {' or '.join(map(str, got))} value"
                                   f"{'' if got == [1] else 's'} for {what}")
    rhs = fns["rhs"] if states else None
    time = rhs.__code__.co_varnames[0] if rhs is not None and getattr(rhs, "__code__", None) \
        and rhs.__code__.co_argcount else None
    model = NativeModel(python=source, states=states, parameters=spec["parameters"],
                        outputs=outputs, forms=forms, inputs=spec["inputs"],
                        regressors=spec["regressors"], covariates=spec["covariates"],
                        t0=spec["t0"], name=spec["name"], reason=reason, values=values, init=init,
                        autonomous=rhs is None or not _reads_time(rhs),
                        switches=_time_switches(tree, time) if time else ())
    precise = reader._Precise(tree)
    _check_compiled(model, rows, precise)
    try:
        compile_model(model)
    except Exception as e:
        raise ModelError(f"Numba cannot compile the code: {_short(e)}") from None
    model.reading = {"form": "compiled as written", "reason": str(reason), "checked_points": len(rows),
                     "spread": spread, "lost_points": len(precise.lost)}
    return model


def _hint(e, states) -> str:
    if isinstance(e, IndexError) and "only integers" in str(e):
        return f" (y holds the states {states} in this order: y[0], not y['name'])"
    if isinstance(e, IndexError) and "out of bounds" in str(e):
        return f" (y holds {len(states)} states {states})"
    return ""


def _form(raw) -> str:
    if isinstance(raw, dict):
        return "dict"
    return "list" if isinstance(raw, (list, tuple)) or np.ndim(raw) > 0 else "scalar"


def _constant_init(ns, states):
    """Initial values given as a list (y0 = [...]) as equations, None otherwise."""
    v = ns.get("init", ns.get("y0"))
    if v is None or callable(v) or not states:
        return None
    values = reader.numbers(v)
    if len(values) != len(states):
        raise ModelError(f"init holds {len(values)} values for {len(states)} states {states}")
    return [repr(float(x)) for x in values]


def _check_compiled(model, rows, precise):
    """The compiled wrappers and code against the code run by Python, at the points of rows; a
    point where the floats of the code miss its value (precise.code_lost) says nothing of Numba
    and is passed over."""
    from .solve import deep_recursion

    ns = {"np": np, "math": math, **model.kernel_globals(jit=True)}
    for code in model.source().values():
        exec(code, ns)  # noqa: S102 - generated wrappers of code checked by reader.parse
    kernels = {k: numba.njit(error_model="numpy")(ns[k]) for k in ("rhs", "out", "init")}
    n, found = model.n_states, {r: [] for r in _ROLES}
    try:
        with deep_recursion():
            for point, row in rows:
                t, y, p = point
                pv = np.array([p[k] for k in model.parameters], float)
                rv = np.array([p[k] for k in model.signals], float)
                if "rhs" in row:
                    dy = np.zeros(n)
                    kernels["rhs"](float(t), y.copy(), pv, rv, np.zeros(n), dy)
                    found["rhs"].append((row["rhs"], dy, point))
                o = np.zeros(model.n_outputs)
                kernels["out"](float(t), y.copy(), pv, rv, o)
                found["observe"].append((row["observe"], o, point))
                if "init" in row:
                    y0 = np.zeros(n)
                    kernels["init"](pv, rv, y0)
                    found["init"].append((row["init"], y0, point))
    except Exception as e:
        raise ModelError(f"Numba cannot compile the code: {_short(e)}") from None
    for role, rs in found.items():
        names = list(model.outputs) if role == "observe" else list(model.states)
        bad = reader.compare(rs, lambda row, js, scale, role=role, names=names:
                             precise.code_lost(role, row, js, scale, names)) if rs else None
        if bad:
            raise ModelError(f"{role}: the compiled code differs from Python ({bad}): a construct "
                             "that Numba reads differently")


# ---------- the code made acceptable to Numba ----------

_DICT_METHODS = {"get", "keys", "values", "items", "copy", "update", "pop", "setdefault"}


def _prepare(tree, names, roles=_ROLES) -> tuple:
    """Code for Numba, and whether p is read as an array (fast) rather than as a typed dict: only
    when every function compiled is found and reads p by its names only."""
    tree = copy.deepcopy(tree)
    defs = _role_defs(tree)
    pargs = {r: a for r, node in defs.items() if (a := _arg(node, 0 if r == "init" else 2))}
    array = all(r in pargs for r in roles) and all(_p_simple(defs[r], a) for r, a in pargs.items())
    for r, a in pargs.items():
        (_PIndex(a, names) if array else _PAttribute(a)).visit(defs[r])
    tree = ast.fix_missing_locations(_Prepare(_aliases(tree)).visit(tree))
    return _HEADER + ast.unparse(tree) + "\n", array


def _role_defs(tree) -> dict:
    """The def or lambda bound at module level to rhs, observe and init (or y0)."""
    defs, out = {}, {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defs[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            defs[node.targets[0].id] = v if isinstance(v, ast.Lambda) else defs.get(v.id) if isinstance(v, ast.Name) else None
    for role, names in (("rhs", ("rhs",)), ("observe", ("observe",)), ("init", ("init", "y0"))):
        node = next((defs[n] for n in names if defs.get(n) is not None), None)
        if node is not None:
            out[role] = node
    return out


def _arg(node, i):
    args = [*node.args.posonlyargs, *node.args.args]
    return args[i].arg if len(args) > i else None


def _p_simple(node, parg) -> bool:
    """True when p is only read as p['name'], p.get('name'[, default]) or p.name."""
    parents = {c: n for n in ast.walk(node) for c in ast.iter_child_nodes(n)}
    for n in ast.walk(node):
        if not (isinstance(n, ast.Name) and n.id == parg):
            continue
        if not isinstance(n.ctx, ast.Load):
            return False
        up = parents.get(n)
        if isinstance(up, ast.Subscript) and up.value is n and isinstance(up.slice, ast.Constant) \
                and isinstance(up.slice.value, str):
            continue
        if isinstance(up, ast.Attribute) and up.value is n:
            call = parents.get(up)
            if up.attr == "get" and isinstance(call, ast.Call) and call.func is up and not call.keywords \
                    and 1 <= len(call.args) <= 2 and isinstance(call.args[0], ast.Constant) \
                    and isinstance(call.args[0].value, str):
                continue
            if up.attr not in _DICT_METHODS:
                continue
        return False
    return True


class _PIndex(ast.NodeTransformer):
    """p['name'], p.get('name', default) and p.name read p as an array in the order of names."""

    def __init__(self, parg, names):
        self.parg, self.index = parg, {n: i for i, n in enumerate(names)}

    def _is_p(self, node):
        return isinstance(node, ast.Name) and node.id == self.parg

    def _at(self, ref, key, default=None):
        if key not in self.index:
            return default if default is not None else ast.Constant(None)
        return ast.copy_location(ast.Subscript(ast.Name(self.parg, ast.Load()),
                                               ast.Constant(self.index[key]), ast.Load()), ref)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if self._is_p(node.value) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            return self._at(node, node.slice.value)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "get" and self._is_p(f.value):
            return self._at(node, node.args[0].value, node.args[1] if len(node.args) > 1 else None)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if self._is_p(node.value) and node.attr not in _DICT_METHODS:
            return self._at(node, node.attr)
        return node


class _PAttribute(ast.NodeTransformer):
    """p.name read as p['name'] where p is a typed dict."""

    def __init__(self, parg):
        self.parg = parg

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == self.parg and node.attr not in _DICT_METHODS:
            return ast.copy_location(ast.Subscript(node.value, ast.Constant(node.attr), ast.Load()), node)
        return node


def _expr(text):
    return ast.parse(text, mode="eval").body


def _norm(a, k):
    x = a[0] if a else k.get("x", "0.0")
    loc = a[1] if len(a) > 1 else k.get("loc", "0.0")
    scale = a[2] if len(a) > 2 else k.get("scale", "1.0")
    return x, loc, scale


_REWRITES = {
    "scipy.special.erf": lambda a, k: f"math.erf({a[0]})",
    "scipy.special.erfc": lambda a, k: f"math.erfc({a[0]})",
    "scipy.special.expit": lambda a, k: f"(1.0 / (1.0 + np.exp(-({a[0]}))))",
    "scipy.special.logit": lambda a, k: f"np.log(({a[0]}) / (1.0 - ({a[0]})))",
    "scipy.special.gammaln": lambda a, k: f"math.lgamma({a[0]})",
    "scipy.special.gamma": lambda a, k: f"math.gamma({a[0]})",
    "scipy.stats.norm.cdf": lambda a, k: "(0.5 * math.erfc(-(({0}) - ({1})) / (({2}) * "
                                         "1.4142135623730951)))".format(*_norm(a, k)),
    "scipy.stats.norm.sf": lambda a, k: "(0.5 * math.erfc((({0}) - ({1})) / (({2}) * "
                                        "1.4142135623730951)))".format(*_norm(a, k)),
    "scipy.stats.norm.pdf": lambda a, k: "(np.exp(-0.5 * ((({0}) - ({1})) / ({2})) ** 2) / "
                                         "(({2}) * 2.5066282746310002))".format(*_norm(a, k)),
}
_FLOAT_TYPES = ("float", "numpy.float64", "numpy.float_", "numpy.double")
_JOINS = ("concatenate", "hstack", "vstack", "stack", "column_stack", "dstack")
_INT_TYPES = ("int", "numpy.int64", "numpy.int_")


def _aliases(tree) -> dict:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                out[a.asname or a.name] = f"{node.module}.{a.name}"
    return out


class _Prepare(ast.NodeTransformer):
    """The code made acceptable to Numba without changing its meaning: decorators, prints and
    np.errstate dropped; imports of scipy, numba and typing removed; scipy's erf, erfc, expit,
    logit, gammaln, gamma and the normal cdf, sf and pdf written with math and numpy; float
    dtypes written np.float64; the values numpy joins made arrays; integer lists that receive
    or return values made float."""

    def __init__(self, aliases):
        self.aliases = aliases

    def _path(self, node) -> str:
        names = reader.chain(node)
        return ".".join([self.aliases.get(names[0], names[0]), *names[1:]]) if names else ""

    def visit_Import(self, node):
        node.names = [a for a in node.names if a.name.split(".")[0] in ("numpy", "math")]
        return node if node.names else None

    def visit_ImportFrom(self, node):
        return node if (node.module or "").split(".")[0] in ("numpy", "math") else None

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        self.generic_visit(node)
        _float_lists(node)
        return node

    def visit_Expr(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call) and self._path(node.value.func) == "print":
            return ast.copy_location(ast.Pass(), node)
        return node

    def visit_With(self, node):
        self.generic_visit(node)
        return node.body

    def visit_ExceptHandler(self, node):
        self.generic_visit(node)
        if node.type is not None and self._path(node.type) != "Exception":
            raise ModelError(f"except {ast.unparse(node.type)}: compiled code catches errors with "
                             "except Exception: only")
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        for k in node.keywords:
            if k.arg == "dtype":
                name = self._path(k.value) or (k.value.value if isinstance(k.value, ast.Constant) else "")
                if name in _FLOAT_TYPES or name in ("float64", "f8"):
                    k.value = _expr("np.float64")
                elif name in _INT_TYPES or name in ("int64", "i8"):
                    k.value = _expr("np.int64")
        path = self._path(node.func)
        if (path.startswith("numpy.") and path.split(".")[-1] in _JOINS and node.args
                and isinstance(node.args[0], (ast.List, ast.Tuple))):
            # Numba joins a tuple of arrays: a number is an array of no dimension, as numpy takes it
            node.args[0] = ast.Tuple([_expr(f"np.asarray({ast.unparse(e)}, dtype=np.float64)")
                                      if isinstance(e, ast.List) else _expr(f"np.asarray({ast.unparse(e)})")
                                      for e in node.args[0].elts], ast.Load())
        if isinstance(node.func, ast.Attribute) and node.func.attr == "astype" and node.args:
            name = self._path(node.args[0]) or (node.args[0].value if isinstance(node.args[0], ast.Constant) else "")
            if name in _FLOAT_TYPES or name in ("float64", "f8"):
                node.args[0] = _expr("np.float64")
            elif name in _INT_TYPES or name in ("int64", "i8"):
                node.args[0] = _expr("np.int64")
        if path == "numpy.clip" and len(node.args) == 3:
            # Numba's clip takes no None: one bound is a maximum or a minimum
            x, lo, hi = node.args
            if isinstance(hi, ast.Constant) and hi.value is None:
                return ast.copy_location(_expr(f"np.maximum({ast.unparse(x)}, {ast.unparse(lo)})"), node)
            if isinstance(lo, ast.Constant) and lo.value is None:
                return ast.copy_location(_expr(f"np.minimum({ast.unparse(x)}, {ast.unparse(hi)})"), node)
        if path in ("sum", "min", "max", "any", "all") and node.args and isinstance(node.args[0], ast.GeneratorExp):
            g = node.args[0]
            node.args[0] = ast.ListComp(g.elt, g.generators)
        if path == "round" and len(node.args) + len(node.keywords) > 1:
            raise ModelError("round(x, digits) of compiled code rounds as numpy does, not as Python: "
                             "write np.round(x, digits), which both compute alike")
        if path in _REWRITES and (node.args or node.keywords):
            args = [ast.unparse(a) for a in node.args]
            kw = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
            if not args:
                args = [kw.pop("x")] if "x" in kw else []
            return ast.copy_location(_expr(_REWRITES[path](args, kw)), node)
        return node


def _float_lists(fn):
    """Integer constants made float in the lists a function fills or returns: Numba types a
    list by its first values."""
    filled = set()
    for n in ast.walk(fn):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AugAssign) else []
        filled |= {t.value.id for t in targets if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)}
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Name):
            filled.add(n.value.id)
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id in filled:
            _floats(n.value)
        elif isinstance(n, ast.Return) and isinstance(n.value, (ast.List, ast.Tuple)):
            _floats(n.value)


def _floats(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        for side in (node.left, node.right):
            _floats(side)
    elif isinstance(node, ast.ListComp) and isinstance(node.elt, ast.Constant) and type(node.elt.value) is int:
        node.elt = ast.Constant(float(node.elt.value))
    elif isinstance(node, (ast.List, ast.Tuple)):
        for i, e in enumerate(node.elts):
            if isinstance(e, ast.Constant) and type(e.value) is int:
                node.elts[i] = ast.Constant(float(e.value))
            elif isinstance(e, ast.UnaryOp) and isinstance(e.operand, ast.Constant) and type(e.operand.value) is int:
                e.operand = ast.Constant(float(e.operand.value))


def _reads_time(fn) -> bool:
    """False only when the code of fn, inner functions included, never names its first argument:
    the model is then autonomous and may reach a steady state."""
    code = getattr(fn, "__code__", None)
    if code is None or code.co_argcount < 1:
        return True
    name = code.co_varnames[0]

    def names(c):
        # Python 3.13 loads or stores two names in one instruction, LOAD_FAST_LOAD_FAST: a tuple
        if any(name in (i.argval if isinstance(i.argval, tuple) else (i.argval,))
               for i in dis.get_instructions(c) if i.opname.startswith(("LOAD", "STORE", "DELETE"))):
            return True
        return any(names(k) for k in c.co_consts if isinstance(k, types.CodeType))
    return names(code)


def _time_switches(tree, time) -> list:
    """The times where a condition of the code on its time switches: in each comparison, chained
    ones included, time alone or plus or minus an expression on one side, numbers, values of p
    and calls on both, written in the equation language. A name assigned once in its function
    reads as its value."""
    out = []
    for scope in [tree, *(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))]:
        known = _assigned_once(scope) if scope is not tree else {}
        for node in ast.walk(scope):
            if isinstance(node, ast.Compare):
                terms = [node.left, *node.comparators]
                for a, b in zip(terms, terms[1:]):
                    for side, other in ((a, b), (b, a)):
                        if (text := _switch(side, other, time, known)) is not None:
                            out.append(text)
    return list(dict.fromkeys(out))


def _assigned_once(scope) -> dict:
    """The value of each name the scope assigns once, by a plain assignment."""
    stores = Counter(n.id for n in ast.walk(scope)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    return {a.targets[0].id: a.value for a in ast.walk(scope)
            if isinstance(a, ast.Assign) and len(a.targets) == 1
            and isinstance(a.targets[0], ast.Name) and stores[a.targets[0].id] == 1}


def _switch(side, other, time, known):
    """The time at which side, holding time, equals other: t, t + x, x + t, t - x or x - t."""
    if not (bound := _p_text(other, known)):
        return None
    for _ in range(20):
        if not (isinstance(side, ast.Name) and side.id != time and side.id in known):
            break
        side = known[side.id]
    if isinstance(side, ast.Name) and side.id == time:
        return bound
    if isinstance(side, ast.BinOp) and isinstance(side.op, (ast.Add, ast.Sub)):
        add = isinstance(side.op, ast.Add)
        if isinstance(side.left, ast.Name) and side.left.id == time and (x := _p_text(side.right, known)):
            return f"({bound}) {'-' if add else '+'} ({x})"
        if isinstance(side.right, ast.Name) and side.right.id == time and (x := _p_text(side.left, known)):
            return f"({bound}) - ({x})" if add else f"({x}) - ({bound})"
    return None


def _p_text(node, known=None, depth=0):
    """p['x'], p["x"] and p.x read as x in an expression of numbers and calls, a name of known
    as its value; None for anything else."""
    known = known or {}
    node = copy.deepcopy(node)

    class Names(ast.NodeTransformer):
        def visit_Subscript(self, n):
            if isinstance(n.value, ast.Name) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                return ast.Name(n.slice.value, ast.Load())
            return n

        def visit_Attribute(self, n):
            return ast.Name(n.attr, ast.Load()) if isinstance(n.value, ast.Name) else n

        def visit_Name(self, n):
            if n.id in known and depth < 20 and (v := _p_text(known[n.id], known, depth + 1)):
                return ast.parse(f"({v})", mode="eval").body
            return n

    node = Names().visit(node)
    allowed = (ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.operator, ast.unaryop, ast.Call)
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            continue
        if not isinstance(n, allowed):
            return None
    return ast.unparse(node)


_MARKERS = ("Unknown attribute", "Untyped global name", "No implementation", "Cannot unify",
            "Use of unsupported", "Invalid use", "NameError", "Cannot determine", "not supported",
            "reflected list")


def _short(e) -> str:
    """The lines of a Numba error that say what failed, without paths."""
    lines = [x.strip() for x in str(e).splitlines() if x.strip()]
    for i, line in enumerate(lines):
        if any(m in line for m in _MARKERS):
            text = " ".join(lines[i:i + 2])
            return re.sub(r"\(<[^>]*>\)|\S*/\S+\.(?:py|so)\b", "", text).strip()[:300]
    return f"{type(e).__name__}: {lines[0][:250] if lines else ''}"
