"""Models written as Python functions.

rhs(t, y, p) returns the derivatives of the states, observe(t, y, p) the outputs and the optional
init(p) (or y0, a function or a list) the initial values; p holds the parameters, covariates and
regressors by name, and the signals of HISTORY (model.py). The functions are run with symbols
in place of numbers, once per branch of each condition, and the result is written as the
equations of model.py, so the model is solved as any other. The equations are then compared with the code around the typical values, at the
numbers the code contains and on both sides of each condition. Valid code that cannot be written
as equations is compiled as written instead (native.py).

The code runs while it is read and checked, with the listed functions of numpy, math and scipy
as its only modules, a few builtins and a time limit. This guards against mistakes, not against
code written to escape: it is not a sandbox.
"""

from __future__ import annotations

import ast
import builtins
import contextvars
import ctypes
import difflib
import functools
import importlib
import inspect
import math
import os
import re
import textwrap
import threading
import time
import types
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanAtom, BooleanFalse, BooleanFunction, BooleanTrue
from sympy.printing.str import StrPrinter

from .model import (FUNCTIONS, HISTORY, Model, ModelError, T, canonical_input, expm1, hazard_map,
                    log1p, normcdf, reserved)

MAX_PATHS = 64  # runs of one function, one per combination of the conditions met
MAX_NODES = 20000  # size of one expression written out, beyond which it is not written
READ_SECONDS = 60  # time allowed to read the code, and again to compile it as written
MIN_POINTS = 10  # points at which the check compares each function at least
PRECISE_DIGITS = 50  # digits of the arithmetic that settles a difference found in floats
NARROW_ROUNDS = 6  # rounds of points drawn closer to the values given, spread halved each
_MAX_EXPONENT = 10000  # integer powers beyond this take too long to compute
_SEED = 20261013


class Untranslatable(ModelError):
    """Valid code that cannot be written as equations; from_python then compiles it as written.
    role names the function at fault (rhs, observe or init) when it is known."""

    def __init__(self, message, role=None):
        super().__init__(message)
        self.role = role


class CodeFails(ModelError):
    """The code itself raises, gives no finite value or does not end, where it is run."""


class TimeUp(CodeFails):
    """The code ran longer than the time allowed."""


class _Guard(Exception):
    """An exception the code raises itself (raise, assert): its branch has no value."""

    def __init__(self, exc):
        super().__init__(exc)
        self.exc = exc


class _Expired(BaseException):
    """Raised in the reading thread when its time is up; not an Exception, so that code catching
    its own errors does not catch it."""


# ---------- entry point ----------

def from_python(code=None, *, rhs=None, observe=None, init=None, states=(), parameters=(),
                outputs=None, covariates=(), regressors=(), interpolated=(), inputs=None,
                t0=None, name="model", values=None, check=True, fallback=True,
                hazards=None) -> Model:
    """Model from Python functions, given as source text, a path to a .py file or a module
    (code), or as functions defined with def at module level (rhs, observe, init).

    states, parameters, covariates and regressors name what y and p hold; outputs names what
    observe returns (the keys of the dict it returns, or out1, out2... by default); inputs maps
    an input to its target state or to {"target", "delay", "scale", "duration"}, written with
    the names of p. values gives typical values of the names of p, around which the equations
    are compared with the code. hazards maps an output to its kind, as in Model. Valid code that
    cannot be written as equations is compiled as written, with a warning, unless fallback is
    False; a model with hazards must be read as equations.
    """
    source = _source(code, rhs, observe, init)
    tree = parse(source)
    spec = _spec(states, parameters, outputs, covariates, regressors, interpolated, inputs, t0,
                 name, hazards)
    spec["values_given"] = bool(values)
    values = _values(values, spec)
    try:
        return within(READ_SECONDS, Untranslatable, lambda: _read_model(tree, spec, values, check))
    except Untranslatable as e:
        reason = e
    except ModelError:
        raise
    except Exception as e:  # a failure of the reader itself: the code compiled as written is right
        reason = Untranslatable(f"the reader failed ({type(e).__name__}: {e}): please report it")
    if not fallback:
        raise reason
    from .native import native_model
    try:
        model = within(READ_SECONDS, TimeUp, lambda: native_model(source, tree, spec, values, reason))
    except CodeFails:
        raise
    except Exception as e:
        cause = e if isinstance(e, ModelError) else f"{type(e).__name__}: {e}"
        raise ModelError(f"{reason}; compiling the code as written failed too: {cause}") from None
    warnings.warn(f"pysaem compiles this model as written, without equations: {reason}",
                  stacklevel=2)
    return model


def _read_model(tree, spec, values, check) -> Model:
    model = _equations(tree, spec)
    model.reading = {"form": "equations",
                     **(_check(model, tree, spec, values) if check else {"checked_points": 0})}
    return model


def _source(code, rhs, observe, init) -> str:
    functions = {"rhs": rhs, "observe": observe, "init": init}
    if code is None:
        return _sources(**functions)
    if any(f is not None for f in functions.values()):
        raise ModelError("give the code, or the functions rhs, observe and init, not both")
    if callable(code):
        raise ModelError("give the functions by name: from_python(rhs=..., observe=...)")
    if isinstance(code, types.ModuleType):
        try:
            return inspect.getsource(code)
        except (OSError, TypeError):
            raise ModelError(f"the source of the module {code.__name__} cannot be read") from None
    if isinstance(code, os.PathLike) or (isinstance(code, str) and "\n" not in code
                                         and code.strip().endswith(".py")):
        try:
            return Path(code).read_text()
        except OSError as e:
            raise ModelError(f"cannot read {code}: {e.strerror}") from None
    if isinstance(code, str):
        return code
    raise ModelError("code is the text of the model, a path to a .py file or a module")


_P_NAME = re.compile(r"""\bp\s*\[\s*(['"])(\w+)\1\s*\]|\bp\.(\w+)\b""")


def _spec(states, parameters, outputs, covariates, regressors, interpolated, inputs, t0, name,
          hazards):
    def names(v, what):
        v = [v] if isinstance(v, str) else list(v or [])
        bad = [x for x in v if not isinstance(x, str)]
        if bad:
            raise ModelError(f"{what}: names are strings, got {bad}")
        return v

    if not isinstance(inputs or {}, dict):
        raise ModelError("inputs maps each input to its target state, as {'input': 'X'}")
    spec = {"states": names(states, "states"), "parameters": names(parameters, "parameters"),
            "outputs": None if outputs is None else names(outputs, "outputs"),
            "covariates": names(covariates, "covariates"),
            "regressors": names(regressors, "regressors"),
            "interpolated": names(interpolated, "interpolated"), "t0": t0, "name": name,
            "inputs": {}, "hazards": hazard_map(hazards)}
    for k, v in (inputs or {}).items():
        v = canonical_input(k, v)
        tg = v.get("target")
        if not (isinstance(tg, str) or isinstance(tg, list) and tg and all(isinstance(x, str) for x in tg)):
            raise ModelError(f"input {k!r}: its target is a state name or a list of them (the "
                             f"amount split among them by their scales), got {tg!r}")
        for key in ("delay", "scale", "duration"):
            # written as in the functions: p['lag'] reads lag
            if isinstance(v.get(key), str):
                v[key] = _P_NAME.sub(lambda m: m[2] or m[3], v[key])
            elif isinstance(v.get(key), list):
                v[key] = [_P_NAME.sub(lambda m: m[2] or m[3], x) if isinstance(x, str) else x
                          for x in v[key]]
        spec["inputs"][k] = v
    declared = spec["states"] + p_names(spec)
    for n in declared:
        if n == "t" or not n.isidentifier():
            raise ModelError(f"{n!r} cannot name a state or a value of p")
        if n in HISTORY:
            kind = next(k for k, v in (("state", "states"), ("parameter", "parameters"),
                                       ("covariate", "covariates"), ("regressor", "regressors"))
                        if n in spec[v])
            raise ModelError(reserved(kind, n))
    twice = sorted({n for n in declared if declared.count(n) > 1})
    if twice:
        raise ModelError(f"declared more than once: {twice}")
    return spec


def _near(name, names):
    """The declared name the user probably meant: the same in another case, or a close one."""
    same = [n for n in names if n.lower() == str(name).lower()]
    close = difflib.get_close_matches(str(name), [str(n) for n in names], 1)
    return (same or close or [None])[0]


def p_names(spec) -> list:
    """The names p holds that are declared, in the order used everywhere: parameters, covariates,
    regressors."""
    return spec["parameters"] + spec["covariates"] + spec["regressors"]


def p_held(spec) -> list:
    """Every name p holds: those declared, then the signals of HISTORY."""
    return p_names(spec) + list(HISTORY)


def _values(values, spec) -> dict:
    """Typical values of the names of p, 1 when not given."""
    values = dict(values or {})
    declared = p_held(spec)
    unknown = [k for k in values if k not in declared]
    if unknown:
        hints = []
        for k in unknown:
            near = _near(k, declared)
            hints.append(f"{k!r}" + (f" (did you mean {near!r}?)" if near else ""))
        raise ModelError(f"values: unknown name(s) {', '.join(hints)}; p holds {declared}")
    out = {}
    for n in declared:
        v = values.get(n, 1.0)
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) \
                or not math.isfinite(v):
            raise ModelError(f"values: {n} must be a finite number, got {v!r}")
        out[n] = float(v)
    return out


# ---------- source and safety ----------

def _sources(**roles) -> str:
    """Source of the given functions, of the functions of their module they call and of the
    constants and modules they read; a function is bound to its role (rhs, observe, init)."""
    roles = {k: f for k, f in roles.items() if f is not None}
    if not roles:
        raise ModelError("give the code as text, a path or a module, or the functions rhs and observe")
    done, parts, imports, constants, binds = set(), [], {}, {}, []
    queue = list(roles.items())
    while queue:
        role, f = queue.pop(0)
        f = inspect.unwrap(getattr(f, "py_func", f))
        if not inspect.isfunction(f) or f.__name__ == "<lambda>":
            raise ModelError(f"{role}: give a function defined with def at module level, or the "
                             "code as text")
        if role in ("rhs", "observe", "init") and f.__name__ != role:
            binds.append(f"{role} = {f.__name__}")
        if f.__name__ in done:
            continue
        done.add(f.__name__)
        if f.__code__.co_freevars:
            raise ModelError(f"{f.__name__} reads {list(f.__code__.co_freevars)} from an enclosing "
                             "function: make them values of p or constants of the module")
        try:
            node = ast.parse(textwrap.dedent(inspect.getsource(f))).body[0]
        except (OSError, TypeError, SyntaxError, IndexError):
            raise ModelError(f"the source of {f.__name__} cannot be read (defined interactively?): "
                             "give the code as text") from None
        node.decorator_list = []
        parts.append(ast.unparse(node))
        for n in sorted(_code_names(f.__code__)):
            if n not in f.__globals__:
                continue
            v = f.__globals__[n]
            v = inspect.unwrap(getattr(v, "py_func", v)) if callable(v) else v
            if inspect.isfunction(v):
                if v.__module__ != f.__module__:
                    raise ModelError(f"{f.__name__} calls {n} from {v.__module__}: define it in the "
                                     "same module, or give the code as text")
                queue.append((n, v))
            elif isinstance(v, types.ModuleType):
                imports[n] = f"import {v.__name__}" + (f" as {n}" if v.__name__ != n else "")
            elif isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
                imports["_pysaem_np"] = "import numpy as _pysaem_np"
                constants[n] = f"{n} = _pysaem_np.array({v.tolist()!r})"
            elif isinstance(v, float) and not math.isfinite(v):
                constants[n] = f"{n} = float({str(v)!r})"
            else:
                try:
                    same = ast.literal_eval(repr(v)) == v
                except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                    same = False
                if same:
                    constants[n] = f"{n} = {v!r}"
    head = "\n".join([*imports.values(), *constants.values()])
    return head + "\n\n" + "\n\n".join(parts) + "\n\n" + "\n".join(dict.fromkeys(binds)) + "\n"


def _code_names(code) -> set:
    """Global names a function reads, its comprehensions and inner functions included."""
    out = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out |= _code_names(c)
    return out


_ROOTS = ("numpy", "math", "scipy", "typing", "numba")
_FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "input", "getattr", "setattr", "delattr",
                    "globals", "locals", "vars", "breakpoint", "exit", "quit", "help", "dir",
                    "memoryview", "type", "super", "object", "classmethod", "staticmethod",
                    "property", "id", "hash", "BaseException"}
# attributes of any object that write files, reach memory or walk the frames of the interpreter
_FORBIDDEN_ATTRS = {"tofile", "dump", "dumps", "ctypes", "format", "format_map", "mro",
                    "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "ag_frame",
                    "ag_code", "f_back", "f_globals", "f_locals", "f_builtins", "f_code",
                    "f_trace", "tb_frame", "tb_next", "co_code", "co_consts", "func_globals",
                    "func_code", "im_func", "im_self"}
_REFUSED_NODES = {ast.ClassDef: "a class", ast.AsyncFunctionDef: "async", ast.AsyncWith: "async",
                  ast.AsyncFor: "async", ast.Await: "await", ast.Yield: "yield",
                  ast.YieldFrom: "yield", ast.Global: "global (a model keeps no state between calls)",
                  ast.Nonlocal: "nonlocal"}


def parse(source: str) -> ast.Module:
    """The code parsed, without what a module file carries besides the model (main block,
    annotations), after refusing what a model never needs: other modules, introspection,
    classes, catching every error."""
    try:
        return _parse(source)
    except ModelError:
        raise
    except SyntaxError as e:
        raise ModelError(f"the code cannot be parsed: line {e.lineno}: {e.msg}") from None
    except (ValueError, RecursionError, MemoryError) as e:
        raise ModelError(f"the code cannot be parsed: {type(e).__name__}: {e}") from None


def _parse(source: str) -> ast.Module:
    tree = ast.parse(textwrap.dedent(source))
    tree.body = [n for n in tree.body if not _is_main_block(n)
                 and not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
    tree = ast.fix_missing_locations(_Annotations().visit(tree))
    for node in ast.walk(tree):
        where = f"line {getattr(node, 'lineno', '?')}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in modules:
                if getattr(node, "level", 0) or m.split(".")[0] not in _ROOTS:
                    raise ModelError(f"{where}: import of {m or '.'!r}; a model imports numpy, math "
                                     "and scipy only")
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                raise ModelError(f"{where}: from {node.module} import *: import the module, as "
                                 "import numpy as np")
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
                raise ModelError(f"{where}: {node.id} cannot be used in a model")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in _FORBIDDEN_ATTRS:
                raise ModelError(f"{where}: .{node.attr} cannot be used in a model")
        elif type(node) in _REFUSED_NODES:
            raise ModelError(f"{where}: {_REFUSED_NODES[type(node)]} cannot be used in a model")
        elif isinstance(node, ast.With):
            if not all(isinstance(i.context_expr, ast.Call) and i.optional_vars is None
                       and chain(i.context_expr.func)[-1:] == ["errstate"] for i in node.items):
                raise ModelError(f"{where}: with cannot be used in a model, except with np.errstate(...)")
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            raise ModelError(f"{where}: catch errors with except Exception:, not with a bare except:")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exp = _constant(node.left), _constant(node.right)
            if base is not None and exp is not None and abs(base) > 1 and exp > _MAX_EXPONENT:
                raise ModelError(f"{where}: an integer power too large to compute")
    return tree


def _is_main_block(node) -> bool:
    if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
        return False
    sides = [node.test.left, *node.test.comparators]
    return any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)


class _Annotations(ast.NodeTransformer):
    """Type annotations removed: they say nothing of the model."""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.returns = None
        a = node.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
            if arg is not None:
                arg.annotation = None
        return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return ast.copy_location(ast.Pass(), node)
        return ast.copy_location(ast.Assign([node.target], node.value), node)


def chain(node) -> list:
    """['np', 'random', 'normal'] for np.random.normal; [] when the root is not a name."""
    out = []
    while isinstance(node, ast.Attribute):
        out.append(node.attr)
        node = node.value
    return [node.id, *out[::-1]] if isinstance(node, ast.Name) else []


def _constant(node):
    """Value of an expression of integer constants, None when it is not one or too large."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _constant(node.operand)
        return None if v is None else -v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)):
        a, b = _constant(node.left), _constant(node.right)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Pow):
            return a**b if 0 <= b <= 64 else None
        return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}[type(node.op)]
    return None


def within(seconds, error, fn):
    """fn() with a time limit. When the time is up an exception is raised in this thread by a
    timer thread, which works off the main thread and leaves signals alone, and error is raised
    in its place. A single long operation in C is not interrupted."""
    ident, lock, state = threading.get_ident(), threading.Lock(), {"on": True}

    def expire():
        with lock:
            if state["on"]:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident),
                                                           ctypes.py_object(_Expired))

    def stop():
        with lock:
            state["on"] = False
            # an exception set but not raised yet is cleared
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), None)
        timer.cancel()

    timer = threading.Timer(seconds, expire)
    timer.daemon = True
    timer.start()
    try:
        try:
            return fn()
        finally:
            stop()
    except _Expired:
        stop()
        raise error(f"the code runs for more than {seconds} s (a loop that never ends, or "
                    "expressions that grow at each step?)") from None


# ---------- modules ----------

class Module:
    """A module as the code sees it: its listed names; any other name gives an object that
    refuses to be used, so that an import left unused does not stop the reading."""

    def __init__(self, path, attrs, refusal, real=None):
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_attrs", attrs)
        object.__setattr__(self, "_refusal", refusal)
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = f"{self._path}.{name}"
        if sub in _MODULES:
            return _module(sub, self._refusal is _symbolic_refusal)
        if name in self._attrs:
            return self._attrs[name]
        if self._refusal is _symbolic_refusal and self._path not in _REFUSALS:
            return _Deferred(sub, self._refusal)
        return _Refused(sub, _REFUSALS.get(self._path, self._refusal))

    def __setattr__(self, name, value):
        raise ModelError("a module cannot be changed in a model")


def real_of(module: Module):
    """The module behind a Module, for Numba only: never an attribute the code can read."""
    return object.__getattribute__(module, "_real")


class _Refused:
    """A name of a module that the code may import but not use."""

    def __init__(self, what, refusal):
        self._what, self._refusal = what, refusal

    def __call__(self, *a, **k):
        raise self._refusal(self._what)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Refused(f"{self._what}.{name}", self._refusal)

    def _refuse(self, *a):
        raise self._refusal(self._what)


class _Deferred(_Refused):
    """A name without symbolic stand-in: called with numbers only, the real function answers;
    called with values that depend on the symbols, it cannot be written as equations."""

    def __call__(self, *a, **k):
        if _symbolic_in([*a, *k.values()]):
            raise self._refusal(self._what)
        root, *parts = self._what.split(".")
        found = _module(root, False)
        for part in parts:
            found = getattr(found, part)
        return found(*[_plain(x) for x in a], **{key: _plain(v) for key, v in k.items()})

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Deferred(f"{self._what}.{name}", self._refusal)


for _op in ("add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "radd", "rsub", "rmul",
            "rtruediv", "rfloordiv", "rmod", "rpow", "neg", "pos", "abs", "float", "int", "index",
            "getitem", "iter", "len", "lt", "le", "gt", "ge", "bool"):
    setattr(_Refused, f"__{_op}__", _Refused._refuse)


def _symbolic_refusal(what):
    return Untranslatable(f"{what} cannot be written as equations")


def _numeric_refusal(what):
    return ModelError(f"{what} cannot be used in a model (pysaem allows the listed functions of "
                      "numpy, math and scipy)")


def _random_refusal(what):
    return ModelError("random draws cannot be used in a model: its predictions must be fixed")


def _solver_refusal(what):
    return ModelError("an ODE solver called inside the model: pysaem integrates rhs itself; "
                      "return the derivatives from rhs")


_REFUSALS = {"numpy.random": _random_refusal, "scipy.integrate": _solver_refusal}


@functools.lru_cache(maxsize=None)
def _module(path: str, symbolic: bool) -> Module:
    refusal = _symbolic_refusal if symbolic else _numeric_refusal
    if path in _REFUSALS:
        return Module(path, {}, _REFUSALS[path])
    if path == "numba":
        return Module(path, _NUMBA, refusal, None if symbolic else _real(path))
    if symbolic:
        return Module(path, _SYMBOLIC[path](), refusal)
    real = _real(path)
    names = _NUMERIC[path]() if callable(_NUMERIC[path]) else _NUMERIC[path]
    return Module(path, {k: getattr(real, k) for k in names if hasattr(real, k)}, refusal, real)


def _real(path):
    if path == "scipy.stats.norm":
        return importlib.import_module("scipy.stats").norm
    if path == "typing":
        return None
    return importlib.import_module(path)


def _decorator(*a, **k):
    """njit and jit of numba: the code is read as Python."""
    if len(a) == 1 and callable(a[0]) and not k:
        return a[0]
    return lambda f: f


_NUMBA = {"njit": _decorator, "jit": _decorator, "prange": range}


def _importer(symbolic):
    def imp(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level or root not in _ROOTS:
            raise ModelError(f"import of {name!r}; a model imports numpy, math and scipy only")
        if not fromlist:
            return _module(root, symbolic)
        found = _module(root, symbolic)
        for part in name.split(".")[1:]:
            found = getattr(found, part)
        return found
    return imp


_BUILTINS = ("abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int",
             "isinstance", "iter", "len", "list", "map", "max", "min", "next", "pow", "range",
             "reversed", "round", "set", "slice", "sorted", "sum", "tuple", "zip", "str", "repr",
             "callable", "Exception", "ValueError", "ArithmeticError", "ZeroDivisionError",
             "RuntimeError", "AssertionError", "KeyError", "IndexError", "TypeError",
             "OverflowError", "FloatingPointError", "NotImplementedError", "StopIteration",
             "LookupError")


def namespace(tree: ast.Module, symbolic: bool) -> dict:
    """The module run once: with the symbolic stand-ins of the listed functions for reading,
    with the real ones for the check."""
    if symbolic:
        try:
            tree = ast.fix_missing_locations(_Rewrite(_module_names(tree)).visit(ast.parse(ast.unparse(tree))))
        except RecursionError:
            raise Untranslatable("the code nests too deeply to be read") from None
    safe = {k: getattr(builtins, k) for k in _BUILTINS}
    safe.update(print=lambda *a, **k: None, __import__=_importer(symbolic))
    # np and math at hand, as in most model code; an import of the code takes precedence
    ns = {"__builtins__": safe, "__name__": "model", "np": _module("numpy", symbolic),
          "math": _module("math", symbolic)}
    if symbolic:
        safe.update(max=_max, min=_min, any=_any, all=_all, sum=_sum, abs=_abs, pow=_pow,
                    divmod=_divmod, isinstance=_isinstance, float=_float, str=_str, repr=_repr,
                    set=_set, frozenset=_set, filter=_filter)
        ns.update(__cmp=_cmp, __and=_and, __or=_or, __sand=_sand, __sor=_sor, __vand=_vand,
                  __vor=_vor, __not=_not, __truth=_truth, __lookup=_lookup, __get=_get,
                  __int=_int, __bool=_bool, __round=_round, __floordiv=_floordiv, __pow=_pow,
                  __arith=_arith, __set=_set, __item=_item, __guard=_Guard, __refuse=_refuse)
    try:
        exec(compile(tree, "<model>", "exec"), ns)  # noqa: S102 - checked by parse, see the module doc
    except ModelError:
        raise
    except Exception as e:
        # with the stand-ins, a failure may be the reader's; with the real functions, the code's
        error = Untranslatable if symbolic else CodeFails
        raise error(f"the code fails when loaded: {type(e).__name__}: {e}") from None
    return ns


def roles(ns) -> dict:
    """rhs, observe and init of the code; init may be named y0, and be a list of values."""
    out = {}
    for role, names in (("rhs", ("rhs",)), ("observe", ("observe",)), ("init", ("init", "y0"))):
        f = next((ns[n] for n in names if n in ns), None)
        if f is None or isinstance(f, (Module, _Refused)):
            out[role] = None
        elif callable(f):
            out[role] = f
        elif role == "init" and isinstance(f, (list, tuple, np.ndarray, int, float)):
            values = list(np.ravel(np.asarray(f, dtype=object)))
            out[role] = lambda p, _v=values: list(_v)
        else:
            raise ModelError(f"{role} is not a function")
    return out


def require(fns, spec, ns):
    """The functions the declarations call for, with the arguments they take."""
    defined = sorted(k for k, v in ns.items() if inspect.isfunction(v) and not k.startswith("__"))
    if fns["observe"] is None:
        raise ModelError("the code defines no observe(t, y, p)"
                         + (f" (it defines {', '.join(defined)})" if defined else ""))
    if spec["states"] and fns["rhs"] is None:
        raise ModelError(f"the code defines no rhs(t, y, p) for the states {spec['states']}")
    for role, f in fns.items():
        if f is None or getattr(f, "__name__", "") == "<lambda>" and role == "init":
            continue
        try:
            sig = inspect.signature(f)
        except (TypeError, ValueError):
            continue
        kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        params = [q for q in sig.parameters.values() if q.kind in kinds]
        required = [q for q in params if q.default is q.empty]
        star = any(q.kind == q.VAR_POSITIONAL for q in sig.parameters.values())
        want = 1 if role == "init" else 3
        wrong_order = role != "init" and len(params) > 1 and params[1].name in ("t", "time")
        if wrong_order or not star and (len(required) > want or len(params) < want):
            raise ModelError(f"{role} takes ({'p' if role == 'init' else 't, y, p'}), found "
                             f"{role}({', '.join(sig.parameters)})"
                             + (": t comes first" if wrong_order else ""))


# symbolic stand-ins: functions of numbers that accept symbols

def _issym(v):
    return isinstance(v, _Cond) or (isinstance(v, sp.Basic) and not v.is_number)


def _is_condition(v):
    return isinstance(v, (Relational, BooleanFunction, BooleanAtom))


def _num(v):
    if isinstance(v, _Cond):
        return v.num()
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    return v


def _rel(v):
    """The condition a value stands for: itself, or that the number is not zero."""
    if isinstance(v, _Cond):
        return v.rel
    if _is_condition(v):
        return v
    if isinstance(v, sp.Basic):
        return sp.Ne(v, 0, evaluate=False) if not v.is_number else (sp.true if v != 0 else sp.false)
    return sp.true if v else sp.false


def _obj(a):
    """An array of objects (lists and numbers included)."""
    if isinstance(a, np.ndarray):
        return a if a.dtype == object else a.astype(object)
    return np.array(a, dtype=object)


def _plain(a):
    """Numbers as numpy holds them: arrays of objects without symbols become arrays of floats."""
    if isinstance(a, (list, tuple)):
        a = _obj(a)
    if isinstance(a, np.ndarray) and a.dtype == object:
        if any(_issym(v) for v in a.ravel()):
            return a
        try:
            return a.astype(float)
        except (TypeError, ValueError):
            return a
    if isinstance(a, sp.Basic) and a.is_number:
        return float(a) if a.is_real else complex(a)
    return a


def _symbolic_in(args):
    for x in args:
        if isinstance(x, (list, tuple)):
            x = _obj(x)
        if isinstance(x, np.ndarray):
            if x.dtype == object and any(_issym(v) for v in x.ravel()):
                return True
        elif _issym(x):
            return True
    return False


_UFUNC_DEFAULTS = {"out": None, "where": True, "casting": "same_kind", "order": "K", "subok": True}


def _keywords(name, k):
    for key, v in k.items():
        default = _UFUNC_DEFAULTS.get(key, ...)
        if key != "dtype" and not (v is default or (isinstance(v, str) and v == default)):
            raise Untranslatable(f"{name} with {key}= cannot be written as equations")


def _ew(f, name):
    """A numpy function: elementwise over arrays, the real function on numbers; where= keeps
    the values of out= where it is false."""
    real = getattr(np, name, None)

    def g(*a, **k):
        a = [_obj(x) if isinstance(x, (list, tuple)) else x for x in a]
        where, out = k.pop("where", True), k.pop("out", None)
        out = out[0] if isinstance(out, tuple) and len(out) == 1 else out
        if not _symbolic_in([*a, where, out]) and real is not None:
            kw = {**k, **({} if out is None else {"out": out}), **({} if where is True else {"where": where})}
            return real(*[_plain(x) for x in a], **kw)
        _keywords(f"numpy.{name}", k)
        if any(isinstance(x, np.ndarray) for x in a):
            value = np.vectorize(lambda *xs: f(*map(_num, xs)), otypes=[object])(*a)
        else:
            value = f(*map(_num, a))
        if where is not True:
            if not isinstance(out, np.ndarray):
                raise Untranslatable(f"numpy.{name} with where= but no out= leaves values undefined")
            value = _where(where, value, out.copy())
        if isinstance(out, np.ndarray):
            out[...] = value
            return out
        return value
    return g


def _scalar(f, name):
    """A math function: numbers only, the real function on numbers."""
    real = getattr(math, name, None)

    def g(*a):
        a = [x.item() if isinstance(x, np.ndarray) and x.size == 1 else x for x in a]
        if any(isinstance(x, (np.ndarray, list, tuple)) for x in a):
            raise TypeError(f"math.{name} takes numbers, not arrays")
        if not any(_issym(x) for x in a) and real is not None:
            return real(*[_plain(x) for x in a])
        return f(*map(_num, a))
    return g


def _log(x, base=None):
    return sp.log(x) if base is None else sp.log(x) / sp.log(base)


def _trunc(x):
    return sp.sign(x) * sp.floor(sp.Abs(x))


def _fmod(a, b):
    # C fmod keeps the sign of a
    return sp.sign(a) * sp.Mod(sp.Abs(a), sp.Abs(b))


def _rint(x):
    """Round half to even, as round() and np.rint."""
    return sp.floor(x + sp.Rational(1, 2)) - sp.Piecewise(
        (1, sp.Eq(sp.Mod(x, 2), sp.Rational(1, 2), evaluate=False)), (0, True))


def _floordiv(a, b):
    if not (_issym(a) or _issym(b)):
        return a // b
    a, b = _num(a), _num(b)
    # Python's a // b for floats is the integer nearest to (a - a % b) / b
    return sp.floor((a - sp.Mod(a, b)) / b + sp.Rational(1, 2))


def _divmod(a, b):
    if not (_issym(a) or _issym(b)):
        return divmod(a, b)
    return _floordiv(a, b), sp.Mod(_num(a), _num(b))


def _pow(a, b, mod=None):
    if mod is not None:
        return pow(a, b, mod)
    if isinstance(a, int) and isinstance(b, int) and abs(a) > 1 and b > _MAX_EXPONENT:
        raise ModelError("an integer power too large to compute")
    return _num(a) ** _num(b)


_FUNCS = {
    "exp": sp.exp, "log": _log, "log10": lambda x: sp.log(x) / sp.log(10),
    "log2": lambda x: sp.log(x) / sp.log(2), "log1p": log1p, "expm1": expm1,
    "exp2": lambda x: 2**x, "sqrt": sp.sqrt,
    "cbrt": lambda x: sp.sign(x) * sp.Abs(x) ** sp.Rational(1, 3), "square": lambda x: x**2,
    "power": lambda a, b: a**b, "pow": lambda a, b: a**b, "float_power": lambda a, b: a**b,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sinh": sp.sinh, "cosh": sp.cosh,
    "tanh": sp.tanh, "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "arcsinh": sp.asinh, "arccosh": sp.acosh, "arctanh": sp.atanh, "asin": sp.asin,
    "acos": sp.acos, "atan": sp.atan, "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "arctan2": sp.atan2, "atan2": sp.atan2, "hypot": lambda a, b: sp.sqrt(a**2 + b**2),
    "abs": sp.Abs, "absolute": sp.Abs, "fabs": sp.Abs, "sign": sp.sign,
    "maximum": sp.Max, "minimum": sp.Min, "fmax": sp.Max, "fmin": sp.Min, "floor": sp.floor,
    "ceil": sp.ceiling, "trunc": _trunc, "fix": _trunc, "rint": _rint, "mod": sp.Mod,
    "remainder": sp.Mod, "fmod": _fmod, "floor_divide": _floordiv, "erf": sp.erf,
    "erfc": sp.erfc, "gamma": sp.gamma, "lgamma": sp.loggamma,
    "logaddexp": lambda a, b: sp.Max(a, b) + sp.log(1 + sp.exp(-sp.Abs(a - b))),
    "deg2rad": lambda x: x * sp.pi / 180, "rad2deg": lambda x: x * 180 / sp.pi,
    "radians": lambda x: x * sp.pi / 180, "degrees": lambda x: x * 180 / sp.pi,
    "add": lambda a, b: a + b, "subtract": lambda a, b: a - b, "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b, "true_divide": lambda a, b: a / b, "negative": lambda a: -a,
    "positive": lambda a: a, "reciprocal": lambda a: 1 / a,
}


def _where(c, a, b):
    if any(isinstance(v, (np.ndarray, list, tuple)) for v in (c, a, b)):
        return np.vectorize(_where, otypes=[object])(*[_obj(v) if isinstance(v, (list, tuple)) else v
                                                       for v in (c, a, b)])
    if _issym(c) or _is_condition(c):
        return sp.Piecewise((_num(a), _rel(c)), (_num(b), True))
    return a if c else b


def _select(condlist, choicelist, default=0):
    out = default
    for c, v in reversed(list(zip(condlist, choicelist))):
        out = _where(c, v, out)
    return out


def _np_piecewise(x, condlist, funclist, *args, **kw):
    if isinstance(condlist, _Cond) or not isinstance(condlist, (list, tuple)):
        condlist = [condlist]
    values = [f(x, *args, **kw) if callable(f) else f for f in funclist]
    out = values[len(condlist)] if len(values) > len(condlist) else 0
    # numpy writes the functions in order: the last true condition wins
    for c, v in zip(condlist, values):
        out = _where(c, v, out)
    return out


def _clip(a, a_min=None, a_max=None, out=None, **k):
    a_min, a_max = k.pop("min", a_min), k.pop("max", a_max)
    _keywords("numpy.clip", {**k, "out": out})
    if not _symbolic_in([a, a_min, a_max]):
        return np.clip(_plain(a), a_min, a_max)
    if isinstance(a, (np.ndarray, list, tuple)):
        return np.vectorize(lambda v: _clip(v, a_min, a_max), otypes=[object])(_obj(a))
    a = _num(a)
    if a_min is not None:
        a = sp.Max(a, _num(a_min))
    return sp.Min(a, _num(a_max)) if a_max is not None else a


def _sinc(x):
    return sp.Piecewise((1, sp.Eq(x, 0, evaluate=False)), (sp.sin(sp.pi * x) / (sp.pi * x), True))


def _heaviside(x, h0):
    return sp.Piecewise((0, x < 0), (h0, sp.Eq(x, 0, evaluate=False)), (1, True))


def _isscalar(x):
    return True if isinstance(x, (sp.Basic, _Cond)) else np.isscalar(x)


def _nan_test(name, module):
    real = getattr(module, name)

    def f(x, *a, **k):
        if _symbolic_in([_obj(x) if isinstance(x, (list, tuple)) else x]):
            raise Untranslatable(f"{name} of a value that depends on the symbols cannot be written "
                                 "as equations")
        return real(_plain(x), *a, **k)
    return f


def _filled(fill):
    def f(shape, dtype=None, order="C", **k):
        out = np.empty(shape, dtype=object)
        out[...] = int(fill) if dtype is int else float(fill)
        return out
    return f


def _filled_like(fill):
    def f(a, dtype=None, order="K", subok=True, shape=None, **k):
        return _filled(fill)(np.shape(a) if shape is None else shape, dtype)
    return f


def _full(shape, fill_value, dtype=None, order="C", **k):
    out = np.empty(shape, dtype=object)
    out[...] = fill_value
    return out


def _full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None, **k):
    return _full(np.shape(a) if shape is None else shape, fill_value)


def _array(a, dtype=None, copy=True, **k):
    # of objects, so that values depending on the symbols can be stored in it
    if isinstance(a, np.ndarray) and a.dtype != object:
        return np.array(a, dtype=dtype)
    return np.array(a, dtype=object)


def _asarray(a, dtype=None, **k):
    return a if isinstance(a, np.ndarray) else _array(a, dtype)


def _numify(arr):
    return np.vectorize(_num, otypes=[object])(arr) if arr.size else arr


def _reduction(real, name, neutral=None):
    """sum, prod, mean...: numpy itself over an array of objects; where= leaves out the
    elements where it is false."""
    def g(a, *args, **k):
        if k.pop("out", None) is not None:
            raise Untranslatable(f"numpy.{name} with out= cannot be written as equations")
        k.pop("dtype", None)
        where, initial = k.pop("where", True), k.pop("initial", None)
        arr = _obj(a)
        if not _symbolic_in([arr, where, initial]):
            kw = {**k, **({} if where is True else {"where": where}),
                  **({} if initial is None else {"initial": initial})}
            return real(_plain(arr), *args, **kw)
        if where is not True:
            if neutral is None or args or k.get("axis") is not None:
                raise Untranslatable(f"numpy.{name} with where= cannot be written as equations")
            count = _sum(_where(where, 1, 0).ravel()) if name == "mean" else None
            arr = _where(where, arr, neutral)
            r = real(_numify(_obj(arr)), **k)
            r = r * len(np.ravel(arr)) / count if count is not None else r
        else:
            r = real(_numify(arr), *args, **k)
        if initial is not None:
            r = r + _num(initial) if name == "sum" else r * _num(initial)
        return r
    return g


def _extreme(op, real):
    """max and min: Max and Min of the elements (along an axis)."""
    def g(a, axis=None, out=None, keepdims=False, initial=None, where=True):
        if out is not None or keepdims:
            raise Untranslatable("max or min with out= or keepdims= cannot be written as equations")
        arr = _obj(a)
        if not _symbolic_in([arr, where, initial]):
            kw = {**({} if initial is None else {"initial": initial}),
                  **({} if where is True else {"where": where})}
            return real(_plain(arr), axis=axis, **kw)
        if where is not True:
            if initial is None or axis is not None:
                raise Untranslatable("max or min with where= needs initial=, along all axes")
            arr = _obj(_where(where, arr, initial))
        if axis is not None and arr.ndim > 1:
            return np.apply_along_axis(lambda v: g(v, initial=initial), axis, arr)
        v = [_num(x) for x in arr.ravel()] + ([_num(initial)] if initial is not None else [])
        if not v:
            raise ValueError("zero-size array to reduction operation")
        return op(*v)
    return g


def _conditions(f, name):
    def g(a, axis=None, out=None, keepdims=False, where=True):
        if axis is not None or out is not None or keepdims or where is not True:
            raise Untranslatable(f"numpy.{name} with options cannot be written as equations")
        return f(*_obj(a).ravel())
    return g


def _logical(f):
    def g(*a, **k):
        _keywords("a logical function of numpy", k)
        a = [_obj(x) if isinstance(x, (list, tuple)) else x for x in a]
        if any(isinstance(x, np.ndarray) for x in a):
            return np.vectorize(f, otypes=[object])(*a)
        return f(*a)
    return g


def _polyval(p, x):
    """np.polyval by Horner's rule."""
    if not _symbolic_in([p, x]):
        return np.polyval(_plain(p), _plain(x))
    x = _obj(x) if isinstance(x, (list, tuple)) else x
    out = 0
    for c in _obj(p).ravel():
        out = out * x + _num(c)
    return out


def _np_round(a, decimals=0, out=None):
    if not _symbolic_in([_obj(a) if isinstance(a, (list, tuple)) else a]):
        return np.round(_plain(a), decimals)
    if decimals or out is not None:
        raise Untranslatable("numpy.round with decimals cannot be written as equations")
    if isinstance(a, (np.ndarray, list, tuple)):
        return np.vectorize(lambda v: _rint(_num(v)), otypes=[object])(_obj(a))
    return _rint(_num(a))


class _NoErrors:
    """np.errstate while reading: nothing to silence."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _apply(sym, x, real):
    """A function of one value: elementwise over arrays, the real one on numbers."""
    if isinstance(x, (np.ndarray, list, tuple)):
        arr = _obj(x)
        if not _symbolic_in([arr]):
            return real(_plain(arr))
        return np.vectorize(lambda v: _apply(sym, v, real), otypes=[object])(arr)
    return sym(_num(x)) if _issym(x) else real(_plain(x))


def _numbers_of(v, what):
    arr = np.asarray(v, dtype=object).ravel()
    if any(_issym(x) for x in arr):
        raise Untranslatable(f"{what} built from values that depend on the symbols")
    return np.asarray(arr, dtype=float)


def _tree(pieces, v, knots):
    """Piecewise over the segments between knots, as a balanced tree of conditions v <= knot:
    pieces[i] is the value on [knots[i], knots[i + 1]]."""
    def build(lo, hi):
        if lo == hi:
            return pieces[lo]
        mid = (lo + hi) // 2
        return sp.Piecewise((build(lo, mid), v <= knots[mid + 1]), (build(mid + 1, hi), True))
    return build(0, len(pieces) - 1)


def _table(xs, ys, kind="linear", below=None, above=None, extrapolate=False):
    """A table read as a function of a symbol: linear or stepwise between its points; outside,
    the fill values, the extrapolation, or the end values held where the code has none."""
    order = np.argsort(np.asarray(xs, float), kind="stable")
    ys = list(ys)
    xs = [float(xs[i]) for i in order]
    ys = [ys[i] if _issym(ys[i]) else float(ys[i]) for i in order]  # values may depend on the symbols
    # a knot given twice is a jump: just before it the first value, at it and after the last
    knots, first, last = [], [], []
    for x, yv in zip(xs, ys):
        if knots and x == knots[-1]:
            last[-1] = yv
        else:
            knots.append(x)
            first.append(yv)
            last.append(yv)
    if len(knots) < 2:
        raise Untranslatable("a table of fewer than two points")
    xs, ys = knots, last
    step = kind in ("previous", "zero", "next", "nearest", "nearest-up")

    def segment(v, i):  # xs[i] <= v <= xs[i + 1]
        a, b, ya, yb = xs[i], xs[i + 1], ys[i], ys[i + 1]
        if not step and first[i + 1] != yb:
            linear = ya + (v - a) * ((first[i + 1] - ya) / (b - a))
            return sp.Piecewise((yb, sp.Eq(v, b, evaluate=False)), (linear, True))
        if kind in ("previous", "zero"):
            return sp.Piecewise((ya, v < b), (yb, True))
        if kind == "next":
            return sp.Piecewise((ya, v <= a), (yb, True))
        if kind == "nearest":
            return sp.Piecewise((ya, v <= (a + b) / 2), (yb, True))
        if kind == "nearest-up":
            return sp.Piecewise((ya, v < (a + b) / 2), (yb, True))
        return ya + (v - a) * ((yb - ya) / (b - a)) if b > a else yb

    def f(v):
        inside = _tree([segment(v, i) for i in range(len(xs) - 1)], v, xs)
        if extrapolate and not step:
            lo = ys[0] + (v - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
            hi = ys[-1] + (v - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            lo = first[0] if below is None else below
            hi = ys[-1] if above is None else above
        return sp.Piecewise((lo, v < xs[0]), (hi, v > xs[-1]), (inside, True))
    return f


def _np_interp(x, xp, fp, left=None, right=None, period=None):
    if period is not None:
        raise Untranslatable("numpy.interp with a period cannot be written as equations")
    xs = _numbers_of(xp, "the points of a table")
    if np.any(np.diff(xs) < 0):
        raise Untranslatable("numpy.interp with points not in increasing order")
    if _symbolic_in([fp, left, right]):
        # values that depend on the symbols at points that are numbers: linear between them
        ys = [_num(v) for v in np.ravel(_obj(fp))]
        sym = _table(xs, ys, below=left, above=right)
        return np.vectorize(lambda v: sym(_num(v)), otypes=[object])(_obj(x)) \
            if isinstance(x, (np.ndarray, list, tuple)) else sym(_num(x))
    ys = _numbers_of(fp, "a table")
    sym = _table(xs, ys, below=left, above=right)
    return _apply(sym, x, lambda v: np.interp(v, xs, ys, left=left, right=right))


def _isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if not _symbolic_in([a, b]):
        return np.isclose(_plain(a), _plain(b), rtol, atol, equal_nan)
    if isinstance(a, (np.ndarray, list, tuple)) or isinstance(b, (np.ndarray, list, tuple)):
        return np.vectorize(lambda x, y: _isclose(x, y, rtol, atol), otypes=[object])(_obj(a), _obj(b))
    a, b = _num(a), _num(b)
    return _Cond(sp.Abs(a - b) <= atol + rtol * sp.Abs(b))


def _math_isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0):
    if not _symbolic_in([a, b]):
        return math.isclose(_plain(a), _plain(b), rel_tol=rel_tol, abs_tol=abs_tol)
    a, b = _num(a), _num(b)
    return _Cond(sp.Abs(a - b) <= sp.Max(rel_tol * sp.Max(sp.Abs(a), sp.Abs(b)), abs_tol))


def _searchsorted(a, v, side="left", sorter=None):
    if not _symbolic_in([v]) or sorter is not None:
        return np.searchsorted(_plain(a), _plain(v), side, sorter)
    xs = _numbers_of(a, "the points searched")
    op = "<" if side == "left" else "<="
    count = lambda u: sum(_num(_cmp1(x, op, u)) for x in xs)  # noqa: E731 - the points below u
    return np.vectorize(count, otypes=[object])(_obj(v)) if isinstance(v, (np.ndarray, list, tuple)) else count(_num(v))


def _ppoly(pp, below=None, above=None):
    """A piecewise polynomial of scipy (splines) as a function of a symbol."""
    if pp.c.ndim != 2:
        raise Untranslatable("a spline of several series cannot be written as equations")
    if pp.extrapolate == "periodic":
        raise Untranslatable("a periodic spline cannot be written as equations")
    xs, k = [float(x) for x in pp.x], pp.c.shape[0] - 1

    def poly(v, i):
        return sum(float(pp.c[m, i]) * (v - xs[i]) ** (k - m) for m in range(k + 1))

    def f(v):
        inside = _tree([poly(v, i) for i in range(len(xs) - 1)], v, xs)
        lo = poly(v, 0) if pp.extrapolate else (below if below is not None else float(pp(xs[0])))
        hi = poly(v, len(xs) - 2) if pp.extrapolate else (above if above is not None else float(pp(xs[-1])))
        return sp.Piecewise((lo, v < xs[0]), (hi, v > xs[-1]), (inside, True))
    return f


class _Interpolant:
    """An interpolating object of scipy built from numbers: a function of a symbol when read."""

    def __init__(self, real, sym):
        self._real, self._sym = real, sym

    def __call__(self, x, nu=0, extrapolate=None):
        if not _symbolic_in([x]):
            args = () if not nu and extrapolate is None else (nu,) if extrapolate is None else (nu, extrapolate)
            return self._real(_plain(x), *args)
        if extrapolate is not None:
            raise Untranslatable("an interpolant called with extrapolate= cannot be written as equations")
        return self.derivative(nu)(x) if nu else _apply(self._sym, x, self._real)

    def derivative(self, nu=1):
        d = self._real.derivative(nu)
        return _Interpolant(d, _ppoly(d))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise Untranslatable(f"the interpolant's {name} cannot be written as equations")


def _interp1d(x, y, kind="linear", axis=-1, copy=True, bounds_error=None, fill_value=np.nan,
              assume_sorted=False):
    from scipy.interpolate import PPoly, interp1d

    xs, ys = _numbers_of(x, "a table"), _numbers_of(y, "a table")
    real = interp1d(xs, ys, kind=kind, axis=axis, bounds_error=bounds_error, fill_value=fill_value,
                    assume_sorted=assume_sorted)
    extrapolate = isinstance(fill_value, str) and fill_value == "extrapolate"
    below = above = None
    if not isinstance(fill_value, str):
        lo, hi = fill_value if isinstance(fill_value, tuple) else (fill_value, fill_value)
        lo, hi = float(np.asarray(lo)), float(np.asarray(hi))
        # outside, the code gives these values, or nan or an error: then the end values are held
        below, above = (None if math.isnan(lo) else lo), (None if math.isnan(hi) else hi)
    if kind in ("linear", "slinear", "previous", "zero", "next", "nearest", "nearest-up"):
        sym = _table(xs, ys, kind, below, above, extrapolate)
    else:
        from scipy.interpolate import make_interp_spline

        order = {"quadratic": 2, "cubic": 3}.get(kind, kind)
        if not isinstance(order, int):
            raise Untranslatable(f"interp1d of kind {kind!r} cannot be written as equations")
        idx = np.argsort(xs, kind="stable")
        pp = PPoly.from_spline(make_interp_spline(xs[idx], ys[idx], k=order))
        pp.extrapolate = extrapolate
        sym = _ppoly(pp, below, above)
    return _Interpolant(real, sym)


def _spline(name):
    def make(*a, **k):
        from scipy import interpolate

        a = [_numbers_of(v, "a spline") if isinstance(v, (list, tuple, np.ndarray)) else v for v in a]
        real = getattr(interpolate, name)(*a, **k)
        return _Interpolant(real, _ppoly(real))
    return make


def _norm(kind):
    def sym(x, loc=0, scale=1):
        z = (_num(x) - _num(loc)) / _num(scale)
        if kind == "pdf":
            return sp.exp(-z**2 / 2) / (_num(scale) * sp.sqrt(2 * sp.pi))
        # normcdf keeps the precision of the tail that erf loses
        return normcdf(z) if kind == "cdf" else normcdf(-z)

    def f(x, loc=0, scale=1):
        if not _symbolic_in([x, loc, scale]):
            from scipy.stats import norm
            return getattr(norm, kind)(_plain(x), _plain(loc), _plain(scale))
        if isinstance(x, (np.ndarray, list, tuple)):
            return np.vectorize(lambda v: sym(v, loc, scale), otypes=[object])(_obj(x))
        return sym(x, loc, scale)
    return f


def _special(f, name):
    def g(*a):
        if not _symbolic_in(a):
            import scipy.special
            return getattr(scipy.special, name)(*[_plain(x) for x in a])
        a = [_obj(x) if isinstance(x, (list, tuple)) else x for x in a]
        if any(isinstance(x, np.ndarray) for x in a):
            return np.vectorize(lambda *xs: f(*map(_num, xs)), otypes=[object])(*a)
        return f(*map(_num, a))
    return g


_NP_TYPES = ("ndarray", "float64", "float32", "int64", "int32", "bool_", "floating", "integer",
             "number", "newaxis", "finfo", "iinfo")
# numpy functions that build, combine or index arrays: themselves, on arrays of objects
_NP_AS_IS = ("arange", "linspace", "concatenate", "stack", "hstack", "vstack", "append",
             "reshape", "ravel", "copy", "diff", "outer", "dot", "matmul", "shape", "size",
             "ndim", "flip", "roll", "tile", "repeat", "column_stack", "atleast_1d", "atleast_2d",
             "array_split", "split", "take", "transpose", "squeeze", "expand_dims",
             "broadcast_to", "eye", "identity", "diag", "trapz", "cross", "kron", "tril", "triu",
             "trace", "insert", "delete", "logspace", "geomspace", "meshgrid", "searchsorted",
             "argmin", "argmax", "sort", "argsort")


def _np_symbolic():
    t = {name: _ew(f, name) for name, f in _FUNCS.items() if hasattr(np, name)}
    t.update({name: getattr(np, name) for name in _NP_AS_IS + _NP_TYPES if hasattr(np, name)})
    t.update({
        "greater": lambda a, b: _cmp1(a, ">", b), "greater_equal": lambda a, b: _cmp1(a, ">=", b),
        "less": lambda a, b: _cmp1(a, "<", b), "less_equal": lambda a, b: _cmp1(a, "<=", b),
        "equal": lambda a, b: _cmp1(a, "==", b), "not_equal": lambda a, b: _cmp1(a, "!=", b),
        "where": _where, "select": _select, "piecewise": _np_piecewise, "interp": _np_interp,
        "polyval": _polyval, "isclose": _isclose, "searchsorted": _searchsorted,
        "array": _array, "asarray": _asarray, "asanyarray": _asarray,
        "ascontiguousarray": _asarray, "zeros": _filled(0), "ones": _filled(1),
        "empty": _filled(0), "full": _full, "zeros_like": _filled_like(0),
        "ones_like": _filled_like(1), "empty_like": _filled_like(0), "full_like": _full_like,
        "clip": _clip, "sinc": _ew(_sinc, "sinc"), "heaviside": _ew(_heaviside, "heaviside"),
        "isscalar": _isscalar, "errstate": _NoErrors,
        "any": _conditions(lambda *v: _or_of(*v), "any"),
        "all": _conditions(lambda *v: _and_of(*v), "all"),
        "max": _extreme(sp.Max, np.max), "amax": _extreme(sp.Max, np.max),
        "min": _extreme(sp.Min, np.min), "amin": _extreme(sp.Min, np.min),
        "sum": _reduction(np.sum, "sum", 0), "prod": _reduction(np.prod, "prod", 1),
        "mean": _reduction(np.mean, "mean", 0), "cumsum": _reduction(np.cumsum, "cumsum"),
        "cumprod": _reduction(np.cumprod, "cumprod"),
        "logical_and": _logical(lambda a, b: _and_of(a, b)),
        "logical_or": _logical(lambda a, b: _or_of(a, b)), "logical_not": _logical(_not),
        "logical_xor": _logical(lambda a, b: _or_of(_and_of(a, _not(b)), _and_of(_not(a), b))),
        "isnan": _nan_test("isnan", np), "isinf": _nan_test("isinf", np),
        "isfinite": _nan_test("isfinite", np), "round": _np_round, "around": _np_round,
        "pi": sp.pi, "e": sp.E, "inf": sp.oo, "nan": sp.nan,
        "float64": _npfloat64, "float32": _npfloat32, "double": _npfloat64,
    })
    return t


_MATH_FUNCS = ("exp", "log", "log10", "log2", "log1p", "expm1", "sqrt", "sin", "cos", "tan",
               "sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh", "acosh", "atanh", "atan2",
               "hypot", "fabs", "floor", "ceil", "trunc", "erf", "erfc", "fmod", "pow", "gamma",
               "lgamma", "radians", "degrees", "exp2", "cbrt")


def _math_symbolic():
    t = {name: _scalar(_FUNCS[name], name) for name in _MATH_FUNCS if hasattr(math, name)}
    t.update(pi=sp.pi, e=sp.E, tau=2 * sp.pi, inf=sp.oo, nan=sp.nan,
             isnan=_nan_test("isnan", math), isinf=_nan_test("isinf", math),
             isfinite=_nan_test("isfinite", math), fsum=lambda it: _sum(it), prod=_prod,
             isclose=_math_isclose)
    return t


def _prod(it, start=1):
    out = start
    for x in it:
        out = _num(out) * _num(x)
    return out


_SPECIAL = {"erf": sp.erf, "erfc": sp.erfc, "expit": lambda x: 1 / (1 + sp.exp(-x)),
            "logit": lambda x: sp.log(x / (1 - x)), "gammaln": sp.loggamma, "gamma": sp.gamma}
_SPLINES = ("CubicSpline", "PchipInterpolator", "Akima1DInterpolator")
_SYMBOLIC = {
    "numpy": _np_symbolic, "numpy.linalg": dict, "math": _math_symbolic, "scipy": dict,
    "scipy.special": lambda: {k: _special(f, k) for k, f in _SPECIAL.items()},
    "scipy.stats": dict,
    "scipy.stats.norm": lambda: {k: _norm(k) for k in ("cdf", "pdf", "sf")},
    "scipy.interpolate": lambda: {"interp1d": _interp1d, **{k: _spline(k) for k in _SPLINES}},
    "typing": dict,
}
# with the real functions: those read as equations, and the numerical functions of numpy that
# Numba compiles
_NP_NUMERIC = {
    "all", "allclose", "amax", "amin", "any", "append", "arange", "arccos", "arccosh", "arcsin",
    "arcsinh", "arctan", "arctan2", "arctanh", "argmax", "argmin", "argsort", "argwhere",
    "around", "array", "array_equal", "array_split", "asarray", "ascontiguousarray",
    "atleast_1d", "atleast_2d", "atleast_3d", "average", "bincount", "broadcast_arrays",
    "broadcast_to", "cbrt", "ceil", "clip", "column_stack", "concatenate", "convolve", "copy",
    "copysign", "corrcoef", "correlate", "cos", "cosh", "count_nonzero", "cov", "cross",
    "cumprod", "cumsum", "deg2rad", "degrees", "diag", "diagflat", "diff", "digitize",
    "divide", "dot", "dstack", "e", "ediff1d", "empty", "empty_like", "equal", "errstate",
    "exp", "exp2", "expand_dims", "expm1", "extract", "eye", "fabs", "finfo", "flatnonzero",
    "flip", "fliplr", "flipud", "float32", "float64", "floor", "floor_divide", "fmax", "fmin",
    "fmod", "full", "full_like", "geomspace", "greater", "greater_equal", "heaviside", "hstack",
    "hypot", "identity", "iinfo", "inf", "inner", "insert", "int32", "int64", "interp",
    "intersect1d", "isclose", "isfinite", "isin", "isinf", "isnan", "isscalar", "kron",
    "ldexp", "less", "less_equal", "linspace", "log", "log10", "log1p", "log2", "logaddexp",
    "logaddexp2", "logical_and", "logical_not", "logical_or", "logical_xor", "logspace",
    "matmul", "max", "maximum", "mean", "median", "meshgrid", "min", "minimum", "mod",
    "multiply", "nan", "nan_to_num", "nanmax", "nanmean", "nanmin", "nansum", "ndarray", "ndim",
    "negative", "newaxis", "nonzero", "not_equal", "ones", "ones_like", "outer", "percentile",
    "pi", "piecewise", "polyval", "positive", "power", "prod", "ptp", "quantile", "rad2deg",
    "radians", "ravel", "reciprocal", "remainder", "repeat", "reshape", "rint", "roll",
    "round", "searchsorted", "select", "shape", "sign", "sin", "sinc", "sinh", "size", "sort",
    "split", "sqrt", "square", "squeeze", "stack", "std", "subtract", "sum", "swapaxes", "take",
    "tan", "tanh", "tile", "trace", "transpose", "trapz", "tri", "tril", "triu", "true_divide",
    "trunc", "unique", "unwrap", "vander", "var", "vstack", "where", "zeros", "zeros_like",
    "add", "absolute", "abs", "float_power", "bool_", "floating", "integer", "number", "fix",
}
_NUMERIC = {
    "numpy": _NP_NUMERIC,
    "numpy.linalg": {"solve", "inv", "det", "norm", "eig", "eigvals", "eigh", "eigvalsh",
                     "matrix_power", "lstsq", "pinv", "cholesky", "qr", "svd", "slogdet", "cond",
                     "matrix_rank"},
    "math": lambda: {n for n in dir(math) if not n.startswith("_")},
    "scipy": set(), "scipy.special": set(_SPECIAL), "scipy.stats": set(),
    "scipy.stats.norm": {"cdf", "pdf", "sf"},
    "scipy.interpolate": {"interp1d", *_SPLINES}, "typing": set(),
}
_MODULES = {*_SYMBOLIC, *_REFUSALS, "numba"}


# ---------- conditions ----------

_ORACLE = contextvars.ContextVar("pysaem_oracle", default=None)
_FLIP = {sp.StrictLessThan: sp.StrictGreaterThan, sp.LessThan: sp.GreaterThan,
         sp.StrictGreaterThan: sp.StrictLessThan, sp.GreaterThan: sp.LessThan,
         sp.Equality: sp.Equality, sp.Unequality: sp.Unequality}


class _Oracle:
    """The truth of the conditions met along one run of a function: implied by those decided
    before when it can be (t > 24 decided true makes t > 12 true), else given by the branch
    explored, false the first time."""

    def __init__(self, prefix, deadline=math.inf):
        self.prefix, self.taken, self.known, self.bounds = list(prefix), [], {}, {}
        self.deadline = deadline

    def decide(self, cond):
        # a loop on the symbols makes each test longer than the last: stop before the time limit
        if time.monotonic() > self.deadline or isinstance(cond, sp.Basic) and _size(cond, MAX_NODES) > MAX_NODES:
            raise Untranslatable("too many conditions to enumerate, or conditions too large (a loop "
                                 "whose turns depend on the symbols?)")
        known = self._implied(cond)
        if known is not None:
            return known
        i = len(self.taken)
        d = self.prefix[i] if i < len(self.prefix) else False
        self.taken.append((cond, d))
        if len(self.taken) > 4 * MAX_PATHS:
            raise Untranslatable("a loop runs on a condition on the symbols (as while t > ...): "
                                 "its number of turns is not fixed")
        self._learn(cond, d)
        return d

    def _implied(self, c):
        if isinstance(c, BooleanTrue):
            return True
        if isinstance(c, BooleanFalse):
            return False
        if c in self.known:
            return self.known[c]
        if isinstance(c, sp.Not):
            r = self._implied(c.args[0])
            return None if r is None else not r
        if isinstance(c, (sp.And, sp.Or)):
            rs = [self._implied(a) for a in c.args]
            absorbing = isinstance(c, sp.Or)
            if absorbing in rs:
                return absorbing
            return (not absorbing) if all(r is not None for r in rs) else None
        if isinstance(c, Relational):
            return self._interval(c)
        return None

    def _learn(self, c, d):
        self.known[c] = d
        self.known[sp.Not(c)] = not d
        if isinstance(c, sp.Not):
            self._learn(c.args[0], not d)
        elif isinstance(c, sp.And) and d or isinstance(c, sp.Or) and not d:
            for a in c.args:
                self._learn(a, d)
        elif isinstance(c, Relational):
            self._narrow(c if d else sp.Not(c))

    @staticmethod
    def _side(rel):
        """(symbol, relation class, number) for a relation of one symbol and a number."""
        if not isinstance(rel, Relational) or type(rel) not in _FLIP:
            return None
        a, b = rel.lhs, rel.rhs
        if isinstance(a, sp.Symbol) and b.is_number and b.is_extended_real:
            return a, type(rel), float(b)
        if isinstance(b, sp.Symbol) and a.is_number and a.is_extended_real:
            return b, _FLIP[type(rel)], float(a)
        return None

    def _interval(self, rel):
        s = self._side(rel)
        if s is None:
            return None
        x, op, c = s
        lo, los, hi, his = self.bounds.get(x, (-math.inf, False, math.inf, False))
        if op is sp.StrictGreaterThan:
            return True if lo > c or (lo == c and los) else False if hi <= c else None
        if op is sp.GreaterThan:
            return True if lo >= c else False if hi < c or (hi == c and his) else None
        if op is sp.StrictLessThan:
            return True if hi < c or (hi == c and his) else False if lo >= c else None
        if op is sp.LessThan:
            return True if hi <= c else False if lo > c or (lo == c and los) else None
        out = c < lo or c > hi or (c == lo and los) or (c == hi and his)
        if op is sp.Equality:
            return False if out else True if lo == hi == c else None
        return True if out else False if lo == hi == c else None

    def _narrow(self, rel):
        s = self._side(rel)
        if s is None:
            return
        x, op, c = s
        lo, los, hi, his = self.bounds.get(x, (-math.inf, False, math.inf, False))
        if op in (sp.StrictGreaterThan, sp.GreaterThan, sp.Equality):
            strict = op is sp.StrictGreaterThan
            if c > lo or (c == lo and strict):
                lo, los = c, strict
        if op in (sp.StrictLessThan, sp.LessThan, sp.Equality):
            strict = op is sp.StrictLessThan
            if c < hi or (c == hi and strict):
                hi, his = c, strict
        self.bounds[x] = (lo, los, hi, his)


class _Cond:
    """A condition on the symbols: decided by the oracle where Python tests it, 1 or 0 in
    arithmetic, as a bool."""
    __array_priority__ = 1000

    def __init__(self, rel):
        self.rel = rel

    def __bool__(self):
        if isinstance(self.rel, (BooleanTrue, BooleanFalse)):
            return bool(self.rel)
        oracle = _ORACLE.get()
        if oracle is None:
            raise Untranslatable("a condition on the symbols is tested outside rhs, observe and init")
        return oracle.decide(self.rel)

    def num(self):
        return sp.Piecewise((1, self.rel), (0, True))

    def _sympy_(self):
        return self.num()

    def __and__(self, o):
        return _Cond(sp.And(self.rel, _rel(o)))

    def __or__(self, o):
        return _Cond(sp.Or(self.rel, _rel(o)))

    def __xor__(self, o):
        o = _rel(o)
        return _Cond(sp.Or(sp.And(self.rel, sp.Not(o)), sp.And(sp.Not(self.rel), o)))

    __rand__, __ror__, __rxor__ = __and__, __or__, __xor__

    def __invert__(self):
        return _Cond(sp.Not(self.rel))

    def __neg__(self):
        return -self.num()

    def __pos__(self):
        return self.num()

    def __abs__(self):
        return self.num()

    def __index__(self):
        raise Untranslatable("a condition on the symbols used as an index")

    # compared by library code (list.count, list.index): a condition, decided when tested
    def __eq__(self, o):
        return _cmp1(self, "==", o)

    def __ne__(self, o):
        return _cmp1(self, "!=", o)

    __hash__ = object.__hash__


for _op in ("add", "sub", "mul", "truediv", "pow", "mod"):
    setattr(_Cond, f"__{_op}__", lambda s, o, _n=_op: getattr(s.num(), f"__{_n}__")(_num(o)))
    setattr(_Cond, f"__r{_op}__", lambda s, o, _n=_op: getattr(s.num(), f"__r{_n}__")(_num(o)))
_Cond.__floordiv__ = lambda s, o: _floordiv(s.num(), o)
_Cond.__rfloordiv__ = lambda s, o: _floordiv(o, s.num())

_RELATIONS = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge,
              "==": lambda a, b: sp.Eq(a, b, evaluate=False),
              "!=": lambda a, b: sp.Ne(a, b, evaluate=False)}
_PYTHON = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
           ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
           "in": lambda a, b: a in b, "not in": lambda a, b: a not in b,
           "is": lambda a, b: a is b, "is not": lambda a, b: a is not b}


def _cmp1(a, op, b):
    if op in ("is", "is not"):
        return _PYTHON[op](a, b)
    if op in ("in", "not in"):
        items = list(b) if isinstance(b, (dict, set, frozenset)) else None
        if items is None and isinstance(b, (list, tuple, np.ndarray, range)):
            items = list(np.ravel(np.asarray(b, dtype=object)))
        if items is None or not (_issym(a) or any(_issym(x) for x in items)):
            return _PYTHON[op](a, b)
        found = _or_of(*[_cmp1(a, "==", x) for x in items])
        return _not(found) if op == "not in" else found
    if isinstance(a, (np.ndarray, list, tuple)) or isinstance(b, (np.ndarray, list, tuple)):
        if not (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
            if not _symbolic_in([a, b]):
                return _PYTHON[op](a, b)  # lists compare as lists
            if op in ("==", "!=") and isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                same = _and_of(*[_cmp1(x, "==", y) for x, y in zip(a, b)]) if len(a) == len(b) else False
                return same if op == "==" else _not(same)
            raise Untranslatable("an ordering of sequences of values depending on the symbols cannot "
                                 "be written as equations")
        return np.vectorize(lambda x, y: _cmp1(x, op, y), otypes=[object])(a, b)
    if not (_issym(a) or _issym(b)):
        return _PYTHON[op](a, b)
    if isinstance(a, (str, type(None))) or isinstance(b, (str, type(None))):
        return _PYTHON[op](a, b)
    a, b = _num(a), _num(b)
    if op in ("==", "!=") and a == b:
        raise Untranslatable(f"x {op} x (a test for nan) cannot be written as equations")
    return _Cond(_RELATIONS[op](a, b))


def _cmp(left, *rest):
    """A chain of comparisons; it stops at the first one false for every value, as in Python."""
    out, a = None, left
    for op, b in zip(rest[::2], rest[1::2]):
        r = _cmp1(a, op, b)
        out = r if out is None else _and_of(out, r)
        if out is False or isinstance(out, np.bool_) and not out:
            return out
        a = b
    return out


def _and_of(*vals):
    if any(_issym(v) or _is_condition(v) for v in vals):
        return _Cond(sp.And(*[_rel(v) for v in vals]))
    return all(bool(v) for v in vals)


def _or_of(*vals):
    if any(_issym(v) or _is_condition(v) for v in vals):
        return _Cond(sp.Or(*[_rel(v) for v in vals]))
    return any(bool(v) for v in vals)


def _and(*thunks):
    """a and b in a test: one condition on the symbols, left to right, stopping at an operand
    false for every value as Python does."""
    conds = []
    for f in thunks:
        v = f()
        if _issym(v) or _is_condition(v):
            conds.append(_rel(v))
        elif not _truth(v):
            return False
    return _Cond(sp.And(*conds)) if conds else True


def _or(*thunks):
    conds = []
    for f in thunks:
        v = f()
        if _issym(v) or _is_condition(v):
            conds.append(_rel(v))
        elif _truth(v):
            return True
    return _Cond(sp.Or(*conds)) if conds else False


def _sand(*thunks):
    """a and b in a test whose operands call functions: each operand is decided before the next
    is run, as Python runs them (a guard in an operand not reached drops nothing)."""
    for f in thunks:
        if not _truth(f()):
            return False
    return True


def _sor(*thunks):
    for f in thunks:
        if _truth(f()):
            return True
    return False


def _vand(*thunks):
    """a and b as a value: the first false operand, else the last, as in Python."""
    v = True
    for f in thunks:
        v = f()
        if not _truth(v):
            return v
    return v


def _vor(*thunks):
    """a or b as a value: the first true operand, else the last, as in Python."""
    v = False
    for f in thunks:
        v = f()
        if _truth(v):
            return v
    return v


def _not(v):
    if _issym(v) or _is_condition(v):
        return _Cond(sp.Not(_rel(v)))
    return not v


def _bool(v=False):
    return _Cond(_rel(v)) if _issym(v) or _is_condition(v) else bool(v)


def _truth(v):
    """The truth of a value where Python tests it: a number is true when it is not zero."""
    if isinstance(v, np.ndarray) and v.size == 1:
        v = v.item()
    if _issym(v) or _is_condition(v):
        return bool(v) if isinstance(v, _Cond) else bool(_Cond(_rel(v)))
    return bool(v)


def _refuse(what):
    raise Untranslatable(f"{what} cannot be written as equations")


def _pieces(pairs, last):
    """Piecewise over exclusive conditions, nested two by two so that its depth stays small;
    entries that are sequences of one length give a list, one such Piecewise per component."""
    entries = [v for v, _ in pairs] + [last]
    if all(isinstance(v, (list, tuple, np.ndarray)) for v in entries):
        rows = [list(np.ravel(np.asarray(v, dtype=object))) if isinstance(v, np.ndarray) else list(v) for v in entries]
        if len({len(r) for r in rows}) != 1:
            raise Untranslatable("entries of different lengths read at a value that depends on the symbols")
        return [_pieces([(row[k], c) for row, (_, c) in zip(rows, pairs)], rows[-1][k]) for k in range(len(rows[0]))]
    if len(pairs) <= 8:
        return sp.Piecewise(*[(_num(v), c) for v, c in pairs], (_num(last), True))
    mid = len(pairs) // 2
    return sp.Piecewise((_pieces(pairs[:mid - 1], pairs[mid - 1][0]), sp.Or(*[c for _, c in pairs[:mid]])),
                        (_pieces(pairs[mid:], last), True))


def _lookup(table, i):
    """table[i], with i depending on the symbols: the entry whose key or index i equals."""
    if not _issym(i):
        if isinstance(i, sp.Basic) and i.is_number:
            i = int(i) if i.is_integer else float(i)
        return table[i]
    if isinstance(table, dict):
        pairs = _keyed(table, i)
        if not pairs:
            raise KeyError(i)
        return _pieces(pairs[:-1], pairs[-1][0])
    if not isinstance(table, (list, tuple, np.ndarray)):
        raise Untranslatable(f"{type(table).__name__} indexed by a value that depends on the symbols")
    arr = np.empty(len(table), dtype=object)
    arr[:] = list(table)  # rows of a 2-D table stay rows
    n, i = len(arr), _num(i)
    # Python reads a negative index from the end
    pairs = [(arr[k], sp.Or(sp.Eq(i, k, evaluate=False), sp.Eq(i, k - n, evaluate=False))) for k in range(n)]
    return _pieces(pairs[:-1], arr[-1])


def _keyed(table, i):
    """(value, condition that i is its key) for the keys of a dict that are numbers."""
    i = _num(i)
    return [(v, sp.Eq(i, _num(k), evaluate=False)) for k, v in table.items()
            if isinstance(k, (bool, int, float, np.number)) or isinstance(k, sp.Basic) and k.is_number]


def _get(obj, *args):
    if isinstance(obj, dict) and args and _issym(args[0]):
        if len(args) > 1:
            return _pieces(_keyed(obj, args[0]), args[1])
        return _lookup(obj, args[0])
    return obj.get(*args)


def _item(x):
    # x.item() of an array element: the element itself
    return x if isinstance(x, sp.Basic) or _issym(x) else x.item()


def _int(x=0, *a):
    if a or not _issym(x):
        return int(x, *a)
    x = _num(x)
    return sp.Piecewise((sp.floor(x), x >= 0), (sp.ceiling(x), True))


def _round(x, ndigits=None):
    if not _issym(x):
        return round(x) if ndigits is None else round(x, ndigits)
    if ndigits:
        raise Untranslatable("round() with digits cannot be written as equations")
    return _rint(_num(x))


class _npfloat64(np.float64):
    """np.float64(): a value depending on the symbols stays one; still a dtype."""

    def __new__(cls, x=0.0):
        return _num(x) if _issym(x) else np.float64(_plain(x))


class _npfloat32(np.float32):
    def __new__(cls, x=0.0):
        return _num(x) if _issym(x) else np.float32(_plain(x))


class _float(float):
    """float(): a value depending on the symbols stays one; still the type float as a dtype."""

    def __new__(cls, x=0.0):
        if isinstance(x, np.ndarray) and x.size == 1:
            x = x.item()
        if _issym(x):
            return _num(x)
        return float(x)


def _isinstance(obj, cls):
    # a value depending on the symbols is a float, a condition a bool
    real = {_float: float, _npfloat64: np.float64, _npfloat32: np.float32}
    cls = tuple(real.get(c, c) for c in (cls if isinstance(cls, tuple) else (cls,)))
    if isinstance(obj, _Cond):
        obj = np.bool_(True)
    elif _issym(obj):
        if isinstance(np.float64(1.0), cls):
            return True
        if any(isinstance(c, type) and issubclass(c, (int, np.integer)) for c in cls):
            # int() of a symbol is a whole number Python would call an int
            raise Untranslatable("isinstance(x, int) of a value depending on the symbols cannot be "
                                 "written as equations")
        return False
    return isinstance(obj, cls)


def _elems(a):
    if isinstance(a, np.ndarray):
        return list(a.ravel())
    return list(a)


def _extremum(a, k, op, real):
    v = _elems(a[0]) if len(a) == 1 else list(a)  # an iterator is read once
    if not any(_issym(x) for x in v):
        return real(v, **k) if len(a) == 1 else real(*a, **k)
    if "key" in k:
        raise Untranslatable(f"{real.__name__}(key=...) of values depending on the symbols cannot "
                             "be written as equations")
    return op(*map(_num, v))


def _max(*a, **k):
    return _extremum(a, k, sp.Max, builtins.max)


def _min(*a, **k):
    return _extremum(a, k, sp.Min, builtins.min)


def _str(x=""):
    if _issym(x) or isinstance(x, (list, tuple, np.ndarray)) and _symbolic_in([x]):
        raise Untranslatable("a value depending on the symbols turned into text cannot be written as equations")
    return str(x)


def _repr(x):
    _str(x)
    return repr(x)


def _set(it=()):
    # Python compares the members of a set as they are written: symbols that may be equal are not
    v = _elems(it)
    if any(_issym(x) for x in v):
        raise Untranslatable("a set of values depending on the symbols cannot be written as equations")
    return set(v)


def _filter(f, it):
    return builtins.filter(_truth if f is None else (lambda x: _truth(f(x))), it)


def _any(it):
    return _or_of(*_elems(it))


def _all(it):
    return _and_of(*_elems(it))


def _sum(it, start=0):
    out = _num(start)
    for x in it:
        out = out + _num(x)
    return out


def _abs(x):
    return sp.Abs(_num(x)) if _issym(x) else builtins.abs(x)


class _States(np.ndarray):
    """y while reading: an array of symbols whose methods keep Python's meaning on them."""

    def __bool__(self):
        if self.size != 1:
            raise ValueError("the truth value of an array with more than one element is ambiguous")
        return _truth(self.item())

    def any(self, axis=None, out=None, keepdims=False, **k):
        return _conditions(lambda *v: _or_of(*v), "any")(np.asarray(self), axis, out, keepdims)

    def all(self, axis=None, out=None, keepdims=False, **k):
        return _conditions(lambda *v: _and_of(*v), "all")(np.asarray(self), axis, out, keepdims)

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        return _extreme(sp.Max, np.max)(np.asarray(self), axis, out, keepdims, initial, where)

    def min(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        return _extreme(sp.Min, np.min)(np.asarray(self), axis, out, keepdims, initial, where)

    def clip(self, min=None, max=None, out=None, **k):
        return _clip(np.asarray(self), min, max, out, **k)

    def __getitem__(self, key):
        if isinstance(key, np.ndarray) and key.dtype == object:
            raise Untranslatable("an array selected by conditions on the symbols cannot be written as equations")
        return np.ndarray.__getitem__(self, key)

    def __setitem__(self, key, value):
        if isinstance(key, np.ndarray) and key.dtype == object and key.shape == self.shape:
            # y[y < 0] = 0: each element takes the value where its condition holds
            values = np.broadcast_to(np.asarray(value, dtype=object), self.shape)
            for i in np.ndindex(self.shape):
                np.ndarray.__setitem__(self, i, _where(key[i], values[i], np.ndarray.__getitem__(self, i)))
            return
        np.ndarray.__setitem__(self, key, value)


_ARITH = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%"}
_ARITH_DO = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
             "/": lambda a, b: a / b, "%": lambda a, b: a % b}


def _arith(a, op, b):
    # a bool next to a value depending on the symbols counts as 0 or 1, as Python counts it
    if isinstance(a, (bool, np.bool_)) and _issym(b) or isinstance(b, (bool, np.bool_)) and _issym(a):
        a, b = _num(a), _num(b)
    return _ARITH_DO[op](a, b)


_PURE_CALLS = {"abs", "min", "max", "len", "float", "int", "round", "sum", "any", "all", "bool",
               "pow", "divmod", "isinstance", "range", "enumerate", "zip", "sorted", "reversed",
               "list", "tuple", "dict"}


def _module_names(tree) -> set:
    """Names the code binds to modules or to their functions."""
    names = {"np", "math"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names}
    return names


class _Rewrite(ast.NodeTransformer):
    """The code made readable with symbols: comparisons through __cmp; tests through __truth,
    and/or in a test as one condition; and, or, not and bool as values through their versions;
    int, round, //, ** and computed indexes too; the code's own raise and assert become guards;
    try and match inside a function refuse the reading when reached."""
    _OPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==",
            ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not in", ast.Is: "is", ast.IsNot: "is not"}

    def __init__(self, modules=frozenset()):
        self.depth, self.modules = 0, set(modules)

    def _pure(self, node) -> bool:
        """No call but to builtins and module functions: nothing raised, nothing changed."""
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                names = chain(n.func)
                if not names or names[0] not in self.modules and not (len(names) == 1 and names[0] in _PURE_CALLS):
                    return False
        return True

    @staticmethod
    def _call(name, args):
        return ast.Call(ast.Name(name, ast.Load()), list(args), [])

    @staticmethod
    def _thunk(node):
        return ast.Lambda(ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
                                        kw_defaults=[], kwarg=None, defaults=[]), node)

    def _cond(self, node):
        if isinstance(node, ast.BoolOp):
            # one condition when the operands only compute; decided one by one when they call
            # functions of the code, which may raise or change things as Python runs them
            pure = all(self._pure(v) for v in node.values)
            fn = ("__and" if pure else "__sand") if isinstance(node.op, ast.And) else ("__or" if pure else "__sor")
            return self._call(fn, [self._thunk(self._cond(v)) for v in node.values])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._call("__not", [self._cond(node.operand)])
        return self.visit(node)

    def _test(self, node):
        return self._call("__truth", [self._cond(node)])

    def _function(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        return node

    visit_FunctionDef = visit_Lambda = _function

    def _tested(self, node):
        test, node.test = node.test, ast.Constant(True)
        self.generic_visit(node)
        node.test = self._test(test)
        return node

    visit_If = visit_While = visit_IfExp = _tested

    def visit_comprehension(self, node):
        ifs, node.ifs = node.ifs, []
        self.generic_visit(node)
        node.ifs = [self._test(i) for i in ifs]
        return node

    def visit_Assert(self, node):
        msg = [self.visit(node.msg)] if node.msg is not None else []
        failed = self._call("__truth", [self._call("__not", [self._cond(node.test)])])
        guard = ast.Raise(self._call("__guard", [self._call("AssertionError", msg)]), None)
        return ast.copy_location(ast.If(failed, [guard], []), node)

    def visit_Raise(self, node):
        self.generic_visit(node)
        if node.exc is not None:
            node.exc = self._call("__guard", [node.exc])
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        fn = "__vand" if isinstance(node.op, ast.And) else "__vor"
        return ast.copy_location(self._call(fn, [self._thunk(v) for v in node.values]), node)

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(self._call("__not", [node.operand]), node)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        args = [node.left]
        for op, c in zip(node.ops, node.comparators):
            args += [ast.Constant(self._OPS[type(op)]), c]
        return ast.copy_location(self._call("__cmp", args), node)

    def visit_BinOp(self, node):
        self.generic_visit(node)
        fn = {ast.FloorDiv: "__floordiv", ast.Pow: "__pow"}.get(type(node.op))
        if fn:
            return ast.copy_location(self._call(fn, [node.left, node.right]), node)
        if type(node.op) in _ARITH:
            return ast.copy_location(self._call("__arith", [node.left, ast.Constant(_ARITH[type(node.op)]), node.right]), node)
        return node

    def visit_Set(self, node):
        self.generic_visit(node)
        return ast.copy_location(self._call("__set", [ast.List(node.elts, ast.Load())]), node)

    def visit_SetComp(self, node):
        self.generic_visit(node)
        return ast.copy_location(self._call("__set", [ast.ListComp(node.elt, node.generators)]), node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        fn = {ast.FloorDiv: "__floordiv", ast.Pow: "__pow"}.get(type(node.op))
        if fn and isinstance(node.target, ast.Name):
            value = self._call(fn, [ast.Name(node.target.id, ast.Load()), node.value])
            return ast.copy_location(ast.Assign([node.target], value), node)
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load) and not isinstance(node.slice, (ast.Constant, ast.Slice, ast.Tuple)):
            return ast.copy_location(self._call("__lookup", [node.value, node.slice]), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        simple = not node.keywords and not any(isinstance(a, ast.Starred) for a in node.args)
        if isinstance(f, ast.Name) and f.id in ("int", "round", "bool") and simple:
            node.func = ast.Name("__" + f.id, ast.Load())
        elif isinstance(f, ast.Attribute) and f.attr == "get" and simple and 1 <= len(node.args) <= 2:
            return ast.copy_location(self._call("__get", [f.value, *node.args]), node)
        elif isinstance(f, ast.Attribute) and f.attr == "item" and simple and not node.args:
            return ast.copy_location(self._call("__item", [f.value]), node)
        return node

    def visit_Try(self, node):
        if not self.depth:
            return self.generic_visit(node)
        return ast.copy_location(ast.Expr(self._call("__refuse", [ast.Constant(
            "try/except around the computations")])), node)

    def visit_Match(self, node):
        if not self.depth:
            return self.generic_visit(node)
        return ast.copy_location(ast.Expr(self._call("__refuse", [ast.Constant(
            "match/case (write it with if/elif)")])), node)

    def visit_With(self, node):
        self.generic_visit(node)
        return node.body  # np.errstate: nothing to silence while reading


# ---------- reading ----------

class _Params(dict):
    """p: the values by name, as a dict and as attributes."""

    def __missing__(self, k):
        near = _near(k, list(self))
        raise ModelError(f"p[{k!r}] is used but {k!r} is not declared"
                         + (f" (did you mean {near!r}?)" if near else f" (declared: {sorted(self)})"))

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError(k)
        return self[k]


def _count(n) -> str:
    return f"{n} value" + ("" if n == 1 else "s")


def _size(e, limit) -> int:
    """Number of nodes of e written out (shared parts counted at each use), up to limit + 1."""
    size, stack = {}, [(e, False)]
    while stack:
        x, done = stack.pop()
        if x in size:
            continue
        if done:
            size[x] = min(limit + 1, 1 + sum(size[a] for a in x.args))
        else:
            stack.append((x, True))
            stack.extend((a, False) for a in x.args if a not in size)
    return size[e]


def _flat(v, names, what):
    """The values a function returns, and the keys of the dict it returns (else None)."""
    keys = None
    if isinstance(v, dict):
        keys = list(v) if names is None else list(names)
        missing = [n for n in keys if n not in v]
        if missing:
            raise ModelError(f"{what} returns no value for {missing}")
        v = [v[n] for n in keys]
    if v is None:
        raise ModelError(f"{what} returns nothing: return the list of its values")
    out = []
    for x in (np.ravel(v) if isinstance(v, np.ndarray) else v if isinstance(v, (list, tuple)) else [v]):
        if isinstance(x, (list, tuple, np.ndarray)):
            out += _flat(x, None, what)[0]
        elif x is None or isinstance(x, (str, dict, set)):
            raise ModelError(f"{what} returns {x!r} among its values, not a number")
        else:
            out.append(sp.sympify(_num(x)))
    return out, keys


def _read(fn, what, args, names):
    """Run fn once per branch and merge the runs into one expression per value; also the keys of
    the dict it returns."""
    runs, stack, guard, keys = [], [[]], None, None
    met, start = 0, time.monotonic()
    while stack:
        prefix = stack.pop()
        oracle = _Oracle(prefix, start + READ_SECONDS / 4)
        token = _ORACLE.set(oracle)
        try:
            raw = fn(*args())
        except _Guard as g:
            raw, guard = _Guard, guard or g.exc
        except ModelError as e:
            e.args = (f"{what}: {e}",)
            if isinstance(e, Untranslatable):
                e.role = e.role or what
            raise
        except RecursionError:
            raise Untranslatable(f"{what}: the expressions nest too deeply", what) from None
        except Exception as e:
            raise Untranslatable(f"{what}: {type(e).__name__}: {e}", what) from None
        finally:
            _ORACLE.reset(token)
        met += len(oracle.taken)
        if met > 20 * MAX_PATHS or time.monotonic() - start > READ_SECONDS / 4:
            raise Untranslatable(f"{what}: too many conditions to enumerate (a loop whose turns "
                                 "depend on the symbols?)", what)
        if raw is not _Guard:
            vals, k = _flat(raw, names if names is not None else keys, what)
            if k is not None:
                if keys is not None and set(k) != set(keys):
                    raise ModelError(f"{what} returns the keys {k} in one branch, {keys} in another")
                keys = keys or k
            if any(_size(v, MAX_NODES) > MAX_NODES for v in vals):
                raise Untranslatable(f"{what}: the expressions grow too large to be written as "
                                     "equations", what)
            runs.append((oracle.taken, vals))
        elif not oracle.taken:
            raise ModelError(f"{what} raises {guard!r} for every value")
        for i in range(len(prefix), len(oracle.taken)):
            stack.append([d for _, d in oracle.taken[:i]] + [True])
        if len(runs) > MAX_PATHS:
            raise Untranslatable(f"{what}: more than {MAX_PATHS} combinations of conditions; combine "
                                 "conditions with and/or, or use np.where", what)
    if not runs:
        raise ModelError(f"{what} raises in every branch ({guard!r})")
    n = len(runs[0][1])
    if any(len(v) != n for _, v in runs):
        raise ModelError(f"{what} returns a different number of values in different branches")
    return [_merge(runs, j, 0) for j in range(n)], keys


def _merge(runs, j, depth):
    """Value j over runs that share their first depth decisions: nested where() in the order
    the code tests its conditions. A branch whose runs all raise has no value: the other
    branch is taken."""
    if len({v[j] for _, v in runs}) == 1:
        return runs[0][1][j]
    if any(len(taken) <= depth or taken[depth][0] != runs[0][0][depth][0] for taken, _ in runs):
        raise Untranslatable("the code does not run the same way twice (does it keep values "
                             "between calls?)")
    cond = runs[0][0][depth][0]
    yes = [r for r in runs if r[0][depth][1]]
    no = [r for r in runs if not r[0][depth][1]]
    if not yes or not no:
        return _merge(yes or no, j, depth + 1)
    return sp.Piecewise((_merge(yes, j, depth + 1), cond), (_merge(no, j, depth + 1), True))


def _equations(tree, spec) -> Model:
    ns = namespace(tree, symbolic=True)
    fns = roles(ns)
    require(fns, spec, ns)
    states = list(spec["states"])
    ys = [sp.Symbol(s, real=True) for s in states]
    ps = {n: sp.Symbol(n, real=True) for n in p_held(spec)}

    def args():
        return T, np.array(ys, dtype=object).view(_States), _Params(ps)

    rhs_x = []
    if states:
        rhs_x, _ = _read(fns["rhs"], "rhs", args, states)
        if not rhs_x:
            # rhs returns nothing: a model without states, whatever states were listed
            states, ys = [], []
        if len(rhs_x) != len(states):
            raise ModelError(f"rhs returns {_count(len(rhs_x))} for {len(states)} states {states}")
    out_x, keys = _read(fns["observe"], "observe", args, spec["outputs"])
    outputs = spec["outputs"] or keys or [f"out{k + 1}" for k in range(len(out_x))]
    if len(out_x) != len(outputs):
        raise ModelError(f"observe returns {_count(len(out_x))} for the outputs {outputs}")
    init_x = None
    if fns["init"] is not None and states:
        init_x, _ = _read(fns["init"], "init", lambda: (_Params(ps),), states)
        if len(init_x) != len(states):
            raise ModelError(f"init returns {_count(len(init_x))} for {len(states)} states {states}")
    common = dict(parameters=spec["parameters"], outputs=outputs, inputs=spec["inputs"],
                  regressors=spec["regressors"], covariates=spec["covariates"],
                  interpolated=spec["interpolated"], t0=spec["t0"], name=spec["name"],
                  hazards=spec["hazards"])
    for o, e in zip(outputs, out_x):
        if o in states and e != ys[states.index(o)]:
            raise ModelError(f"output {o!r} has the name of a state but another value")
    aux_names = [o for o in outputs if o not in states]
    # the names first, with equations that cannot fail; a failure of the equations written
    # after that is a failure of the writing
    Model(states={s: "0" for s in states}, odes={s: "0" for s in states},
          aux={o: "0" for o in aux_names}, **common)
    declared = set(states) | set(ps)
    aux = {o: _write(e, f"output {o}", declared, "observe")
           for o, e in zip(outputs, out_x) if o not in states}
    odes = {s: _write(e, f"equation of {s}", declared, "rhs") for s, e in zip(states, rhs_x)}
    init = {s: _write(init_x[i], f"initial value of {s}", declared, "init") if init_x else "0"
            for i, s in enumerate(states)}
    try:
        return Model(states=init, odes=odes, aux=aux, **common)
    except (ModelError, RecursionError) as e:
        raise Untranslatable(f"the equations written cannot be read back: {e}") from None


# ---------- writing ----------

_RELATION_TEXT = {sp.StrictLessThan: "<", sp.LessThan: "<=", sp.StrictGreaterThan: ">",
                  sp.GreaterThan: ">=", sp.Equality: "==", sp.Unequality: "!="}
_CALLS = {**{f: n for n, f in FUNCTIONS.items() if isinstance(f, type)}, sp.exp: "exp",
          sp.log: "log", sp.Abs: "abs", sp.Max: "max", sp.Min: "min", sp.Mod: "mod",
          sp.floor: "floor", sp.ceiling: "ceil", sp.loggamma: "lgamma"}
_WRITTEN = (sp.Symbol, sp.Rational, sp.Float, sp.Add, sp.Mul, sp.Pow, sp.Piecewise, sp.And,
            sp.Or, sp.Not, sp.ITE, BooleanTrue, BooleanFalse, sp.sign)


class _Writer(StrPrinter):
    """A SymPy expression of the reader in the equation language of model.py; any other kind
    of expression is refused as Untranslatable."""

    def __init__(self, declared):
        super().__init__()
        self.declared = declared

    def _print(self, expr, **kwargs):
        if not (isinstance(expr, _WRITTEN) or type(expr) in _RELATION_TEXT or type(expr) in _CALLS
                or expr in (sp.pi, sp.E, sp.oo, sp.S.NegativeInfinity)):
            raise Untranslatable(f"{type(expr).__name__} cannot be written as an equation")
        return super()._print(expr, **kwargs)

    def _print_Float(self, e):
        return repr(float(e))

    def _print_Pi(self, e):
        # a name of the model may be pi
        return "acos(-1)" if "pi" in self.declared else "pi"

    def _print_Exp1(self, e):
        return "exp(1)"

    def _print_Infinity(self, e):
        if "inf" in self.declared:
            raise Untranslatable("infinity cannot be written where inf names a value")
        return "inf"

    def _print_NegativeInfinity(self, e):
        return f"(-{self._print_Infinity(e)})"

    def _print_BooleanTrue(self, e):
        return "(0 < 1)"

    def _print_BooleanFalse(self, e):
        return "(1 < 0)"

    def _print_Piecewise(self, e):
        *head, (last, _) = e.args
        out = self._print(last)
        for val, cond in reversed(head):
            out = f"where({self._print(cond)}, {self._print(val)}, {out})"
        return out

    def _relation(self, e):
        return f"({self._print(e.lhs)} {_RELATION_TEXT[type(e)]} {self._print(e.rhs)})"

    _print_StrictLessThan = _print_LessThan = _print_StrictGreaterThan = _relation
    _print_GreaterThan = _print_Equality = _print_Unequality = _relation

    def _print_And(self, e):
        return "(" + " and ".join(self._print(a) for a in e.args) + ")"

    def _print_Or(self, e):
        return "(" + " or ".join(self._print(a) for a in e.args) + ")"

    def _print_Not(self, e):
        return f"(not {self._print(e.args[0])})"

    def _print_ITE(self, e):
        c, a, b = (self._print(x) for x in e.args)
        return f"(({c} and {a}) or ((not {c}) and {b}))"

    def _print_sign(self, e):
        x = self._print(e.args[0])
        return f"where({x} > 0, 1, where({x} < 0, -1, 0))"

    def _print_Function(self, e):
        return f"{_CALLS[type(e)]}(" + ", ".join(self._print(a) for a in e.args) + ")"

    _print_Abs = _print_Max = _print_Min = _print_Mod = _print_floor = _print_ceiling = _print_Function
    _print_loggamma = _print_exp = _print_log = _print_gamma = _print_Function


def _write(e, where, declared, role) -> str:
    # whole numbers written as such: beta, not 1.0*beta
    e = e.xreplace({f: sp.Integer(int(f)) for f in e.atoms(sp.Float)
                    if f.is_finite and f == int(f) and abs(f) < 1e15})
    if e.has(sp.nan, sp.zoo, sp.I):
        raise Untranslatable(f"{where}: the value is not always a finite real number (nan, a "
                             "complex number or a division by zero)", role)
    if _size(e, MAX_NODES) > MAX_NODES:
        raise Untranslatable(f"{where}: the expression is too large to be written as an equation", role)
    try:
        return _Writer(declared).doprint(e)
    except RecursionError:
        raise Untranslatable(f"{where}: the expression nests too deeply", role) from None
    except Untranslatable as err:
        raise Untranslatable(f"{where}: {err}", role) from None


# ---------- check ----------

def _check(model, tree, spec, values) -> dict:
    """Compare the equations with the code at points around the typical values, at the numbers
    the code contains, and on both sides of each condition of the equations; a difference in
    floats is settled in PRECISE_DIGITS-digit arithmetic."""
    ns = namespace(tree, symbolic=False)
    fns = roles(ns)
    states, outputs = list(model.states), list(model.outputs)
    counts = {"rhs": len(states), "observe": len(outputs), "init": len(states)}
    used = [r for r in ("rhs", "observe", "init") if fns[r] is not None and counts[r]]
    kernels = model.python_functions()
    relations = _relations(model)
    truth = model.evaluator(relations) if relations else None
    seen = np.zeros((len(relations), 2), bool)
    rows = {r: [] for r in used}
    with np.errstate(all="ignore"):
        for spread, pts in point_rounds(tree, spec, values, model):
            for t, y, p in pts:
                pv = np.array([p[n] for n in model.parameters], float)
                rv = np.array([p[n] for n in model.signals], float)
                compared = False
                for role in used:
                    ref = reference(fns[role], role, t, y, p, outputs if role == "observe" else states)
                    if ref is not None:
                        rows[role].append((ref, _kernel(kernels, role, t, y, pv, rv, counts[role]), (t, y, p)))
                        compared = True
                if compared and truth is not None:
                    try:
                        v = truth(t, y, pv, rv) != 0
                        seen[np.arange(len(relations)), v.astype(int)] = True
                    except Exception:
                        pass
            if all(len(rs) >= MIN_POINTS for rs in rows.values()):
                break
    precise = _Precise(tree, model)
    for role, rs in rows.items():
        if len(rs) < MIN_POINTS:
            raise ModelError(too_few(f"{role} gives a finite value", len(rs), spec, spread,
                                     " to check the reading"))
        bad = compare(rs, lambda row, js, scale, role=role: precise.settles(role, row, js, scale))
        if bad:
            raise Untranslatable(f"{role}: the equations read differ from the code ({bad}): a "
                                 "reading error of pysaem, please report it", role)
    return {"checked_points": min(len(rs) for rs in rows.values()),
            "conditions": [int(seen.all(axis=1).sum()), len(relations)], "spread": spread,
            "precise_points": len(precise.settled), "lost_points": len(precise.lost)}


def too_few(what, k, spec, spread, purpose="") -> str:
    """The refusal of a check with fewer than MIN_POINTS points."""
    head = f"{what} at {k} of the points drawn only, {MIN_POINTS} are needed{purpose}"
    if not spec["values_given"]:
        return f"{head}: give values= (typical values of {p_names(spec)})"
    return (f"{head}, even drawn closer to the values given (spread {spread:g} on the log scale): "
            "check that the code has a finite value at these values")


def _relations(model, limit=200) -> list:
    s = model.symbolic
    found = set()
    for e in [*s.rhs, *s.out, *s.init]:
        found |= {r for r in e.atoms(Relational) if r.free_symbols}
    return sorted(found, key=str)[:limit]


def _kernel(kernels, role, t, y, pv, rv, n):
    out = np.zeros(n)
    try:
        if role == "rhs":
            kernels["rhs"](t, y.copy(), pv, rv, np.zeros(n), out)
        elif role == "observe":
            kernels["out"](t, y.copy(), pv, rv, out)
        else:
            kernels["init"](pv, rv, out)
    except Exception:
        out[:] = np.nan
    return out


def call(fn, role, t, y, p):
    """What the code returns at one point (t, y, p)."""
    p = _Params({k: np.float64(v) for k, v in p.items()})
    return fn(p) if role == "init" else fn(np.float64(t), y.copy(), p)


def reference(fn, role, t, y, p, names):
    """The values the code gives at one point; None where it gives none: an error, a value that
    is not a finite real number, or a nan met on the way (not (nan < x) is true in Python,
    nan >= x is false: the code's own branches are not defined there). A refusal of the code
    is not a missing value."""
    try:
        with np.errstate(invalid="raise"):
            v = numbers(call(fn, role, t, y, p), names)
    except ModelError:
        raise
    except Exception:
        return None
    return v if np.all(np.isfinite(v)) else None


def numbers(v, names=None) -> np.ndarray:
    """The values a function returns as floats, a dict in the order of names."""
    if isinstance(v, dict):
        v = [v[n] for n in names] if names is not None else list(v.values())
    flat = []
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, (list, tuple)):
            stack.extend(reversed(x))
        elif isinstance(x, np.ndarray):
            stack.extend(reversed(x.ravel().tolist()))
        elif isinstance(x, (complex, np.complexfloating)) or isinstance(x, (str, bytes, dict)) or x is None:
            raise ValueError(f"{x!r} is not a real number")
        else:
            flat.append(float(x))
    return np.array(flat, float)


def compare(rows, settle=None):
    """The first difference between the values of the code and those compared, None when they
    agree within 1e-7 of each value and 1e-9 of the median size of each component.
    settle(row, components, scale), when given, clears the differences of one point."""
    for ref, got, point in rows:
        if len(ref) != len(got):
            return f"{len(got)} values against {len(ref)} from the code at {point_text(point)}"
    ref = np.array([r for r, _, _ in rows])
    got = np.array([g for _, g, _ in rows])
    scale = np.median(np.abs(ref), axis=0)  # one extreme point does not loosen the others
    bad = ~agree(got, ref, scale)
    for i in np.flatnonzero(bad.any(axis=1)):
        js = np.flatnonzero(bad[i])
        if settle is None or not settle(rows[i], js, scale[js]):
            j = js[0]
            return (f"value {j + 1}: {got[i, j]:.10g} against {ref[i, j]:.10g} from the code at "
                    f"{point_text(rows[i][2])}")
    return None


def agree(got, ref, scale):
    """Values that agree with those of the code: within 1e-7 of each and 1e-9 of scale."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(got) & (np.abs(got - ref) <= 1e-7 * np.abs(ref) + 1e-9 * scale)


# ---------- check in PRECISE_DIGITS-digit arithmetic ----------

class _NotPrecise(Exception):
    """A name without a stand-in in PRECISE_DIGITS-digit arithmetic: the point is not settled."""


class _PreciseModule:
    """math and numpy as seen in PRECISE_DIGITS-digit arithmetic: their scalar functions only."""

    def __init__(self, attrs):
        object.__setattr__(self, "_attrs", attrs)

    def __getattr__(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            raise _NotPrecise(name) from None


def _precise_functions() -> dict:
    """mpmath stand-ins of the scalar functions of math and numpy, with their meaning on floats;
    built in the precision they serve."""

    def fmod(a, b):  # the sign of a, as in C
        a, b = mp.mpf(a), mp.mpf(b)
        if not b:
            raise ValueError("fmod by zero")
        q = a / b
        return a - b * (mp.floor(q) if q >= 0 else mp.ceil(q))

    def copysign(a, b):
        if not b:
            raise _NotPrecise("copysign")  # the sign of a zero
        return abs(mp.mpf(a)) * (1 if b > 0 else -1)

    f = {"exp": mp.exp, "log": lambda x, base=None: mp.log(x) if base is None else mp.log(x, base),
         "log10": lambda x: mp.log(x, 10), "log2": lambda x: mp.log(x, 2), "log1p": mp.log1p,
         "expm1": mp.expm1, "exp2": lambda x: mp.mpf(2) ** x, "sqrt": mp.sqrt,
         "cbrt": lambda x: mp.sign(x) * mp.cbrt(abs(x)), "hypot": mp.hypot, "floor": mp.floor,
         "ceil": mp.ceil, "trunc": int, "fmod": fmod, "copysign": copysign, "sign": mp.sign,
         "pow": lambda a, b: mp.mpf(a) ** b, "power": lambda a, b: mp.mpf(a) ** b,
         "square": lambda x: mp.mpf(x) ** 2, "fabs": lambda x: abs(mp.mpf(x)), "abs": abs,
         "absolute": abs, "maximum": max, "minimum": min, "erf": mp.erf, "erfc": mp.erfc,
         "gamma": mp.gamma, "lgamma": lambda x: mp.log(abs(mp.gamma(x))), "radians": mp.radians,
         "degrees": mp.degrees, "isnan": mp.isnan, "isinf": mp.isinf, "isfinite": mp.isfinite,
         "where": lambda c, a, b: a if c else b, "pi": +mp.pi, "e": +mp.e, "tau": 2 * mp.pi,
         "inf": mp.inf, "nan": mp.nan}
    for n in ("sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh",
              "acosh", "atanh", "atan2"):
        f[n] = getattr(mp, n)
        if n.startswith("a"):
            f["arc" + n[1:]] = f[n]
    return f


def _precise_namespace(tree) -> dict:
    """The module run in PRECISE_DIGITS-digit arithmetic: math and numpy as _PreciseModule, float
    as mpf, any other module refused."""
    module = _PreciseModule(_precise_functions())

    def imp(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in ("math", "numpy"):
            raise _NotPrecise(name)
        return module

    safe = {k: getattr(builtins, k) for k in _BUILTINS}
    safe.update(print=lambda *a, **k: None, __import__=imp, float=lambda x=0.0: mp.mpf(x))
    ns = {"__builtins__": safe, "__name__": "model", "np": module, "math": module}
    exec(compile(tree, "<model>", "exec"), ns)  # noqa: S102 - checked by parse, see the module doc
    return ns


def _precise_args(model, point) -> tuple:
    """(t, y, p, r) of an evaluator of the model, in PRECISE_DIGITS-digit numbers."""
    t, y, p = point

    def mpf(vs):
        return np.array([mp.mpf(float(v)) for v in vs], dtype=object)
    return (mp.mpf(float(t)), mpf(y), mpf(p[n] for n in model.parameters),
            mpf(p[n] for n in model.signals))


class _Precise:
    """The code and, given a model, its equations evaluated in PRECISE_DIGITS-digit arithmetic at
    the points where floats disagree. settled holds the points where the two agree in that
    arithmetic; lost those where the floats of both have lost their values (the floats cannot
    judge the reading there)."""

    def __init__(self, tree, model=None):
        self.tree, self.model, self.settled, self.lost = tree, model, set(), set()
        self._fns, self._eqs = None, {}

    def code(self, role, point, names):
        """The values of the code at point, rounded to floats."""
        if self._fns is None:
            try:
                self._fns = roles(_precise_namespace(self.tree))
            except Exception:
                self._fns = {}
        if self._fns.get(role) is None:
            raise _NotPrecise(role)
        t, y, p = point
        pp = _Params({k: mp.mpf(float(v)) for k, v in p.items()})
        if role == "init":
            return numbers(self._fns[role](pp), names)
        ys = np.array([mp.mpf(float(v)) for v in y], dtype=object)
        return numbers(self._fns[role](mp.mpf(float(t)), ys, pp), names)

    def equations(self, role, point, js):
        """The values of components js of the equations at point, rounded to floats."""
        key = (role, tuple(js))
        if key not in self._eqs:
            s = self.model.symbolic
            exprs = {"rhs": s.rhs, "observe": s.out, "init": s.init}[role]
            self._eqs[key] = self.model.evaluator([exprs[j] for j in js],
                                                  _PreciseModule(_precise_functions()))
        return numbers(list(self._eqs[key](*_precise_args(self.model, point))))

    def settles(self, role, row, js, scale) -> bool:
        """Whether the difference in components js at the point of row is rounding: in
        PRECISE_DIGITS-digit arithmetic the equations agree with the code (a point within
        rounding of a condition, a cancellation), or the floats of both miss their values (a
        float that has lost its digits). The equations must have finite floats there."""
        ref, got, point = row
        if not np.all(np.isfinite(got[js])):
            return False
        names = list(self.model.outputs) if role == "observe" else list(self.model.states)
        try:
            with mp.workdps(PRECISE_DIGITS), np.errstate(all="ignore"):
                code = self.code(role, point, names)[js]
                eqs = self.equations(role, point, js)
        except Exception:
            return False
        if np.all(agree(eqs, code, scale)):
            self.settled.add(id(point))
            return True
        if np.all(np.isfinite(code)) and np.all(np.isfinite(eqs)) and not np.any(
                agree(ref[js], code, scale) | agree(got[js], eqs, scale)):
            self.lost.add(id(point))
            return True
        return False

    def code_lost(self, role, row, js, scale, names) -> bool:
        """Whether the floats of the code miss its values in components js at the point of row:
        its PRECISE_DIGITS-digit values are finite and differ from them."""
        ref, _, point = row
        try:
            with mp.workdps(PRECISE_DIGITS), np.errstate(all="ignore"):
                code = self.code(role, point, names)[js]
        except Exception:
            return False
        if not np.all(np.isfinite(code)) or np.any(agree(ref[js], code, scale)):
            return False
        self.lost.add(id(point))
        return True


def point_text(point) -> str:
    t, y, p = point
    shown = ", ".join(f"{k}={v:.6g}" for k, v in list(p.items())[:8])
    return f"t={t:.6g}, y={np.round(y, 6).tolist()}, p: {shown}"


def point_rounds(tree, spec, values, model=None):
    """(spread, points) of each round of draw_points: spread 1, then, when values were given,
    NARROW_ROUNDS rounds of points drawn closer to them, spread halved each round. The caller
    stops when it has enough points."""
    yield 1.0, draw_points(tree, spec, values, model)
    if spec["values_given"]:
        for k in range(1, NARROW_ROUNDS + 1):
            yield 2.0**-k, draw_points(tree, spec, values, model, spread=2.0**-k)


def draw_points(tree, spec, values, model=None, n=60, spread=1.0) -> list:
    """Points (t, y, p) at which the code is compared: the typical values, values drawn around
    them, zero and negative states, the numbers the code contains, and, given the model read,
    the edges of each of its conditions. With spread below 1, the points are drawn closer to
    the typical values (states 1): each value kept, or moved by a factor exp(N(0, spread)) (by
    N(0, spread) from zero); times keep their range."""
    rng = np.random.default_rng(_SEED + round(-math.log2(spread)))
    names = p_held(spec)
    n_states = model.n_states if model is not None else len(spec["states"])
    data = set(spec["covariates"]) | set(spec["regressors"]) | set(HISTORY)
    consts = sorted({float(c.value) for c in ast.walk(tree) if isinstance(c, ast.Constant)
                     and isinstance(c.value, (int, float)) and not isinstance(c.value, bool)
                     and abs(c.value) < 1e12})[:60]
    probe = sorted({*consts, *(c + d for c in consts for d in (-0.5, 0.5)), 0.0, 1.0, 2.0, 3.0, -1.0})

    def near(typical):
        if rng.uniform() < 0.5:
            return typical
        return typical * float(np.exp(rng.normal(0, spread))) if typical else float(rng.normal(0, spread))

    def value(name):
        if spread < 1:
            return near(values[name])
        u, typical = rng.uniform(), values[name]
        if name in data:
            if u < 0.5:
                return float(rng.choice(probe))
            if u < 0.65:
                return typical + float(rng.normal(0, 1)) * max(1.0, abs(typical))
        else:
            # a parameter keeps the sign of its typical value, as a log-normal one does
            signed = [c for c in probe if c * typical > 0] if typical else probe
            if u < 0.15 and signed:
                return float(rng.choice(signed))
            if u < 0.3:
                return typical * float(np.exp(rng.normal(0, 1.5))) if typical else float(rng.normal(0, 1))
        return typical * float(np.exp(rng.normal(0, 0.3))) if typical else float(rng.normal(0, 1))

    def state():
        if spread < 1:
            return near(1.0)
        u = rng.uniform()
        if u < 0.1:
            return 0.0
        if u < 0.2:
            return -float(rng.lognormal(0, 1))
        if u < 0.3:
            return float(rng.choice(probe))
        return float(rng.lognormal(0, 3 if u < 0.4 else 1))

    def time():
        u = rng.uniform()
        if u < 0.25:
            return float(rng.choice(probe))
        if u < 0.3:
            return -float(rng.uniform(0, 10))
        return float(rng.choice([1.0, 10.0, 100.0, 1000.0, 10000.0]) * rng.uniform())

    pts = [(1.0, np.ones(n_states), dict(values))] if spread == 1 else []
    for _ in range(n):
        t = time()
        y = np.array([state() for _ in range(n_states)])
        pts.append((t, y, {k: value(k) for k in names}))
    if model is not None:
        pts += _edges(model, pts)
    return pts


def _edges(model, pts, limit=600) -> list:
    """Points on both sides of each condition, and on it: a variable of the condition (t, a
    state or a value of p) moved from a point drawn to where the two sides are equal. The point
    on it is kept only where the two sides are equal in PRECISE_DIGITS-digit arithmetic (x = 65
    for x >= 65): elsewhere it lies within rounding of the condition, where the code and the
    equations may take either side, and the points on both sides check the two branches."""
    relations = _relations(model)
    if not relations:
        return []
    gaps = model.evaluator([r.lhs - r.rhs for r in relations])
    exact = {}
    sy = model._symbols
    place = {T: ("t", None)}
    place.update({sy[s]: ("y", i) for i, s in enumerate(model.states)})
    place.update({sy[n]: ("p", n) for n in model.parameters + model.signals})

    def args(point):
        t, y, p = point
        return (t, y, np.array([p[n] for n in model.parameters], float),
                np.array([p[n] for n in model.signals], float))

    def moved(point, where, v):
        t, y, p = point
        kind, key = where
        if kind == "t":
            return v, y, p
        if kind == "y":
            y = y.copy()
            y[key] = v
            return t, y, p
        return t, y, {**p, key: v}

    def gap(point, i):
        try:
            return float(gaps(*args(point))[i])
        except Exception:
            return math.nan

    def on(point, i):
        try:
            with mp.workdps(PRECISE_DIGITS):
                if i not in exact:
                    exact[i] = model.evaluator([relations[i].lhs - relations[i].rhs],
                                               _PreciseModule(_precise_functions()))
                return exact[i](*_precise_args(model, point))[0] == 0
        except Exception:
            return False

    out = []
    with np.errstate(all="ignore"):
        bases = [pt for pt in pts[:8] if np.all(np.isfinite(gaps(*args(pt))))][:2] if pts else []
        for i, rel in enumerate(relations):
            for sym in sorted(rel.free_symbols & set(place), key=str)[:3]:
                where = place[sym]
                for base in bases:
                    v = {"t": base[0], "y": base[1][where[1]] if where[0] == "y" else 0.0,
                         "p": base[2].get(where[1], 0.0)}[where[0]]
                    v = float(v)
                    for _ in range(8):
                        g = gap(moved(base, where, v), i)
                        if not math.isfinite(g) or abs(g) <= 1e-12 * (1 + abs(v)):
                            break
                        h = 1e-6 * max(1.0, abs(v))
                        slope = (gap(moved(base, where, v + h), i) - g) / h
                        if not math.isfinite(slope) or slope == 0:
                            break
                        v -= g / slope
                    if not math.isfinite(v):
                        continue
                    nice = float(f"{v:.12g}")
                    if abs(gap(moved(base, where, nice), i)) <= abs(gap(moved(base, where, v), i)):
                        v = nice
                    d = 1e-6 * max(1.0, abs(v))
                    out += [moved(base, where, v - d), moved(base, where, v + d)]
                    if on(moved(base, where, v), i):
                        out.append(moved(base, where, v))
                    if len(out) >= limit:
                        return out
    return out


Model.from_python = staticmethod(from_python)
