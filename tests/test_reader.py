"""Models written as Python functions: read as equations that give the code's own values, or
compiled as written, or refused with the cause."""

import json
import math
import threading
import warnings

import numpy as np
import pytest

from pysaem import reader
from pysaem.model import Model, ModelError
from pysaem.reader import CodeFails, TimeUp, Untranslatable, from_python


def integral(text):
    # SymPy 1.13 and later print a float that holds an integer as 1.0, earlier versions as 1
    import re
    return re.sub(r"\b(\d+)\.0\b", r"\1", text)


def read(code, **kw):
    """The model read as equations (no fallback) and the code run by Python."""
    m = from_python(code, fallback=False, **kw)
    ns = {}
    exec(code, ns)  # noqa: S102 - test code
    return m, ns


def outputs(m, t, y=(), p=()):
    o = np.zeros(m.n_outputs)
    pv = [p[n] for n in m.parameters]
    rv = [p[n] for n in m.regressors + m.covariates]
    m.python_functions()["out"](t, np.array(y, float), np.array(pv, float), np.array(rv, float), o)
    return o


def derivatives(m, t, y, p):
    dy = np.zeros(m.n_states)
    pv = [p[n] for n in m.parameters]
    rv = [p[n] for n in m.regressors + m.covariates]
    m.python_functions()["rhs"](t, np.array(y, float), np.array(pv, float), np.array(rv, float),
                                np.zeros(m.n_states), dy)
    return dy


def same(m, ns, points):
    """The equations give the code's values at each (t, y, p)."""
    for t, y, p in points:
        ref = np.ravel(np.asarray(ns["observe"](t, np.array(y, float), dict(p)), float))
        assert np.allclose(outputs(m, t, y, p), ref, rtol=1e-12, atol=1e-12), (t, y, p)


DECAY = """
import numpy as np

def rhs(t, y, p):
    return [-p['k'] * y[0]]

def observe(t, y, p):
    return y[0] / p['v']
"""


def test_functions_are_read_as_equations_and_checked():
    m = from_python(DECAY, states=["A"], parameters=["k", "v"], inputs={"u": "A"})
    assert m.odes == {"A": "-A*k"} and m.aux == {"out1": "A/v"}
    assert m.reading["form"] == "equations" and m.reading["checked_points"] >= reader.MIN_POINTS
    assert Model.from_dict(m.to_dict()).odes == m.odes  # equations: no code to run again


def test_conditions_become_where_in_the_order_of_the_code():
    code = """
def rhs(t, y, p):
    if t < p['t1']:
        k = p['k']
    elif y[0] > p['c']:
        k = 2 * p['k']
    else:
        k = 0.0
    return [-k * y[0]]

def observe(t, y, p):
    return y[0]
"""
    m = from_python(code, states=["A"], parameters=["k", "t1", "c"])
    assert m.odes["A"] == "where((t < t1), -A*k, where((A > c), -2*A*k, 0))"
    assert m.reading["conditions"][0] == m.reading["conditions"][1] == 2


def test_the_truth_of_a_number_is_that_it_is_not_zero():
    code = """
import numpy as np

def rhs(t, y, p):
    return [0.0]

def observe(t, y, p):
    if not y[0]:
        first = 0.0
    else:
        first = np.log(y[0] / p['v'])
    second = (y[0] / p['v']) or 1e-12
    return [first, second, bool(y[0]) + 0, sum(1 for c in y if c)]
"""
    m, ns = read(code, states=["X"], parameters=["v"], outputs=["a", "b", "c", "d"])
    same(m, ns, [(0.0, [0.0], {"v": 2.0}), (0.0, [3.0], {"v": 2.0})])


def test_membership_and_lookups_on_a_covariate():
    code = """
EFFECT = {1: 1.0, 2: 0.5, 3: 0.8}
SCALE = [1.0, 1.5, 2.0]

def observe(t, y, p):
    a = 0.5 if p['g'] in (2, 3) else 1.0
    b = EFFECT[p['g']] + EFFECT.get(p['g'], 9.0)
    c = SCALE[int(p['g']) - 1]
    return [a, b, c]
"""
    m, ns = read(code, covariates=["g"], outputs=["a", "b", "c"])
    same(m, ns, [(0.0, [], {"g": g}) for g in (1.0, 2.0, 3.0)])


def test_arrays_are_copied_and_numpy_keywords_are_kept():
    code = """
import numpy as np

def rhs(t, y, p):
    return [0.0, 0.0]

def observe(t, y, p):
    c = np.array(y)
    c[0] = max(c[0], 0.0)
    total = np.sum(y, where=y > 0)
    logs = np.log(y / p['v'], out=np.full_like(y, -20.0), where=y > 0)
    return [y[0] + c[0], total, *logs]
"""
    m, ns = read(code, states=["X", "Z"], parameters=["v"], outputs=["a", "b", "c", "d"])
    same(m, ns, [(0.0, y, {"v": 2.0}) for y in ([-0.5, 0.0], [2.0, 3.0])])


@pytest.mark.parametrize("expr, points", [
    ("round(y[0])", [0.5, 1.5, 2.5, -0.5, -2.5]),
    ("np.rint(y[0])", [0.5, 1.5, 2.5, -1.5]),
    ("y[0] // 2.0", [-7.0, -0.5, 3.0, 7.5]),
    ("math.fmod(y[0], 3.0)", [-5.0, -3.0, 4.0]),
    ("y[0] % 3.0", [-5.0, 4.0]),
    ("int(y[0])", [-2.5, 2.5]),
])
def test_rounding_and_remainders_follow_python(expr, points):
    code = f"import math\nimport numpy as np\n\ndef rhs(t, y, p):\n    return [0.0]\n\ndef observe(t, y, p):\n    return {expr}\n"
    m, ns = read(code, states=["X"])
    for x in points:
        assert outputs(m, 0.0, [x]) == pytest.approx(float(ns["observe"](0.0, np.array([x]), {}))), x


def test_tables_are_read_at_their_knots_and_outside():
    code = """
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

XS = [0.0, 24.0, 48.0, 96.0]
YS = [10.0, 5.0, 2.0, 1.0]
STEP = interp1d(XS, YS, kind='previous', bounds_error=False, fill_value=(10.0, 1.0))
NEAR = interp1d(XS, YS, kind='nearest', fill_value='extrapolate')
SPLINE = CubicSpline(XS, YS)

def observe(t, y, p):
    return [np.interp(t, XS, YS), STEP(t), NEAR(t), SPLINE(t)]
"""
    m, ns = read(code, outputs=["a", "b", "c", "d"])
    same(m, ns, [(t, [], {}) for t in (-5.0, 0.0, 12.0, 24.0, 30.0, 72.0, 96.0, 150.0)])


def test_the_code_s_own_errors_drop_their_branch_only():
    code = """
import numpy as np

def observe(t, y, p):
    if p['k'] <= 0:
        raise ValueError('k must be positive')
    assert p['v'] > 0, 'v must be positive'
    c = y[0] / p['v']
    if c > p['cmax']:
        return -p['k'] * y[0] - np.clip(c - p['cmax'], 0, None)
    return -p['k'] * y[0]
"""
    m, ns = read(code.replace("def observe", "def rhs(t, y, p):\n    return [0.0]\n\ndef observe"),
                 states=["X"], parameters=["k", "v", "cmax"], values={"cmax": 100.0})
    same(m, ns, [(0.0, [y], {"k": 1.0, "v": 1.0, "cmax": 100.0}) for y in (50.0, 200.0)])
    with pytest.raises(ModelError, match="for every value"):
        from_python("def observe(t, y, p):\n    raise RuntimeError('no')\n")


def test_and_or_stop_where_python_stops():
    code = """
TIMES = [0.02, 24.0]

def observe(t, y, p):
    given = [d for d in TIMES if t >= d]
    if len(given) > 0 and sum(given) / len(given) > 1.0:
        return 2.0
    return 1.0
"""
    m, ns = read(code)
    same(m, ns, [(t, [], {}) for t in (0.01, 0.5, 30.0)])


def test_initial_values_from_init_or_a_list():
    code = """
def rhs(t, y, p):
    return [-y[0]]

def observe(t, y, p):
    return y[0]

def init(p):
    return [p['dose'] if p['route'] in (1, 3) else 0.0]
"""
    m = from_python(code, states=["A"], parameters=["dose"], covariates=["route"])
    assert integral(m.states["A"]) == "where(((route == 1) or (route == 3)), dose, 0)"
    m = from_python("y0 = [100.0]\n" + code.split("def init")[0], states=["A"])
    assert integral(m.states["A"]) == "100"


def test_p_as_attributes_get_with_a_default_and_dict_returns():
    code = """
def rhs(t, y, p):
    return {'B': p.k * y[0], 'A': -p.k * y[0]}

def observe(t, y, p):
    return {'E': p.get('e0', 1.0) + y[0], 'C': y[1] / p['v']}
"""
    m = from_python(code, states=["A", "B"], parameters=["k", "v"])
    assert m.odes == {"A": "-A*k", "B": "A*k"} and m.outputs == ["E", "C"]
    assert {k: integral(v) for k, v in m.aux.items()} == {"E": "A + 1", "C": "B/v"}


def test_natural_python_is_read():
    code = '''
from __future__ import annotations
import math
import numpy as np
from typing import Sequence

K = np.array([[-1.0, 0.5], [1.0, -0.5]])

def helper(x: float, p: dict) -> float:
    return x / (p['km'] + x)

def rhs(t: float, y: Sequence[float], p: dict) -> list:
    flux = K @ y
    with np.errstate(divide='ignore'):
        sat = helper(y[0], p)
    dy = np.zeros_like(y)
    for i in range(len(y)):
        dy[i] = flux[i] - (sat if i == 0 else 0.0)
    return dy

def observe(t, y, p):
    a, b = y
    return math.exp(-t) * a + max(b, 0.0) + np.clip(a, 0.0, 1.0)

if __name__ == "__main__":
    import matplotlib
'''
    m = from_python(code, states=["A", "B"], parameters=["km"])
    assert m.reading["form"] == "equations"


def test_names_of_constants_and_functions_may_name_values():
    code = """
import numpy as np

def rhs(t, y, p):
    return [-p['gamma'] * y[0] + p['pi'] * np.sin(2 * np.pi * t / 24)]

def observe(t, y, p):
    return y[0]
"""
    m, ns = read(code, states=["A"], parameters=["gamma", "pi"])
    p = {"gamma": 0.3, "pi": 7.0}
    for t in (3.0, 6.0):
        ref = ns["rhs"](t, np.array([2.0]), p)[0]
        assert derivatives(m, t, [2.0], p)[0] == pytest.approx(ref)


def test_the_check_catches_a_wrong_reading(monkeypatch):
    # fmod read as Python's %: a reading error the check must find at a negative value
    monkeypatch.setitem(reader._FUNCS, "fmod", lambda a, b: __import__("sympy").Mod(a, b))
    reader._module.cache_clear()
    try:
        code = "import math\n\ndef rhs(t, y, p):\n    return [0.0]\n\ndef observe(t, y, p):\n    return math.fmod(y[0], 24.0)\n"
        with pytest.raises(Untranslatable, match="differ from the code"):
            from_python(code, states=["X"], fallback=False)
    finally:
        monkeypatch.undo()
        reader._module.cache_clear()


def test_the_check_needs_points_where_the_code_has_a_value():
    code = "import numpy as np\n\ndef observe(t, y, p):\n    return np.sqrt(p['v'] - 50.0)\n"
    with pytest.raises(ModelError, match="values="):
        from_python(code, parameters=["v"])
    assert from_python(code, parameters=["v"], values={"v": 100.0}).reading["form"] == "equations"
    with pytest.raises(ModelError, match="did you mean 'v'"):
        from_python(code, parameters=["v"], values={"V": 100.0})


def test_points_are_drawn_closer_to_the_values_given_until_the_code_has_a_value():
    # finite within 1e-2 of a = 1 only, as code that overflows away from its typical values
    code = "import math\n\ndef observe(t, y, p):\n    return math.sqrt(1e-4 - (p['a'] - 1.0) ** 2)\n"
    m = from_python(code, parameters=["a"], values={"a": 1.0}, fallback=False)
    assert m.reading["form"] == "equations" and m.reading["checked_points"] >= reader.MIN_POINTS
    assert m.reading["spread"] < 1
    with pytest.raises(ModelError, match="finite value at these values"):
        from_python(code.replace("1e-4", "-1e-4"), parameters=["a"], values={"a": 1.0})


# the check of floats against floats: code and equations differ by rounding only
CONDITION_ROUNDED = """
def observe(t, y, p):
    if p["x"] < p["g"] * 0.1 * 0.7:
        return 1.0
    return 0.0
"""
CANCELLATION = """
import math

def observe(t, y, p):
    z = (math.log(t + 1e-6) - p["mu"]) / math.exp(p["ls"])
    s = 1.0 - 0.5 * math.erfc(-z / math.sqrt(2.0))
    return math.exp(-0.5 * z * z) / ((t + 1e-6) * math.exp(p["ls"]) * math.sqrt(2 * math.pi)) / (s + 1e-30)
"""
SUM_ROUNDED = "def rhs(t, y, p):\n    return [(y[0] + 1e10 * p['z']) - 1e10 * p['z']]\n\ndef observe(t, y, p):\n    return y[0]\n"
CLOCK = """
import math

def observe(t, y, p):
    tacro, tnadir = math.exp(p["la"]), math.exp(p["ln"])
    t24 = (t - tacro) - 24.0 * math.floor((t - tacro) / 24.0)
    L1 = (tnadir - tacro) - 24.0 * math.floor((tnadir - tacro) / 24.0)
    phase = (t24 <= L1) * (t24 / (L1 + 1e-9)) + (t24 > L1) * (1.0 + (t24 - L1) / (24.0 - L1 + 1e-9))
    return p["b"] * (1.0 - 0.0024 * (p["AGE"] - 49.9)) + p["a"] * math.cos(math.pi * phase)
"""
ROUNDED = {
    "condition": (CONDITION_ROUNDED, dict(parameters=["x", "g"], values={"x": 1.0, "g": 1.3})),
    "cancellation": (CANCELLATION, dict(parameters=["mu", "ls"], values={"mu": 2.79, "ls": -0.37})),
    "sum": (SUM_ROUNDED, dict(states=["A"], parameters=["z"], values={"z": 1.0})),
    # la drawn at 49.9, a number of the code: (t - exp(la)) modulo 24 has no digit left in floats
    "clock": (CLOCK, dict(parameters=["la", "ln", "a", "b"], covariates=["AGE"],
                          values={"la": 2.24, "ln": 2.64, "a": 32.1, "b": 238.9, "AGE": 40.0})),
}


@pytest.mark.parametrize("case", ROUNDED)
def test_a_difference_of_rounding_is_not_a_reading_error(case):
    # a point within rounding of a condition (Mondick 2018), 1 - pnorm(z) at large z (Claret
    # 2014), a sum that loses its digits (Ippolito 2024), a clock modulo 24 h of exp(50)
    # (Gonzalez-Sales 2015): settled in 50-digit arithmetic, or no point on the condition
    code, kw = ROUNDED[case]
    m = from_python(code, fallback=False, **kw)
    assert m.reading["form"] == "equations" and m.reading["checked_points"] >= reader.MIN_POINTS
    if case in ("cancellation", "sum"):
        assert m.reading["precise_points"] > 0
    if case == "clock":
        assert m.reading["lost_points"] > 0


@pytest.mark.parametrize("case", ["cancellation", "clock"])
def test_a_small_reading_error_is_caught_where_floats_lose_digits(case, monkeypatch):
    # exp read 1e-6 too large: 50-digit arithmetic settles rounding, not this
    sp = __import__("sympy")
    monkeypatch.setitem(reader._FUNCS, "exp", lambda x: sp.exp(x) * sp.Float(1 + 1e-6))
    reader._module.cache_clear()
    try:
        code, kw = ROUNDED[case]
        with pytest.raises(Untranslatable, match="differ from the code"):
            from_python(code, fallback=False, **kw)
    finally:
        monkeypatch.undo()
        reader._module.cache_clear()


def test_a_reading_error_exactly_on_a_condition_is_caught(monkeypatch):
    # >= read as >: the equations differ from the code at AGE = 65 only, where both sides are
    # equal in exact arithmetic
    monkeypatch.setitem(reader._RELATIONS, ">=", reader.sp.Gt)
    code = "def observe(t, y, p):\n    if p['AGE'] >= 65.0:\n        return 2.0 * p['k']\n    return p['k']\n"
    with pytest.raises(Untranslatable, match="differ from the code"):
        from_python(code, parameters=["k"], covariates=["AGE"], values={"k": 1.0, "AGE": 50.0},
                    fallback=False)


def test_untranslatable_code_is_compiled_as_written_with_a_warning():
    code = """
import math

def rhs(t, y, p):
    r = y[0] / p['v']
    if math.isnan(r):
        r = 0.0
    return [-p['k'] * r]

def observe(t, y, p):
    return y[0]
"""
    with pytest.warns(UserWarning, match="compiles this model as written"):
        m = from_python(code, states=["A"], parameters=["k", "v"])
    assert m.reading["form"] == "compiled as written" and "isnan" in m.reading["reason"]
    with pytest.raises(Untranslatable):
        from_python(code, states=["A"], parameters=["k", "v"], fallback=False)


@pytest.mark.parametrize("code, kw, message", [
    ("def observe(t, y, p):\n    return p['v']\n", {}, "'v' is not declared"),
    ("def rhs(t, y, p):\n    return [0.0]\n\ndef observe(t, y, p):\n    return y[0]\n",
     {"states": ["A", "B"]}, "1 value for 2 states"),
    ("import numpy as np\n\ndef observe(t, y, p):\n    return np.random.normal()\n", {}, "random"),
    ("from scipy.integrate import odeint\n\ndef observe(t, y, p):\n    return odeint(None, 0, [0])\n",
     {}, "pysaem integrates rhs itself"),
    ("def rhs(y, t, p):\n    return [0.0]\n\ndef observe(t, y, p):\n    return y[0]\n",
     {"states": ["A"]}, "t comes first"),
    ("def rhs(t, y, p):\n    return [-y['A']]\n\ndef observe(t, y, p):\n    return y[0]\n",
     {"states": ["A"]}, r"y\[0\], not y\['name'\]"),
    ("import os\n\ndef observe(t, y, p):\n    return 1.0\n", {}, "import of 'os'"),
    ("def observe(t, y, p):\n    return eval('1')\n", {}, "eval cannot be used"),
    ("def model(t, y, p):\n    return 1.0\n", {}, "defines no observe.*it defines model"),
])
def test_errors_of_the_code_are_reported_with_their_cause(code, kw, message):
    with pytest.raises(ModelError, match=message):
        from_python(code, **kw)


def test_modules_give_their_listed_functions_only(tmp_path):
    target = tmp_path / "written"
    for code in (f"from numpy import save\n\ndef observe(t, y, p):\n    save({str(target)!r}, [1.0])\n    return 1.0\n",
                 f"from scipy.io import savemat\n\ndef observe(t, y, p):\n    savemat({str(target)!r}, {{}})\n    return 1.0\n",
                 f"import numpy.distutils\n\ndef observe(t, y, p):\n    numpy.distutils.exec_command.exec_command('touch {target}')\n    return 1.0\n",
                 "def observe(t, y, p):\n    return [y.tofile('x')]\n"):
        with pytest.raises(ModelError):
            from_python(code)
    assert not list(tmp_path.iterdir())


def test_modules_hide_what_is_behind_them():
    for path in reader._MODULES:
        for symbolic in (True, False):
            module = reader._module(path, symbolic)
            assert isinstance(getattr(module, "real"), reader._Refused), (path, symbolic)


@pytest.mark.parametrize("code", ["def observe(t, y, p):\n    return 1.0\x00\n",
                                  "def observe(t, y, p):\n    return " + "-" * 2000 + "p['a']\n"])
def test_code_python_cannot_parse_is_refused_with_a_model_error(code):
    with pytest.raises(ModelError, match="cannot be parsed"):
        from_python(code, parameters=["a"])


def test_code_that_never_ends_is_stopped_in_any_thread(monkeypatch):
    monkeypatch.setattr(reader, "READ_SECONDS", 1)
    code = "def observe(t, y, p):\n    x = p['a']\n    while x > 0.5:\n        pass\n    return x\n"
    errors = []

    def run():
        try:
            from_python(code, parameters=["a"])
        except ModelError as e:
            errors.append(e)
    thread = threading.Thread(target=run)
    thread.start()
    thread.join(10)
    assert not thread.is_alive() and isinstance(errors[0], TimeUp)


def test_readings_in_threads_do_not_mix():
    code = """
def rhs(t, y, p):
    a = p['k1'] if y[0] > p['c1'] else p['k2']
    b = p['k3'] if y[1] > p['c2'] else p['k4']
    return [-a * y[0], a * y[0] - b * y[1]]

def observe(t, y, p):
    return y[1]
"""
    kw = dict(states=["A", "B"], parameters=["k1", "k2", "k3", "k4", "c1", "c2"], check=False)
    expected = from_python(code, **kw).odes
    found = []
    threads = [threading.Thread(target=lambda: found.append(from_python(code, **kw).odes))
               for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert found == [expected] * 6


def test_expressions_that_grow_are_compiled_as_written():
    code = """
def free(c0, r, kd):
    c = c0
    for _ in range(30):
        c = c - (c + r * c / (kd + c) - c0) / (1 + r * kd / (kd + c) ** 2)
    return c

def rhs(t, y, p):
    return [-p['k'] * free(y[0], p['r'], p['kd'])]

def observe(t, y, p):
    return y[0]
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = from_python(code, states=["A"], parameters=["k", "r", "kd"])
    assert m.reading["form"] == "compiled as written" and "too large" in m.reading["reason"]


def test_a_specification_holding_code_is_read_on_request_only():
    d = {"python": DECAY, "states": ["A"], "parameters": ["k", "v"], "outputs": "C"}
    with pytest.raises(ModelError, match="python=True"):
        Model.from_dict(d)
    m = Model.from_dict(d, python=True)
    assert m.outputs == ["C"] and m.odes == {"A": "-A*k"}


def _rhs(t, y, p):
    return [-p["k"] * y[0]]


def _observe(t, y, p):
    return {"C": y[0] / p["v"]}


def test_functions_given_as_callables_are_read():
    m = from_python(rhs=_rhs, observe=_observe, states=["A"], parameters=["k", "v"])
    assert m.odes == {"A": "-A*k"} and m.outputs == ["C"]
    with pytest.raises(ModelError, match="def at module level"):
        from_python(rhs=lambda t, y, p: [0.0], observe=_observe, states=["A"], parameters=["k", "v"])


@pytest.mark.parametrize("body, times", [
    ("DOSES = [0.0, 24.0, 48.0]\n\ndef observe(t, y, p):\n"
     "    return p['a'] * (t - max((d for d in DOSES if d <= t), default=-1.0))\n", [-5.0, 30.0, 50.0]),
    ("def observe(t, y, p):\n    return float(np.piecewise(t, [t >= 0, t >= 24, t >= 48], [p['a'], 2.0, 3.0]))\n",
     [10.0, 30.0, 60.0]),
    ("def observe(t, y, p):\n    return p['a'] * np.interp(t, [0.0, 24.0, 24.0, 48.0], [1.0, 1.0, 2.0, 2.0])\n",
     [12.0, 24.0, 30.0]),
    ("W = [1.0, 2.0]\n\ndef observe(t, y, p):\n    return p['a'] * all(W) + any([0.0, 3.0]) + np.logical_and(2.0, 3.0)\n",
     [1.0]),
    ("N = 3\nW = np.median([60.0, 70.0, 80.0])\nM = np.linalg.inv(np.array([[2.0, 0.0], [0.0, 4.0]]))\n\n"
     "def observe(t, y, p):\n    return np.float64(p['a']) * t ** N / math.factorial(N) * W * M[1, 1]\n", [1.0, 2.0]),
    ("D = np.array([0.0, 24.0, 48.0])\n\ndef observe(t, y, p):\n    i = np.searchsorted(D, t, side='right') - 1\n"
     "    return p['a'] * (t - D[max(i, 0)]) * (2.0 if np.isclose(t % 24.0, 0.0) else 1.0) + round(t / 24.0, 0)\n",
     [10.0, 24.0, 30.0, 48.0]),
    ("def observe(t, y, p):\n    return np.interp(t, [0.0, 24.0], [p['a'], 2 * p['a']]) + np.log1p(p['a'] * 1e-12)"
     " + np.expm1(p['a'] * 1e-12) + norm.cdf(-10 * p['a']) + norm.sf(10 * p['a'])\n", [6.0, 30.0]),
])
def test_python_semantics_of_iterators_tables_and_functions_are_kept(body, times):
    code = "import math\nimport numpy as np\nfrom scipy.stats import norm\n\n" + body
    m, ns = read(code, parameters=["a"], values={"a": 1.0})
    for t in times:
        ref = float(ns["observe"](t, np.array([]), {"a": 1.0}))
        assert outputs(m, t, [], {"a": 1.0})[0] == pytest.approx(ref, rel=1e-12, abs=0), t


@pytest.mark.parametrize("expr", [
    "[p['a'] > 0.5, p['b'] > 0.5].count(True)",
    "len(list(filter(None, [p['a'], p['b']])))",
    "2.0 if [p['a'], p['b']] == [1.0, 1.0] else 1.0",
    "{True: 0.6, False: 1.0}[p['a'] == 1] + {True: 2.0}.get(p['b'] == 1, 5.0)",
    "(1.0 if y.any() else 0.0) + y.clip(0)[0] + y[0].item()",
])
def test_truth_equality_and_counts_of_containers_follow_python(expr):
    code = f"def rhs(t, y, p):\n    return [0.0]\n\ndef observe(t, y, p):\n    return [{expr}]\n"
    m, ns = read(code, states=["X"], covariates=["a", "b"], values={"a": 1.0, "b": 0.0})
    same(m, ns, [(1.0, [x], {"a": a, "b": b}) for a in (0.0, 1.0) for b in (0.0, 1.0) for x in (-1.0, 0.0, 2.0)])


def test_a_guard_in_an_operand_python_does_not_reach_is_not_raised():
    code = """
def adult(wt):
    assert wt >= 40, "adult formula"
    return wt < 120

def observe(t, y, p):
    if p['WT'] >= 40 and adult(p['WT']):
        return p['CL'] * p['WT'] / 70
    return p['CL']
"""
    m, ns = read(code, parameters=["CL"], covariates=["WT"], values={"CL": 1.0, "WT": 70.0})
    same(m, ns, [(1.0, [], {"CL": 1.0, "WT": w}) for w in (20.0, 70.0, 150.0)])


def test_dict_returns_built_in_another_order_keep_their_names():
    code = """
def rhs(t, y, p):
    return [-p['k'] * y[0]]

def observe(t, y, p):
    if t < p['tlag']:
        return {'Cc': 0.0, 'E': p['E0']}
    out = {}
    out['E'] = p['E0'] + y[0]
    out['Cc'] = y[0] / p['V']
    return out
"""
    vals = {"k": 0.1, "tlag": 0.5, "E0": 10.0, "V": 5.0}
    m, ns = read(code, states=["A"], parameters=list(vals), values=vals)
    assert sorted(m.outputs) == ["Cc", "E"]
    for t in (0.2, 2.0):
        ref = ns["observe"](t, np.array([10.0]), vals)
        assert outputs(m, t, [10.0], vals).tolist() == [ref[k] for k in m.outputs]


@pytest.mark.parametrize("expr", ["CL.get(str(int(p['G'])), 2.0)", "len({p['G'], p['H']})"])
def test_texts_and_sets_of_values_are_not_read_as_equations(expr):
    code = f"CL = {{'1': 2.0, '2': 3.5}}\n\ndef rhs(t, y, p):\n    return [-{expr} * y[0]]\n\ndef observe(t, y, p):\n    return y[0]\n"
    with pytest.raises(Untranslatable):
        from_python(code, states=["A"], covariates=["G", "H"], values={"G": 1.0, "H": 1.0}, fallback=False)


def test_one_extreme_point_does_not_loosen_the_check_of_the_others():
    point = (0.0, np.array([]), {})
    rows = [(np.array([1e300]), np.array([1e300]), point), (np.array([1.0]), np.array([1.0]), point),
            (np.array([2.0]), np.array([2.5]), point)]
    assert "2.5" in reader.compare(rows)


HAZARD = """
import numpy as np

def rhs(t, y, p):
    return [-p["k"] * y[0]]

def observe(t, y, p):
    c = y[0] / p["V"]
    return {"C": c, "haz": p["lam0"] * p["kw"] * t ** (p["kw"] - 1) * np.exp(p["beta"] * c)}
"""
HAZARD_KW = dict(states=["A"], parameters=["k", "V", "lam0", "beta", "kw"], inputs={"u": "A"},
                 values={"k": 0.1, "V": 10.0, "lam0": 0.01, "beta": 0.1, "kw": 1.5},
                 hazards={"haz": "repeated"})


def test_a_hazard_returned_by_observe_is_read_as_equations():
    m, ns = read(HAZARD, **HAZARD_KW)
    assert m.hazards == {"haz": "repeated"} and m.outputs == ["C", "haz"]
    assert m.reading["form"] == "equations" and m.reading["checked_points"] >= reader.MIN_POINTS
    vals = HAZARD_KW["values"]
    for t, a in ((0.5, 3.0), (4.0, 30.0)):
        ref = ns["observe"](t, np.array([a]), vals)
        assert np.allclose(outputs(m, t, [a], vals), [ref[k] for k in m.outputs], rtol=1e-12)
    assert Model.from_json(m.to_json()).hash == m.hash
    assert Model.from_dict({"python": HAZARD, **HAZARD_KW}, python=True).hash == m.hash


def test_hazards_need_equations_and_a_bad_name_is_not_compiled_as_written():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ModelError, match="'hz' is not an output") as e:
            from_python(HAZARD, **{**HAZARD_KW, "hazards": {"hz": "repeated"}})
    assert not isinstance(e.value, Untranslatable)
    assert not [w for w in caught if "as written" in str(w.message)]
    guarded = HAZARD.replace('    c = y[0] / p["V"]\n',
                             '    try:\n        c = y[0] / p["V"]\n    except Exception:\n'
                             '        c = 0.0\n')
    with pytest.raises(ModelError, match="hazards need a model read as equations"):
        from_python(guarded, **HAZARD_KW)


def test_an_input_target_is_a_state_name_or_a_list_of_them():
    code = "def rhs(t, y, p):\n    return [-p['k'] * y[0], -p['k'] * y[1]]\n\n\ndef observe(t, y, p):\n    return y[0] + y[1]\n"
    with pytest.raises(ModelError, match="input 'input': its target is a state name or a list"):
        from_python(code, states=["A", "B"], parameters=["k", "f"], outputs=["Y"],
                    inputs={"input": {"target": ["A", 2]}}, values={"k": 0.1, "f": 0.5})
    m = from_python(code, states=["A", "B"], parameters=["k", "f"], outputs=["Y"],
                    inputs={"input": {"target": ["A", "B"], "scale": ["p['f']", "1 - p['f']"]}},
                    values={"k": 0.1, "f": 0.5})
    assert m.inputs["input"] == {"target": ["A", "B"], "scale": ["f", "1 - f"]}
    # one object per target, as language models write it: the same input
    m = from_python(code, states=["A", "B"], parameters=["k", "f"], outputs=["Y"],
                    inputs={"input": [{"target": "A", "fraction": "p['f']"}, {"target": "B", "scale": "1 - p['f']", "delay": "2"}]},
                    values={"k": 0.1, "f": 0.5})
    assert m.inputs["input"] == {"target": ["A", "B"], "scale": ["f", "1 - f"], "delay": [0, "2"]}
    # or a map of shares
    m = from_python(code, states=["A", "B"], parameters=["k", "f"], outputs=["Y"],
                    inputs={"input": {"target": {"A": "p['f']", "B": "1 - p['f']"}}},
                    values={"k": 0.1, "f": 0.5})
    assert m.inputs["input"] == {"target": ["A", "B"], "scale": ["f", "1 - f"]}
    with pytest.raises(ModelError, match="the shares are given twice"):
        from_python(code, states=["A", "B"], parameters=["k", "f"], outputs=["Y"],
                    inputs={"input": {"target": {"A": 0.5, "B": 0.5}, "scale": [1, 1]}},
                    values={"k": 0.1, "f": 0.5})


# ---------- signals of the amounts received ----------

WEIBULL = """
import numpy as np

def rhs(t, y, p):
    s = t - p['t_last']
    w = p['amount_last'] * p['b'] / p['a'] * (s / p['a']) ** (p['b'] - 1) * np.exp(-(s / p['a']) ** p['b'])
    return [-w, w - p['k'] * y[1]]

def observe(t, y, p):
    return [y[1] + 0 * p['n_amounts'] * p['t_first']]
"""


def _history_predictions(model):
    import pandas as pd
    from pysaem.data import Dataset
    from pysaem.solve import Problem
    rows = [{"ID": 1, "TIME": t, "DV": ".", "AMT": a} for t, a in ((0, 100), (12, 50))]
    rows += [{"ID": 1, "TIME": t, "DV": 1.0, "AMT": "."} for t in (1, 5, 12, 13, 20)]
    prob = Problem(model, Dataset.from_records(pd.DataFrame(rows).sort_values("TIME", kind="stable")))
    pred, status = prob.predict(np.array([[3.0, 1.8, 0.15]]))
    assert (status == 0).all()
    return prob.gather(pred)[0]


def test_the_signals_of_the_amounts_are_names_of_p():
    kw = dict(states=["Ad", "Ac"], parameters=["a", "b", "k"], inputs={"dose": "Ad"},
              values={"a": 3.0, "b": 1.8, "k": 0.15})
    m, _ = read(WEIBULL, **kw)
    assert m.symbolic.history and "t_last" in m.odes["Ac"] and m.reading["checked_points"] >= 10
    eq = Model(states={"Ad": "0", "Ac": "0"}, parameters=["a", "b", "k"],
               aux={"w": "amount_last*b/a*((t - t_last)/a)**(b - 1)*exp(-((t - t_last)/a)**b)"},
               odes={"Ad": "-w", "Ac": "w - k*Ac"}, outputs=["Ac"],
               inputs={"dose": {"target": "Ad"}})
    ref = _history_predictions(eq)
    np.testing.assert_allclose(_history_predictions(m), ref, rtol=1e-9)
    # compiled as written: the code reads them from r too
    guard = WEIBULL.replace("    s = t", "    if np.isnan(y[0]):\n        y = y * 0.0\n    s = t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        native = from_python(guard, **kw)
    assert native.reading["form"] == "compiled as written" and native.symbolic.history
    np.testing.assert_allclose(_history_predictions(native), ref, rtol=1e-6)


@pytest.mark.parametrize("field", ["parameters", "covariates", "regressors", "states"])
def test_a_value_of_p_cannot_take_the_name_of_a_signal_of_the_amounts(field):
    kw = {"states": ["A"], "parameters": ["k"], field: ["amount_last"]}
    with pytest.raises(ModelError, match="'amount_last' is reserved: it is the amount of"):
        from_python("def rhs(t, y, p):\n    return [-p['k'] * y[0]]\n\n"
                    "def observe(t, y, p):\n    return [y[0]]\n", **kw)
