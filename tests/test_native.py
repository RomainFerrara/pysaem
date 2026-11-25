"""Models compiled as written (native.py): the same predictions as the equations of the same
code, through the wrappers the solver calls."""

import json
import math
from dataclasses import replace
import warnings

import numpy as np
import pandas as pd
import pytest

from pysaem.data import Dataset
from pysaem.model import Model, ModelError
from pysaem.native import NativeModel
from pysaem.reader import from_python
from pysaem.solve import Problem
from test_reader import CLOCK, ROUNDED

# a test of nan on a value of the states cannot be written as equations: the code is compiled
GUARD = "    if math.isnan(y[0]):\n        y = y * 0.0\n"

TWO = """
import math
import numpy as np

def rhs(t, y, p):
{guard}    a, b = y[0], y[1]
    k = p['k1'] * p['k2'] / (p['ts'] + p['k2'])
    return [-k * a, k * a - p['c'] / p['v'] * b * (p['w'] / 70.0) ** 0.75]

def observe(t, y, p):
    return [y[1] / p['v'], y[0] + p['r']]
"""
KW = dict(states=["A", "B"], parameters=["k1", "k2", "ts", "c", "v"], covariates=["w"],
          regressors=["r"], outputs=["C", "D"], inputs={"u": {"target": "A", "delay": "ts"}},
          values={"k1": 1.0, "k2": 0.5, "ts": 2.0, "c": 2.0, "v": 10.0, "w": 70.0})


def both(code, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        native = from_python(code.replace("{guard}", GUARD), **kw)
    return native, from_python(code.replace("{guard}", ""), **kw)


def predictions(model, df, psi, covariates=(), regressors=(), method="auto"):
    prob = Problem(model, Dataset.from_records(df, covariates=covariates, regressors=regressors),
                   method=method)
    pred, status = prob.predict(np.atleast_2d(np.asarray(psi, float)))
    return prob.gather(pred)[0], status


def frame(times, doses, **columns):
    rows = [{"ID": 1, "TIME": t, "DV": ".", "AMT": a, **columns} for t, a in doses]
    rows += [{"ID": 1, "TIME": t, "DV": 1.0, "AMT": ".", **columns} for t in times]
    return pd.DataFrame(rows).sort_values("TIME", kind="stable")


@pytest.mark.parametrize("method", ["explicit", "implicit"])
def test_native_predictions_match_the_equations_of_the_same_code(method):
    native, eq = both(TWO, **KW)
    assert isinstance(native, NativeModel) and native.reading["form"] == "compiled as written"
    assert not isinstance(eq, NativeModel)
    # code compiled as written is not read: every parameter may reach every output
    assert native.dependencies("D") == native.parameters
    assert eq.dependencies("D") == ["k1", "k2", "ts"]
    df = frame([0.5, 1, 3, 6, 12], [(0, 100)], w=85.0, r=0.3)
    psi = [1.2, 0.4, 2.0, 2.5, 12.0]
    f_native, s_native = predictions(native, df, psi, ["w"], ["r"], method)
    f_eq, s_eq = predictions(eq, df, psi, ["w"], ["r"], method)
    assert (s_native == 0).all() and (s_eq == 0).all()
    assert np.allclose(f_native, f_eq, rtol=1e-5)


def test_a_switch_in_time_of_the_code_stops_the_solver_as_the_equations_do():
    code = TWO.replace("p['k1'] * p['k2'] / (p['ts'] + p['k2'])", "p['k1'] if t < p['ts'] else p['k2']")
    native, eq = both(code, **KW)
    assert [str(b) for b in native.symbolic.breakpoints] == [str(b) for b in eq.symbolic.breakpoints] == ["_p2"]
    df = frame([0.5, 1, 3, 6, 12], [(0, 100)], w=85.0, r=0.3)
    psi = [1.2, 0.4, 2.0, 2.5, 12.0]
    for method in ("explicit", "implicit"):
        f_native, _ = predictions(native, df, psi, ["w"], ["r"], method)
        f_eq, _ = predictions(eq, df, psi, ["w"], ["r"], method)
        assert np.allclose(f_native, f_eq, rtol=1e-5)


@pytest.mark.parametrize("returns", [
    "return {'B': k * y[0] - y[1], 'A': -k * y[0]}",
    "dy = [0] * 2\n    dy[0] = -k * y[0]\n    dy[1] = k * y[0] - y[1]\n    return dy",
    "return np.array([-k * y[0], k * y[0] - y[1]])",
])
def test_native_reads_the_forms_python_returns(returns):
    code = ("import math\nimport numpy as np\n\ndef rhs(t, y, p):\n{guard}    k = p.k\n    "
            + returns + "\n\ndef observe(t, y, p):\n    return {'X': y[1], 'Y': y[0]}\n")
    native, eq = both(code, states=["A", "B"], parameters=["k"])
    assert native.outputs == eq.outputs == ["X", "Y"]
    df = frame([0.5, 2, 5], [(0, 10)])
    f_native, _ = predictions(native, df.assign(AMT="."), [0.7])
    f_eq, _ = predictions(eq, df.assign(AMT="."), [0.7])
    assert np.allclose(f_native, f_eq, rtol=1e-6)


def test_native_counts_and_errors_of_the_code_are_reported():
    with pytest.raises(ModelError, match="1 value for 2 states"):
        from_python("import math\n\ndef rhs(t, y, p):\n" + GUARD + "    return [-y[0]]\n\n"
                    "def observe(t, y, p):\n    return y[0]\n", states=["A", "B"])


def test_an_error_of_the_compiled_code_fails_the_solve():
    code = ("import math\n\nRATE = [1.0, 2.0, 3.0]\n\ndef rhs(t, y, p):\n" + GUARD
            + "    return [RATE[int(t // 24)] - p['k'] * y[0]]\n\ndef observe(t, y, p):\n    return y[0]\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = from_python(code, states=["R"], parameters=["k"])
    f, status = predictions(m, frame([10.0, 30.0, 100.0], []), [0.1])
    assert (status != 0).any() and np.isnan(f[-1])  # Python raises past the table too


def test_a_native_model_without_time_in_rhs_reaches_a_steady_state():
    code = ("import math\n\ndef rhs(t, y, p):\n" + GUARD + "    return [-p['k'] * y[0]]\n\n"
            "def observe(t, y, p):\n    return y[0]\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = from_python(code, states=["A"], parameters=["k"], inputs={"u": "A"})
    assert m.symbolic.autonomous
    t = np.array([0.5, 2.0, 6.0])
    df = pd.DataFrame([{"ID": 1, "TIME": 0, "DV": ".", "AMT": 100, "SS": 1, "II": 12.0}]
                      + [{"ID": 1, "TIME": x, "DV": 1.0, "AMT": ".", "SS": 0, "II": "."} for x in t])
    f, status = predictions(m, df, [0.2])
    assert (status == 0).all()
    assert np.allclose(f, 100 * np.exp(-0.2 * t) / (1 - np.exp(-0.2 * 12)), rtol=1e-4)


def test_a_native_model_comes_back_from_its_json():
    native, _ = both(TWO, **KW)
    with pytest.warns(UserWarning, match="as written"):
        again = Model.from_json(native.to_json(), python=True)
    assert isinstance(again, NativeModel) and again.hash == native.hash
    assert json.loads(native.to_json())["values"]["v"] == 10.0
    with pytest.raises(ModelError, match="no equations"):
        native.equations()


SWITCH = """
import math

def rhs(t, y, p):
{guard}    rate = p['R'] if t > p['T1'] else 0.0
    return [rate - p['k'] * y[0], p['kf'] * (y[0] - y[1])]

def observe(t, y, p):
    return y[1]
"""
SWITCH_KW = dict(states=["A", "B"], parameters=["R", "T1", "k", "kf"],
                 values={"R": 10.0, "T1": 2.0, "k": 0.1, "kf": 1e4})


@pytest.mark.parametrize("binding", ["rhs, other = _rhs, 0\n", "def make():\n    return _rhs\n\nrhs = make()\n"])
def test_time_in_rhs_is_seen_wherever_rhs_is_bound(binding):
    native, eq = both(SWITCH.replace("def rhs(", "def _rhs(") + binding, **{**SWITCH_KW, "values": {
        "R": 10.0, "T1": 2.0, "k": 0.1, "kf": 1.0}})
    assert not native.symbolic.autonomous
    df = frame([1.0, 10.0], [])
    for method in ("explicit", "implicit"):
        assert np.allclose(predictions(native, df, [10.0, 2.0, 0.1, 1.0], method=method)[0],
                           predictions(eq, df, [10.0, 2.0, 0.1, 1.0], method=method)[0], rtol=1e-6)


def test_a_stiff_model_switched_on_from_states_at_zero():
    native, eq = both(SWITCH, **SWITCH_KW)
    df = frame([1.0, 10.0, 20.0], [])
    for method in ("auto", "implicit"):
        f_native, status = predictions(native, df, [10.0, 2.0, 0.1, 1e4], method=method)
        assert (status == 0).all()
        assert np.allclose(f_native, predictions(eq, df, [10.0, 2.0, 0.1, 1e4], method=method)[0], rtol=1e-6)


@pytest.mark.parametrize("test", ["p['T1'] < t < p['T2']", "t - p['T1'] > 0 and p['T2'] - t > 0"])
def test_switches_of_chained_and_shifted_comparisons_stop_the_solver(test):
    code = SWITCH.replace("rate = p['R'] if t > p['T1'] else 0.0", f"rate = p['R'] if {test} else 0.0")
    kw = {**SWITCH_KW, "parameters": ["R", "T1", "T2", "k", "kf"],
          "values": {"R": 10.0, "T1": 2.0, "T2": 6.0, "k": 0.1, "kf": 1.0}}
    native, eq = both(code, **kw)
    assert sorted(str(b) for b in native.symbolic.breakpoints) == sorted(f"_p{1 + i}" for i in range(2))
    df = frame([1.0, 4.0, 10.0], [])
    for method in ("explicit", "implicit"):
        f_native, status = predictions(native, df, [10.0, 2.0, 6.0, 0.1, 1.0], method=method)
        assert (status == 0).all()
        assert np.allclose(f_native, predictions(eq, df, [10.0, 2.0, 6.0, 0.1, 1.0], method=method)[0], rtol=1e-6)


LAG = """
import math

def rhs(t, y, p):
{guard}    lag = math.exp(p['llag'])
    tad = t - p['t_last']
    ka = p['ka1'] if tad < lag else p['ka2']
    return [-ka * y[0], ka * y[0] - p['k'] * y[1]]

def observe(t, y, p):
    lag = 0.0
    return y[1] + lag
"""


def test_a_switch_time_through_names_of_the_code_and_t_last_stops_the_solver():
    # names assigned once in their function read as their value: the switch is at t_last + lag
    kw = dict(states=["A", "C"], parameters=["llag", "ka1", "ka2", "k"], outputs=["C"],
              inputs={"dose": {"target": "A"}}, t0=0.0,
              values={"llag": 1.0, "ka1": 0.2, "ka2": 1.0, "k": 0.15})
    native, eq = both(LAG, **kw)
    assert isinstance(native, NativeModel)
    assert [str(b) for b in native.symbolic.breakpoints] == [str(b) for b in eq.symbolic.breakpoints] \
        == ["_r0 + exp(_p0)"]
    df = frame([1.0, 3.0, 6.0, 13.0, 15.5, 25.0, 27.0, 36.0], [(0, 100), (12, 50), (24, 80)])
    psi = [math.log(3.1), 0.2, 1.0, 0.15]
    for method in ("explicit", "implicit"):
        f_native, status = predictions(native, df, psi, method=method)
        assert (status == 0).all()
        assert np.allclose(f_native, predictions(eq, df, psi, method=method)[0], rtol=1e-6)

def test_each_jacobian_entry_shows_the_change_of_its_own_row():
    # states at zero after the switch: the step that shows the change of dB/dt loses that of
    # dA/dt = 10 - 0.1 A in rounding
    from pysaem.solve import _jac_fd, compile_model
    native, _ = both(SWITCH, **SWITCH_KW)
    J = np.zeros((2, 2))
    _jac_fd(compile_model(native)[0], 2.5, np.zeros(2), np.array([10.0, 2.0, 0.1, 1e4]), np.zeros(0),
            J, np.full(2, 1e-294))
    assert np.allclose(J, [[-0.1, 0.0], [1e4, -1e4]], rtol=1e-6)


def test_a_short_array_returned_by_the_code_fails_the_solve():
    code = ("import math\nimport numpy as np\n\ndef rhs(t, y, p):\n" + GUARD
            + "    if 7.0 < t < 7.4:\n        return np.array([-y[0]])\n"
            "    return np.array([-y[0], y[0] - y[1]])\n\ndef observe(t, y, p):\n    return y[1]\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = from_python(code, states=["A", "B"], inputs={"d": "A"})
    for method in ("explicit", "implicit"):
        assert (predictions(m, frame([6.0, 8.0], [(0.0, 100.0)]), np.zeros(0), method=method)[1] != 0).all()


@pytest.mark.parametrize("body", [
    "    a = y[ORDER[1]]\n    return [-p['k'] * a, p['k'] * a - y[1]]",
    "    dy = [0 for _ in range(2)]\n    dy[0] = -p['k'] * y[0]\n    dy[1] = p['k'] * y[0] - y[1]\n    return dy",
    "    z = y.astype(float)\n    return [-p['k'] * z[0], p['k'] * z[0] - z[1]]",
    "    a = np.clip(y[0], 0.0, None)\n    return [-p['k'] * a, p['k'] * a - y[1] + 0.0 * sum(c for c in y)]",
    "    w = norm.cdf(x=y[1], loc=1.0, scale=2.0)\n    return [-p['k'] * y[0], p['k'] * y[0] - y[1] * w]",
    "    a = -p['k'] * y[0].item()\n    return np.hstack([a, -a - y[1]])",
    "    a = -p['k'] * y[0]\n    return np.stack((a, -a - y[1]))",
])
def test_numpy_idioms_compile_as_written(body):
    code = ("import math\nimport numpy as np\nfrom scipy.stats import norm\nORDER = [1, 0]\n\n"
            "def rhs(t, y, p):\n{guard}" + body + "\n\ndef observe(t, y, p):\n    return y[1]\n")
    native, eq = both(code, states=["A", "B"], parameters=["k"], inputs={"d": "A"})
    assert isinstance(native, NativeModel)
    df = frame([1.0, 5.0], [(0.0, 100.0)])
    assert np.allclose(predictions(native, df, [0.5])[0], predictions(eq, df, [0.5])[0], rtol=1e-6)


@pytest.mark.parametrize("body, message", [
    ("    return round(y[0], 1)\n", "write np.round"),
    ("    try:\n        return y[0] / p['k']\n    except ZeroDivisionError:\n        return 0.0\n",
     "except Exception: only"),
    ("    try:\n        return y[0] / p['k']\n    except:\n        return 0.0\n", "not with a bare except"),
])
def test_what_compiled_code_cannot_do_is_said(body, message):
    code = ("import math\n\ndef rhs(t, y, p):\n" + GUARD + "    return [-y[0]]\n\n"
            "def observe(t, y, p):\n" + body)
    with pytest.raises(ModelError, match=message):
        from_python(code, states=["A"], parameters=["k"])


def test_a_point_where_the_floats_of_the_code_lose_its_value_is_passed_over():
    # Gonzalez-Sales 2015: (t - exp(la)) modulo 24 at la = 49.9 has no digit left in floats,
    # and Numba and Python round it differently; the 50-digit code shows it
    code = CLOCK.replace("    tacro, tnadir", "    if math.isnan(t):\n        t = 0.0\n    tacro, tnadir")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = from_python(code, **ROUNDED["clock"][1])
    assert m.reading["form"] == "compiled as written" and m.reading["lost_points"] > 0


def test_native_points_are_drawn_closer_to_the_values_given():
    code = ("import math\n\ndef observe(t, y, p):\n    if math.isnan(t):\n        t = 0.0\n"
            "    return math.sqrt(1e-4 - (p['a'] - 1.0) ** 2)\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = from_python(code, parameters=["a"], values={"a": 1.0})
    assert m.reading["form"] == "compiled as written" and m.reading["spread"] < 1


@pytest.mark.slow
def test_code_with_hundreds_of_conditions_compiles_as_written():
    # Numba's analysis of the control flow recurses once per block, past Python's default limit
    lines = ["def rhs(t, y, p):", "    A = y[0]", "    k = p['k']"]
    lines += [f"    k = k * (1.001 if A > {i + 1}.0 else 0.999)" for i in range(480)]
    lines += ["    return [-k * A]", "", "def observe(t, y, p):", "    return y[0]"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = from_python("\n".join(lines), states=["A"], parameters=["k"], inputs={"d": "A"},
                        values={"k": 0.1})
    f, status = predictions(m, frame([1.0], [(0.0, 100.0)]), [0.1])
    assert m.reading["form"] == "compiled as written" and (status == 0).all() and 0 < f[0] < 100


@pytest.mark.slow
def test_a_fit_of_the_native_model_gives_the_estimates_of_its_equations():
    import pysaem
    from pysaem import Population, Settings

    code = """
import math

def rhs(t, y, p):
{guard}    return [-p['ka'] * y[0], p['ka'] * y[0] - p['CL'] / p['V'] * y[1]]

def observe(t, y, p):
    return y[1] / p['V']
"""
    native, eq = both(code, states=["Ad", "Ac"], parameters=["ka", "CL", "V"], inputs={"u": "Ad"})
    rng = np.random.default_rng(7)
    times = [0.5, 1, 2, 4, 6, 8, 12, 24.0]
    df = pd.concat([frame(times, [(0, 100.0)]).assign(ID=i) for i in range(1, 41)])
    psi = np.array([1.0, 4.0, 30.0]) * np.exp(rng.normal(0, 0.3, (40, 3)))
    prob = Problem(eq, Dataset.from_records(df))
    pred, _ = prob.predict(psi)
    y = prob.gather(pred)[0]
    df.loc[df["AMT"] == ".", "DV"] = y * (1 + 0.1 * rng.normal(size=y.size))
    settings = Settings(n_exploratory=150, n_smoothing=80, exploratory_autostop=False,
                        smoothing_autostop=False, seed=1, simulations=False)
    # a native model has no exact derivatives: its equations are fitted with differences too
    fits = [pysaem.fit(m, Dataset.from_records(df),
                       Population.build(m, init={"ka": 0.5, "CL": 2.0, "V": 20.0}, omega=0.5,
                                        error="proportional"), replace(settings, sensitivities="fd"))
            for m in (eq, native)]
    a, b = fits[0].parameters, fits[1].parameters
    assert max(abs(a[k] - b[k]) / abs(a[k]) for k in a) < 1e-4
    assert fits[0].ofv == pytest.approx(fits[1].ofv, abs=0.01)
