"""Model specification: parsing, validation, symbolic analysis."""

import json

import numpy as np
import pytest

from pysaem.model import Model, ModelError

TWO_CMT_MM = dict(
    states={"Ac": "0", "Ap": "0"},
    parameters=["V1", "V2", "Q", "Vm", "Km"],
    aux={"Cc": "Ac/V1"},
    odes={
        "Ac": "-Vm*Cc/(Km + Cc) - Q*(Cc - Ap/V2)",
        "Ap": "Q*(Cc - Ap/V2)",
    },
    outputs=["Cc"],
    inputs={"dose": {"target": "Ac"}},
)

ORAL_LAG = dict(
    states={"Ad": "0", "Ac": "0"},
    parameters=["ka", "CL", "V", "Tlag"],
    aux={"Cc": "Ac/V"},
    odes={"Ad": "-ka*Ad", "Ac": "ka*Ad - CL/V*Ac"},
    outputs=["Cc"],
    inputs={"dose": {"target": "Ad", "delay": "Tlag"}},
)


def test_valid_model_exposes_its_structure():
    m = Model(**TWO_CMT_MM)
    assert m.n_states == 2 and m.n_outputs == 1 and m.n_parameters == 5
    assert m.symbolic.linear is False
    assert m.symbolic.autonomous is True


def test_linear_system_is_detected():
    assert Model(**ORAL_LAG).symbolic.linear is True


def test_algebraic_model_needs_no_state():
    m = Model(states={}, odes={}, parameters=["a", "k"], aux={"y": "a*exp(-k*t)"},
              outputs=["y"])
    assert m.n_states == 0 and m.symbolic.rhs == []


def test_unknown_symbol_is_named_in_the_error():
    spec = dict(TWO_CMT_MM, odes={"Ac": "-kel*Ac", "Ap": "0"})
    with pytest.raises(ModelError, match="kel"):
        Model(**spec)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('true')",
    "Ac.real",
    "open('x')",
    "[Ac for Ac in (1,)]",
    "Ac if Ac else (lambda: 1)()",
])
def test_code_is_never_executed(expr):
    spec = dict(TWO_CMT_MM, odes={"Ac": expr, "Ap": "0"})
    with pytest.raises(ModelError):
        Model(**spec)


def test_common_habits_are_accepted():
    m = Model.from_dict({
        "states": {"X": "X0"}, "parameters": ["X0", "k", "h"],
        "aux": {"drive": "np.exp(-k*t) + math.sqrt(X^2 + 1) + ln(h) + 0*pi"},
        "odes": {"dX/dt": "-drive / (1 + X^h)"}, "outputs": "X"})
    f = m.python_functions()
    dy = np.zeros(1)
    f["rhs"](2.0, np.array([3.0]), np.array([3.0, 0.5, 2.0]), np.zeros(0), np.zeros(1), dy)
    drive = np.exp(-1.0) + np.sqrt(10.0) + np.log(2.0)
    assert dy[0] == pytest.approx(-drive / (1 + 9.0))
    assert m.outputs == ["X"] and list(m.odes) == ["X"]
    m = Model.from_dict({"states": {"X": "1"}, "parameters": "k", "covariates": "WT",
                         "odes": {"X": "-k*WT*X"}, "outputs": "X"})
    assert m.parameters == ["k"] and m.covariates == ["WT"]


@pytest.mark.parametrize("key", ["X", "dX", "dXdt", "dX/dt", "ddt_X"])
def test_derivative_keys_are_normalised(key):
    m = Model.from_dict({"states": {"X": "1"}, "parameters": ["k"], "odes": {key: "-k*X"},
                         "outputs": ["X"]})
    assert list(m.odes) == ["X"]


def test_state_without_equation_is_rejected():
    spec = dict(TWO_CMT_MM, odes={"Ac": "-Ac"})
    with pytest.raises(ModelError, match="Ap"):
        Model(**spec)


def test_output_must_be_a_state_or_an_auxiliary():
    with pytest.raises(ModelError, match="Cp"):
        Model(**dict(TWO_CMT_MM, outputs=["Cp"]))


def test_initial_condition_cannot_depend_on_a_state():
    with pytest.raises(ModelError, match="initial"):
        Model(**dict(TWO_CMT_MM, states={"Ac": "Ap", "Ap": "0"}))


def test_python_keyword_is_a_valid_parameter_name():
    m = Model(states={"X": "X0"}, odes={"X": "lambda*X"}, parameters=["X0", "lambda"],
              outputs=["X"])
    assert m.parameters == ["X0", "lambda"]


def test_time_switch_becomes_a_breakpoint():
    m = Model(states={"S": "S0"}, parameters=["S0", "alpha", "beta", "ts"],
              odes={"S": "where(t <= 0, alpha*S, (alpha - beta)*S) + where(t > ts, 1, 0)"},
              outputs=["S"])
    # the kernel writes the times, then their periods (0: met once)
    b = np.zeros(4)
    m.python_functions()["brk"](np.array([1.0, 0.1, 0.3, 7.5]), np.zeros(0), b)
    assert sorted(b[:2]) == [0.0, 7.5] and list(b[2:]) == [0.0, 0.0]


def test_a_switch_time_of_covariates_and_amount_signals_is_a_breakpoint():
    # a covariate and t_last move a time switch; a regressor does not, and those of t_last come last
    m = Model(states={"S": "1"}, parameters=["k", "lag"], covariates=["TSTART"], regressors=["X"],
              odes={"S": "where(t >= TSTART, -k*S, 0) + where(t - t_last > lag, -k*S, 0)"
                          " + where(t > X, -k*S, 0)"},
              outputs=["S"])
    from pysaem.model import history_breakpoints
    assert [str(b) for b in m.symbolic.breakpoints] == ["_r1", "_p1 + _r2"]
    assert history_breakpoints(m) == 1
    b = np.zeros(4)
    m.python_functions()["brk"](np.array([0.1, 2.0]), np.array([0.0, 5.0, 12.0, 0, 0, 0]), b)
    assert list(b) == [5.0, 14.0, 0.0, 0.0]


def test_state_condition_is_not_a_breakpoint():
    m = Model(states={"TS": "TS0"}, parameters=["TS0", "kge"],
              odes={"TS": "where(TS > 1e12, 0, kge*TS)"}, outputs=["TS"])
    assert m.symbolic.breakpoints == []
    assert m.symbolic.linear is False


def test_json_round_trip_keeps_the_model_and_its_hash():
    m = Model(**ORAL_LAG)
    again = Model.from_json(m.to_json())
    assert again.to_dict() == m.to_dict() and again.hash == m.hash
    json.loads(m.to_json())


def test_generated_jacobian_matches_finite_differences():
    m = Model(**TWO_CMT_MM)
    fns = m.python_functions()
    y = np.array([300.0, 120.0])
    p = np.array([40.0, 30.0, 2.0, 25.0, 5.0])
    r = np.zeros(0)
    u = np.zeros(2)
    J = np.zeros((2, 2))
    fns["jac"](0.0, y, p, r, J)
    num = np.zeros((2, 2))
    f0, f1 = np.zeros(2), np.zeros(2)
    for j in range(2):
        h = 1e-6 * y[j]
        yp, ym = y.copy(), y.copy()
        yp[j] += h
        ym[j] -= h
        fns["rhs"](0.0, yp, p, r, u, f0)
        fns["rhs"](0.0, ym, p, r, u, f1)
        num[:, j] = (f0 - f1) / (2 * h)
    assert np.allclose(J, num, rtol=1e-6)


def test_input_rate_enters_the_target_state_only():
    m = Model(**TWO_CMT_MM)
    fns = m.python_functions()
    dy0, dy1 = np.zeros(2), np.zeros(2)
    y, p, r = np.zeros(2), np.array([40.0, 30.0, 2.0, 25.0, 5.0]), np.zeros(0)
    fns["rhs"](0.0, y, p, r, np.zeros(2), dy0)
    fns["rhs"](0.0, y, p, r, np.array([7.0, 9.0]), dy1)
    assert np.allclose(dy1 - dy0, [7.0, 0.0])


def test_interpolated_signal_must_be_a_regressor():
    with pytest.raises(ModelError, match="interpolated"):
        Model(states={"X": "1"}, odes={"X": "-X"}, parameters=[], outputs=["X"],
              regressors=["C"], interpolated=["W"])
    m = Model(states={"X": "1"}, odes={"X": "-C*X"}, parameters=[], outputs=["X"],
              regressors=["C"], interpolated=["C"])
    assert Model.from_dict(m.to_dict()).interpolated == ["C"]
    assert list(m.lowered().states) == ["X", "C"] and m.lowered().regressors == ["_at_C", "_slope_C"]



def test_generated_kernels_keep_the_meaning_of_the_equations():
    # % binds as * does; x**-2 at zero gives inf as numpy does, where a compiled kernel would
    # raise; integers beyond 64 bits compile as floats
    import pandas as pd

    from pysaem.data import Dataset
    from pysaem.solve import Problem

    m = Model(states={"A": "0"}, odes={"A": "0"}, parameters=["k"], outputs=["c", "e", "g"],
              aux={"c": "exp(-k*mod(t, 24))", "e": "2 - 1/(1 + 1/A**2)", "g": "A*10**20 + 1"})
    df = pd.DataFrame({"ID": 1, "TIME": 30.0, "DV": 1.0, "YTYPE": [1, 2, 3]})
    prob = Problem(m, Dataset.from_records(df, output="YTYPE"))
    pred, status = prob.predict(np.array([[0.1]]))
    assert (status == 0).all()
    assert np.allclose(prob.gather(pred)[0], [np.exp(-0.1 * 6), 2.0, 1.0])
    # constants keep all their digits: floor(t/168.0) is exact at the multiples of 168
    m = Model(states={"A": "0"}, odes={"A": "0"}, parameters=["k"], outputs=["n"], aux={"n": "k*floor(t/168.0)"})
    prob = Problem(m, Dataset.from_records(pd.DataFrame({"ID": 1, "TIME": [168.0, 336.0, 504.0], "DV": 1.0})))
    pred, status = prob.predict(np.array([[1.0]]))
    assert prob.gather(pred)[0].tolist() == [1.0, 2.0, 3.0]


JOINT = dict(states={"X": "X0"}, parameters=["X0", "g", "h0", "beta"], odes={"X": "g*X"},
             aux={"h": "h0*exp(beta*X)"}, outputs=["X", "h"])


def test_a_model_without_hazards_keeps_its_hash_and_is_solved_as_written():
    m = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"}, aux={"C": "A/V"},
              outputs=["C"], inputs={"dose": {"target": "A"}})
    assert m.hash == "4be96c1504c56432"
    assert m.hazards == {} and "hazards" not in m.to_dict()
    assert m.lowered() is m


def test_hazards_give_each_output_a_kind_and_survive_json():
    m = Model.from_dict({"equations": "X(0) = X0\nd/dt(X) = g*X\nh = h0*exp(beta*X)",
                         "parameters": ["X0", "g", "h0", "beta"], "outputs": ["X", "h"],
                         "hazards": {"h": "terminal"}})
    again = Model.from_json(m.to_json())
    assert again.hazards == {"h": "terminal"} and again.hash == m.hash
    # kept in the order of the outputs: one model, one hash
    two = dict(JOINT, aux={"h": "h0*exp(beta*X)", "d": "h0"}, outputs=["X", "h", "d"])
    a = Model(**two, hazards={"d": "repeated", "h": "single"})
    assert list(a.hazards) == ["h", "d"]
    assert a.hash == Model(**two, hazards={"h": "single", "d": "repeated"}).hash
    with pytest.raises(ModelError, match="'hz' is not an output"):
        Model(**JOINT, hazards={"hz": "terminal"})
    with pytest.raises(ModelError, match="kind 'once' is not one of"):
        Model(**JOINT, hazards={"h": "once"})
    # a kind is never assumed
    for no_kind in (["h"], "h"):
        with pytest.raises(ModelError, match="to its kind"):
            Model(**JOINT, hazards=no_kind)


def test_each_hazard_is_integrated_by_a_last_state_and_read_after_the_outputs():
    # a state of the user is named _cum_h: the integral of h is named __cum_h
    m = Model(states={"X": "X0", "_cum_h": "0"}, parameters=["X0", "k", "lam", "mu"],
              odes={"X": "-k*C*X", "_cum_h": "0"}, regressors=["C"], interpolated=["C"],
              aux={"h": "lam*X", "d": "mu"}, outputs=["d", "X", "h", "_cum_h"],
              hazards={"h": "single", "d": "repeated"})
    low = m.lowered()
    assert list(low.states) == ["X", "_cum_h", "C", "_cum_d", "__cum_h"]
    assert (low.states["_cum_d"], low.odes["_cum_d"]) == ("0", "d")
    assert (low.states["__cum_h"], low.odes["__cum_h"]) == ("0", "h")
    assert low.outputs == ["_cum_d", "X", "__cum_h", "_cum_h", "d", "h"]
    assert low.regressors == ["_at_C", "_slope_C"] and low.hazards == {}
    # the equations and kernels of the model itself do not change
    assert m.source() == Model(**{**m.to_dict(), "hazards": None}).source()


@pytest.mark.parametrize("hazard, closed", [
    ("lam", True), ("h0*A", True), ("h0*R", True), ("h0*exp(lam*A)", False),
    ("h0*kw*t**(kw - 1)", False)])
def test_a_hazard_linear_in_the_states_keeps_the_closed_form(hazard, closed):
    from pysaem.solve import _CLOSED_FORM, _HAZARDS, compile_model

    m = Model(states={"A": "1"}, parameters=["k", "lam", "h0", "kw"], odes={"A": "-k*A"},
              regressors=["R"], aux={"h": hazard}, outputs=["A", "h"], hazards={"h": "single"})
    # rhs, jac, has_jac, out, init, brk, inp, structure
    structure = compile_model(m.lowered(), len(m.hazards))[7]
    assert bool(structure & _CLOSED_FORM) is closed and structure // _HAZARDS == 1


def test_dependencies_follow_the_states_an_output_reads():
    m = Model(states={"A": "A0", "B": "0", "Z": "1"},
              parameters=["A0", "ka", "k", "V", "F", "lag", "D1", "h0", "kz"],
              odes={"A": "-ka*A", "B": "ka*A - k*B", "Z": "-kz*Z"},
              aux={"C": "B/V", "h": "h0*C"}, outputs=["C", "h", "A", "Z"],
              inputs={"first": {"target": "A", "scale": "F", "delay": "lag"},
                      "second": {"target": "B", "duration": "D1"}},
              hazards={"h": "terminal"})
    assert m.dependencies("A") == ["A0", "ka", "F", "lag"]
    assert m.dependencies("C") == ["A0", "ka", "k", "V", "F", "lag", "D1"]
    assert m.dependencies("h") == m.dependencies("C") + ["h0"]
    assert m.dependencies("Z") == ["kz"]
    with pytest.raises(ModelError, match="not an output"):
        m.dependencies("B")


def test_a_modulo_written_by_hand_switches_where_mod_does():
    # t - 24*(t // 24) is mod(t, 24): its comparison with 8 switches at 8 + 24 j
    from pysaem import from_python
    code = ("def rhs(t, y, p):\n    h = t - 24 * (t // 24)\n    return [(p['kin'] if h < 8 else 0.0) - p['k'] * y[0]]\n\n\n"
            "def observe(t, y, p):\n    return y[0]\n\n\ndef init(p):\n    return [1.0]\n")
    m = from_python(code, states=["A"], parameters=["kin", "k"], outputs=["A"], values={"kin": 1.0, "k": 0.1})
    s = m.lowered().symbolic
    assert sorted((str(b), q) for b, q in zip(s.breakpoints, s.periods)) == [("0", 24.0), ("8", 24.0)]
