"""Compiled solver against closed-form solutions and a SciPy reference."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from pysaem.data import Dataset
from pysaem.model import Model
from pysaem.solve import Problem

TOL = 2e-5  # relative error allowed at the default rtol of 1e-6


def predict(model, df, psi, **kw):
    kw.setdefault("covariates", ())
    opts = {k: kw.pop(k) for k in ("method", "rtol", "atol") if k in kw}
    prob = Problem(model, Dataset.from_records(df, **kw), **opts)
    pred, status = prob.predict(np.atleast_2d(np.asarray(psi, float)))
    assert (status == 0).all()
    return prob.gather(pred)[0]


def records(times, doses=(), rate=None, id=1):
    rows = [{"ID": id, "TIME": t, "DV": ".", "AMT": a, "RATE": rate or "."} for t, a in doses]
    rows += [{"ID": id, "TIME": t, "DV": 1.0, "AMT": ".", "RATE": "."} for t in times]
    return pd.DataFrame(rows).sort_values("TIME", kind="stable")


ONE_CMT = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"},
                aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A"}})
ORAL = Model(states={"Ad": "0", "Ac": "0"}, parameters=["ka", "CL", "V", "Tlag"],
             odes={"Ad": "-ka*Ad", "Ac": "ka*Ad - CL/V*Ac"}, aux={"C": "Ac/V"},
             outputs=["C"], inputs={"dose": {"target": "Ad", "delay": "Tlag"}})


def test_bolus_matches_the_exponential():
    t = np.array([0.5, 1, 2, 4, 8, 24.0])
    f = predict(ONE_CMT, records(t, [(0, 100)]), [2.0, 10.0])
    assert np.allclose(f, 100 / 10 * np.exp(-0.2 * t), rtol=TOL)


def test_infusion_matches_the_closed_form():
    t = np.array([0.25, 0.5, 1, 2, 6, 12.0])
    cl, v, amt, rate = 2.0, 10.0, 100.0, 200.0
    k, tinf = cl / v, amt / rate
    during = rate / cl * (1 - np.exp(-k * np.minimum(t, tinf)))
    exact = during * np.exp(-k * np.maximum(t - tinf, 0))
    f = predict(ONE_CMT, records(t, [(0, amt)], rate=rate), [cl, v])
    assert np.allclose(f, exact, rtol=TOL)


def test_repeated_doses_superpose():
    t = np.array([1, 11.9, 12.5, 23, 25, 40.0])
    exact = sum(np.where(t > td, 10 * np.exp(-0.2 * (t - td)), 0) for td in (0, 12, 24))
    f = predict(ONE_CMT, records(t, [(0, 100), (12, 100), (24, 100)]), [2.0, 10.0])
    assert np.allclose(f, exact, rtol=TOL)


def test_observation_at_event_time_is_taken_before_the_event():
    df = records([12.0], [(0, 100), (12, 100)])
    f = predict(ONE_CMT, df, [2.0, 10.0])
    assert np.allclose(f, 10 * np.exp(-0.2 * 12), rtol=TOL)


@pytest.mark.parametrize("method", ["auto", "explicit"])
def test_reset_starts_the_subject_again(method):
    # EVID 4 at 24 gives a fresh first dose; EVID 3 at 5 cancels an infusion under way
    t = np.array([1, 4, 25, 26, 30.0])
    fresh = 100 / 10 * np.exp(-0.2 * np.where(t < 24, t, t - 24))
    df = pd.DataFrame([{"ID": 1, "TIME": 0, "DV": ".", "AMT": 100, "RATE": ".", "EVID": 1},
                       {"ID": 1, "TIME": 24, "DV": ".", "AMT": 100, "RATE": ".", "EVID": 4}]
                      + [{"ID": 1, "TIME": x, "DV": 1.0, "AMT": ".", "RATE": ".", "EVID": 0} for x in t])
    assert np.allclose(predict(ONE_CMT, df.sort_values("TIME", kind="stable"), [2.0, 10.0], method=method),
                       fresh, rtol=TOL)
    cut = pd.DataFrame([{"ID": 1, "TIME": 0, "DV": ".", "AMT": 100, "RATE": 10, "EVID": 1},
                        {"ID": 1, "TIME": 5, "DV": ".", "AMT": ".", "RATE": ".", "EVID": 3},
                        {"ID": 1, "TIME": 24, "DV": ".", "AMT": 100, "RATE": ".", "EVID": 1}]
                       + [{"ID": 1, "TIME": x, "DV": 1.0, "AMT": ".", "RATE": ".", "EVID": 0} for x in (6, 20, 26)])
    f = predict(ONE_CMT, cut.sort_values("TIME", kind="stable"), [2.0, 10.0], method=method)
    assert f[0] == f[1] == 0.0 and f[2] == pytest.approx(10 * np.exp(-0.4), rel=TOL)


@pytest.mark.parametrize("method", ["auto", "explicit"])
def test_input_with_a_duration_is_given_at_amount_over_duration(method):
    # the record has no rate: the duration of the input sets it, and may be estimated
    zero = Model(states={"A": "0"}, parameters=["CL", "V", "Tk0"], odes={"A": "-CL/V*A"},
                 aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A", "duration": "Tk0"}})
    t = np.array([0.5, 1, 2, 3, 6, 12.0])
    cl, v, tk0 = 2.0, 10.0, 2.5
    k = cl / v
    exact = 100 / tk0 / cl * (1 - np.exp(-k * np.minimum(t, tk0))) * np.exp(-k * np.maximum(t - tk0, 0))
    assert np.allclose(predict(zero, records(t, [(0, 100)]), [cl, v, tk0], method=method), exact, rtol=TOL)


def steady(times, rate=None, ii=12.0, n_before=0):
    """A dose at 0 at steady state of a dose every ii, or the same dose given n_before times
    before it."""
    head = ([{"ID": 1, "TIME": 0, "DV": ".", "AMT": 100, "RATE": rate or ".", "SS": 1, "II": ii}]
            if not n_before else
            [{"ID": 1, "TIME": -ii * j, "DV": ".", "AMT": 100, "RATE": rate or ".", "SS": 0, "II": "."}
             for j in range(n_before, -1, -1)])
    return pd.DataFrame(head + [{"ID": 1, "TIME": x, "DV": 1.0, "AMT": ".", "RATE": ".", "SS": 0,
                                 "II": "."} for x in times])


@pytest.mark.parametrize("method", ["auto", "explicit"])
def test_steady_state_matches_the_closed_form_and_long_repeated_dosing(method):
    t = np.array([0.5, 2, 6, 11.9])
    f = predict(ONE_CMT, steady(t), [2.0, 10.0], method=method)
    assert np.allclose(f, 10 * np.exp(-0.2 * t) / (1 - np.exp(-0.2 * 12)), rtol=TOL)
    # an infusion, and a delayed first-order input whose lag shifts the cycle
    for model, psi, rate in ((ONE_CMT, [2.0, 10.0], 50.0), (ORAL, [1.0, 2.0, 10.0, 1.5], None)):
        assert np.allclose(predict(model, steady(t, rate), psi, method=method),
                           predict(model, steady(t, rate, n_before=80), psi, method=method), rtol=TOL)
    # saturable elimination: cycles until the pre-input state settles (reference: 1000 doses)
    mm = Model(states={"A": "0"}, parameters=["Vm", "Km", "V"], odes={"A": "-Vm*(A/V)/(Km + A/V)"},
               aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A"}})
    assert np.allclose(predict(mm, steady(t), [10.0, 2.0, 20.0], method=method),
                       [12.49172, 11.8476, 10.15577, 7.74719], rtol=TOL)


def test_delayed_first_order_input_matches_the_bateman_function():
    ka, cl, v, lag = 1.5, 2.0, 10.0, 0.4
    k = cl / v
    t = np.array([0.25, 0.4, 0.5, 1, 2, 4, 8, 24.0])
    s = np.maximum(t - lag, 0)
    exact = 100 / v * ka / (ka - k) * (np.exp(-k * s) - np.exp(-ka * s))
    f = predict(ORAL, records(t, [(0, 100)]), [ka, cl, v, lag])
    assert np.allclose(f, exact, rtol=TOL, atol=1e-12)
    assert f[0] == 0.0 and f[1] == 0.0


def test_input_scale_multiplies_the_amount():
    m = Model(states={"A": "0"}, parameters=["CL", "V", "F"], odes={"A": "-CL/V*A"},
              aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A", "scale": "F"}})
    f = predict(m, records([1.0], [(0, 100)]), [2.0, 10.0, 0.6])
    assert np.allclose(f, 0.6 * 10 * np.exp(-0.2), rtol=TOL)


def test_growth_model_without_event():
    m = Model(states={"X": "X0"}, parameters=["X0", "K", "r"], odes={"X": "r*X*(1 - X/K)"},
              outputs=["X"])
    t = np.array([0, 1, 5, 10, 30.0])
    x0, K, r = 5.0, 1000.0, 0.4
    exact = K / (1 + (K - x0) / x0 * np.exp(-r * t))
    f = predict(m, pd.DataFrame({"ID": 1, "TIME": t, "DV": 1.0}), [x0, K, r])
    assert np.allclose(f, exact, rtol=TOL)


def test_model_without_state_is_evaluated_directly():
    m = Model(states={}, odes={}, parameters=["a", "k"], aux={"y": "a*exp(-k*t)"},
              outputs=["y"])
    t = np.array([0, 1, 2.5])
    f = predict(m, pd.DataFrame({"ID": 1, "TIME": t, "DV": 1.0}), [3.0, 0.7])
    assert np.allclose(f, 3 * np.exp(-0.7 * t), rtol=1e-12)


def test_time_switch_and_negative_times():
    m = Model(states={"S": "S0"}, parameters=["S0", "a", "b"],
              odes={"S": "where(t <= 0, a*S, (a - b)*S)"}, outputs=["S"])
    t = np.array([-20, -5, 0, 3, 10.0])
    s0, a, b = 2.0, 0.05, 0.3
    exact = np.where(t <= 0, s0 * np.exp(a * (t + 20)),
                     s0 * np.exp(a * 20) * np.exp((a - b) * t))
    f = predict(m, pd.DataFrame({"ID": 1, "TIME": t, "DV": 1.0}), [s0, a, b])
    assert np.allclose(f, exact, rtol=TOL)


def test_declared_start_time_holds_the_initial_state_before_it():
    m = Model(states={"X": "X0"}, parameters=["X0", "k"], odes={"X": "-k*X"}, outputs=["X"],
              t0=0)
    t = np.array([-3, 0, 2.0])
    f = predict(m, pd.DataFrame({"ID": 1, "TIME": t, "DV": 1.0}), [4.0, 0.5])
    assert np.allclose(f, [4.0, 4.0, 4 * np.exp(-1)], rtol=TOL)


def test_regressor_drives_the_dynamics_piecewise():
    m = Model(states={"X": "X0"}, parameters=["X0", "k"], regressors=["C"],
              odes={"X": "-k*C*X"}, outputs=["X"])
    df = pd.DataFrame({"ID": 1, "TIME": [0, 1, 2, 4.0], "DV": 1.0, "C": [2.0, 0.5, 0.0, 0.0]})
    f = predict(m, df, [10.0, 0.3], regressors=["C"])
    exact = 10 * np.exp(-0.3 * np.array([0, 2.0, 2.5, 2.5]))
    assert np.allclose(f, exact, rtol=TOL)


def test_two_outputs_on_their_own_time_grids():
    m = Model(states={"T": "T0", "D": "0"}, parameters=["T0", "g", "lam", "kd"],
              odes={"T": "g*T", "D": "lam*T - kd*D"}, outputs=["T", "D"])
    df = pd.DataFrame({
        "ID": 1,
        "TIME": [0, 2, 2, 5, 7.0],
        "DV": 1.0,
        "YTYPE": [1, 1, 2, 2, 1],
    })
    t0_, g, lam, kd = 3.0, 0.1, 0.5, 0.8
    tum = lambda t: t0_ * np.exp(g * t)
    dna = lambda t: lam * t0_ / (g + kd) * (np.exp(g * t) - np.exp(-kd * t))
    f = predict(m, df, [t0_, g, lam, kd], output="YTYPE")
    assert np.allclose(f, [tum(0), tum(2), dna(2), dna(5), tum(7)], rtol=TOL)


def test_stiff_system_switches_to_the_implicit_solver():
    # Robertson kinetics, stiffness ratio around 1e9
    m = Model(states={"a": "1", "b": "0", "c": "0"}, parameters=["k1", "k2", "k3"],
              odes={"a": "-k1*a + k3*b*c", "b": "k1*a - k3*b*c - k2*b**2", "c": "k2*b**2"},
              outputs=["a", "b", "c"], t0=0)
    t = np.array([0.4, 4, 40, 400.0])
    df = pd.DataFrame({"ID": 1, "TIME": np.repeat(t, 3), "DV": 1.0, "YTYPE": [1, 2, 3] * 4})
    psi = [0.04, 3e7, 1e4]
    fns = m.python_functions()

    def fun(tt, y):
        dy = np.zeros(3)
        fns["rhs"](tt, y, np.array(psi), np.zeros(0), np.zeros(3), dy)
        return dy
    ref = solve_ivp(fun, (0, 400), [1, 0, 0], method="Radau", t_eval=t, rtol=1e-10,
                    atol=1e-14).y.T.ravel()
    prob = Problem(m, Dataset.from_records(df, output="YTYPE"), atol=1e-12)
    pred, status = prob.predict(np.array([psi]))
    assert (status == 0).all()
    assert np.allclose(prob.gather(pred)[0], ref, rtol=1e-4)
    assert prob.last_stats["implicit"] > 0
    assert prob.totals == {"solves": 1, "exact": 0, "implicit": 1, "failed": 0, "events": 0,
                           "unlocated": 0}


@pytest.mark.parametrize("method", ["explicit", "implicit"])
def test_both_integrators_agree_with_scipy_on_a_nonlinear_system(method):
    m = Model(states={"Ac": "0", "Ap": "0"}, parameters=["V1", "V2", "Q", "Vm", "Km"],
              aux={"Cc": "Ac/V1"},
              odes={"Ac": "-Vm*Cc/(Km + Cc) - Q*(Cc - Ap/V2)", "Ap": "Q*(Cc - Ap/V2)"},
              outputs=["Cc"], inputs={"dose": {"target": "Ac"}})
    t = np.array([0.5, 1, 2, 4, 8, 12, 18, 24, 48, 72.0])
    psi = np.array([40.0, 30.0, 2.0, 25.0, 5.0])
    fns = m.python_functions()

    def fun(tt, y, rate):
        dy = np.zeros(2)
        fns["rhs"](tt, y, psi, np.zeros(0), np.array([rate, 0.0]), dy)
        return dy
    tinf = 1000 / 3000
    a = solve_ivp(fun, (0, tinf), [0, 0], args=(3000.0,), method="LSODA", rtol=1e-12,
                  atol=1e-14)
    b = solve_ivp(fun, (tinf, 72), a.y[:, -1], args=(0.0,), method="LSODA", t_eval=t,
                  rtol=1e-12, atol=1e-14)
    f = predict(m, records(t, [(0, 1000)], rate=3000), psi, method=method)
    assert np.allclose(f, b.y[0] / 40, rtol=TOL)


@pytest.mark.parametrize("other", [None, "1"])
def test_newton_from_states_at_zero_iterates_until_it_converges(other):
    # the first correction of a state at zero is measured on its absolute tolerance alone, alone
    # or beside a state that is not at zero: the rate compares two corrections on one scale, or
    # the step is taken before Newton converges (dA/dA = 0 at A = 0)
    states, odes = {"A": "0"}, {"A": "where(t > T1, R, 0) - Vm*A**2"}
    if other:
        states["B"], odes["B"] = other, "-0.01*B"
    m = Model(states=states, parameters=["R", "T1", "Vm"], odes=odes, outputs=["A"], t0=0)
    t = np.array([2.2, 3.0, 10.0])
    ref = solve_ivp(lambda tt, y: [10.0 - 20.0 * y[0] ** 2], (2.0, 10.0), [0.0], method="Radau",
                    t_eval=t, rtol=1e-12, atol=1e-16).y[0]
    assert np.allclose(predict(m, records(t), [10.0, 2.0, 20.0], method="implicit"), ref, rtol=TOL)


def test_implicit_steps_stay_near_scipy_when_the_lu_pivots():
    # Tornoe 2007 GnRH agonist: a sharp Emax couples Ac to Pool (dPool'/dAc near -765), so the
    # LU of the Newton matrix pivots at the steps SciPy takes (near 0.1); 112000 attempts when
    # the solve applied the row swaps out of order
    m = Model(states={"Ad": "0", "Ac": "0", "Pool": "ksyn/krel", "LH": "ksyn/kLH",
                      "Tst": "kinT/koutT"},
              parameters=["ka", "V", "Cl", "Emax", "EC50", "ksyn", "krel", "kLH", "kinT", "koutT"],
              aux={"Cc": "Ac/V", "rel": "krel*(1 + Emax*Cc/(EC50 + Cc))*Pool"},
              odes={"Ad": "-ka*Ad", "Ac": "ka*Ad - Cl/V*Ac", "Pool": "ksyn - rel",
                    "LH": "rel - kLH*LH", "Tst": "kinT*LH*kLH/ksyn - koutT*Tst"},
              outputs=["Tst"], inputs={"sc": {"target": "Ad"}})
    psi = np.array([1.0, 100.0, 200.0, 20.0, 0.001, 10.0, 0.05, 10.0, 30.0, 1.5])
    t = np.array([0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 14, 21, 35, 42, 56])
    fns = m.python_functions()

    def fun(tt, y):
        dy = np.zeros(5)
        fns["rhs"](tt, y, psi, np.zeros(0), np.zeros(1), dy)
        return dy
    y0 = [3.75, 0.0, 200.0, 1.0, 20.0]
    ref = solve_ivp(fun, (0, 56), y0, method="Radau", t_eval=t, rtol=1e-12, atol=1e-16).y[4]
    n_scipy = solve_ivp(fun, (0, 56), y0, method="Radau", rtol=1e-6, atol=1e-12).t.size - 1
    prob = Problem(m, Dataset.from_records(records(t, [(0, 3.75)]), covariates=()),
                   method="implicit", max_steps=3 * n_scipy)
    pred, status = prob.predict(np.atleast_2d(psi))
    assert (status == 0).all()
    assert np.allclose(prob.gather(pred)[0], ref, rtol=1e-5)


def test_the_lu_solve_matches_numpy_on_matrices_that_pivot():
    # random matrices pivot after their first column, where swaps applied during the forward
    # pass gave wrong solutions; real for the closed form, complex for Radau
    from pysaem.solve import _lu, _lu_solve
    rng = np.random.default_rng(0)
    late = 0
    for dtype in (float, complex):
        for n in (3, 5, 8):
            for _ in range(20):
                A = rng.normal(size=(n, n)).astype(dtype)
                b = rng.normal(size=n).astype(dtype)
                if dtype is complex:
                    A += 1j * rng.normal(size=(n, n))
                    b += 1j * rng.normal(size=n)
                LU, piv, x = A.copy(), np.empty(n, np.int64), b.copy()
                assert _lu(LU, piv)
                late += (piv[1:] != np.arange(1, n)).any()
                _lu_solve(LU, piv, x)
                assert np.allclose(x, np.linalg.solve(A, b), rtol=1e-10, atol=1e-12)
    assert late > 90


def test_newton_converges_when_a_large_state_is_coupled_strongly_to_a_tiny_one():
    # Anbari 2023: p at 4e-25 (tolerance 1e-30) binds M at 3e4 at kon*M = 1.2e16; unscaled, the
    # LU pivoted the column of p onto the row of M, whose rounding left Newton at rate 1
    m = Model(states={"p": "p0", "M": "M0"}, parameters=["kp", "kon", "k1", "p0", "M0"],
              odes={"p": "-kp*p", "M": "-k1*M - kon*M*p"}, outputs=["M"], t0=0)
    th = [195.0, 3.6e11, 36.0, 4e-25, 32786.0]
    t = np.array([0.01, 0.0625])
    ref = solve_ivp(lambda tt, y: [-th[0] * y[0], -th[2] * y[1] - th[1] * y[1] * y[0]], (0, t[-1]),
                    [th[3], th[4]], method="Radau", t_eval=t, rtol=1e-12, atol=[1e-40, 1e-14]).y[1]
    assert np.allclose(predict(m, records(t), th, method="implicit"), ref, rtol=TOL)


def test_steady_state_of_a_linear_system_with_gains_matches_repeated_doses():
    # gains above the mass balance make I - T pivot after its first column: the exact steady
    # state came out negative when the LU solve applied the row swaps during the forward pass
    m = Model(states={"Ad": "0", "Ac": "0", "E1": "0", "E2": "0"},
              parameters=["ka", "ke", "g1", "k1", "g2", "k2"],
              odes={"Ad": "-ka*Ad", "Ac": "ka*Ad - ke*Ac", "E1": "g1*Ac - k1*E1",
                    "E2": "g2*E1 - k2*E2"},
              outputs=["E2"], inputs={"dose": {"target": "Ad"}})
    t = np.array([0.5, 2, 6, 11.9])
    psi = [0.2, 0.5, 60.0, 0.4, 1.0, 2.0]
    assert np.allclose(predict(m, steady(t), psi), predict(m, steady(t, n_before=300), psi),
                       rtol=1e-9)


def test_the_solve_ends_at_the_last_observation():
    # a switching time long after the data changes no prediction: no integration up to it
    m = Model(states={"A": "0"}, parameters=["R", "T2", "Vm"], odes={"A": "where(t < T2, R, 0) - Vm*A**2"},
              outputs=["A"], t0=0)
    t = np.array([0.05, 0.1, 10.0])
    f = predict(m, records(t), [10.0, 1e6, 20.0], method="explicit")
    assert np.allclose(f, np.sqrt(0.5) * np.tanh(np.sqrt(200.0) * t), rtol=TOL)


def test_invalid_parameters_are_flagged_not_raised():
    df = records([1.0, 2.0], [(0, 100)])
    prob = Problem(ONE_CMT, Dataset.from_records(df))
    pred, status = prob.predict(np.array([[2.0, 0.0], [np.nan, 10.0], [2.0, 10.0]])[:, None, :])
    assert status[:, 0].tolist()[2] == 0 and (status[:2, 0] != 0).all()
    assert np.isfinite(pred[2]).all()


def test_batch_of_chains_and_subjects_matches_single_solves():
    rng = np.random.default_rng(0)
    df = pd.concat([records([0.5, 2, 6.0], [(0, 100 * (i + 1))], id=i) for i in range(5)])
    prob = Problem(ONE_CMT, Dataset.from_records(df))
    psi = np.exp(rng.normal(size=(4, 5, 2)) * 0.3) * [2.0, 10.0]
    pred, status = prob.predict(psi)
    assert pred.shape == (4, 15, 1) and status.shape == (4, 5)
    for c in range(4):
        for i in range(5):
            cl, v = psi[c, i]
            exact = 100 * (i + 1) / v * np.exp(-cl / v * np.array([0.5, 2, 6.0]))
            assert np.allclose(pred[c, 3 * i:3 * i + 3, 0], exact, rtol=TOL)


def test_data_with_events_need_a_declared_input():
    m = Model(states={"X": "1"}, parameters=["k"], odes={"X": "-k*X"}, outputs=["X"])
    with pytest.raises(ValueError, match="input"):
        Problem(m, Dataset.from_records(records([1.0], [(0, 100)])))


def test_covariates_enter_equations_initial_values_and_inputs_in_any_form():
    m = Model(states={"A": "A0 * WT"}, parameters=["CL", "V", "A0", "slope"],
              covariates=["WT", "SEX"],
              aux={"cl": "CL * (WT/70)**0.75 * (1 + slope*SEX)", "C": "A/V"},
              odes={"A": "-cl/V*A"}, outputs=["C"],
              inputs={"dose": {"target": "A", "scale": "0.5 if SEX > 0.5 else 1"}}, t0=0)
    rows = []
    for i, (wt, sex) in enumerate([(50.0, "F"), (90.0, "M")]):
        rows.append({"ID": i, "TIME": 0.0, "DV": ".", "AMT": 100.0, "WT": wt, "SEX": sex})
        rows += [{"ID": i, "TIME": t, "DV": 1.0, "AMT": ".", "WT": wt, "SEX": sex}
                 for t in (1.0, 4.0)]
    data = Dataset.from_records(pd.DataFrame(rows), covariates=["WT", "SEX"])
    assert data.covariate_levels == {"SEX": ["F", "M"]}
    prob = Problem(m, data)
    pred, status = prob.predict(np.tile([2.0, 10.0, 0.1, 0.4], (2, 1)))
    assert (status == 0).all()
    f = prob.gather(pred)[0].reshape(2, 2)
    for i, (wt, sex) in enumerate([(50.0, 0.0), (90.0, 1.0)]):
        cl = 2.0 * (wt / 70) ** 0.75 * (1 + 0.4 * sex)
        start = 0.1 * wt + 100.0 * (0.5 if sex else 1.0)
        assert np.allclose(f[i], start / 10 * np.exp(-cl / 10 * np.array([1.0, 4.0])), rtol=TOL)


def test_missing_covariate_is_reported():
    m = Model(states={"X": "1"}, parameters=["k"], covariates=["AGE"], odes={"X": "-k*AGE*X"},
              outputs=["X"])
    with pytest.raises(ValueError, match="AGE"):
        Problem(m, Dataset.from_records(pd.DataFrame({"ID": 1, "TIME": [0.0, 1.0], "DV": 1.0})))


def test_stiff_chain_at_equilibrium_is_handed_to_the_implicit_solver():
    # the fast states are empty after 1e-3: the stiffness test alone never fires
    m = Model(states={"T1": "0", "T2": "0", "A": "0"}, parameters=["ktr", "k"],
              odes={"T1": "-ktr*T1", "T2": "ktr*T1 - ktr*T2", "A": "ktr*T2 - k*A/(1 + 1e-12*A)"},
              outputs=["A"], inputs={"input": {"target": "T1"}}, t0=0)
    t = np.array([1.0, 6.0, 24.0])
    prob = Problem(m, Dataset.from_records(records(t, [(0, 100)])))
    pred, status = prob.predict(np.array([[[1e4, 0.2]]]))
    assert (status == 0).all() and prob.last_stats["implicit"] == 1
    assert np.allclose(prob.gather(pred)[0], 100 * np.exp(-0.2 * t), rtol=1e-3)


def exact(model, df, psi, **kw):
    """Predictions by the default method, which must be the closed form."""
    prob = Problem(model, Dataset.from_records(df, **kw))
    pred, status = prob.predict(np.atleast_2d(np.asarray(psi, float)))
    assert (status == 0).all() and prob.last_stats["exact"] == prob.data.n_subjects
    return prob.gather(pred)[0]


def test_linear_systems_are_solved_in_closed_form():
    t = np.array([0.25, 1.0, 3.0, 8.0, 30.0, 200.0])
    # one state with an input rate, during and after it
    f = exact(ONE_CMT, records(t, [(0, 100)], rate=50), [2.0, 10.0])
    k = 0.2
    during = 50 / 2.0 * (1 - np.exp(-k * t))
    after = 50 / 2.0 * (1 - np.exp(-k * 2)) * np.exp(-k * (t - 2))
    assert np.allclose(f, np.where(t <= 2, during, after), rtol=1e-11)
    # two rates equal: the matrix has a single eigenvector
    chain = Model(states={"A": "0", "B": "0"}, parameters=["k"],
                  odes={"A": "-k*A", "B": "k*A - k*B"}, outputs=["B"],
                  inputs={"input": {"target": "A"}}, t0=0)
    assert np.allclose(exact(chain, records(t, [(0, 100)]), [0.7]),
                       100 * 0.7 * t * np.exp(-0.7 * t), rtol=1e-11)
    # complex eigenvalues: a damped oscillation that changes sign
    swing = Model(states={"x": "x0", "v": "0"}, parameters=["x0", "g", "w"],
                  odes={"x": "v", "v": "-2*g*v - w**2*x"}, outputs=["x"], t0=0)
    wd = np.sqrt(1.0 - 0.1**2)
    assert np.allclose(exact(swing, records(t), [10.0, 0.1, 1.0]),
                       10 * np.exp(-0.1 * t) * (np.cos(wd * t) + 0.1 / wd * np.sin(wd * t)),
                       rtol=1e-10, atol=1e-12)
    # a constant term: production at a fixed rate
    turnover = Model(states={"X": "x0"}, parameters=["x0", "kin", "kout"],
                     odes={"X": "kin - kout*X"}, outputs=["X"], t0=0)
    assert np.allclose(exact(turnover, records(t), [5.0, 3.0, 0.4]),
                       3 / 0.4 + (5 - 3 / 0.4) * np.exp(-0.4 * t), rtol=1e-12)
    # rates four decades apart
    stiff = Model(states={"T": "0", "A": "0"}, parameters=["kf", "k"],
                  odes={"T": "-kf*T", "A": "kf*T - k*A"}, outputs=["A"],
                  inputs={"input": {"target": "T"}}, t0=0)
    kf = 1e4
    assert np.allclose(exact(stiff, records(t, [(0, 100)]), [kf, 0.2]),
                       100 * kf / (kf - 0.2) * (np.exp(-0.2 * t) - np.exp(-kf * t)), rtol=1e-9)


def test_closed_form_agrees_with_the_integrator_on_events_regressors_and_covariates():
    m = Model(states={"D": "0", "C": "0", "P": "0"}, parameters=["ka", "k", "q", "lag"],
              regressors=["R"], covariates=["W"],
              odes={"D": "-ka*D", "C": "ka*D - k*R*C - q*C + q*P/W", "P": "q*C - q*P/W"},
              outputs=["C"], inputs={"input": {"target": "D", "delay": "lag", "scale": "W/2"},
                                     "direct": {"target": "C"}})
    rows = []
    for sid, w in ((1, 2.0), (2, 3.5)):
        for time in np.arange(0.0, 60.0, 1.5):
            rows.append({"ID": sid, "TIME": time, "DV": 1.0, "AMT": ".", "RATE": ".", "W": w,
                         "R": 1.0 + (time > 20) + 0.5 * (time > 41), "ADM": 1})
        for time, amount, rate, adm in ((0, 100, ".", 1), (12, 100, ".", 1), (24, 80, 20, 2),
                                        (25, 50, 100, 2), (48, 100, ".", 1)):
            rows.append({"ID": sid, "TIME": time, "DV": ".", "AMT": amount, "RATE": rate, "W": w,
                         "R": 1.0 + (time > 20) + 0.5 * (time > 41), "ADM": adm})
    df = pd.DataFrame(rows).sort_values(["ID", "TIME"], kind="stable")
    psi = np.tile([1.1, 0.15, 0.3, 0.4], (2, 1))
    kw = {"regressors": ["R"], "covariates": ["W"], "input": "ADM"}
    reference = predict(m, df, psi, method="explicit", rtol=1e-11, **kw)
    assert np.allclose(exact(m, df, psi, **kw), reference, rtol=1e-8, atol=1e-10)


def test_closed_form_is_kept_for_linear_systems_with_constant_coefficients():
    used = {}
    for name, ode in (("saturable", "-k*A/(1 + A)"), ("time", "-k*A*exp(-t)"),
                      ("switch", "-k*A*where(t < 2, 1, 2)"), ("linear", "-k*A")):
        m = Model(states={"A": "0"}, parameters=["k"], odes={"A": ode}, outputs=["A"],
                  inputs={"input": {"target": "A"}})
        prob = Problem(m, Dataset.from_records(records([1.0, 4.0], [(0, 10)])))
        prob.predict(np.array([[0.3]]))
        used[name] = prob.last_stats["exact"]
    assert used == {"saturable": 0, "time": 0, "switch": 0, "linear": 1}
    forced = Problem(ONE_CMT, Dataset.from_records(records([1.0], [(0, 10)])), method="explicit")
    forced.predict(np.array([[2.0, 10.0]]))
    assert forced.last_stats["exact"] == 0


def test_matrix_exponential_matches_a_high_precision_reference():
    import mpmath as mp
    from pysaem.solve import _expm
    mp.mp.dps = 80
    rng = np.random.default_rng(0)
    checked = 0
    for trial in range(160):
        m = int(rng.integers(1, 7))
        A = rng.standard_normal((m, m)) * 10 ** rng.uniform(-3, 3)
        if trial % 4 == 1:
            A = -np.abs(A)
        elif trial % 4 == 2:  # one eigenvalue, repeated
            A = np.triu(A)
            np.fill_diagonal(A, -abs(A[0, 0]))
        elif trial % 4 == 3:  # transfers between states with a loss from each
            R = np.abs(rng.standard_normal((m, m))) * 10 ** rng.uniform(-3, 3, (m, m))
            np.fill_diagonal(R, 0)
            A = R - np.diag(R.sum(0) + np.abs(rng.standard_normal(m)))
        h = 10 ** rng.uniform(-2, 1.5)
        exact = mp.expm(mp.matrix(A) * h)
        top = max(abs(v) for v in exact)
        if not mp.mpf("1e-250") < top < mp.mpf("1e250"):
            continue
        L = np.empty((7, m, m))
        assert _expm(np.ascontiguousarray(A), h, L, np.empty(m, np.int64), np.empty(m))
        error = max(abs(mp.mpf(float(L[0][i, j])) - exact[i, j])
                    for i in range(m) for j in range(m)) / top
        # rounding grows with the norm through the squarings
        assert error <= 1e-14 * max(1.0, np.abs(A).sum(0).max() * h)
        checked += 1
    assert checked > 120


def test_interpolated_regressor_follows_the_line_between_records():
    m = Model(states={"X": "X0"}, parameters=["X0", "k"], regressors=["C"], interpolated=["C"],
              odes={"X": "-k*C*X"}, outputs=["X"])
    df = pd.DataFrame({"ID": 1, "TIME": [0, 1, 2, 4.0], "DV": 1.0, "C": [2.0, 0.5, 0.0, 0.0]})
    f = predict(m, df, [10.0, 0.3], regressors=["C"])
    # integral of the broken line: 1.25 on [0, 1], 0.25 on [1, 2], 0 after
    exact = 10 * np.exp(-0.3 * np.array([0, 1.25, 1.5, 1.5]))
    assert np.allclose(f, exact, rtol=TOL)
    # before the first record the regressor holds its first value, the line does not extend
    m0 = Model(**(m.to_dict() | {"t0": -1.0}))
    assert np.allclose(predict(m0, df, [10.0, 0.3], regressors=["C"]), exact * np.exp(-0.6), rtol=TOL)


# hazard outputs: the solver integrates each hazard as one more state (Model.lowered)

def solved(model, df, psi, method="auto", **kw):
    """The problem and the predictions of one chain, every column, every subject solved."""
    kw.setdefault("covariates", ())
    prob = Problem(model, Dataset.from_records(df, **kw), method=method)
    pred, status = prob.predict(np.atleast_2d(np.asarray(psi, float)))
    assert (status == 0).all()
    return prob, pred[0]


def grid(prob, i):
    d = prob.data
    return slice(d.grid_ptr[i], d.grid_ptr[i + 1]), d.grid_time[d.grid_ptr[i]:d.grid_ptr[i + 1]]


def test_a_constant_hazard_is_integrated_from_the_start_of_each_solve():
    m = Model(states={"A": "0"}, parameters=["k", "lam"], odes={"A": "-k*A"}, aux={"h": "lam"},
              outputs=["A", "h"], hazards={"h": "terminal"}, inputs={"dose": {"target": "A"}})
    # subject 1 starts at its first record, subject 2 at its input
    df = pd.DataFrame([
        {"ID": 1, "TIME": 0.5, "DV": 0.0, "AMT": ".", "YTYPE": 2},
        {"ID": 1, "TIME": 1.0, "DV": 1.0, "AMT": ".", "YTYPE": 1},
        {"ID": 1, "TIME": 4.0, "DV": 0.0, "AMT": ".", "YTYPE": 2},
        {"ID": 2, "TIME": -2.0, "DV": ".", "AMT": 10.0, "YTYPE": 1},
        {"ID": 2, "TIME": 0.0, "DV": 0.0, "AMT": ".", "YTYPE": 2},
        {"ID": 2, "TIME": 3.0, "DV": 2.0, "AMT": ".", "YTYPE": 1},
        {"ID": 2, "TIME": 7.5, "DV": 1.0, "AMT": ".", "YTYPE": 2}])
    for t0, start in ((None, [0.5, -2.0]), (-5.0, [-5.0, -5.0])):
        model = m if t0 is None else Model(**(m.to_dict() | {"t0": t0}))
        prob, pred = solved(model, df, [[0.3, 0.05]] * 2, output="YTYPE")
        # A, the integral of h in the column of h, then h itself
        assert prob.n_columns == pred.shape[1] == 3 and prob.last_stats["exact"] == 2
        for i in range(2):
            rows, t = grid(prob, i)
            assert np.allclose(pred[rows, 1], 0.05 * (t - start[i]), rtol=0, atol=1e-12)
            assert (pred[rows, 2] == 0.05).all()


def test_hazards_linear_in_the_states_keep_the_closed_form_after_an_input():
    m = Model(states={"X": "0"}, parameters=["k", "h0", "lam"], odes={"X": "-k*X"},
              aux={"h": "h0*X", "g": "lam"}, outputs=["X", "h", "g"],
              hazards={"h": "single", "g": "repeated"}, inputs={"dose": {"target": "X"}})
    k, h0, lam, td, amount = 0.4, 0.02, 0.1, 1.0, 50.0
    df = pd.DataFrame([{"ID": 1, "TIME": td, "DV": ".", "AMT": amount, "YTYPE": 1}]
                      + [{"ID": 1, "TIME": t, "DV": 0.0, "AMT": ".", "YTYPE": y}
                         for t, y in ((0.0, 2), (0.0, 3), (0.5, 1), (1.0, 1), (2.0, 3), (3.0, 2),
                                      (4.0, 1), (6.0, 3), (9.0, 2))])
    prob, pred = solved(m, df.sort_values("TIME", kind="stable"), [k, h0, lam], output="YTYPE")
    assert prob.last_stats["exact"] == 1
    t = prob.data.grid_time
    x = np.where(t > td, amount * np.exp(-k * (t - td)), 0.0)
    integral = np.where(t > td, h0 * amount / k * -np.expm1(-k * (t - td)), 0.0)
    # outputs X, h, g with the integrals of h and g in their columns, then h and g themselves
    assert np.allclose(pred, np.column_stack([x, integral, lam * t, h0 * x, np.full_like(t, lam)]),
                       rtol=1e-12, atol=1e-15)


def test_an_exp_link_matches_the_exponential_integral():
    from scipy.special import expi
    m = Model(states={"X": "X0"}, parameters=["X0", "g", "h0", "beta"], odes={"X": "g*X"},
              aux={"h": "h0*exp(beta*X)"}, outputs=["X", "h"], hazards={"h": "terminal"})
    rng = np.random.default_rng(1)
    n = 30
    psi = np.column_stack([5.0 * np.exp(0.3 * rng.standard_normal(n)),
                           0.1 * np.exp(0.3 * rng.standard_normal(n)), np.full(n, 0.01),
                           np.full(n, 0.15)])
    end = rng.uniform(1.0, 12.0, n)
    rows = []
    for i in range(n):
        rows += [{"ID": i, "TIME": t, "DV": dv, "YTYPE": 2} for t, dv in ((0.0, 0.0), (end[i], 1.0))]
        rows += [{"ID": i, "TIME": t, "DV": 1.0, "YTYPE": 1} for t in np.arange(0.0, end[i], 2.0)]
    prob, pred = solved(m, pd.DataFrame(rows), psi, output="YTYPE")
    assert prob.last_stats["exact"] == 0
    for i, (x0, g, h0, beta) in enumerate(psi):
        rows, t = grid(prob, i)
        exact = h0 / g * (expi(beta * x0 * np.exp(g * t)) - expi(beta * x0))
        H = pred[rows, 1]
        assert H[0] == 0.0 and np.allclose(H[1:], exact[1:], rtol=2e-5, atol=0)
        # the increment over the follow-up, the quantity the likelihood reads
        assert H[-1] - H[0] == pytest.approx(exact[-1], rel=1e-6)


@pytest.mark.parametrize("shape", [0.5, 0.8, 1.5])
def test_a_weibull_hazard_from_the_origin_stays_on_the_explicit_integrator(shape):
    # below shape 1 the hazard is infinite at the first record: the integral there is resolved
    # to an absolute 1e-3 * rtol, not to its own size (zero)
    m = Model(states={}, odes={}, parameters=["lam", "kw"], aux={"h": "lam*kw*t**(kw - 1)"},
              outputs=["h"], hazards={"h": "single"})
    df = pd.DataFrame([{"ID": i, "TIME": t, "DV": 0.0} for i, end in enumerate((2.0, 5.0, 10.0))
                       for t in (0.0, 0.5, 1.0, 2.0, end)])
    prob, pred = solved(m, df, [[0.1, shape]] * 3)
    assert prob.last_stats == {"implicit": 0, "exact": 0, "failed": 0, "events": 0, "unlocated": 0}
    t = prob.data.grid_time
    # 2.2e-6 at most here (shape 0.8), a few times rtol
    assert np.allclose(pred[:, 0], 0.1 * t ** shape, rtol=5e-6, atol=0)
    # the hazard itself, infinite at the origin below shape 1, fails no solve
    assert np.isinf(pred[t == 0, 1]).all() == (shape < 1)


@pytest.mark.parametrize("method", ["auto", "explicit", "implicit"])
def test_resets_and_steady_states_leave_the_hazard_integral_as_it_was(method):
    # SS at 0 (every 12), EVID 3 at 5.5: the system restarts, the risk already run stays
    m = Model(states={"A": "0"}, parameters=["k", "V", "lam", "b"], odes={"A": "-k*A"},
              aux={"C": "A/V", "h": "lam + b*C"}, outputs=["C", "h"], hazards={"h": "terminal"},
              inputs={"dose": {"target": "A"}})
    k, V, lam, b = 0.2, 10.0, 0.05, 0.02
    df = pd.DataFrame(
        [{"ID": 1, "TIME": t, "DV": ".", "AMT": a, "EVID": e, "SS": ss, "II": ii, "YTYPE": 1}
         for t, a, e, ss, ii in ((0.0, 100.0, 1, 1, 12.0), (5.5, 0.0, 3, 0, "."))]
        + [{"ID": 1, "TIME": t, "DV": 0.0, "AMT": ".", "EVID": 0, "SS": 0, "II": ".", "YTYPE": y}
           for t, y in ((0.0, 2), (1.0, 1), (1.0, 2), (5.0, 2), (6.0, 1), (10.0, 2))])
    prob, pred = solved(m, df.sort_values("TIME", kind="stable"), [k, V, lam, b], method=method,
                        output="YTYPE")
    assert prob.last_stats["exact"] == (method == "auto")
    t = prob.data.grid_time
    a_ss = 100.0 / -np.expm1(-12 * k)
    exact = lam * t + b * a_ss / (V * k) * -np.expm1(-k * np.minimum(t, 5.5))
    assert np.allclose(pred[:, 1], exact, rtol=0, atol=1e-12 if method == "auto" else 2e-6)


def test_a_hazard_of_time_does_not_prevent_a_steady_state_of_the_system():
    # the steady state needs equations free of t; a hazard of t is integrated, not cycled
    from scipy.integrate import quad
    m = Model(states={"A": "0"}, parameters=["k", "V", "lam", "kw", "beta"], odes={"A": "-k*A"},
              aux={"C": "A/V", "h": "lam*kw*t**(kw - 1)*exp(beta*C)"}, outputs=["C", "h"],
              hazards={"h": "single"}, inputs={"dose": {"target": "A"}})
    k, V, lam, kw, beta = 0.2, 10.0, 0.05, 0.5, 0.3
    df = pd.DataFrame(
        [{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": 100.0, "SS": 1, "II": 12.0, "YTYPE": 1}]
        + [{"ID": 1, "TIME": t, "DV": 0.0, "AMT": ".", "SS": 0, "II": ".", "YTYPE": 2}
           for t in (0.0, 1.0, 5.0, 10.0)])
    prob, pred = solved(m, df, [k, V, lam, kw, beta], output="YTYPE")
    c_ss = 100.0 / -np.expm1(-12 * k) / V
    exact = [quad(lambda s: lam * kw * s ** (kw - 1) * np.exp(beta * c_ss * np.exp(-k * s)), 0, t,
                  epsabs=0, epsrel=1e-12, limit=200)[0] for t in (1.0, 5.0, 10.0)]
    assert pred[0, 1] == 0.0 and np.allclose(pred[1:, 1], exact, rtol=2e-6, atol=0)


def test_occasions_and_regressors_keep_the_closed_form_of_a_hazard():
    m = Model(states={}, odes={}, parameters=["lam", "beta"], regressors=["X"],
              aux={"h": "lam*exp(beta*X)"}, outputs=["h"], hazards={"h": "repeated"})
    df = pd.DataFrame([{"ID": 1, "TIME": t, "DV": 0, "X": x, "OCC": o} for t, x, o in
                       ((0.0, 0.0, 1), (2.0, 1.0, 1), (5.0, 1.0, 2), (7.0, 0.5, 2), (9.0, 0.5, 2))])
    lam1, lam2, beta = 0.2, 0.5, 0.7
    prob, pred = solved(m, df, [[[lam1, beta], [lam2, beta]]], regressors=["X"])
    assert prob.last_stats["exact"] == 1
    # occasion 2 starts at 5; X is 0 on [0, 2), 1 on [2, 7), 0.5 from 7
    pieces = [lam1 * 2.0, lam1 * np.exp(beta) * 3.0, lam2 * np.exp(beta) * 2.0,
              lam2 * np.exp(0.5 * beta) * 2.0]
    assert np.allclose(pred[:, 0], np.concatenate([[0.0], np.cumsum(pieces)]), rtol=0, atol=1e-12)


def test_a_state_with_an_infinite_scale_has_a_zero_jacobian_column():
    # a hazard integral during steady-state cycles: no equation reads it
    from pysaem.solve import _jac_fd, compile_model
    m = Model(states={"X": "1", "H": "0"}, parameters=["k"], odes={"X": "-k*X", "H": "X**2"},
              outputs=["X"])
    J = np.empty((2, 2))
    _jac_fd(compile_model(m)[0], 0.0, np.array([2.0, 5.0]), np.array([0.5]), np.zeros(0), J,
            np.array([1e-6, np.inf]))
    assert np.allclose(J[:, 0], [-0.5, 4.0], rtol=1e-6) and (J[:, 1] == 0.0).all()


@pytest.mark.parametrize("method", ["auto", "explicit", "implicit"])
def test_states_at_zero_fed_by_a_function_of_time(method):
    # no initial state or input sets a scale (a transit input t^5 exp(-3t)): the solve must not collapse
    m = Model(states={"Ad": 0, "Ac": 0}, odes={"Ad": "100*t**5*exp(-3*t) - ka*Ad", "Ac": "ka*Ad - k*Ac"},
              parameters=["ka", "k"], outputs=["Ac"], t0=0)
    times = [0.25, 0.5, 1, 2, 4, 8, 24]
    prob = Problem(m, Dataset.from_records(pd.DataFrame({"ID": 1, "TIME": times, "DV": 1.0})), method=method)
    pred, status = prob.predict(np.array([[1.0, 0.2]]))
    assert status.ravel().tolist() == [0]
    ref = solve_ivp(lambda t, y: [100 * t**5 * np.exp(-3 * t) - y[0], y[0] - 0.2 * y[1]], (0, 24), [0, 0],
                    t_eval=times, method="Radau", rtol=1e-10, atol=1e-14).y[1]
    np.testing.assert_allclose(pred[0, :, 0], ref, rtol=1e-4)


def test_an_infusion_after_a_smooth_segment_starts_with_a_fresh_step():
    # the step of the long segment before 36 carried over the infusion start left 36.409 8e-5 off
    k, Vm, Km, A0, amt, dur = 0.1178125, 0.51112, 4.9825, 1.472, 7.17, 1.506
    doses, times = [0.0, 24.0, 36.0], [33.357, 36.409, 36.866, 39.436]
    m = Model(states={"A": "A0"}, odes={"A": "-k*A - Vm*A/(Km + A)"}, parameters=["k", "Vm", "Km", "A0"],
              outputs=["A"], inputs={"dose": {"target": "A"}}, t0=0)
    rows = [{"ID": 1, "TIME": d, "AMT": amt, "TINF": dur, "EVID": 1, "MDV": 1, "DV": np.nan} for d in doses]
    rows += [{"ID": 1, "TIME": t, "AMT": np.nan, "TINF": np.nan, "EVID": 0, "MDV": 0, "DV": 1.0} for t in times]
    data = Dataset.from_records(pd.DataFrame(rows))
    pred, status = Problem(m, data).predict(np.array([[k, Vm, Km, A0]]))
    assert status.ravel().tolist() == [0]
    stops = sorted({0.0, *times, *doses, *[d + dur for d in doses]})
    y, ref = [A0], {}
    for a, b in zip(stops[:-1], stops[1:]):
        rate = sum(amt / dur for d in doses if d <= a < d + dur)
        y = solve_ivp(lambda t, y: [-k * y[0] - Vm * y[0] / (Km + y[0]) + rate], (a, b), y,
                      method="Radau", rtol=1e-12, atol=1e-14).y[:, -1]
        ref[b] = y[0]
    np.testing.assert_allclose(pred[0][data.obs_grid, 0], [ref[t] for t in times], rtol=1e-5)


def test_switches_on_a_periodic_function_of_time_are_breakpoints():
    # t % 24 and floor(t / 24) jump every day: the solver stops there, as on t < 8
    m = Model(states={"A": "10"}, odes={"A": "where(mod(t, 24) < 8, kin, 0) - k*(1 + 0.1*floor(t/24))*A"},
              parameters=["kin", "k"], outputs=["A"], t0=0)
    assert [(str(b), q) for b, q in zip(m.symbolic.breakpoints, m.symbolic.periods)] == [
        ("0", 24.0), ("8", 24.0)]
    times = [3.0, 7.9, 8.1, 20.0, 30.0, 47.5, 56.2, 71.0]
    pred, status = Problem(m, Dataset.from_records(pd.DataFrame({"ID": 1, "TIME": times, "DV": 1.0}))).predict(
        np.array([[2.0, 0.3]]))
    assert status.ravel().tolist() == [0]
    stops = sorted({0.0, *times, *[d + h for d in (0.0, 24.0, 48.0) for h in (0.0, 8.0)], 24.0, 48.0, 72.0})
    y, ref = [10.0], {}
    for a, b in zip(stops[:-1], stops[1:]):
        kin, day = (2.0 if a % 24 < 8 else 0.0), math.floor(a / 24)
        y = solve_ivp(lambda t, y: [kin - 0.3 * (1 + 0.1 * day) * y[0]], (a, b), y, method="Radau",
                      rtol=1e-12, atol=1e-14).y[:, -1]
        ref[b] = y[0]
    np.testing.assert_allclose(pred[0, :, 0], [ref[t] for t in times], rtol=1e-5)


def test_a_switch_time_given_as_a_covariate_is_a_breakpoint():
    # the kill starts at a time of each subject; stepped across, it was 1e-4 off at rtol 1e-6
    m = Model(states={"TS": "TS0"}, parameters=["TS0", "beta", "TSmax", "kkill"], covariates=["TSTART"],
              odes={"TS": "beta*TS*log(TSmax/TS) - where(t >= TSTART, kkill, 0)*TS"}, outputs=["TS"], t0=0)
    times = [0.5, 1, 2, 3, 5, 8, 10, 13, 16, 20, 24, 28, 35, 42, 49, 56]
    th, starts = [100.0, 0.03, 1000.0, 0.08], {1: 5.0, 2: 11.3}
    df = pd.DataFrame([{"ID": i, "TIME": t, "DV": 1.0, "TSTART": s} for i, s in starts.items() for t in times])
    got = predict(m, df, [th, th], covariates=["TSTART"])
    ref = []
    for ts in starts.values():
        def f(t, y, kill):
            return [th[1] * y[0] * math.log(th[2] / y[0]) - kill * y[0]]
        a = solve_ivp(f, (0, ts), [th[0]], args=(0.0,), method="Radau", rtol=1e-12, atol=1e-12, dense_output=True)
        b = solve_ivp(f, (ts, 56), a.y[:, -1], args=(th[3],), method="Radau", rtol=1e-12, atol=1e-12,
                      dense_output=True)
        ref += [(a if t < ts else b).sol(t)[0] for t in times]
    np.testing.assert_allclose(got, ref, rtol=1e-5)


def test_a_switch_at_a_lag_after_each_amount_is_a_breakpoint():
    # the absorption rate changes lag after each amount, at t_last + lag: 5e-5 off when stepped across
    ka = "where(t - t_last < lag, ka1, ka2)"
    m = Model(states={"Ad": "0", "Ac": "0"}, parameters=["ka1", "ka2", "lag", "k"],
              odes={"Ad": f"-{ka}*Ad", "Ac": f"{ka}*Ad - k*Ac"}, outputs=["Ac"],
              inputs={"dose": {"target": "Ad"}}, t0=0)
    doses, lag = [(0.0, 100.0), (12.0, 50.0), (24.0, 80.0)], 3.1
    times = [1.0, 3.0, 6.0, 11.0, 13.0, 15.5, 20.0, 25.0, 27.0, 30.0, 36.0, 48.0]
    th = [0.2, 1.0, lag, 0.15]
    got = predict(m, records(times, doses), th)
    stops = sorted({0.0, *times, *[d + lag for d, _ in doses], *[d for d, _ in doses]})
    y, ref = np.zeros(2), {}
    for a, b in zip(stops[:-1], stops[1:]):
        y = y + [sum(x for d, x in doses if d == a), 0.0]
        last = max(d for d, _ in doses if d <= a)
        k1 = th[0] if a < last + lag else th[1]
        y = solve_ivp(lambda t, y: [-k1 * y[0], k1 * y[0] - th[3] * y[1]], (a, b), y, method="Radau",
                      rtol=1e-12, atol=1e-14).y[:, -1]
        ref[b] = y[1]
    np.testing.assert_allclose(got, [ref[t] for t in times], rtol=1e-5)

def test_rates_nine_decades_apart_leave_the_closed_form_for_the_integrators():
    # squaring exp(h M) 40 times left the slow state 1e-3 off; the integrators take over from there
    from scipy.linalg import expm
    m = Model(states={"A1": "0", "A2": "0", "A3": "0"},
              odes={"A1": "-ktr*A1", "A2": "ktr*A1 - ktr*A2", "A3": "ktr*A2 - k*A3"},
              parameters=["ktr", "k"], outputs=["A3"], inputs={"dose": {"target": "A1"}}, t0=0)
    times = [1.0, 24.0, 168.0]
    rows = [{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": 100.0}] + [{"ID": 1, "TIME": t, "DV": 1.0, "AMT": "."} for t in times]
    prob = Problem(m, Dataset.from_records(pd.DataFrame(rows)))
    pred, status = prob.predict(np.array([[1e10, 0.1]]))
    assert status.ravel().tolist() == [0] and prob.last_stats["exact"] == 0
    A = np.array([[-1e10, 0, 0], [1e10, -1e10, 0], [0, 1e10, -0.1]])
    ref = [(expm(A * t) @ [100.0, 0, 0])[2] for t in times]
    np.testing.assert_allclose(prob.gather(pred)[0], ref, rtol=1e-5)
    # rates two decades apart keep the closed form
    prob.predict(np.array([[10.0, 0.1]]))
    assert prob.last_stats["exact"] == 1


def test_a_state_far_below_the_largest_keeps_its_own_tolerance():
    # P near 18 beside R at 2e36: the floor borrowed from R left P uncontrolled
    m = Model(states={"P": "P0", "R": "R0"}, aux={"K": "Kmax*R/R0"}, odes={"P": "r*P*(1 - P/K)", "R": "-c*P"},
              parameters=["r", "Kmax", "c", "P0", "R0"], outputs=["P"], t0=0)
    times = [0.25, 1.0, 10.0, 48.0, 168.0]
    th = [0.1766, 1338.3, 2.39e-10, 17.69, 2.2e36]
    pred, status = Problem(m, Dataset.from_records(pd.DataFrame({"ID": 1, "TIME": times, "DV": 1.0}))).predict(
        np.array([th]))
    assert status.ravel().tolist() == [0]
    ref = solve_ivp(lambda t, y: [th[0] * y[0] * (1 - y[0] / (th[1] * y[1] / th[4])), -th[2] * y[0]], (0, 168),
                    [th[3], th[4]], method="Radau", rtol=1e-10, atol=[1e-14, 1e-14 * th[4]], t_eval=times).y[0]
    np.testing.assert_allclose(pred[0, :, 0], ref, rtol=1e-5)


def test_an_input_with_two_targets_splits_each_amount_by_their_scales():
    # one record reaches A (fraction f, after a lag) and B (the rest): the same as two inputs
    # each fed by its own copy of the record
    odes = {"A": "-ka*A", "B": "-kb*B", "C": "ka*A + kb*B - k*C"}
    split = Model(states={"A": "0", "B": "0", "C": "0"}, odes=odes, parameters=["ka", "kb", "k", "f", "lag"],
                  outputs=["C"], inputs={"dose": {"target": ["A", "B"], "scale": ["f", "1 - f"], "delay": ["lag", 0]}})
    two = Model(states={"A": "0", "B": "0", "C": "0"}, odes=odes, parameters=["ka", "kb", "k", "f", "lag"],
                outputs=["C"], inputs={"a": {"target": "A", "scale": "f", "delay": "lag"},
                                       "b": {"target": "B", "scale": "1 - f"}})
    times = [0.5, 1, 2, 6, 12, 24, 30, 48]
    rows = [{"ID": 1, "TIME": t, "DV": ".", "AMT": 100.0, "RATE": r, "EVID": e}
            for t, r, e in ((0.0, 0, 1), (12.0, 50.0, 4))]
    rows += [{"ID": 1, "TIME": t, "DV": 1.0, "AMT": ".", "RATE": ".", "EVID": 0} for t in times]
    psi = np.array([[1.2, 0.3, 0.1, 0.6, 0.75]])
    p1, s1 = Problem(split, Dataset.from_records(pd.DataFrame(rows))).predict(psi)
    both = pd.DataFrame([dict(r, CMT=c) for r in rows for c in (("a", "b") if r["EVID"] else ("a",))])
    p2, s2 = Problem(two, Dataset.from_records(both, input="CMT", input_labels=["a", "b"])).predict(psi)
    assert s1.ravel().tolist() == s2.ravel().tolist() == [0]
    np.testing.assert_allclose(p1, p2, rtol=1e-10)
    ss = pd.DataFrame(rows).assign(SS=[1, 0] + [0] * len(times), II=[24, 0] + [0] * len(times))
    with pytest.raises(ValueError, match="several targets"):
        Problem(split, Dataset.from_records(ss))


def switched(f, g, span, y0, times, side):
    """SciPy reference that locates the switch g = 0: f(t, y, side) on the side of sign side, the
    other side from the crossing that leaves it on."""
    a, y, out = span[0], np.asarray(y0, float), {}
    while a < span[1]:
        ev = lambda t, y, side: g(t, y)  # noqa: E731
        ev.terminal, ev.direction = True, -side
        s = solve_ivp(f, (a, span[1]), y, args=(side,), events=ev, dense_output=True, method="DOP853",
                      rtol=1e-12, atol=1e-14)
        b = s.t[-1]
        out.update({t: s.sol(t) for t in times if a <= t <= b})
        a, y, side = b, s.y[:, -1], -side
    return np.array([out[t] for t in times])


@pytest.mark.parametrize("method", ["auto", "implicit"])
def test_a_switch_on_a_state_is_located(method):
    # elimination linear below Km, at its cap above: 1.4e-4 off when the kink was crossed under error control
    m = Model(states={"A": "0"}, parameters=["k", "Km"], odes={"A": "-where(A < Km, k*A, k*Km)"},
              outputs=["A"], inputs={"dose": {"target": "A"}})
    assert [str(g) for g in m.symbolic.switches] == ["_p1 - _y0"]
    t = np.array([0.5, 1, 2, 3, 4, 5, 6, 8, 12, 24.0])
    k, Km, A0 = 0.3, 20.0, 100.0
    t1 = (A0 - Km) / (k * Km)
    prob = Problem(m, Dataset.from_records(records(t, [(0, A0)]), covariates=()), method=method)
    pred, status = prob.predict(np.array([[k, Km]]))
    assert status.ravel().tolist() == [0] and prob.last_stats["events"] == 1
    exact = np.where(t < t1, A0 - k * Km * t, Km * np.exp(-k * (t - t1)))
    np.testing.assert_allclose(prob.gather(pred)[0], exact, rtol=1e-5)


def test_a_threshold_effect_max_of_a_state_is_located():
    # E driven by max(C - thr, 0): the kink where C falls below thr is a switch
    m = Model(states={"A": "0", "E": "E0"}, parameters=["k", "V", "thr", "kout", "s", "E0"], aux={"C": "A/V"},
              odes={"A": "-k*A", "E": "kout*E0*(1 + s*max(C - thr, 0)) - kout*E"}, outputs=["E"],
              inputs={"dose": {"target": "A"}})
    th = [0.2, 10.0, 2.0, 0.5, 0.3, 5.0]
    times = [1.0, 3.0, 6.0, 8.0, 10.0, 16.0, 24.0]
    prob = Problem(m, Dataset.from_records(records(times, [(0, 100.0)]), covariates=()))
    pred, status = prob.predict(np.array([th]))
    assert status.ravel().tolist() == [0] and prob.last_stats["events"] == 1
    k, V, thr, kout, s, E0 = th
    f = lambda t, y, side: [-k * y[0], kout * E0 * (1 + s * (y[0] / V - thr if side > 0 else 0.0)) - kout * y[1]]  # noqa: E731
    ref = switched(f, lambda t, y: y[0] / V - thr, (0.0, 24.0), [100.0, E0], times, 1)[:, 1]
    np.testing.assert_allclose(prob.gather(pred)[0], ref, rtol=1e-5)


def test_a_switch_crossed_during_a_delayed_infusion_is_located():
    # X crosses Km during an infusion that starts at the lag: 3.4e-5 off without location
    k, Vmax, Km, X0, lag = 0.125, 125.0, 40.0, 12.5, 0.8
    times = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    rows = [{"ID": 1, "TIME": 0.0, "AMT": 300.0, "TINF": 2.0, "EVID": 1, "MDV": 1, "DV": np.nan}]
    rows += [{"ID": 1, "TIME": t, "AMT": np.nan, "TINF": np.nan, "EVID": 0, "MDV": 0, "DV": 1.0} for t in times]
    m = Model(states={"X": "X0"}, odes={"X": "k*X*(1 - X/Vmax) if X < Km else k*X*(Km/(X + Km))"},
              parameters=["k", "Vmax", "Km", "Tlag", "X0"], outputs=["X"],
              inputs={"dose": {"target": "X", "delay": "Tlag"}}, t0=0.0)
    prob = Problem(m, Dataset.from_records(pd.DataFrame(rows)))
    pred, status = prob.predict(np.array([[k, Vmax, Km, lag, X0]]))
    assert status.ravel().tolist() == [0] and prob.last_stats["events"] == 1
    y, ref = [X0], []
    for a, b, u in ((0.0, lag, 0.0), (lag, lag + 2.0, 150.0), (lag + 2.0, 4.0, 0.0)):
        f = lambda t, z, side, u=u: [(k * z[0] * (1 - z[0] / Vmax) if side < 0 else k * z[0] * Km / (z[0] + Km)) + u]  # noqa: E731
        inside = [t for t in times if a < t <= b] + [b]
        side = -1 if y[0] < Km else 1
        out = switched(f, lambda t, z: z[0] - Km, (a, b), y, inside, side)
        ref += list(out[:-1, 0])
        y = out[-1]
    np.testing.assert_allclose(prob.gather(pred)[0], ref, rtol=1e-5)


def test_a_switch_on_a_state_in_python_code_is_located():
    # loss above a threshold written as a statement, read as equations (fuzz_375: 5.7e-4 off)
    from pysaem.reader import from_python
    code = '''
def rhs(t, y, p):
    A = y[0]
    loss = p["k1"] * A
    if A > p["thr"]:
        loss += p["kex"] * (A - p["thr"])
    return [-loss]

def observe(t, y, p):
    return [y[0]]
'''
    m = from_python(code, states=["A"], parameters=["k1", "kex", "thr"], inputs={"dose": "A"},
                    values={"k1": 0.1, "kex": 0.5, "thr": 20.0})
    assert m.reading["form"] == "equations" and len(m.symbolic.switches) == 1
    k1, kex, thr, A0 = 0.1, 0.5, 20.0, 100.0
    t = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 24.0])
    prob = Problem(m, Dataset.from_records(records(t, [(0, A0)]), covariates=()))
    pred, status = prob.predict(np.array([[k1, kex, thr]]))
    assert status.ravel().tolist() == [0] and prob.last_stats["events"] == 1
    eq = kex * thr / (k1 + kex)
    t1 = math.log((A0 - eq) / (thr - eq)) / (k1 + kex)
    exact = np.where(t < t1, eq + (A0 - eq) * np.exp(-(k1 + kex) * t), thr * np.exp(-k1 * (t - t1)))
    np.testing.assert_allclose(prob.gather(pred)[0], exact, rtol=1e-5)


def test_switches_met_again_and_again_are_no_longer_located():
    # an oscillator crossing x = 0 twice a period: after _MAX_EVENTS in one segment the subject goes on
    # without locating; sliding along x = c ends at once, located or not, without a hang
    from pysaem.solve import _MAX_EVENTS
    m = Model(states={"x": "1", "v": "0", "z": "0"}, parameters=["w"],
              odes={"x": "v", "v": "-w**2*x", "z": "where(x > 0, 1, 0)"}, outputs=["x", "z"], t0=0)
    times = [10.0, 60.0]
    df = pd.DataFrame([{"ID": 1, "TIME": t, "DV": 1.0, "YTYPE": y} for t in times for y in (1, 2)])
    prob = Problem(m, Dataset.from_records(df, covariates=(), output="YTYPE"))
    pred, status = prob.predict(np.array([[2 * math.pi]]))
    assert status.ravel().tolist() == [0]
    assert prob.last_stats["events"] == _MAX_EVENTS and prob.last_stats["unlocated"] == 1
    # half of the time above zero, the second record as accurate as without location (1e-3)
    np.testing.assert_allclose(prob.gather(pred)[0][1::2], [5.0, 30.0], rtol=1e-3)
    slide = Model(states={"A": "1"}, parameters=["c", "v"], odes={"A": "where(A > c, -v, v)"}, outputs=["A"])
    prob = Problem(slide, Dataset.from_records(records([0.25, 0.5, 1.0, 5.0]), covariates=()))
    prob.predict(np.array([[0.4, 1.0]]))
    assert prob.last_stats["events"] <= _MAX_EVENTS


@pytest.mark.parametrize("g", [1.0, 0.7])
def test_a_power_of_a_state_at_zero_keeps_the_implicit_integrator_finite(g):
    # d(C**g)/dC came out as g*C**g/C, nan at C = 0 even for g = 1; for g < 1 it is infinite
    m = Model(states={"B": "1e7", "C": "0"}, parameters=["k", "EC50", "g"],
              odes={"B": "k*(1 - C**g/(EC50**g + C**g))*B*(1 - B/1e10)", "C": "-C"}, outputs=["B"], t0=0)
    times = [1.0, 6.0, 24.0, 96.0]
    data = Dataset.from_records(pd.DataFrame({"ID": 1, "TIME": times, "DV": 1.0}))
    psi = np.array([[0.5, 0.65, g]])
    preds = [Problem(m, data, method=k).predict(psi) for k in ("implicit", "explicit")]
    assert [s.ravel().tolist() for _, s in preds] == [[0], [0]]
    np.testing.assert_allclose(preds[0][0], preds[1][0], rtol=1e-4)


# ---------- signals of the amounts received (model.HISTORY) ----------

DOSES = [(0, 100.0), (12, 50.0), (20, 200.0)]


def restarted(rate, k, doses, times):
    """SciPy reference of dA/dt = rate(amount, t - t_dose) - k A, the input restarted at each
    amount from the most recent one."""
    starts = [d[0] for d in doses] + [np.inf]
    out, a = {}, 0.0
    for j, (td, amt) in enumerate(doses):
        end = min(starts[j + 1], max(times))
        inside = [x for x in times if td < x <= starts[j + 1]]
        if end > td:
            sol = solve_ivp(lambda t, y: [rate(amt, t - td) - k * y[0]], (td, end), [a],
                            method="LSODA", rtol=1e-11, atol=1e-12, dense_output=True)
            out.update({x: sol.sol(x)[0] for x in inside})
            a = sol.y[0, -1]
    return np.array([out[x] for x in times])


def test_a_model_without_input_reads_the_amounts_through_the_signals():
    # an observation that only decays from each amount: no state receives it (a blind model of
    # 1 November 2026 needed a bookkeeping state to accept the AMT column)
    m = Model(states={}, odes={}, parameters=["k"], aux={"y": "amount_last*exp(-k*(t - t_last))"},
              outputs=["y"])
    times = [1.0, 11.0, 13.0, 25.0]
    f = predict(m, records(times, DOSES), [0.2])
    ref = [100 * np.exp(-0.2), 100 * np.exp(-2.2), 50 * np.exp(-0.2), 200 * np.exp(-1.0)]
    assert np.allclose(f, ref, rtol=1e-12)
    plain = Model(states={}, odes={}, parameters=["m"], aux={"y": "m"}, outputs=["y"])
    with pytest.raises(ValueError, match="declares no input and reads none of"):
        predict(plain, records(times, DOSES), [1.0])


@pytest.mark.parametrize("method", ["explicit", "implicit"])
def test_a_weibull_input_restarted_at_each_amount_matches_scipy(method):
    m = Model(states={"Ad": "0", "Ac": "0"}, parameters=["a", "b", "k"],
              aux={"w": "amount_last*b/a*((t - t_last)/a)**(b - 1)*exp(-((t - t_last)/a)**b)"},
              odes={"Ad": "-w", "Ac": "w - k*Ac"}, outputs=["Ac"], inputs={"dose": {"target": "Ad"}})
    assert m.symbolic.history and not m.symbolic.autonomous
    a, b, k = 3.0, 1.8, 0.15
    times = [0.5, 2, 5, 11.9, 12, 13, 16, 19.99, 21, 25, 40.0]
    f = predict(m, records(times, DOSES), [a, b, k], method=method)
    ref = restarted(lambda amt, s: amt * b / a * (s / a) ** (b - 1) * np.exp(-(s / a) ** b), k,
                    DOSES, times)
    assert np.allclose(f, ref, rtol=1e-4, atol=1e-8)


def test_the_savic_transit_input_under_repeated_amounts_matches_scipy():
    rate = "amount_last*ktr*exp(n*log(ktr*(t - t_last)) - ktr*(t - t_last) - lgamma(n + 1))"
    m = Model(states={"Aa": "0", "Ac": "0"}, parameters=["MTT", "n", "ka", "k"],
              aux={"ktr": "(n + 1)/MTT", "w": rate}, odes={"Aa": "w - ka*Aa", "Ac": "ka*Aa - k*Ac"},
              outputs=["Ac"], inputs={"dose": {"target": "Ac", "scale": "0"}})
    mtt, n, ka, k = 2.0, 3.5, 1.2, 0.1
    ktr = (n + 1) / mtt
    times = [0.5, 1, 2, 4, 11.5, 12.5, 14, 20, 20.5, 22, 30.0]
    f = predict(m, records(times, DOSES), [mtt, n, ka, k])

    def ref():
        starts = [d[0] for d in DOSES] + [np.inf]
        out, y = {}, [0.0, 0.0]
        for j, (td, amt) in enumerate(DOSES):
            end = min(starts[j + 1], max(times))
            sol = solve_ivp(lambda t, z: [amt * ktr * math.exp(n * math.log(ktr * max(t - td, 1e-300))
                                                               - ktr * (t - td) - math.lgamma(n + 1))
                                          - ka * z[0], ka * z[0] - k * z[1]],
                            (td, end), y, method="LSODA", rtol=1e-11, atol=1e-12, dense_output=True)
            out.update({x: sol.sol(x)[1] for x in times if td < x <= starts[j + 1]})
            y = list(sol.y[:, -1])
        return np.array([out[x] for x in times])
    assert np.allclose(f, ref(), rtol=1e-4, atol=1e-8)


def test_a_scale_that_depends_on_the_amount_is_read_at_its_record():
    # bioavailability falling with the amount, and the same in an equation
    m = Model(states={"Ad": "0", "Ac": "0"}, parameters=["ka", "k", "A50"],
              odes={"Ad": "-ka*Ad", "Ac": "ka*Ad - k*Ac"}, outputs=["Ac"],
              inputs={"dose": {"target": "Ad", "scale": "1/(1 + amount_last/A50)"}})
    ka, k, a50 = 1.0, 0.2, 100.0
    t = np.array([1, 6, 12, 13, 18, 20, 22, 30.0])
    f = predict(m, records(t, DOSES), [ka, k, a50])
    exact = sum(np.where(t > td, amt / (1 + amt / a50) * ka / (ka - k)
                         * (np.exp(-k * (t - td)) - np.exp(-ka * (t - td))), 0.0) for td, amt in DOSES)
    assert np.allclose(f, exact, rtol=TOL)
    # the parameter of an equation follows the most recent amount, whatever the state holds
    e = Model(states={"A": "0"}, parameters=["k", "A50"], odes={"A": "-k*A/(1 + amount_last/A50)"},
              outputs=["A"], inputs={"dose": {"target": "A"}})
    t = np.array([6, 12, 16, 20, 25.0])
    f = predict(e, records(t, DOSES), [0.2, 100.0])
    kk = [0.2 / (1 + amt / 100.0) for _, amt in DOSES]
    a12 = 100 * math.exp(-kk[0] * 12) + 50
    a20 = a12 * math.exp(-kk[1] * 8) + 200
    exact = [100 * math.exp(-kk[0] * 6), 100 * math.exp(-kk[0] * 12),
             a12 * math.exp(-kk[1] * 4), a12 * math.exp(-kk[1] * 8), a20 * math.exp(-kk[2] * 5)]
    assert np.allclose(f, exact, rtol=TOL)


def test_the_count_and_the_first_time_of_the_amounts():
    m = Model(states={"A": "0"}, parameters=["k"], odes={"A": "-k*A"},
              aux={"N": "n_amounts", "T1": "t_first", "TL": "t_last", "AL": "amount_last"},
              outputs=["N", "T1", "TL", "AL"], inputs={"dose": {"target": "A"}})
    times = [0.5, 1, 2, 3.5, 5, 9]
    rows = [{"ID": 1, "TIME": t, "DV": 1.0, "AMT": ".", "CMT": c} for t in times for c in (1, 2, 3, 4)]
    rows += [{"ID": 1, "TIME": t, "DV": ".", "AMT": a, "CMT": "."} for t, a in ((1, 10), (3, 30), (3, 40))]
    df = pd.DataFrame(rows).sort_values("TIME", kind="stable")
    f = predict(m, df, [0.1], output="CMT").reshape(len(times), 4)
    # before the first amount: the start of the solve (the first record, 0.5) and zeros; an
    # observation at the time of a record sees it; two records at one time count twice
    assert f.tolist() == [[0, 0.5, 0.5, 0], [1, 1, 1, 10], [1, 1, 1, 10], [3, 1, 3, 40],
                          [3, 1, 3, 40], [3, 1, 3, 40]]


@pytest.mark.parametrize("delay", [0.0, 1.5])
def test_a_reset_without_amount_does_not_count_and_additional_amounts_do(delay):
    m = Model(states={"A": "0"}, parameters=["k", "lag"], odes={"A": "-k*A"},
              aux={"N": "n_amounts", "TL": "t_last"}, outputs=["N", "TL"],
              inputs={"dose": {"target": "A", "delay": "lag"}})
    times = [1, 4, 5, 7, 9]
    rows = [{"ID": 1, "TIME": t, "DV": 1.0, "AMT": ".", "EVID": 0, "ADDL": ".", "II": ".", "CMT": c}
            for t in times for c in (1, 2)]
    rows += [{"ID": 1, "TIME": 0, "DV": ".", "AMT": 100, "EVID": 1, "ADDL": 2, "II": 4, "CMT": "."},
             {"ID": 1, "TIME": 6, "DV": ".", "AMT": ".", "EVID": 3, "ADDL": ".", "II": ".", "CMT": "."}]
    df = pd.DataFrame(rows).sort_values("TIME", kind="stable")
    f = predict(m, df, [0.1, delay], output="CMT").reshape(len(times), 2)
    # the record time, not the time the delayed amount arrives
    assert f.tolist() == [[1, 0], [2, 4], [2, 4], [2, 4], [3, 8]]


def test_a_steady_state_is_refused_with_the_signals_of_the_amounts():
    m = Model(states={"A": "0"}, parameters=["k"], odes={"A": "-k*A/(1 + n_amounts)"},
              outputs=["A"], inputs={"dose": {"target": "A"}})
    with pytest.raises(ValueError, match="steady state needs a model that does not read"):
        Problem(m, Dataset.from_records(steady([1.0])))


@pytest.mark.parametrize("field", ["parameters", "regressors", "covariates", "states"])
def test_a_name_of_the_signals_of_the_amounts_cannot_be_declared(field):
    from pysaem.model import ModelError
    kw = dict(states={"A": "0"}, parameters=["k"], odes={"A": "-k*A"}, outputs=["A"])
    if field == "states":
        kw.update(states={"A": "0", "t_last": "0"}, odes={"A": "-k*A", "t_last": "0"})
    else:
        kw[field] = kw.get(field, []) + ["t_last"]
    with pytest.raises(ModelError, match="'t_last' is reserved: it is the time of the most recent"):
        Model(**kw)


def test_a_model_without_the_signals_has_the_same_timeline_and_signal_rows():
    data = Dataset.from_records(records([1, 5, 13, 25.0], DOSES))
    prob = Problem(ONE_CMT, data)
    assert not ONE_CMT.symbolic.history and prob._reg[2].shape == (0, 0)
    from pysaem.solve import _INPUT_HISTORY
    assert not prob._kernel[7] & _INPUT_HISTORY


def test_a_fractional_power_of_a_state_decayed_to_zero_stays_finite():
    # Cc decays to 1e-33 and is returned at about -1e-17, within its tolerance: read as 0, not nan
    m = Model(states={"Ad": "0", "Ac": "0"}, aux={"Cc": "Ac/V", "E": "Cc**g"},
              odes={"Ad": "-ka*Ad", "Ac": "ka*Ad - Vm*Cc/(Km + Cc)"}, parameters=["ka", "V", "Vm", "Km", "g"],
              outputs=["E"], inputs={"dose": {"target": "Ad"}}, t0=0)
    times = [1.0, 2.0, 168.0]
    rows = [{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": 100.0}] + [{"ID": 1, "TIME": t, "DV": 1.0, "AMT": "."} for t in times]
    pred, status = Problem(m, Dataset.from_records(pd.DataFrame(rows))).predict(np.array([[1.0, 10.0, 10.0, 2.0, 2.5]]))
    assert status.ravel().tolist() == [0]
    ref = solve_ivp(lambda t, y: [-y[0], y[0] - 10.0 * (y[1] / 10) / (2.0 + y[1] / 10)], (0, 168), [100.0, 0.0],
                    method="Radau", rtol=1e-12, atol=1e-30, t_eval=times).y[1] / 10
    np.testing.assert_allclose(pred[0, :2, 0], ref[:2] ** 2.5, rtol=1e-5)
    assert pred[0, 2, 0] < 1e-20


def test_the_largest_value_of_a_state_is_followed_between_events():
    # one dose then a long segment: the central state rises to its peak and decays within it;
    # its tolerance follows the peak, so the same model solved with an observation at the peak agrees
    m = Model(states={"Ad": "0", "Ac": "0"}, odes={"Ad": "-ka*Ad", "Ac": "ka*Ad - k*Ac"}, parameters=["ka", "k"],
              outputs=["Ac"], inputs={"dose": {"target": "Ad"}}, t0=0)
    psi = np.array([[1.0, 0.05]])
    for times in ([100.0], [1.5, 100.0]):
        rows = [{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": 100.0}] + [{"ID": 1, "TIME": t, "DV": 1.0, "AMT": "."} for t in times]
        pred, status = Problem(m, Dataset.from_records(pd.DataFrame(rows)), method="explicit").predict(psi)
        exact = 100 * 1.0 / (1.0 - 0.05) * (np.exp(-0.05 * 100.0) - np.exp(-100.0))
        assert status.ravel().tolist() == [0]
        np.testing.assert_allclose(pred[0, -1, 0], exact, rtol=1e-5)


@pytest.mark.slow
def test_equations_with_hundreds_of_conditions_compile():
    # Boger 2018: Numba's analysis of the control flow recurses once per block, past Python's
    # default limit; C = A at A = 0.5, where every condition is false
    aux = "A*" + "*".join(f"where(A > {i + 1}, 1.001, 0.999)" for i in range(500))
    m = Model(states={"A": "0"}, odes={"A": "-k*A"}, parameters=["k"], outputs=["C"], aux={"C": aux},
              inputs={"dose": "A"})
    out = predict(m, records([1.0], [(0.0, 0.5)]), [0.1])
    np.testing.assert_allclose(out[0], 0.5 * math.exp(-0.1) * 0.999**500, rtol=1e-5)


LAGGED = Model(states={"A": "0"}, parameters=["CL", "V", "lag"], odes={"A": "-CL/V*A"},
               aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A", "delay": "lag"}})


def lagged_series(t, lag, k, v, ii=12.0, dur=0.0, amt=100.0, n=400):
    """Doses at lag + j ii, j = -n..0, superposed: bolus, or infusion of length dur."""
    c = np.zeros_like(t)
    for j in range(-n, 1):
        s = t - (lag + j * ii)
        if dur == 0.0:
            c += np.where(s > 0, amt / v * np.exp(-k * np.maximum(s, 0)), 0.0)
        else:
            c += np.where(s > 0, amt / dur / (k * v) * (1 - np.exp(-k * np.clip(s, 0, dur)))
                          * np.exp(-k * np.maximum(s - dur, 0)), 0.0)
    return c


@pytest.mark.parametrize("method", ["auto", "explicit"])
@pytest.mark.parametrize("lag, rate", [(13.0, None), (12.0, None), (37.0, None), (11.0, 50.0),
                                       (5.0, 10.0), (1.5, None)])
def test_a_steady_state_series_goes_on_after_the_record_with_its_lag(method, lag, rate):
    # the inputs of the series at or after the record (lag >= II) and the rest of an infusion under
    # way at the record arrive; reference: the doses lag + j II superposed
    t = np.array([0.5, 1.5, 3, 6, 11.9, 13.5, 26, 40.0])
    f = predict(LAGGED, steady(t, rate), [2.0, 10.0, lag], method=method)
    exact = lagged_series(t, lag, 0.2, 10.0, dur=100 / rate if rate else 0.0)
    assert np.allclose(f, exact, rtol=TOL)


@pytest.mark.parametrize("method", ["auto", "explicit"])
def test_an_input_of_amount_zero_is_no_input_whatever_its_rate(method):
    # 100 at 20/h from 0; an amount of 0 at 5/h at 2 h, and one too short to outlast rounding,
    # leave it as it is; a reset row with a rate (EVID 3) gives nothing
    t = np.array([1.0, 3, 6, 12, 24])
    k = 0.2
    exact = 20 / 2.0 * (1 - np.exp(-k * np.minimum(t, 5))) * np.exp(-k * np.maximum(t - 5, 0))
    rows = [{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": 100.0, "RATE": 20.0, "EVID": 1},
            {"ID": 1, "TIME": 2.0, "DV": ".", "AMT": 0.0, "RATE": 5.0, "EVID": 1},
            {"ID": 1, "TIME": 2.5, "DV": ".", "AMT": 1e-30, "RATE": 1.0, "EVID": 1}]
    rows += [{"ID": 1, "TIME": x, "DV": 1.0, "AMT": ".", "RATE": ".", "EVID": 0} for x in t]
    df = pd.DataFrame(rows).sort_values("TIME", kind="stable")
    assert np.allclose(predict(ONE_CMT, df, [2.0, 10.0], method=method), exact, rtol=TOL)
    rows = [{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": 100.0, "RATE": ".", "EVID": 1},
            {"ID": 1, "TIME": 5.0, "DV": ".", "AMT": ".", "RATE": 40.0, "EVID": 3}]
    rows += [{"ID": 1, "TIME": x, "DV": 1.0, "AMT": ".", "RATE": ".", "EVID": 0} for x in (2.0, 6, 48)]
    df = pd.DataFrame(rows).sort_values("TIME", kind="stable")
    f = predict(ONE_CMT, df, [2.0, 10.0], method=method)
    assert f[0] == pytest.approx(10 * np.exp(-0.4), rel=TOL) and f[1] == f[2] == 0.0


PHASED = {
    "covariate": Model(states={"A": "0"}, parameters=["phi", "k"],
                       odes={"A": "where(mod(t - phi, 24) < 12, 1, 0) - k*A"}, outputs=["A"],
                       inputs={"dose": {"target": "A"}}),
    "history": Model(states={"A": "0"}, parameters=["phi", "k"],
                     odes={"A": "where(mod(t - t_last - phi, 24) < 12, 1, 0) - k*A"},
                     outputs=["A"], inputs={"dose": {"target": "A"}}),
}


@pytest.mark.parametrize("phi", [1e19, -1e25, 1e21])
@pytest.mark.parametrize("kind", sorted(PHASED))
def test_a_periodic_breakpoint_far_from_the_span_stays_within_its_slots(kind, phi, monkeypatch):
    import pysaem.solve as S
    t = np.array([1.0, 24, 48, 96, 168])
    rows = [{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": 1.0}]
    rows += [{"ID": 1, "TIME": x, "DV": 1.0, "AMT": "."} for x in t]
    prob = Problem(PHASED[kind], Dataset.from_records(pd.DataFrame(rows), covariates=()))
    psi = np.array([[[phi, 0.1]]])
    # t - phi rounds to -phi over the span: the input is constant
    u = float(np.mod(-phi, 24.0) < 12)
    exact = np.exp(-0.1 * t) + u / 0.1 * (1 - np.exp(-0.1 * t))
    # the kernel run as Python checks every index of the timeline
    pred, st = prob.empty(1), np.empty((1, 1), np.int64)
    imp, loc = st.copy(), np.empty((1, 1, 2), np.int64)
    with monkeypatch.context() as mp:
        mp.setattr(S, "_solve_one", S._solve_one.py_func)
        S._predict.py_func(*prob.kernel_args(1), pred, st, imp, loc, psi)
    assert st.ravel().tolist() == [0]
    assert np.allclose(prob.gather(pred)[0], exact, rtol=TOL)
    pred, st = prob.predict(psi)
    assert st.ravel().tolist() == [0]
    assert np.allclose(prob.gather(pred)[0], exact, rtol=TOL)


def test_a_periodic_breakpoint_reduced_modulo_its_period_keeps_its_times():
    # phase 24e6 + 5: switches at 5 + 12 j; reference: the exact piecewise solution
    m = PHASED["covariate"]
    t = np.array([1.0, 6, 17, 20, 30, 41, 50])
    df = pd.DataFrame([{"ID": 1, "TIME": x, "DV": 1.0} for x in [0.0, *t]])
    f = predict(m, df, [24e6 + 5, 0.3])
    k, a, now, exact = 0.3, 0.0, 0.0, []
    for x in t:
        for s in [*(b for b in np.arange(5.0, x, 12.0) if b > now), x]:
            u = float((0.5 * (now + s) - 5) % 24 < 12)
            a = u / k + (a - u / k) * np.exp(-k * (s - now))
            now = s
        exact.append(a)
    assert np.allclose(f[1:], exact, rtol=TOL)


def test_the_closed_form_keeps_a_constant_term_that_only_a_later_occasion_has():
    m = Model(states={"A": "0"}, parameters=["kin", "k"], odes={"A": "kin - k*A"}, outputs=["A"])
    t = np.array([0.0, 1, 4, 6, 8, 12])
    df = pd.DataFrame([{"ID": 1, "TIME": x, "DV": 1.0, "OCC": 1 if x < 5 else 2} for x in t])
    prob = Problem(m, Dataset.from_records(df, covariates=()))
    pred, st = prob.predict(np.array([[[0.0, 0.3], [2.0, 0.3]]]))
    assert st.ravel().tolist() == [0] and prob.last_stats["exact"] == 1
    exact = np.where(t < 6, 0.0, 2.0 / 0.3 * (1 - np.exp(-0.3 * (t - 6))))
    assert np.allclose(prob.gather(pred)[0], exact, rtol=1e-12, atol=1e-14)


def test_a_steady_state_is_refused_with_an_interpolated_regressor():
    m = Model(states={"A": "0"}, parameters=["k", "kin"], regressors=["W"], interpolated=["W"],
              odes={"A": "kin*W/70 - k*A"}, outputs=["A"], inputs={"dose": {"target": "A"}})
    df = steady([0.5, 2.0]).assign(W=70.0)
    with pytest.raises(ValueError, match=r"interpolated regressor \['W'\]: their signal before"):
        Problem(m, Dataset.from_records(df, regressors=["W"]))
    # the same data without steady state is solved
    df["SS"], df["II"] = 0, "."
    assert predict(m, df, [0.2, 1.0], regressors=["W"])[0] == pytest.approx(
        100 * np.exp(-0.1) + 5 * (1 - np.exp(-0.1)), rel=TOL)


# regressor rows and the timeline (review of 89ae536)

LINE = Model(states={"X": "X0"}, parameters=["X0", "k"], regressors=["C"], interpolated=["C"],
             odes={"X": "-k*X"}, aux={"Cout": "C"}, outputs=["Cout"], inputs={"dose": {"target": "X"}})


@pytest.mark.parametrize("method", ["auto", "explicit", "implicit"])
def test_a_reset_leaves_an_interpolated_regressor_on_its_line(method):
    rec_t, rec_c = np.array([0.0, 4.0, 8.0]), np.array([0.0, 4.0, 0.0])
    obs_t = np.array([1.0, 2.0, 2.9, 3.5, 5.0, 7.0])
    data = Dataset.from_tables(pd.DataFrame({"id": 1, "time": obs_t, "value": 1.0}),
                               events=pd.DataFrame({"id": 1, "time": [0.0, 2.5], "amount": [1.0, 0.0],
                                                    "reset": [0, 1]}),
                               regressors=pd.DataFrame({"id": 1, "time": rec_t, "C": rec_c}))
    prob = Problem(LINE, data, method=method)
    pred, status = prob.predict(np.array([[[1.0, 0.1]]]))
    assert status.ravel().tolist() == [0]
    # until 1 November 2026 the reset put C back to 0, its value at the record at 0
    np.testing.assert_allclose(prob.gather(pred)[0], np.interp(obs_t, rec_t, rec_c), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", ["auto", "explicit"])
def test_regressor_records_before_the_start_give_the_value_in_force_there(method):
    obs_t = np.array([0.0, 1.0, 3.0, 5.0])
    line = Model(states={"X": "0"}, parameters=["k"], regressors=["C"], interpolated=["C"],
                 odes={"X": "-k*X"}, aux={"Cout": "C"}, outputs=["Cout"])
    data = Dataset.from_tables(pd.DataFrame({"id": 1, "time": obs_t, "value": 1.0}),
                               regressors=pd.DataFrame({"id": 1, "time": [-2.0, 4.0], "C": [0.0, 8.0]}))
    prob = Problem(line, data, method=method)
    pred, status = prob.predict(np.array([[0.1]]))
    assert status.ravel().tolist() == [0]
    np.testing.assert_allclose(prob.gather(pred)[0], np.interp(obs_t, [-2.0, 4.0], [0.0, 8.0]), rtol=1e-12)
    # a model t0 after the first record
    late = Model(**(line.to_dict() | {"t0": 1.0}))
    obs = np.array([1.0, 2.0, 5.0])
    data = Dataset.from_tables(pd.DataFrame({"id": 1, "time": obs, "value": 1.0}),
                               regressors=pd.DataFrame({"id": 1, "time": [0.0, 4.0], "C": [0.0, 8.0]}))
    prob = Problem(late, data, method=method)
    pred, _ = prob.predict(np.array([[0.1]]))
    np.testing.assert_allclose(prob.gather(pred)[0], np.interp(obs, [0.0, 4.0], [0.0, 8.0]), rtol=1e-12)
    # an initial value read from a piecewise-constant regressor: the value in force at the start
    m = Model(states={"X": "C"}, parameters=["k"], regressors=["C"], odes={"X": "-k*X"}, outputs=["X"])
    rows = [{"ID": 1, "TIME": -2.0, "DV": ".", "EVID": 2, "C": 1.0},
            {"ID": 1, "TIME": -1.0, "DV": ".", "EVID": 2, "C": 5.0}]
    rows += [{"ID": 1, "TIME": x, "DV": 1.0, "EVID": 0, "C": 5.0} for x in obs_t]
    f = predict(m, pd.DataFrame(rows), [0.1], regressors=["C"], method=method)
    np.testing.assert_allclose(f, 5.0 * np.exp(-0.1 * obs_t), rtol=TOL)


def test_a_subject_without_regressor_record_is_refused_with_its_name():
    from pysaem.data import DataError
    m = Model(states={"A": "W"}, parameters=["k"], regressors=["W"], odes={"A": "-k*A"}, outputs=["A"])
    obs = pd.DataFrame({"id": [1, 1, 2, 2], "time": [1.0, 2.0, 1.0, 2.0], "value": 1.0})
    # first and last subject: until 1 November 2026 the next subject's W, or memory past the array
    for regs in (pd.DataFrame({"id": [2], "time": [0.0], "W": [123.0]}),
                 pd.DataFrame({"id": [1], "time": [0.0], "W": [7.0]})):
        absent = 3 - regs["id"][0]
        with pytest.raises(DataError, match=rf"subject\(s\) \[{absent}\] have no record"):
            Problem(m, Dataset.from_tables(obs, regressors=regs))
    both = pd.DataFrame({"id": [1, 2], "time": [0.0, 0.0], "W": [7.0, 123.0]})
    prob = Problem(m, Dataset.from_tables(obs, regressors=both))
    pred, status = prob.predict(np.array([[0.1], [0.1]]))
    assert status.ravel().tolist() == [0, 0]
    # the solve starts at the first observation, 1
    np.testing.assert_allclose(prob.gather(pred)[0], np.repeat([7.0, 123.0], 2) * np.exp(-0.1 * np.tile([0.0, 1.0], 2)),
                               rtol=TOL)


def test_the_timeline_sort_is_stable_and_n_log_n():
    import time

    from pysaem.solve import _sort_timeline
    rng = np.random.default_rng(5)
    for _ in range(300):
        m = int(rng.integers(0, 200))
        at = rng.integers(0, max(1, m // 5), m + 3).astype(float)
        ak, ai, av = rng.integers(0, 8, m + 3), np.arange(m + 3), rng.standard_normal(m + 3)
        order = np.concatenate([np.argsort(at[:m], kind="stable"), np.arange(m, m + 3)])
        ref = [x[order] for x in (at, ak, ai, av)]
        _sort_timeline(at, ak, ai, av, m)
        assert all(np.array_equal(x, y) for x, y in zip((at, ak, ai, av), ref))

    def cost(n):
        # observations filled before the inputs, the order of _solve_one
        at = np.concatenate([12.0 * np.arange(n) + 6, 12.0 * np.arange(n)])
        z = np.zeros(2 * n, np.int64)
        best = np.inf
        for _ in range(3):
            a = at.copy()
            t0 = time.perf_counter()
            _sort_timeline(a, z.copy(), z.copy(), np.zeros(2 * n), 2 * n)
            best = min(best, time.perf_counter() - t0)
        return best
    cost(10)
    # ten times the entries: 12 times the time for n log n, 100 for the insertion sort used
    # until 1 November 2026
    assert cost(40_000) < 30 * cost(4_000)


def test_an_amount_recorded_before_t0_is_refused():
    from pysaem.data import DataError
    m = Model(**(ONE_CMT.to_dict() | {"t0": 0.0}))
    rows = [{"ID": 1, "TIME": -2.0, "DV": ".", "AMT": 100.0, "EVID": 1}] + \
           [{"ID": 1, "TIME": x, "DV": 1.0, "AMT": ".", "EVID": 0} for x in (1.0, 4.0)]
    # until 1 November 2026 the amount was given whole at t0
    with pytest.raises(DataError, match="subject 1: an amount recorded at -2, before the start"):
        Problem(m, Dataset.from_records(pd.DataFrame(rows), covariates=()))
    # a reset without amount before t0 changes nothing and is kept; without t0 the solve starts at -2
    rows[0] |= {"AMT": ".", "EVID": 3}
    np.testing.assert_allclose(predict(m, pd.DataFrame(rows), [2.0, 10.0]), [0.0, 0.0])
    rows[0] |= {"AMT": 100.0, "EVID": 1}
    np.testing.assert_allclose(predict(ONE_CMT, pd.DataFrame(rows), [2.0, 10.0]),
                               10 * np.exp(-0.2 * np.array([3.0, 6.0])), rtol=TOL)
