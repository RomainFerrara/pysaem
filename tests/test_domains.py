"""Models outside pharmacokinetics: epidemic, predator-prey, tumour and cell-free DNA.

Each test simulates a population with SciPy from known values, independently of
the pysaem solver, and checks that the fit recovers them.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

import pysaem
from pysaem import Dataset, ErrorModel, Model, Population, Settings

SETTINGS = Settings(n_exploratory=400, n_smoothing=200, exploratory_autostop=False,
                    smoothing_autostop=False, seed=3)


def lognormal(rng, value, omega):
    return value * np.exp(omega * rng.standard_normal())


def check(result, truth, tolerance):
    for name, value in truth.items():
        assert result.parameters[name] == pytest.approx(value, rel=tolerance), name


@pytest.mark.slow
def test_epidemic_in_forty_communities():
    model = Model(states={"S": "1 - I0", "I": "I0"}, parameters=["beta", "gamma", "I0"],
                  odes={"S": "-beta*S*I", "I": "beta*S*I - gamma*I"}, outputs=["I"], t0=0)
    rng = np.random.default_rng(0)
    times = np.arange(2.0, 61.0, 4.0)
    rows = []
    for i in range(40):
        beta, i0 = lognormal(rng, 0.4, 0.2), lognormal(rng, 0.002, 0.3)
        sol = solve_ivp(lambda t, y: [-beta * y[0] * y[1], beta * y[0] * y[1] - 0.15 * y[1]],
                        (0, 60), [1 - i0, i0], t_eval=times, rtol=1e-10, atol=1e-14)
        y = sol.y[1] * (1 + 0.1 * rng.standard_normal(times.size))
        rows += [{"ID": i, "TIME": t, "DV": v} for t, v in zip(times, y)]
    pop = Population.build(model, init={"beta": 0.2, "gamma": 0.3, "I0": 0.01}, omega=0.5,
                           no_iiv=["gamma"], error="proportional")
    res = pysaem.fit(model, Dataset.from_records(pd.DataFrame(rows)), pop, SETTINGS)
    check(res, {"beta_pop": 0.4, "gamma_pop": 0.15, "b": 0.1}, 0.08)
    check(res, {"I0_pop": 0.002, "omega_beta": 0.2, "omega_I0": 0.3}, 0.35)


@pytest.mark.slow
def test_predator_and_prey_observed_together():
    model = Model(states={"x": "x0", "y": "y0"}, parameters=["a", "b", "c", "d", "x0", "y0"],
                  odes={"x": "a*x - b*x*y", "y": "-c*y + d*x*y"}, outputs=["x", "y"], t0=0)
    rng = np.random.default_rng(1)
    times = np.arange(0.5, 20.1, 1.5)
    rows = []
    for i in range(30):
        a, x0 = lognormal(rng, 1.0, 0.1), lognormal(rng, 10.0, 0.2)
        sol = solve_ivp(lambda t, z: [a * z[0] - 0.1 * z[0] * z[1], -1.5 * z[1] + 0.075 * z[0] * z[1]],
                        (0, 20), [x0, 5.0], t_eval=times, rtol=1e-10, atol=1e-12)
        for k, t in enumerate(times):
            rows.append({"ID": i, "TIME": t, "DV": sol.y[0, k] * (1 + 0.08 * rng.standard_normal()),
                         "KIND": "prey"})
            rows.append({"ID": i, "TIME": t, "DV": sol.y[1, k] + 0.3 * rng.standard_normal(),
                         "KIND": "predator"})
    data = Dataset.from_records(pd.DataFrame(rows), output="KIND",
                                output_labels=["prey", "predator"])
    pop = Population.build(model, init={"a": 0.8, "b": 0.15, "c": 1.2, "d": 0.1, "x0": 8, "y0": 4},
                           omega=0.3, no_iiv=["b", "c", "d", "y0"],
                           error={"x": ErrorModel("proportional", b=0.2),
                                  "y": ErrorModel("constant", a=1.0)})
    res = pysaem.fit(model, data, pop, SETTINGS)
    check(res, {"a_pop": 1.0, "b_pop": 0.1, "c_pop": 1.5, "d_pop": 0.075, "x0_pop": 10.0,
                "y0_pop": 5.0}, 0.06)
    check(res, {"b_x": 0.08, "a_y": 0.3, "omega_x0": 0.2}, 0.3)


def cfdna_model():
    """Joint tumour size and cell-free DNA kinetics (structure of the cfDNA-TK model)."""
    return Model(
        states={"Ts": "Ts0", "Tr": "Tr0", "Ds": "Ds0", "Dl": "Dl0"},
        parameters=["Ts0", "Tr0", "Ds0", "Dl0", "beta", "alpha", "lambda_s", "lambda_l", "kDs",
                    "kDl"],
        aux={"T": "Tr + Ts", "release": "beta*Ts + alpha*T"},
        odes={"Ts": "where(t <= 0, alpha*Ts, (alpha - beta)*Ts)", "Tr": "alpha*Tr",
              "Ds": "lambda_s*release - kDs*Ds", "Dl": "lambda_l*release - kDl*Dl"},
        outputs=["T", "Ds", "Dl"])


CFDNA_TRUTH = {"Ts0": 40.0, "Tr0": 3.0, "Ds0": 8.0, "Dl0": 3.0, "beta": 0.05, "alpha": 0.012,
               "lambda_s": 0.6, "lambda_l": 0.25, "kDs": 0.12, "kDl": 0.15}
CFDNA_IIV = {"Ts0": 0.4, "Tr0": 0.5, "beta": 0.3, "lambda_s": 0.3}


def cfdna_data(n=80, seed=5):
    """Simulated cohort: tumour size, short and long fragments, three error models."""
    rng = np.random.default_rng(seed)
    names = list(CFDNA_TRUTH)
    t_size = np.array([-28.0, 0, 21, 42, 63, 84, 126])
    t_dna = np.array([-28.0, 0, 7, 21, 42, 63, 84, 126])
    fns = cfdna_model().python_functions()
    rows = []
    for i in range(n):
        p = np.array([lognormal(rng, CFDNA_TRUTH[k], CFDNA_IIV.get(k, 0.0)) for k in names])

        def rhs(t, y):
            dy = np.zeros(4)
            fns["rhs"](t, y, p, np.zeros(0), np.zeros(4), dy)
            return dy
        grid = np.union1d(t_size, t_dna)
        y0 = np.array([p[0], p[1], p[2], p[3]])
        before = solve_ivp(rhs, (-28, 0), y0, t_eval=grid[grid <= 0], rtol=1e-10, atol=1e-12)
        after = solve_ivp(rhs, (1e-12, 126), before.y[:, -1], t_eval=grid[grid > 0], rtol=1e-10,
                          atol=1e-12)
        y = np.hstack([before.y, after.y])
        size, short, long_ = y[0] + y[1], y[2], y[3]
        for t in t_size:
            k = np.searchsorted(grid, t)
            rows.append({"ID": i, "TIME": t, "DV": size[k] * (1 + 0.1 * rng.standard_normal()),
                         "YTYPE": 1})
        for t in t_dna:
            k = np.searchsorted(grid, t)
            rows.append({"ID": i, "TIME": t, "DV": short[k] * (1 + 0.15 * rng.standard_normal()),
                         "YTYPE": 2})
            rows.append({"ID": i, "TIME": t, "YTYPE": 3,
                         "DV": long_[k] + (0.3 + 0.1 * long_[k]) * rng.standard_normal()})
    return pd.DataFrame(rows)


def cfdna_population(model):
    init = {k: v * (1.6 if i % 2 else 0.6) for i, (k, v) in enumerate(CFDNA_TRUTH.items())}
    return Population.build(
        model, init=init, omega=0.5, no_iiv=[k for k in CFDNA_TRUTH if k not in CFDNA_IIV],
        error={"T": ErrorModel("proportional", b=0.2), "Ds": ErrorModel("proportional", b=0.2),
               "Dl": ErrorModel("combined1", a=1.0, b=0.2)})


@pytest.mark.slow
def test_tumour_and_cell_free_dna_with_three_outputs():
    model = cfdna_model()
    data = Dataset.from_records(cfdna_data(), output="YTYPE")
    assert data.n_obs_by_output.tolist() == [560, 640, 640]
    res = pysaem.fit(model, data, cfdna_population(model), SETTINGS)
    check(res, {f"{k}_pop": v for k, v in CFDNA_TRUTH.items() if k in ("Ts0", "beta", "alpha")},
          0.12)
    check(res, {"b_T": 0.1, "b_Ds": 0.15}, 0.15)
    ratio_s = res.parameters["lambda_s_pop"] / res.parameters["kDs_pop"]
    assert ratio_s == pytest.approx(0.6 / 0.12, rel=0.2)
    assert set(res.tests) == {"1", "2", "3"} and len(res.vpc["output"].unique()) == 3
    for label in ("1", "2", "3"):
        iw = res.predictions.loc[res.predictions["output"] == label, "iwres_mode"]
        assert abs(iw.mean()) < 0.2 and 0.75 < iw.std() < 1.25
