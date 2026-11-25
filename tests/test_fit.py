"""Estimation against models whose likelihood is known."""

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

import exact
import pysaem
from pysaem import Dataset, FitError, Model, Population, Settings

INTERCEPT = Model(states={}, odes={}, parameters=["m"], aux={"y": "m"}, outputs=["y"])
ONE_CMT = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"},
                aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A"}})
FAST = Settings(n_exploratory=300, n_smoothing=150, exploratory_autostop=False,
                smoothing_autostop=False, seed=1)


def hessian(f, x, h=1e-4):
    n = len(x)
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            def at(si, sj):
                z = np.array(x, float)
                z[i] += si * h * abs(x[i])
                z[j] += sj * h * abs(x[j])
                return f(z)
            out[i, j] = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / (
                4 * h * abs(x[i]) * h * abs(x[j])) if i != j else (
                at(1, 0) - 2 * f(np.array(x, float)) + at(-1, 0)) / (h * abs(x[i])) ** 2
    return out


@pytest.fixture(scope="module")
def intercept_fit():
    df, y = exact.random_intercept()
    pop = Population.build(INTERCEPT, init={"m": 5.0}, distribution="normal", omega=1.0,
                           error="constant")
    return pysaem.fit(INTERCEPT, Dataset.from_records(df), pop, FAST), y


def test_random_intercept_reaches_the_closed_form_mle(intercept_fit):
    res, y = intercept_fit
    mu, omega, a = exact.random_intercept_mle(y)
    p = res.parameters
    assert res.converged
    assert p["m_pop"] == pytest.approx(mu, abs=0.01 * omega)
    assert p["omega_m"] == pytest.approx(omega, rel=0.02)
    assert p["a"] == pytest.approx(a, rel=0.02)


def test_importance_sampling_matches_the_exact_likelihood(intercept_fit):
    res, y = intercept_fit
    p = res.parameters
    ll = exact.random_intercept_loglik(p["m_pop"], p["omega_m"], p["a"], y)
    assert res.ll == pytest.approx(ll, abs=0.3)
    assert res.ofv == pytest.approx(-2 * res.ll)
    n_tot = y.size
    assert res.aic == pytest.approx(-2 * res.ll + 2 * 3)
    assert res.bic == pytest.approx(-2 * res.ll + 3 * np.log(y.shape[0]))
    assert res.bicc == pytest.approx(-2 * res.ll + 2 * np.log(y.shape[0]) + np.log(n_tot))


def test_exponential_error_is_the_exact_likelihood_of_the_data():
    # log y is a random intercept: the density of y is that of log y divided by y
    df, z = exact.random_intercept(mu=1.0, omega=0.4, a=0.2)
    df["DV"] = np.exp(df["DV"])
    model = Model(states={}, odes={}, parameters=["m"], aux={"y": "exp(m)"}, outputs=["y"])
    pop = Population.build(model, init={"m": 0.5}, distribution="normal", omega=1.0,
                           error="exponential")
    res = pysaem.fit(model, Dataset.from_records(df), pop, FAST)
    mu, omega, a = exact.random_intercept_mle(z)
    p = res.parameters
    assert p["m_pop"] == pytest.approx(mu, abs=0.01 * omega)
    assert p["omega_m"] == pytest.approx(omega, rel=0.02) and p["a"] == pytest.approx(a, rel=0.02)
    ll = exact.random_intercept_loglik(p["m_pop"], p["omega_m"], p["a"], z) - z.sum()
    assert res.ll == pytest.approx(ll, abs=0.3)
    assert np.allclose(res.predictions["iwres_mode"], (z.ravel() - np.log(res.predictions["ind_pred_mode"]))
                       / p["a"])


def test_inter_occasion_variability_reaches_the_exact_maximum():
    from scipy.optimize import minimize
    df, y = exact.occasions()
    pop = Population.build(INTERCEPT, init={"m": 5.0}, distribution="normal", omega=1.0,
                           error="constant", iov={"m": 0.5})
    res = pysaem.fit(INTERCEPT, Dataset.from_records(df), pop, FAST)
    best = minimize(lambda x: -exact.occasions_loglik([x[0], *np.exp(x[1:])], y, 3), [9.0, 0, 0, 0],
                    method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    mu, omega, gamma, a = best.x[0], *np.exp(best.x[1:])
    p = res.parameters
    assert set(p) == {"m_pop", "omega_m", "gamma_m", "a"}
    assert p["m_pop"] == pytest.approx(mu, abs=0.05 * omega)
    for name, ref in (("omega_m", omega), ("gamma_m", gamma), ("a", a)):
        assert p[name] == pytest.approx(ref, rel=0.03), name
    theta = [p["m_pop"], p["omega_m"], p["gamma_m"], p["a"]]
    assert res.ll == pytest.approx(exact.occasions_loglik(theta, y, 3), abs=0.3)
    info = -hessian(lambda z: exact.occasions_loglik(z, y, 3), theta)
    for name, ref in zip(("m_pop", "omega_m", "gamma_m", "a"), np.sqrt(np.diag(np.linalg.inv(info)))):
        assert res.se[name] == pytest.approx(ref, rel=0.15), name
    assert res.population["parameters"]["m"]["gamma"] == pytest.approx(p["gamma_m"])


def test_standard_errors_match_the_exact_information(intercept_fit):
    res, y = intercept_fit
    p = res.parameters
    x = [p["m_pop"], p["omega_m"], p["a"]]
    info = -hessian(lambda z: exact.random_intercept_loglik(z[0], z[1], z[2], y), x)
    se = np.sqrt(np.diag(np.linalg.inv(info)))
    for name, ref in zip(("m_pop", "omega_m", "a"), se):
        assert res.se[name] == pytest.approx(ref, rel=0.15)
        assert res.rse[name] == pytest.approx(100 * ref / abs(p[name]), rel=0.15)


def test_individual_estimates_and_shrinkage_of_the_random_intercept(intercept_fit):
    res, y = intercept_fit
    p = res.parameters
    n = y.shape[1]
    w = p["omega_m"] ** 2 / (p["omega_m"] ** 2 + p["a"] ** 2 / n)
    blup = p["m_pop"] + w * (y.mean(1) - p["m_pop"])
    ind = res.individual
    assert np.allclose(ind["m_mode"], blup, atol=2e-3)
    # the conditional mean is a Monte Carlo estimate: its error is judged against the
    # posterior sd (measured: median 0.04 sd, largest of 80 subjects 0.27 sd)
    error = np.abs(ind["m_mean"] - blup) / (np.sqrt(w) * p["a"] / np.sqrt(n))
    assert np.median(error) < 0.1 and error.max() < 0.35
    shrink = 1 - np.std(blup - p["m_pop"], ddof=1) / p["omega_m"]
    assert res.shrinkage["m"]["mode"] == pytest.approx(shrink, abs=0.01)


@pytest.fixture(scope="module")
def bolus_fit():
    df, y = exact.simulate()
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5,
                           error="proportional")
    return pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, FAST), y


def test_bolus_model_reaches_the_quadrature_mle(bolus_fit):
    res, y = bolus_fit
    p = res.parameters
    start = [p["CL_pop"], p["V_pop"], p["omega_CL"], p["omega_V"], p["b"]]
    best, ll_best = exact.mle(start, y)
    assert res.converged
    for name, ref, tol in zip(("CL_pop", "V_pop", "omega_CL", "omega_V", "b"), best,
                              (0.01, 0.01, 0.05, 0.05, 0.03)):
        assert p[name] == pytest.approx(ref, rel=tol), name
    assert res.ll == pytest.approx(exact.loglik(start, y), abs=0.5)
    assert res.ll <= ll_best + 0.5


def test_bolus_standard_errors_match_the_quadrature_information(bolus_fit):
    res, y = bolus_fit
    names = ("CL_pop", "V_pop", "omega_CL", "omega_V", "b")
    x = [res.parameters[k] for k in names]
    c = exact.centers(x, y)
    info = -hessian(lambda z: exact.loglik(z, y, centers=c), x, h=2e-3)
    se = np.sqrt(np.diag(np.linalg.inv(info)))
    for name, ref in zip(names, se):
        assert res.se[name] == pytest.approx(ref, rel=0.2), name


def test_parameter_without_variability_is_estimated():
    df, y = exact.simulate(iiv_v=False, seed=11)
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, no_iiv=["V"],
                           error="proportional")
    res = pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, FAST)
    p = res.parameters
    assert "omega_V" not in p
    start = [p["CL_pop"], p["V_pop"], p["omega_CL"], p["b"]]
    best, _ = exact.mle(start, y, iiv=(0,))
    for name, ref, tol in zip(("CL_pop", "V_pop", "omega_CL", "b"), best,
                              (0.01, 0.01, 0.05, 0.03)):
        assert p[name] == pytest.approx(ref, rel=tol), name
    assert res.ll == pytest.approx(exact.loglik(start, y, iiv=(0,)), abs=0.5)
    assert np.isfinite(res.se["V_pop"]) and res.se["V_pop"] > 0


def test_fixed_population_is_evaluated_not_estimated(bolus_fit):
    first, y = bolus_fit
    df, _ = exact.simulate()
    pop = Population.build(ONE_CMT, init={"CL": first.parameters["CL_pop"],
                                          "V": first.parameters["V_pop"]},
                           omega={"CL": first.parameters["omega_CL"],
                                  "V": first.parameters["omega_V"]}, error="proportional")
    for p in pop.parameters.values():
        p.fixed = p.omega_fixed = True
    pop.error["C"].b = first.parameters["b"]
    pop.error["C"].fixed = True
    res = pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, FAST)
    assert res.parameters == pytest.approx(first.parameters)
    assert res.ll == pytest.approx(first.ll, abs=0.5)
    assert all(np.isnan(v) for v in res.se.values())


def test_same_seed_gives_the_same_fit(bolus_fit):
    first, _ = bolus_fit
    df, _ = exact.simulate()
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5,
                           error="proportional")
    again = pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, FAST)
    assert again.parameters == first.parameters and again.ll == first.ll


def test_model_that_cannot_be_evaluated_fails_loudly():
    df, _ = exact.simulate(n=10)
    bad = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"},
                aux={"C": "log(-V)*A"}, outputs=["C"], inputs={"dose": {"target": "A"}})
    pop = Population.build(bad, error="proportional")
    with pytest.raises(FitError, match="initial"):
        pysaem.fit(bad, Dataset.from_records(df), pop, FAST)


def test_progress_is_reported_and_can_cancel():
    df, _ = exact.simulate(n=10)
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, error="proportional")
    seen = []

    def on_progress(event):
        seen.append(event)
        return len(seen) < 20
    with pytest.raises(FitError, match="cancel"):
        pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, FAST, on_progress=on_progress)
    assert seen[0]["phase"] in ("burn-in", "exploratory") and "parameters" in seen[-1]


def test_combined_error_estimates_and_standard_errors_are_exact():
    df, y = exact.simulate(a=0.15, b=0.1, seed=5, n=80)
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5,
                           error=pysaem.ErrorModel("combined1", a=1.0, b=0.3))
    res = pysaem.fit(ONE_CMT, Dataset.from_records(df), pop,
                     Settings(n_exploratory=400, n_smoothing=200, exploratory_autostop=False,
                              smoothing_autostop=False, seed=1))
    names = ("CL_pop", "V_pop", "omega_CL", "omega_V", "a", "b")
    x = [res.parameters[k] for k in names]
    best, _ = exact.mle(x, y)
    c = exact.centers(x, y)
    se = np.sqrt(np.diag(np.linalg.inv(-hessian(lambda z: exact.loglik(z, y, centers=c), x,
                                                h=2e-3))))
    for name, ref, ref_se, tol in zip(names, best, se, (0.01, 0.01, 0.05, 0.05, 0.05, 0.05)):
        assert res.parameters[name] == pytest.approx(ref, rel=tol), name
        assert res.se[name] == pytest.approx(ref_se, rel=0.15), name


def test_result_survives_a_json_round_trip(bolus_fit):
    res, _ = bolus_fit
    again = pysaem.FitResult.from_json(res.to_json())
    assert again.parameters == res.parameters and again.ofv == res.ofv
    assert again.predictions.shape == res.predictions.shape
    assert list(again.individual.columns) == list(res.individual.columns)
    assert again.ci["CL_pop"] == pytest.approx(res.ci["CL_pop"])
    # how the subjects were solved: a linear model in closed form only
    assert again.solver == res.solver and res.solver["solves"] == res.solver["exact"] > 0


def test_covariate_effect_matches_least_squares_on_subject_means():
    rng = np.random.default_rng(8)
    n_subjects, n_obs = 100, 5
    c = rng.normal(size=n_subjects)
    y = (10 + 1.5 * c + 2.0 * rng.standard_normal(n_subjects))[:, None] + rng.standard_normal(
        (n_subjects, n_obs))
    df = pd.DataFrame({"ID": np.repeat(np.arange(n_subjects), n_obs),
                       "TIME": np.tile(np.arange(n_obs, dtype=float), n_subjects),
                       "DV": y.ravel(), "C": np.repeat(c, n_obs)})
    pop = Population.build(INTERCEPT, init={"m": 5.0}, distribution="normal", omega=1.0,
                           error="constant", covariates={"m": ["C"]})
    res = pysaem.fit(INTERCEPT, Dataset.from_records(df, covariates=["C"]), pop, FAST)
    slope, intercept = np.polyfit(c, y.mean(1), 1)
    resid = y.mean(1) - intercept - slope * c
    assert res.parameters["beta_m_C"] == pytest.approx(slope, abs=0.01)
    assert res.parameters["m_pop"] == pytest.approx(intercept, abs=0.01)
    assert res.se["beta_m_C"] == pytest.approx(resid.std() / (c.std() * n_subjects**0.5), rel=0.15)
    assert res.n_parameters == 4
    # BICc: m_pop, omega and the coefficient penalised by log(N), a by log(n_obs)
    assert res.bicc == pytest.approx(res.ofv + 3 * np.log(n_subjects) + np.log(n_subjects * n_obs))
    # a fixed parameter keeps its coefficients too
    pop.parameters["m"].init, pop.parameters["m"].covariates = 9.0, {"C": 1.0}
    pop.parameters["m"].fixed = True
    held = pysaem.fit(INTERCEPT, Dataset.from_records(df, covariates=["C"]), pop, FAST)
    assert held.parameters["beta_m_C"] == 1.0 and held.parameters["m_pop"] == 9.0
    assert held.n_parameters == 2


def test_left_censored_observations_reach_the_exact_mle():
    df, y = exact.random_intercept(n_subjects=100, n_obs=5, mu=10.0, omega=2.0, a=1.5, seed=12)
    lloq = 9.0
    cens = y < lloq
    df["CENS"] = cens.ravel().astype(int)
    df["DV"] = np.where(cens, lloq, y).ravel()
    assert 0.2 < cens.mean() < 0.5
    pop = Population.build(INTERCEPT, init={"m": 8.0}, distribution="normal", omega=1.0,
                           error="constant")
    res = pysaem.fit(INTERCEPT, Dataset.from_records(df), pop, FAST)
    p = res.parameters
    x = [p["m_pop"], p["omega_m"], p["a"]]
    best, ll_best = exact.censored_intercept_mle(x, np.where(cens, lloq, y), cens)
    for name, ref, tol in zip(("m_pop", "omega_m", "a"), best, (0.01, 0.04, 0.04)):
        assert p[name] == pytest.approx(ref, rel=tol), name
    assert res.ll == pytest.approx(exact.censored_intercept_loglik(x, np.where(cens, lloq, y), cens),
                                   abs=0.5)
    assert res.ll <= ll_best + 0.5


POWER = Model(states={"A": "0"}, parameters=["CL", "V", "power"], covariates=["WT"],
              aux={"cl": "CL * (WT/70)**power", "C": "A/V"}, odes={"A": "-cl/V*A"},
              outputs=["C"], inputs={"dose": {"target": "A"}})


def power_population():
    return Population.build(POWER, init={"CL": 1.0, "V": 5.0, "power": 0.0}, omega=0.5,
                            distribution={"power": "normal"}, no_iiv=["power"],
                            error="proportional")


@pytest.fixture(scope="module")
def power_case():
    """Data where CL goes as (WT/70)**0.75, and the fit of the linear form of that
    effect: log(WT/70) on log CL."""
    rng = np.random.default_rng(14)
    rows = []
    for i in range(80):
        wt = rng.uniform(40, 120)
        cl = 2.0 * (wt / 70) ** 0.75 * np.exp(0.2 * rng.standard_normal())
        v = 10.0 * np.exp(0.2 * rng.standard_normal())
        y = exact.conc(cl, v) * (1 + 0.1 * rng.standard_normal(exact.TIMES.size))
        rows.append({"ID": i, "TIME": 0.0, "DV": ".", "AMT": exact.DOSE, "WT": wt})
        rows += [{"ID": i, "TIME": t, "DV": yy, "AMT": ".", "WT": wt}
                 for t, yy in zip(exact.TIMES, y)]
    df = pd.DataFrame(rows).assign(LW=lambda d: np.log(d["WT"] / 70))
    linear = pysaem.fit(
        ONE_CMT, Dataset.from_records(df, covariates=["LW"]),
        Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional",
                         covariates={"CL": ["LW"]}), FAST)
    return Dataset.from_records(df, covariates=["WT"]), linear


def test_free_form_covariate_effect_matches_its_linear_form(power_case):
    # the two ways of writing the effect must give the same estimate, standard error
    # and likelihood, whatever the seed
    data, linear = power_case
    reference, se = linear.parameters["beta_CL_LW"], linear.se["beta_CL_LW"]
    assert reference == pytest.approx(0.75, abs=3 * se)
    for seed in (1, 2):
        free = pysaem.fit(POWER, data, power_population(), replace(FAST, seed=seed))
        assert free.fim_status == "ok"
        assert free.parameters["power_pop"] == pytest.approx(reference, abs=0.1 * se)
        assert free.se["power_pop"] == pytest.approx(se, rel=0.1)
        assert free.ll == pytest.approx(linear.ll, abs=0.5)
        for name in ("CL_pop", "V_pop", "omega_CL", "omega_V", "b"):
            assert free.parameters[name] == pytest.approx(linear.parameters[name], rel=0.03)


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 30.0])
def test_information_finds_the_step_of_a_structural_effect(power_case, monkeypatch, scale):
    # the scale of such an effect is unknown: a first step far too small or too large
    # must lead to the same standard error
    from pysaem import post
    data, linear = power_case
    steps = post._Marginal.steps

    def scaled(self):
        h = steps(self)
        h[self.solve_cols] *= scale
        return h

    monkeypatch.setattr(post._Marginal, "steps", scaled)
    free = pysaem.fit(POWER, data, power_population(), replace(FAST, simulations=False))
    assert free.fim_status == "ok"
    assert free.se["power_pop"] == pytest.approx(linear.se["beta_CL_LW"], rel=0.1)
    assert free.se["CL_pop"] == pytest.approx(linear.se["CL_pop"], rel=0.1)


def test_unresolved_direction_keeps_a_conservative_standard_error():
    from pysaem.post import _covariance
    info = np.array([[4.0, 1.9, 0.0], [1.9, 1.0, 0.0], [0.0, 0.0, 0.0]])
    full = np.sqrt(np.diag(np.linalg.inv(info[:2, :2])))
    # Monte Carlo error small against the smallest eigenvalue: the plain inverse
    cov, none, low = _covariance(info, 0.002 * np.eye(3))
    assert list(none) == [False, False, True] and not low.any()
    assert np.sqrt(np.diag(cov))[:2] == pytest.approx(full)
    # error larger than that eigenvalue: same standard errors, reported as imprecise
    cov, none, low = _covariance(info, np.array([[0.2, -0.1, 0], [-0.1, 0.05, 0], [0, 0, 0]]))
    assert list(low) == [True, True, False]
    assert np.sqrt(np.diag(cov))[:2] == pytest.approx(full)
    # a negative eigenvalue takes the floor set by the error: finite, imprecise
    info[0, 1] = info[1, 0] = 2.1
    cov, none, low = _covariance(info, np.array([[0.2, -0.1, 0], [-0.1, 0.05, 0], [0, 0, 0]]))
    assert low[:2].all() and np.isfinite(np.diag(cov)[:2]).all()


def test_error_parameter_at_zero_has_no_information_and_does_not_crash():
    # with b at 1e-17 its curvature is one rounding unit, which passed for information
    # and overflowed the confidence interval
    from pysaem import post, saem
    from pysaem.engine import Engine
    df, _ = exact.simulate(n=40, a=0.3, b=0.0, seed=3)
    pop = Population.build(ONE_CMT, init={"CL": 2.0, "V": 10.0}, omega=0.3,
                           error=pysaem.ErrorModel("combined1", a=0.3, b=0.1))
    eng = Engine(ONE_CMT, Dataset.from_records(df), pop)
    trace = saem.run(eng, FAST)
    eng.eb[0] = 1e-17
    cond = post.conditional(eng, FAST, trace)
    sample = post.loglik(eng, FAST, cond)[2]
    info, index, _ = post.information(eng, sample, post.modes(eng, cond.best))
    values = saem.estimates(eng)
    se, se_u, _, status = post.standard_errors(eng, info, index, list(values))
    assert status == "no information on b" and np.isnan(se["b"]) and np.isfinite(se["a"])
    assert np.isnan(post.intervals(eng, index, values, se_u)["b"]).all()


def test_screen_points_to_the_covariate_that_explains_a_random_effect():
    rng = np.random.default_rng(21)
    n_subjects, n_obs = 120, 5
    c = rng.uniform(-1, 1, n_subjects)
    site = rng.integers(0, 3, n_subjects)
    m = 10 + 1.5 * c + rng.standard_normal(n_subjects)
    y = m[:, None] + 0.5 * rng.standard_normal((n_subjects, n_obs))
    df = pd.DataFrame({"ID": np.repeat(np.arange(n_subjects), n_obs),
                       "TIME": np.tile(np.arange(n_obs, dtype=float), n_subjects), "DV": y.ravel(),
                       "C": np.repeat(c, n_obs), "NOISE": np.repeat(rng.standard_normal(n_subjects),
                                                                   n_obs),
                       "SITE": np.repeat(np.array(["x", "y", "z"])[site], n_obs)})
    data = Dataset.from_records(df, covariates=["C", "NOISE", "SITE"])

    def screen(covariates):
        pop = Population.build(INTERCEPT, init={"m": 5.0}, distribution="normal", omega=1.0,
                               error="constant", covariates=covariates)
        table = pysaem.fit(INTERCEPT, data, pop, FAST).covariate_screen
        return table.set_index("covariate")

    without = screen(None)
    assert without.index[0] == "C"
    assert without.loc["C", "association"] == pytest.approx(np.corrcoef(m, c)[0, 1], abs=0.05)
    assert without.loc["C", "p_value"] < 1e-8
    assert without.loc["NOISE", "p_value"] > 0.05 and without.loc["SITE", "p_value"] > 0.05
    assert without.loc["SITE", "measure"] == "correlation ratio"
    # once the effect is in the model, nothing is left for the covariate to explain
    assert screen({"m": ["C"]}).loc["C", "p_value"] > 0.05


def test_screen_compares_a_covariate_with_few_values_by_level():
    # the middle level stands apart: no trend for a correlation to see
    rng = np.random.default_rng(5)
    n_subjects, n_obs = 90, 5
    level = np.arange(n_subjects) % 3
    m = 10 + 1.2 * (level == 1) + rng.standard_normal(n_subjects)
    y = m[:, None] + 0.5 * rng.standard_normal((n_subjects, n_obs))
    df = pd.DataFrame({"ID": np.repeat(np.arange(n_subjects), n_obs),
                       "TIME": np.tile(np.arange(n_obs, dtype=float), n_subjects), "DV": y.ravel(),
                       "CODE": np.repeat(level + 1, n_obs),
                       "LABEL": np.repeat(np.array(["a", "b", "c"])[level], n_obs)})
    pop = Population.build(INTERCEPT, init={"m": 5.0}, distribution="normal", omega=1.0,
                           error="constant")
    table = pysaem.fit(INTERCEPT, Dataset.from_records(df, covariates=["CODE", "LABEL"]), pop,
                       FAST).covariate_screen.set_index("covariate")
    assert table.loc["CODE", "measure"] == table.loc["LABEL", "measure"] == "correlation ratio"
    assert table.loc["CODE", "p_value"] == pytest.approx(table.loc["LABEL", "p_value"])
    assert table.loc["CODE", "p_value"] < 1e-4
    groups = [m[level == k] for k in range(3)]
    between = sum(len(g) * (g.mean() - m.mean()) ** 2 for g in groups)
    assert table.loc["CODE", "association"] == pytest.approx(
        np.sqrt(between / ((m - m.mean()) ** 2).sum()), abs=0.08)


def test_fit_stopped_too_early_is_reported_as_still_moving(bolus_fit):
    assert bolus_fit[0].converged and not bolus_fit[0].messages
    df, _ = exact.simulate()
    pop = Population.build(ONE_CMT, init={"CL": 20.0, "V": 0.5}, omega=0.5, error="proportional")
    res = pysaem.fit(ONE_CMT, Dataset.from_records(df), pop,
                     replace(FAST, n_exploratory=15, n_smoothing=20, fim=False, simulations=False))
    assert not res.converged and "still moving" in res.messages[0]


def test_time_budget_stops_the_fit_with_the_current_estimates():
    df, _ = exact.simulate(n=20)
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, error="proportional")
    with pytest.raises(FitError, match="time budget.*CL_pop"):
        pysaem.fit(ONE_CMT, Dataset.from_records(df), pop,
                   Settings(n_exploratory=100000, exploratory_autostop=False, max_seconds=0.5))


def test_time_budget_can_leave_a_provisional_fit_whose_likelihood_bounds_the_maximum(bolus_fit):
    full, _ = bolus_fit
    df, _ = exact.simulate()
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional")
    res = pysaem.fit(ONE_CMT, Dataset.from_records(df), pop,
                     replace(FAST, n_exploratory=100000, exploratory_autostop=False, max_seconds=0.5,
                             on_budget="estimate"))
    assert res.provisional and not res.converged and not full.provisional
    assert any("ran out" in m and "provisional" in m for m in res.messages)
    assert not any("still moving" in m for m in res.messages)
    assert all(np.isnan(v) for v in res.se.values()) and np.isfinite(res.ofv)
    # the -2LL at any point is above its value at the maximum, up to Monte Carlo error
    assert res.ofv >= full.ofv - 4 * (res.ll_se + full.ll_se)
    assert res.n_parameters == full.n_parameters


def test_correlated_random_effects_reach_the_quadrature_mle():
    df, y = exact.simulate(n=100, rho=0.6, seed=9)
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional",
                           correlations=[["CL", "V"]])
    res = pysaem.fit(ONE_CMT, Dataset.from_records(df), pop,
                     Settings(n_exploratory=400, n_smoothing=200, exploratory_autostop=False,
                              smoothing_autostop=False, seed=1))
    names = ("CL_pop", "V_pop", "omega_CL", "omega_V", "corr_CL_V", "b")
    x = [res.parameters[k] for k in names]
    best, _ = exact.mle(x, y, correlated=True)
    c = exact.centers(x, y, correlated=True)
    se = np.sqrt(np.diag(np.linalg.inv(-hessian(
        lambda z: exact.loglik(z, y, centers=c, correlated=True), x, h=2e-3))))
    assert res.converged and res.fim_status == "ok"
    for name, ref, ref_se, tol in zip(names, best, se, (0.01, 0.01, 0.05, 0.05, 0.05, 0.03)):
        assert res.parameters[name] == pytest.approx(ref, rel=tol), name
        assert res.se[name] == pytest.approx(ref_se, rel=0.15), name
    assert res.ll == pytest.approx(exact.loglik(x, y, correlated=True), abs=0.5)
    lo, hi = res.ci["corr_CL_V"]
    assert -1 < lo < res.parameters["corr_CL_V"] < hi < 1
    assert res.aic == pytest.approx(res.ofv + 2 * 6)
    eta = res.individual[["eta_CL_mode", "eta_V_mode"]].to_numpy()
    assert np.corrcoef(eta.T)[0, 1] == pytest.approx(res.parameters["corr_CL_V"], abs=0.15)


def test_covariance_factor_matches_lapack():
    from pysaem.engine import _factor
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(1, 7))
        root = rng.standard_normal((n, n + 2))
        cov = root @ root.T * 10 ** rng.uniform(-4, 2)
        chol, prec, ok = _factor(cov)
        assert ok and np.allclose(chol, np.linalg.cholesky(cov), rtol=1e-10)
        assert np.allclose(prec, np.linalg.inv(cov), rtol=1e-7)
    assert not _factor(np.array([[1.0, 2.0], [2.0, 1.0]]))[2]


@pytest.mark.slow
def test_three_state_fit_does_not_depend_on_the_seed():
    # started a factor two away, with an error parameter far above the data: the
    # variances must neither run away nor drift to zero, whatever the seed
    from scipy.linalg import expm
    rng = np.random.default_rng(9)
    truth = {"CL": 35.0, "V1": 4.0, "V2": 12.0, "V3": 100.0, "Q2": 40.0, "Q3": 8.0}
    omega = {"CL": 0.3, "V1": 0.25, "V2": 0.3, "V3": 0.35, "Q2": 0.3, "Q3": 0.35}
    times = np.array([0.05, 0.15, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0])
    rows = []
    for i in range(60):
        cl, v1, v2, v3, q2, q3 = (truth[k] * np.exp(omega[k] * rng.standard_normal()) for k in truth)
        A = np.array([[-(cl + q2 + q3) / v1, q2 / v2, q3 / v3], [q2 / v1, -q2 / v2, 0.0],
                      [q3 / v1, 0.0, -q3 / v3]])
        f = np.array([(expm(A * t) @ [100.0, 0.0, 0.0])[0] / v1 for t in times])
        y = f + (0.003 + 0.15 * f) * rng.standard_normal(times.size)
        rows.append({"ID": i, "TIME": 0.0, "DV": ".", "AMT": 100.0})
        rows += [{"ID": i, "TIME": t, "DV": v, "AMT": "."} for t, v in zip(times, y)]
    model = Model(states={"A1": 0, "A2": 0, "A3": 0}, parameters=list(truth), aux={"C": "A1/V1"},
                  odes={"A1": "-CL/V1*A1 - Q2/V1*A1 + Q2/V2*A2 - Q3/V1*A1 + Q3/V3*A3",
                        "A2": "Q2/V1*A1 - Q2/V2*A2", "A3": "Q3/V1*A1 - Q3/V3*A3"},
                  outputs=["C"], inputs={"input": {"target": "A1"}}, t0=0)
    data = Dataset.from_records(pd.DataFrame(rows))
    start = {k: v * (2.0 if n % 2 == 0 else 0.5) for n, (k, v) in enumerate(truth.items())}
    fits = []
    for seed in (1, 2, 3, 4):
        pop = Population.build(model, init=start, omega=0.5,
                               error=pysaem.ErrorModel("combined1", a=5.0, b=0.2))
        fits.append(pysaem.fit(model, data, pop, Settings(
            n_exploratory=1000, n_smoothing=500, exploratory_autostop=False,
            smoothing_autostop=False, seed=seed, fim=False, simulations=False)))
    ofv = [f.ofv for f in fits]
    assert max(ofv) - min(ofv) < 2.0 and not any(f.messages for f in fits)
    for k in truth:
        typical = np.array([f.parameters[f"{k}_pop"] for f in fits])
        spread = np.array([f.parameters[f"omega_{k}"] for f in fits])
        assert np.ptp(typical) < 0.15 * typical.mean() and np.ptp(spread) < 0.15
        # sampling stops at 24: the slow third state is the one the data inform least
        assert typical.mean() == pytest.approx(truth[k], rel=0.4 if k == "V3" else 0.15)
        assert spread.min() > 0.5 * omega[k]


def test_a_subject_caught_in_a_wrong_mode_comes_back():
    # delayed feedback model of Friberg et al. (2002): a transit time seven times too
    # short is a second mode of a subject's conditional distribution, and random-walk
    # steps do not cross the valley; the prior draw of one component does
    from pysaem.engine import Engine
    from pysaem.solve import Problem
    model = Model(
        states={"A": 0, "P": "B0", "T1": "B0", "T2": "B0", "T3": "B0", "B": "B0"},
        parameters=["k", "B0", "MTT", "gam", "slope"], aux={"ktr": "4/MTT", "E": "slope*A"},
        odes={"A": "-k*A", "P": "ktr*P*(1 - E)*(B0/B)**gam - ktr*P", "T1": "ktr*(P - T1)",
              "T2": "ktr*(T1 - T2)", "T3": "ktr*(T2 - T3)", "B": "ktr*(T3 - B)"},
        outputs=["B"], inputs={"input": {"target": "A"}}, t0=0)
    truth = np.array([0.17, 5.0, 100.0, 0.16, 0.3])
    times = np.arange(0.0, 601.0, 48.0)
    rows = []
    for i in range(8):
        rows.append({"ID": i, "TIME": 0.0, "DV": ".", "AMT": 3.3})
        rows += [{"ID": i, "TIME": t, "DV": 1.0, "AMT": "."} for t in times]
    df = pd.DataFrame(rows)
    design = Dataset.from_records(df)
    problem = Problem(model, design, method="explicit", rtol=1e-9)
    f = problem.gather(problem.predict(np.tile(truth, (8, 1))[None])[0])[0]
    rng = np.random.default_rng(0)
    df.loc[df["DV"] != ".", "DV"] = f * (1 + 0.1 * rng.standard_normal(f.size))
    pop = Population.build(model, init=dict(zip(model.parameters, truth)), omega=0.3,
                           no_iiv=["k", "gam"], error=pysaem.ErrorModel("proportional", b=0.1))
    sampled = np.array([1, 2, 4])

    def transit_effect(mcmc):
        eng = Engine(model, Dataset.from_records(df), pop)
        eta = np.zeros((1, 8, 5))
        eta[0, :, 2] = -2.0
        states = eng.rng_states(1, stream=0)
        scale = 0.1 * np.ones(5)
        for it in range(40):
            eng.estep(eta, states, eng.omega, sampled, mcmc, 2, scale, scale)
        return eta[0, :, 2]

    # without joint prior draws, which informative data on every parameter make useless
    assert (transit_effect((0, 2, 2, 0)) < -1.0).any()
    assert (transit_effect((0, 2, 2, 1)) > -1.0).all()


def test_group_draws_from_the_prior_leave_a_correlated_prior_unchanged():
    # data that say nothing: the draws of the fourth kernel alone must keep N(0, Omega)
    from pysaem.engine import Engine
    model = Model(states={}, odes={}, parameters=["a", "b", "c", "d"], aux={"y": "0*a*b*c*d + 1"},
                  outputs=["y"])
    df = pd.DataFrame({"ID": np.repeat(np.arange(400), 2), "TIME": np.tile([0.0, 1.0], 400),
                       "DV": 1.0})
    pop = Population.build(model, init={k: 1.0 for k in "abcd"},
                           omega={"a": 0.5, "b": 1.0, "c": 0.3, "d": 2.0}, error="constant",
                           correlations=[["a", "b", "c"]])
    pop.correlation_values = {"corr_a_b": 0.6, "corr_a_c": -0.4, "corr_b_c": 0.2}
    eng = Engine(model, Dataset.from_records(df), pop)
    eta, states, sampled = np.zeros((1, 400, 4)), eng.rng_states(1, 0), np.arange(4)
    draws = []
    for it in range(300):
        eng.estep(eta, states, eng.omega, sampled, (0, 0, 0, 1), it % 3 + 2, np.ones(4), np.ones(4))
        if it >= 50:
            draws.append(eta[0].copy())
    x = np.concatenate(draws)
    assert (np.abs(x.mean(0)) / eng.omega).max() < 0.02
    # 75 000 draws: 0.02 on the scale of the standard deviations is five Monte Carlo errors
    scale = np.outer(eng.omega, eng.omega)
    assert np.allclose(np.cov(x.T) / scale, eng.prior(eng.omega, sampled)[0] / scale, atol=0.02)


def test_with_all_random_effects_correlated_the_parametrisation_does_not_matter():
    """CL and V vary independently; k = CL/V. With a diagonal Omega, (k, V) is another model;
    with all random effects correlated it is the same model and reaches the same likelihood."""
    df, _ = exact.simulate()
    data = Dataset.from_records(df)
    rate = Model(states={"A": "0"}, parameters=["k", "V"], odes={"A": "-k*A"}, aux={"C": "A/V"},
                 outputs=["C"], inputs={"dose": {"target": "A"}})
    fits = {}
    for name, model, init in (("CL", ONE_CMT, {"CL": 1.0, "V": 5.0}), ("k", rate, {"k": 0.2, "V": 5.0})):
        for corr in (None, [list(model.parameters)]):
            pop = Population.build(model, init=init, omega=0.5, error="proportional", correlations=corr)
            fits[name, corr is not None] = pysaem.fit(model, data, pop, FAST)
    full_cl, full_k = fits["CL", True], fits["k", True]
    noise = 4 * (full_cl.ll_se + full_k.ll_se)
    assert full_k.ofv == pytest.approx(full_cl.ofv, abs=max(noise, 0.5))
    assert fits["k", False].ofv > full_k.ofv + 10  # the diagonal (k, V) forbids the correlation the data carry


def test_several_starts_keep_the_best_and_report_them_all():
    df, _ = exact.simulate()
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 10.0}, omega=0.3, error="proportional")
    res = pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, replace(FAST, n_starts=3, start_spread=1.0))
    assert len(res.starts) == 3 and res.starts[0]["start"] == 0
    ofv = [r["ofv"] for r in res.starts if r["parameters"] is not None]
    assert res.ofv <= min(ofv) + 2.0
    assert res.converged
    assert pysaem.FitResult.from_json(res.to_json()).starts == res.starts


def test_box_cox_written_in_the_model_reaches_the_likelihood_of_the_truth():
    rng = np.random.default_rng(3)
    n, lam, mu, om, v = 200, 0.5, 2.0, 0.6, 10.0
    cl = (1 + lam * (mu + om * rng.standard_normal(n))) ** (1 / lam)
    rows = []
    for i in range(n):
        rows.append({"ID": i + 1, "TIME": 0.0, "DV": ".", "AMT": 100.0})
        for t in (0.5, 2.0, 6.0, 12.0):
            f = 100 / v * np.exp(-cl[i] / v * t)
            rows.append({"ID": i + 1, "TIME": t, "DV": f * (1 + 0.1 * rng.standard_normal()), "AMT": "."})
    data = Dataset.from_records(pd.DataFrame(rows))
    m = Model(states={"A": "0"}, parameters=["phi", "lam", "V"], odes={"A": "-CL/V*A"},
              aux={"CL": "(1 + lam*phi)**(1/lam)", "C": "A/V"}, outputs=["C"],
              inputs={"dose": {"target": "A"}})
    pop = Population.build(m, init={"phi": 1.5, "lam": 0.8, "V": 8.0}, distribution={"phi": "normal"},
                           no_iiv=["lam", "V"], omega={"phi": 0.4}, error="proportional")
    res = pysaem.fit(m, data, pop, replace(FAST, simulations=False))
    truth = Population.build(m, init={"phi": mu, "lam": lam, "V": v}, distribution={"phi": "normal"},
                             no_iiv=["lam", "V"], omega={"phi": om}, error="proportional")
    for p in truth.parameters.values():
        p.fixed = p.omega_fixed = True
    truth.error["C"].b, truth.error["C"].fixed = 0.1, True
    at_truth = pysaem.fit(m, data, truth, replace(FAST, fim=False, simulations=False))
    assert res.converged and 0 < res.parameters["lam_pop"] < 1.5
    assert res.ofv <= at_truth.ofv + 3.0


def test_a_typical_value_rounded_onto_its_edge_is_kept_inside_and_reported():
    # expit of a large Gaussian value rounds to 1.0, exp of a very negative one to 0.0: the
    # population at the estimates must still be valid, and the fit says the parameter is at
    # the edge
    from pysaem.population import inside
    assert inside("logitnormal", (0.0, 1.0), 1.0) == (np.nextafter(1.0, 0.0), True)
    assert inside("logitnormal", (0.0, 1.0), 0.0) == (np.nextafter(0.0, 1.0), True)
    assert inside("logitnormal", (0.0, 1.0), 0.5) == (0.5, False)
    assert inside("lognormal", (0.0, 1.0), 0.0) == (np.finfo(float).tiny, True)
    assert inside("lognormal", (0.0, 1.0), 2.0) == (2.0, False)
    assert inside("normal", (0.0, 1.0), -3.0) == (-3.0, False)
    # the reported value itself, on the bound, starts or evaluates a new fit
    pop = Population(INTERCEPT, {"m": pysaem.Parameter(init=1.0, distribution="logitnormal")},
                     {"y": pysaem.ErrorModel("constant", a=0.1)})
    assert pop.parameters["m"].init == np.nextafter(1.0, 0.0)


def test_a_singular_correlation_block_is_a_boundary_estimate_and_said():
    # the two random effects are simulated equal, so the maximum sits where their correlation
    # matrix is singular: the fit ends at the floor, says so, and the standard errors of the
    # other quantities remain
    rng = np.random.default_rng(5)
    rows = []
    for i in range(60):
        eta = 0.3 * rng.standard_normal()
        cl, v = 1.0 * np.exp(eta), 10.0 * np.exp(eta)
        rows.append({"ID": i + 1, "TIME": 0.0, "DV": ".", "AMT": 100.0})
        for t in (0.5, 1.0, 2.0, 4.0, 8.0, 12.0):
            f = 100 / v * np.exp(-cl / v * t)
            rows.append({"ID": i + 1, "TIME": t, "DV": f * (1 + 0.1 * rng.standard_normal()), "AMT": "."})
    data = Dataset.from_records(pd.DataFrame(rows))
    pop = Population.build(ONE_CMT, init={"CL": 1.5, "V": 8.0}, omega=0.4, error="proportional",
                           correlations=[["CL", "V"]])
    res = pysaem.fit(ONE_CMT, data, pop, replace(FAST, simulations=False))
    assert res.parameters["corr_CL_V"] > 0.99
    assert any("singular" in m and "CL, V" in m for m in res.messages)
    assert np.isnan(res.se["corr_CL_V"])
    assert np.isfinite(res.se["CL_pop"]) and np.isfinite(res.se["omega_CL"]) and np.isfinite(res.se["b"])
    assert "corr_CL_V" not in res.fim_status


def _oral_engine():
    from pysaem.engine import Engine
    model = Model(states={"Ad": "0", "Ac": "0"}, parameters=["ka", "k", "V"],
                  odes={"Ad": "-ka*Ad", "Ac": "ka*Ad - k*Ac"}, aux={"C": "Ac/V"}, outputs=["C"],
                  inputs={"dose": {"target": "Ad"}})
    rng = np.random.default_rng(2)
    times = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 24, 36])
    rows = []
    for i in range(20):
        ka, k, v = np.exp(np.log([1.0, 0.2, 10.0]) + 0.3 * rng.standard_normal(3))
        c = 100 * ka / (v * (ka - k)) * (np.exp(-k * times) - np.exp(-ka * times))
        rows.append({"ID": i + 1, "TIME": 0.0, "DV": ".", "AMT": 100.0})
        rows += [{"ID": i + 1, "TIME": t, "DV": y * (1 + 0.03 * rng.standard_normal()), "AMT": "."} for t, y in zip(times, c)]
    pop = Population.build(model, init={"ka": 1.0, "k": 0.2, "V": 10.0}, omega=0.3,
                           error=pysaem.ErrorModel("proportional", b=0.03))
    eng = Engine(model, Dataset.from_records(pd.DataFrame(rows)), pop)
    return eng


def test_mode_jump_takes_a_chain_out_of_the_flip_flop_mode():
    # one-compartment oral model: ka and k swapped, V scaled by k/ka, give the same curve, so a
    # subject has a second mode that random walks do not leave; the jump kernel moves a chain
    # started there to the mode reached from the population mean
    from pysaem import saem
    from pysaem.post import modes
    eng = _oral_engine()
    main = modes(eng, np.zeros((eng.N, eng.P)))
    phi = eng.mean() + main
    swapped = phi.copy()
    swapped[0, 0], swapped[0, 1] = phi[0, 1], phi[0, 0]
    swapped[0, 2] = phi[0, 2] + phi[0, 1] - phi[0, 0]
    eta = (swapped - eng.mean())[None]
    eta = modes(eng, eta[0])[None]
    sampled = np.flatnonzero(eng.iiv)
    post_density = lambda e: (-eng.nll(eng.predict(eng.mean() + e)[0]) + eng.log_prior(e, eng.omega, sampled))[0, 0]
    assert post_density(eta) < post_density(main[None]) - 1
    d = saem.jump_vectors(eng, eta)
    step = np.random.default_rng(0)
    for _ in range(10):
        eta = saem.jump(eng, eta, eng.omega, sampled, d, step)
    assert np.abs(eta[0, 0] - main[0]).max() < 0.05
    # the other subjects sit at their mode: a jump moves them by rounding only
    assert np.abs(eta[0, 1:] - main[1:]).max() < 1e-3


def test_the_mode_search_does_not_solve_again_a_subject_whose_search_has_ended(monkeypatch):
    # a derivative solve that crawls or fails would be paid in full at every step of the search:
    # the rows of an ended subject are not finite, which ends its solve at once
    from pysaem import post
    eng = _oral_engine()
    asked = []
    exact = post.sensitivities

    def recorded(eng, phi, idx, **options):
        asked.append(phi[0].copy())
        return exact(eng, phi, idx, **options)

    monkeypatch.setattr(post, "sensitivities", recorded)
    start = np.zeros((eng.N, eng.P))
    start[:, np.flatnonzero(eng.iiv)] = 0.05
    found = post.modes(eng, start)
    own = np.array([np.isfinite(phi).all(1) for phi in asked])
    assert own[0].all() and not own[-1].all()  # everyone at first, fewer subjects at the end
    assert not (own[1:] & ~own[:-1]).any()  # a search that has ended stays ended
    monkeypatch.undo()
    # the modes are those of a search started from them: nothing was lost for the ended subjects
    assert np.allclose(post.modes(eng, found), found, atol=1e-4)


# hazard outputs: on their records, the scoring step takes the slopes of each record's
# log-likelihood by the chain rule; a parameter without random effect moves by that step only
HAZARD = Settings(seed=1, simulations=False)
JOINT_ESTIMATES = ["X0_pop", "g_pop", "omega_X0", "omega_g", "h0_pop", "beta_pop", "b"]


def hazard_model(expression, kind="single", parameters=("lam",)):
    return Model(states={}, odes={}, parameters=list(parameters), aux={"h": expression},
                 outputs=["h"], hazards={"h": kind})


def joint_fit(link):
    """A marker and its terminal event (exact.joint_simulate), fitted from initial values away
    from the truth."""
    df, subjects = exact.joint_simulate(link=link)
    model = Model(states={"X": "X0"}, odes={"X": "g*X"}, parameters=["X0", "g", "h0", "beta"],
                  aux={"h": "h0*exp(beta*X)" if link == "exp" else "h0 + beta*X"},
                  outputs=["X", "h"], hazards={"h": "terminal"})
    pop = Population.build(model, init={"X0": 4.0, "g": 0.08, "h0": 0.02 if link == "exp" else 0.01,
                                        "beta": 0.1 if link == "exp" else 0.01},
                           omega={"X0": 0.5, "g": 0.5}, no_iiv=["h0", "beta"],
                           distribution={"beta": "normal"},
                           error={"X": pysaem.ErrorModel("proportional", b=0.2)})
    res = pysaem.fit(model, Dataset.from_records(df, output="YTYPE"), pop, HAZARD)
    return res, np.array([res.parameters[k] for k in JOINT_ESTIMATES]), subjects


def test_an_exponential_hazard_gives_the_events_over_the_time_at_risk():
    df, time, d = exact.weibull(omega=0.0)
    model = hazard_model("lam")
    res = pysaem.fit(model, Dataset.from_records(df),
                     Population.build(model, init={"lam": 1.0}, no_iiv=["lam"]), HAZARD)
    lam = d.sum() / time.sum()
    assert res.parameters["lam_pop"] == pytest.approx(lam, rel=1e-3)
    assert res.se["lam_pop"] == pytest.approx(lam / np.sqrt(d.sum()), rel=0.01)
    assert res.ll == pytest.approx(d.sum() * np.log(lam) - lam * time.sum(), abs=1e-3)


def test_a_weibull_hazard_without_random_effect_reaches_the_exact_maximum():
    # differencing each record's log-likelihood in the scoring step left lam 0.5% off
    from scipy.optimize import minimize
    df, time, d = exact.weibull(omega=0.0)
    model = hazard_model("lam*k*t**(k - 1)", parameters=("lam", "k"))
    res = pysaem.fit(model, Dataset.from_records(df),
                     Population.build(model, init={"lam": 0.2, "k": 1.0}, no_iiv=["lam", "k"]),
                     HAZARD)

    def nll(u):
        lam, k = np.exp(u)
        return -(d * np.log(lam * k * time ** (k - 1)) - lam * time**k).sum()
    best = minimize(nll, np.log([0.05, 1.5]), method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    for name, ref in zip(("lam_pop", "k_pop"), np.exp(best.x)):
        assert res.parameters[name] == pytest.approx(ref, rel=2e-3), name


def test_repeated_events_with_a_frailty_reach_the_quadrature_mle():
    df, k = exact.counts()
    model = hazard_model("lam", "repeated")
    res = pysaem.fit(model, Dataset.from_records(df),
                     Population.build(model, init={"lam": 1.0}, omega=1.0), HAZARD)
    est = [res.parameters["lam_pop"], res.parameters["omega_lam"]]
    best, _ = exact.frailty_mle(exact.counts_loglik, est, k)
    assert est[0] == pytest.approx(best[0], rel=0.02)
    assert est[1] == pytest.approx(best[1], rel=0.05)
    assert res.ll == pytest.approx(exact.counts_loglik(est, k), abs=0.5)
    se = np.sqrt(np.diag(np.linalg.inv(-hessian(lambda z: exact.counts_loglik(z, k), best))))
    for name, ref in zip(("lam_pop", "omega_lam"), se):
        assert res.se[name] == pytest.approx(ref, rel=0.2), name


def test_a_frailty_on_one_event_per_subject_reaches_the_quadrature_maximum_with_its_chains():
    # the default gives it 10 chains: with 1, the fit ended 0.6 below the maximum, along the
    # ridge between the frailty and the shape of the hazard
    df, time, d = exact.weibull()
    model = hazard_model("lam*k*t**(k - 1)", parameters=("lam", "k"))
    pop = Population.build(model, init={"lam": 0.2, "k": 1.0}, omega=1.0, no_iiv=["k"])
    res = pysaem.fit(model, Dataset.from_records(df), pop, replace(HAZARD, fim=False))
    est = [res.parameters[k] for k in ("lam_pop", "k_pop", "omega_lam")]
    _, best = exact.frailty_mle(exact.weibull_loglik, est, time, d)
    assert exact.weibull_loglik(est, time, d) > best - 0.1


def test_a_small_coefficient_of_a_hazard_reaches_the_quadrature_maximum():
    # a hazard linear in the marker, beta about 0.005: differencing each record's
    # log-likelihood in the scoring step left the -2LL 2.9 above the maximum
    res, est, subjects = joint_fit("linear")
    _, best = exact.joint_mle(est, subjects, "linear")
    assert -2 * exact.joint_loglik(est, subjects, "linear") < -2 * best + 0.1


def test_a_marker_and_its_terminal_event_reach_the_quadrature_mle_and_its_information():
    res, est, subjects = joint_fit("exp")
    best, ll_best = exact.joint_mle(est, subjects)
    centers = exact.joint_centers(est, subjects)
    se = np.sqrt(np.diag(np.linalg.inv(
        -hessian(lambda z: exact.joint_loglik(z, subjects, centers=centers), est, h=2e-3))))
    for name, value, ref, ref_se in zip(JOINT_ESTIMATES, est, best, se):
        if name in ("h0_pop", "beta_pop"):
            assert abs(value - ref) < 0.25 * ref_se, name
        else:
            assert value == pytest.approx(ref, rel=0.05 if "omega" in name else 0.01), name
        assert res.se[name] == pytest.approx(ref_se, rel=0.2), name
    ll = exact.joint_loglik(est, subjects)
    assert -2 * ll < -2 * ll_best + 0.5
    assert res.ofv == pytest.approx(-2 * ll, abs=0.5)


def test_a_relative_error_resolves_predictions_down_to_the_smallest_observation():
    # observed from 60 down to 1e-10 with a proportional error: the default dynamic range of
    # 1e-12 left the last records unresolved
    from pysaem.engine import Engine
    from pysaem.solve import Problem
    model = Model(states={"A": "0"}, parameters=["Vm", "Km"], odes={"A": "-Vm*A/(Km + A)"},
                  outputs=["A"], inputs={"dose": {"target": "A"}})
    times = [0.5, 2.0, 8.0, 16.0, 24.0, 28.0]
    df = pd.DataFrame([{"ID": 1, "TIME": 0.0, "AMT": 100.0, "DV": np.nan}]
                      + [{"ID": 1, "TIME": t, "AMT": np.nan, "DV": 100.0 * np.exp(-t)}
                         for t in times])
    data = Dataset.from_records(df)
    psi = np.array([[[1e3, 1e3]]])
    tight = Problem(model, data, rtol=1e-11, dynamic_range=1e-24).predict(psi)[0]
    tight = tight[0, data.obs_grid, 0]
    for error, resolved in (("proportional", True), ("constant", False)):
        pop = Population.build(model, init={"Vm": 1e3, "Km": 1e3}, omega=0.3, error=error)
        eng = Engine(model, data, pop)
        assert (eng.prob.dyn[0] < 1e-14) == resolved
        f = eng.prob.predict(psi)[0][0, data.obs_grid, 0]
        if resolved:
            np.testing.assert_allclose(f, tight, rtol=1e-3)
    assert abs(f[-1] / tight[-1] - 1) > 1e-2


def test_subject_seen_once_gets_a_curve_to_the_end_of_the_study():
    decay = Model(states={"A": "A0"}, parameters=["A0", "k"], odes={"A": "-k*A"},
                  aux={"y": "A"}, outputs=["y"])
    df = pd.DataFrame({"ID": [1, 1, 1, 2], "TIME": [0.0, 5.0, 10.0, 0.0],
                       "DV": [100.0, 60.0, 37.0, 80.0]})
    pop = Population.build(decay, init={"A0": 90.0, "k": 0.1}, omega=0.3, error="proportional")
    eng = pysaem.api._engine(decay, Dataset.from_records(df), pop, Settings())
    c = pysaem.post.curves(eng, np.tile(eng.mean(), (2, 1)))
    once = c[c["id"] == 2]
    assert len(once) == pysaem.post.CURVE_POINTS and once["time"].max() == 10.0
    np.testing.assert_allclose(once["ind_pred_mode"], 90.0 * np.exp(-0.1 * once["time"]), rtol=1e-4)


def test_individual_curves_pass_through_the_individual_predictions(bolus_fit):
    fit = bolus_fit[0]
    c, p = fit.curves, fit.predictions
    assert set(c["id"]) == set(p["id"]) and c.groupby("id").size().min() >= pysaem.post.CURVE_POINTS
    at = p.merge(c, on=["id", "time", "output"], suffixes=("", "_curve"))
    assert len(at) == len(p)
    np.testing.assert_allclose(at["ind_pred_mode_curve"], at["ind_pred_mode"], rtol=1e-6)
    again = pysaem.FitResult.from_json(fit.to_json()).curves
    pd.testing.assert_frame_equal(again.reset_index(drop=True), c.reset_index(drop=True), check_dtype=False)


GROWTH = Model(states={"N": "N0"}, parameters=["r", "N0"], odes={"N": "r*N"}, outputs=["N"])


def test_a_late_first_record_with_initial_values_is_reported():
    df = pd.DataFrame({"ID": [1, 1, 2, 2], "TIME": [1.0, 2.0, 1.0, 3.0], "DV": [2.0, 3.0, 2.0, 4.0]})
    pop = Population.build(GROWTH, init={"r": 0.3, "N0": 1.0}, error="proportional")
    with pytest.warns(UserWarning, match="after time 0 for 2 of 2 subjects"):
        pysaem.engine.Engine(GROWTH, Dataset.from_records(df), pop)
    at_zero = Model(states={"N": "N0"}, parameters=["r", "N0"], odes={"N": "r*N"},
                    outputs=["N"], t0=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pysaem.engine.Engine(at_zero, Dataset.from_records(df), pop)


def test_population_names_are_checked_and_fixed_holds_a_typical_value():
    with pytest.raises(ValueError, match=r"init: \['Cl'\] not among the parameters"):
        Population.build(ONE_CMT, init={"Cl": 0.04})
    with pytest.raises(ValueError, match="no_iiv"):
        Population.build(ONE_CMT, no_iiv=["cl"])
    with pytest.raises(ValueError, match="error"):
        Population.build(ONE_CMT, error={"Cc": "constant"})
    pop = Population.build(ONE_CMT, no_iiv="V", fixed=["V"])
    assert pop.parameters["V"].fixed and not pop.parameters["V"].iiv
    assert not pop.parameters["CL"].fixed and pop.parameters["CL"].iiv


def test_a_prior_on_a_typical_value_gives_the_posterior_mode_and_sd():
    df, y = exact.random_intercept()
    mu, omega, a = exact.random_intercept_mle(y)
    N, n = y.shape
    m0, s = mu + 2.0, 0.3
    pop = Population.build(INTERCEPT, init={"m": 5.0}, distribution="normal", omega=omega,
                           error=pysaem.ErrorModel("constant", a=a, fixed=True), prior={"m": (m0, s)})
    pop.parameters["m"].omega_fixed = True
    res = pysaem.fit(INTERCEPT, Dataset.from_records(df), pop, FAST)
    # the marginal likelihood of m is normal with precision N n / (a^2 + n omega^2)
    precision = N * n / (a**2 + n * omega**2)
    posterior_mode = (precision * mu + m0 / s**2) / (precision + 1 / s**2)
    assert res.parameters["m_pop"] == pytest.approx(posterior_mode, abs=0.03)
    assert res.se["m_pop"] == pytest.approx(1 / np.sqrt(precision + 1 / s**2), rel=0.2)
    assert any(m.startswith("prior on the typical value of m") for m in res.messages)
    assert pysaem.FitResult.from_json(res.to_json()).population["parameters"]["m"]["prior"] == [m0, s]


def test_a_tight_prior_holds_a_typical_value_with_or_without_variability():
    df, _ = exact.simulate()
    data = Dataset.from_records(df)
    free = pysaem.fit(ONE_CMT, data, Population.build(
        ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional"), FAST)
    # sd 0.001 on the log scale: a precision of 1e6, far above that of the data, with or
    # without a random effect on V
    for no_iiv in ((), ("V",)):
        target = {"CL": 1.3 * free.parameters["CL_pop"], "V": 0.8 * free.parameters["V_pop"]}
        pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, no_iiv=no_iiv,
                               error="proportional", prior={k: (v, 0.001) for k, v in target.items()})
        res = pysaem.fit(ONE_CMT, data, pop, FAST)
        assert res.converged
        assert res.parameters["CL_pop"] == pytest.approx(target["CL"], rel=0.05)
        assert res.parameters["V_pop"] == pytest.approx(target["V"], rel=0.05)
        assert res.se["V_pop"] < 0.03 * target["V"]


def test_a_prior_is_checked_and_read_from_a_run_file(tmp_path):
    with pytest.raises(ValueError, match="sd of the prior must be positive"):
        Population.build(ONE_CMT, prior={"CL": (1.0, 0.0)})
    with pytest.raises(ValueError, match="lognormal prior must be positive"):
        Population.build(ONE_CMT, prior={"CL": (-1.0, 0.2)})
    with pytest.raises(ValueError, match=r"prior must be \(m, s\)"):
        Population.build(ONE_CMT, prior={"CL": 1.0})
    pop = Population.build(ONE_CMT, prior={"CL": [1.0, 0.2]})
    assert Population.from_dict(ONE_CMT, pop.to_dict()).parameters["CL"].prior == (1.0, 0.2)
    from pysaem import runfile
    example = Path(__file__).parent.parent / "examples" / "theophylline.toml"
    text = example.read_text().replace("ka = { init = 1.0, omega = 0.5 }",
                                       "ka = { init = 1.0, omega = 0.5, prior = [1.5, 0.3] }")
    (tmp_path / "run.toml").write_text(text)
    (tmp_path / "theophylline.csv").write_bytes((example.parent / "theophylline.csv").read_bytes())
    assert runfile.load(tmp_path / "run.toml")["population"].parameters["ka"].prior == (1.5, 0.3)
def grouped_intercepts(n=90, seed=11):
    """y = m + shift of the group + eta + eps, three groups of equal size; X is unrelated."""
    rng = np.random.default_rng(seed)
    group = np.repeat(["A", "B", "C"], n // 3)
    shift = {"A": 0.0, "B": 2.0, "C": 5.0}
    eta = 0.5 * rng.standard_normal(n)
    rows = []
    for i in range(n):
        for t in range(4):
            rows.append({"ID": i, "TIME": float(t), "G": group[i], "X": rng.standard_normal(),
                         "DV": 1.0 + shift[group[i]] + eta[i] + 0.3 * rng.standard_normal()})
    df = pd.DataFrame(rows)
    df["X"] = df.groupby("ID")["X"].transform("first")
    return df


def test_a_covariate_with_several_levels_is_estimated_as_shifts_from_the_first():
    data = Dataset.from_records(grouped_intercepts(), covariates=["G", "X"])
    pop = Population.build(INTERCEPT, init={"m": 2.0}, distribution="normal", omega=1.0,
                           error="constant", covariates={"m": ["G"]})
    res = pysaem.fit(INTERCEPT, data, pop, FAST)
    assert res.parameters["m_pop"] == pytest.approx(1.0, abs=0.3)
    assert res.parameters["beta_m_G_B"] == pytest.approx(2.0, abs=0.4)
    assert res.parameters["beta_m_G_C"] == pytest.approx(5.0, abs=0.4)
    assert list(res.population["parameters"]["m"]["covariates"]) == ["G_B", "G_C"]
    # the screen tests the group as one covariate, by its levels, and never its columns
    screen = res.covariate_screen
    assert set(screen["covariate"]) <= {"G", "X"}
    assert screen.loc[screen["covariate"] == "G", "measure"].tolist() == ["correlation ratio"]
    # a fit started from the estimates, with the columns named, gives the same model
    again = pysaem.fit(INTERCEPT, data, Population.from_dict(INTERCEPT, res.population), FAST)
    assert again.parameters["beta_m_G_C"] == pytest.approx(res.parameters["beta_m_G_C"], abs=0.2)
    with pytest.raises(ValueError, match="given twice"):
        pysaem.engine.Engine(INTERCEPT, data, Population.build(
            INTERCEPT, distribution="normal", error="constant", covariates={"m": ["G", "G_B"]}))
