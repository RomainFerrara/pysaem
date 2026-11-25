"""Convergence from several starts, likelihood profile and bootstrap, on models with a known
likelihood; and the bootstrap samples of a dataset."""

import numpy as np
import pandas as pd
import pytest

import exact
import pysaem
from pysaem import Dataset, Model, Population, Settings
from pysaem.assess import bootstrap, convergence, profile
from pysaem.datasets import theophylline

INTERCEPT = Model(states={}, odes={}, parameters=["m"], aux={"y": "m"}, outputs=["y"])
FAST = Settings(n_exploratory=300, n_smoothing=150, exploratory_autostop=False,
                smoothing_autostop=False, seed=1, simulations=False)


@pytest.fixture(scope="module")
def intercept():
    df, y = exact.random_intercept()
    data = Dataset.from_records(df)
    pop = Population.build(INTERCEPT, init={"m": 5.0}, distribution="normal", omega=1.0,
                           error="constant")
    return data, pop, pysaem.fit(INTERCEPT, data, pop, FAST), y


def test_take_keeps_every_record_of_the_subjects_taken_in_their_order():
    table = theophylline()
    table["OCC"] = np.where(table["TIME"] < 12, 1, 2)
    table["RATE"] = np.nan
    d = Dataset.from_records(table, covariates=["WEIGHT", "SEX"], regressors=["WEIGHT"])
    same = d.take(np.arange(d.n_subjects))
    for name in ("obs_ptr", "obs_y", "obs_grid", "grid_time", "ev_ptr", "ev_amt", "occ_ptr",
                 "occ_time", "reg_ptr", "reg_value", "t_first", "ids"):
        np.testing.assert_array_equal(getattr(same, name), getattr(d, name))
    assert same.occ_labels == d.occ_labels and same.covariates.keys() == d.covariates.keys()
    # a sample with a repeat equals the data built from the records of those subjects
    idx = [3, 0, 3, 11]
    ids = table["ID"].unique()
    pieces = []
    for k, i in enumerate(idx):
        one = table[table["ID"] == ids[i]].copy()
        one["ID"] = f"s{k}"
        pieces.append(one)
    rebuilt = Dataset.from_records(pd.concat(pieces), covariates=["WEIGHT", "SEX"],
                                   regressors=["WEIGHT"])
    taken = d.take(idx)
    for name in ("obs_ptr", "obs_y", "obs_time", "obs_out", "grid_ptr", "grid_time", "ev_ptr",
                 "ev_time", "ev_amt", "occ_ptr", "occ_time", "reg_ptr", "reg_time", "reg_value",
                 "t_first"):
        np.testing.assert_array_equal(getattr(taken, name), getattr(rebuilt, name), err_msg=name)
    np.testing.assert_array_equal(taken.obs_grid - taken.grid_ptr[taken.obs_subject],
                                  rebuilt.obs_grid - rebuilt.grid_ptr[rebuilt.obs_subject])
    for c in d.covariates:
        np.testing.assert_array_equal(taken.covariates[c], rebuilt.covariates[c])
    assert taken.ids.tolist() == [ids[3], ids[0], f"{ids[3]}.2", ids[11]]


def test_convergence_runs_agree_on_a_model_with_one_maximum(intercept):
    data, pop, res, _ = intercept
    seen = []
    runs = convergence(INTERCEPT, data, pop, FAST, n=3, spread=0.5,
                       on_fit=lambda k, r: seen.append(k))
    assert seen == [0, 1, 2] and runs.table["converged"].all()
    assert runs.table["seed"].tolist() == [1, 2, 3]
    summary = runs.summary().set_index("estimate")
    assert summary.loc["ofv", "range"] < 1.0
    assert (summary.drop("ofv")["range"] < 3.0).all()
    assert abs(runs.table["m_pop"] - res.parameters["m_pop"]).max() < 0.05


def test_the_profile_of_a_quadratic_likelihood_gives_the_wald_interval(intercept):
    data, pop, res, y = intercept
    at = Population.from_dict(INTERCEPT, res.population)
    prof = profile(INTERCEPT, data, at, FAST, parameter="m_pop", se=res.se["m_pop"], n_points=7)
    assert len(prof.table) == 7 and prof.table["converged"].all()
    # the -2LL of this model is exactly quadratic in m
    mu, omega, a = exact.random_intercept_mle(y)
    exact_ofv = [-2 * exact.random_intercept_loglik(v, omega, a, y) for v in prof.table["value"]]
    np.testing.assert_allclose(prof.table["ofv"], exact_ofv, atol=1.0)
    lo, hi = prof.interval(0.95)
    half = 1.96 * res.se["m_pop"]
    assert lo == pytest.approx(res.parameters["m_pop"] - half, abs=0.2 * half)
    assert hi == pytest.approx(res.parameters["m_pop"] + half, abs=0.2 * half)
    with pytest.raises(ValueError, match="not a typical value"):
        profile(INTERCEPT, data, at, FAST, parameter="m", values=[1.0])
    with pytest.raises(ValueError, match="give values"):
        profile(INTERCEPT, data, at, FAST, parameter="m_pop")


def test_the_bootstrap_sd_agrees_with_the_standard_error(intercept):
    data, pop, res, _ = intercept
    at = Population.from_dict(INTERCEPT, res.population)
    boot = bootstrap(INTERCEPT, data, at, FAST, n=24, seed=2)
    assert boot.failed == 0 and len(boot.samples) == 24
    assert all(len(s) == data.n_subjects for s in boot.samples)
    summary = boot.summary().set_index("estimate")
    assert summary.loc["m_pop", "sd"] == pytest.approx(res.se["m_pop"], rel=0.5)
    assert summary.loc["m_pop", "lower"] < res.parameters["m_pop"] < summary.loc["m_pop", "upper"]
    assert "ofv" not in summary.index and boot.estimates["ofv"].isna().all()


def test_a_stratified_bootstrap_keeps_the_size_of_each_level():
    df, _ = exact.random_intercept(n_subjects=30)
    df["G"] = np.where(df["ID"] % 3 == 0, "a", "b")
    data = Dataset.from_records(df, covariates=["G"])
    pop = Population.build(INTERCEPT, init={"m": 10.0}, distribution="normal", omega=2.0,
                           error="constant")
    boot = bootstrap(INTERCEPT, data, pop, FAST, n=2, seed=5, strata="G", loglik=True)
    g = np.asarray(data.covariates["G"])
    for idx in boot.samples:
        assert (g[idx] == 1).sum() == (g == 1).sum() and len(idx) == 30
    assert boot.estimates["ofv"].notna().all()
    with pytest.raises(ValueError, match="not a covariate"):
        bootstrap(INTERCEPT, data, pop, FAST, n=1, strata="H")
