"""Covariate search on data with known effects."""

import numpy as np
import pandas as pd
import pytest

import exact
from pysaem import Dataset, Model, Population, Settings
from pysaem.covariates import _draws, search, stepwise, supported

ONE_CMT = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"},
                aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A"}})
FAST = Settings(n_exploratory=300, n_smoothing=150, exploratory_autostop=False,
                smoothing_autostop=False, seed=1, simulations=False)


def frame(n=100, seed=4):
    """CL goes with X1, V with X2; X3 follows X1 closely, X4 to X6 are unrelated."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x = rng.standard_normal(6)
        x[2] = 0.8 * x[0] + 0.6 * x[2]
        cl = 2.0 * np.exp(0.3 * x[0] + 0.2 * rng.standard_normal())
        v = 10.0 * np.exp(0.25 * x[1] + 0.2 * rng.standard_normal())
        y = exact.conc(cl, v) * (1 + 0.1 * rng.standard_normal(exact.TIMES.size))
        cov = {f"X{k + 1}": x[k] for k in range(6)}
        rows.append({"ID": i, "TIME": 0.0, "DV": ".", "AMT": exact.DOSE, **cov})
        rows += [{"ID": i, "TIME": t, "DV": yy, "AMT": ".", **cov} for t, yy in zip(exact.TIMES, y)]
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_search_finds_the_two_effects_and_only_them():
    covariates = [f"X{k}" for k in range(1, 7)]
    data = Dataset.from_records(frame(), covariates=covariates)
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional")
    found = stepwise(ONE_CMT, data, pop, FAST)
    assert found.effects == [("CL", "X1"), ("V", "X2")]
    assert stepwise(ONE_CMT, data, pop, FAST, criterion="lrt").effects == found.effects
    assert found.result.parameters["beta_CL_X1"] == pytest.approx(0.3, abs=0.06)
    assert found.result.parameters["beta_V_X2"] == pytest.approx(0.25, abs=0.06)
    assert list(found.history["action"])[0] == "start" and len(found.history) < 15
    # X3 follows X1 (correlation 0.8): swapped in on CL, refitted, clearly worse
    swap = found.history[found.history["action"] == "swap"]
    assert ("CL", "X3") in set(zip(swap["parameter"], swap["covariate"]))
    assert not found.alternatives and set(swap["decision"]) == {"dropped"}
    # the correlated covariate is tried on CL at most after the true one is in
    first = found.history[found.history["action"] == "add"].iloc[0]
    assert (first["parameter"], first["decision"]) in {("CL", "kept"), ("V", "kept")}


def test_search_can_be_restricted_and_starts_from_given_effects():
    data = Dataset.from_records(frame(n=60), covariates=["X1", "X2", "X4"])
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional",
                           covariates={"V": ["X4"]})
    found = stepwise(ONE_CMT, data, pop, FAST, candidates={"CL": ["X1"]})
    assert ("CL", "X1") in found.effects and ("V", "X2") not in found.effects
    # the effect given at the start is unrelated and goes at the first challenge
    assert ("V", "X4") not in found.effects


def test_a_proxy_kept_first_is_swapped_for_the_better_covariate():
    data = Dataset.from_records(frame(), covariates=["X1", "X2", "X3"])
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional",
                           covariates={"CL": ["X3"], "V": ["X2"]})
    # a closed screen leaves the swap as the only way from the proxy X3 to X1
    found = stepwise(ONE_CMT, data, pop, FAST, p_screen=1e-300)
    assert found.effects == [("CL", "X1"), ("V", "X2")]
    swap = found.history[found.history["action"] == "swap"]
    assert list(zip(swap["parameter"], swap["covariate"], swap["decision"]))[0] == ("CL", "X1", "kept")
    # a swap not adopted and within `equivalent` is reported, once, with its gap
    every = stepwise(ONE_CMT, data, pop, FAST, p_screen=1e-300, equivalent=1e9)
    alt, = every.alternatives
    assert (alt["parameter"], alt["kept"], alt["alternative"]) == ("CL", "X1", "X3")
    assert alt["delta_bicc"] > 0 and alt["correlation"] > 0.7


def test_an_effect_that_does_not_hold_on_its_own_is_removed_at_retention():
    data = Dataset.from_records(frame(), covariates=["X1", "X2", "X4"])
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional",
                           covariates={"CL": ["X1"], "V": ["X2", "X4"]})
    # a closed screen adds nothing, so only the retention step can remove the unrelated X4
    found = stepwise(ONE_CMT, data, pop, FAST, p_screen=1e-300)
    assert found.effects == [("CL", "X1"), ("V", "X2")]
    gone, = found.retention_removed
    assert (gone["parameter"], gone["covariate"]) == ("V", "X4") and gone["rise_of_minus_2ll"] < 10.83
    kept = stepwise(ONE_CMT, data, pop, FAST, p_screen=1e-300, p_retain=None)
    assert ("V", "X4") in kept.effects and not kept.retention_removed


@pytest.mark.slow
def test_samba_finds_the_two_effects_and_only_them():
    covariates = [f"X{k}" for k in range(1, 7)]
    data = Dataset.from_records(frame(), covariates=covariates)
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional")
    found = search(ONE_CMT, data, pop, FAST)
    assert found.effects == [("CL", "X1"), ("V", "X2")]
    assert found.result.parameters["beta_CL_X1"] == pytest.approx(0.3, abs=0.06)
    assert found.result.parameters["beta_V_X2"] == pytest.approx(0.25, abs=0.06)
    # X3 follows X1 (correlation 0.8) but adds nothing to it in the regression of the draws;
    # each effect found is then fitted out, and both hold
    assert list(found.history["action"]) == ["start", "proposal", "remove", "remove"]
    assert list(found.history["decision"]) == ["", "kept", "", ""]
    # at the last iteration, any covariate coming in or going out raises its regression's criterion
    near = found.near.set_index(["parameter", "covariate"])["change"]
    assert near[("CL", "X1")] > 0 and near[("V", "X2")] > 0 and (near > 0).all()
    assert found.start.n_parameters < found.result.n_parameters
    # L = min(10, ceil(max(500 / 100, 5000 / 1000))) = 5 of the draws of each fit are used
    draws = found.result.draws
    assert len(draws) == 100 * (draws["draw"].max() + 1) and (draws[["CL", "V"]] > 0).all().all()


def test_samba_keeps_to_the_candidates_and_refuses_an_unknown_method():
    data = Dataset.from_records(frame(n=60), covariates=["X1", "X2", "X4"])
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional")
    found = search(ONE_CMT, data, pop, FAST, candidates={"CL": ["X1", "X4"]})
    assert found.effects == [("CL", "X1")]
    # a parameter whose random effect the fit removed (section 8.7) is no longer searched
    found.result.population["parameters"]["V"]["iiv"] = False
    assert set(_draws(found.result, data.n_obs)) == {"CL"}
    with pytest.raises(ValueError):
        search(ONE_CMT, data, pop, FAST, method="scm")


def test_a_change_of_model_needs_positive_evidence_beyond_the_monte_carlo_error():
    assert supported((100, 0.1), (97, 0.1), 1) and not supported((100, 0.1), (98.5, 0.1), 1)
    assert supported((100, 0.1), (101, 0.1), -1) and not supported((100, 0.1), (103, 0.1), -1)
    assert supported((100, 0.1), (97, 0.1), 0) and not supported((100, 0.1), (99, 0.1), 0)
    assert not supported((100, 1.0), (97, 1.0), 1)  # 2 + 4 * sqrt(2) = 7.7 needed


INTERCEPT = Model(states={}, odes={}, parameters=["m"], aux={"y": "m"}, outputs=["y"])


def test_the_searches_add_a_group_of_levels_as_one_covariate():
    from test_fit import grouped_intercepts
    data = Dataset.from_records(grouped_intercepts(), covariates=["G", "X"])
    pop = Population.build(INTERCEPT, init={"m": 2.0}, distribution="normal", omega=1.0,
                           error="constant")
    quick = Settings(n_exploratory=150, n_smoothing=80, exploratory_autostop=False,
                     smoothing_autostop=False, seed=1, simulations=False, n_importance=2000)
    found = stepwise(INTERCEPT, data, pop, quick)
    assert found.effects == [("m", "G")]
    assert set(found.population.parameters["m"].covariates) == {"G_B", "G_C"}
    assert found.result.parameters["beta_m_G_C"] == pytest.approx(5.0, abs=0.4)
    by_samba = search(INTERCEPT, data, pop, quick, method="samba")
    assert by_samba.effects == [("m", "G")]


class _Quadratic:
    """Least squares on the covariates and their squares."""

    def _design(self, X):
        X = np.asarray(X, float)
        return np.column_stack([np.ones(len(X)), X, X ** 2])

    def fit(self, X, y):
        self.coef, *_ = np.linalg.lstsq(self._design(X), np.asarray(y, float), rcond=None)
        return self

    def predict(self, X):
        return self._design(X) @ self.coef


def curved_frame(n=120, seed=7):
    """CL follows the square of X1, a curve no line sees; X2 and X3 are unrelated."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x = rng.standard_normal(3)
        cl = 2.0 * np.exp(0.35 * (x[0] ** 2 - 1) + 0.15 * rng.standard_normal())
        v = 10.0 * np.exp(0.2 * rng.standard_normal())
        y = exact.conc(cl, v) * (1 + 0.1 * rng.standard_normal(exact.TIMES.size))
        cov = {f"X{k + 1}": x[k] for k in range(3)}
        rows.append({"ID": i, "TIME": 0.0, "DV": ".", "AMT": exact.DOSE, **cov})
        rows += [{"ID": i, "TIME": t, "DV": yy, "AMT": ".", **cov} for t, yy in zip(exact.TIMES, y)]
    return pd.DataFrame(rows)


def test_learned_search_keeps_a_curved_effect_and_reports_its_learner():
    data = Dataset.from_records(curved_frame(), covariates=["X1", "X2", "X3"])
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional")
    found = search(ONE_CMT, data, pop, FAST, method="learned", learner=_Quadratic(), parameters=["CL"])
    assert found.effects == [("CL", "score_CL")]
    assert 0.5 < found.result.parameters["beta_CL_score_CL"] < 1.5
    assert found.result.parameters["omega_CL"] < 0.6 * found.start.parameters["omega_CL"]
    assert list(found.scores.columns) == ["id", "score_CL"] and len(found.scores) == data.n_subjects
    assert abs(found.scores["score_CL"].mean()) < 1e-9 and "CL" in found.centres
    assert found.features == {"CL": ["X1", "X2", "X3"]} and "score_CL" in found.data.covariates
    x = found.data.covariates
    assert np.allclose(found.learners["CL"].predict(np.column_stack([x["X1"], x["X2"], x["X3"]])).mean(),
                       found.centres["CL"], atol=0.05)
    assert found.history.decision.eq("kept").sum() == 1


def test_learned_search_drops_a_score_learned_on_noise():
    data = Dataset.from_records(frame(n=60), covariates=["X4", "X5", "X6"])
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional")
    found = search(ONE_CMT, data, pop, FAST, method="learned", learner=_Quadratic())
    assert found.effects == [] and found.learners == {} and list(found.scores.columns) == ["id"]
    assert found.result.bicc == found.start.bicc


def test_learned_search_checks_its_arguments():
    data = Dataset.from_records(frame(n=20), covariates=["X1"])
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional")
    with pytest.raises(ValueError, match="learner"):
        search(ONE_CMT, data, pop, FAST, method="learned")
    with pytest.raises(ValueError, match="parameters"):
        search(ONE_CMT, data, pop, FAST, method="learned", learner=_Quadratic(), parameters=["ka"])
    with pytest.raises(ValueError, match="folds"):
        search(ONE_CMT, data, pop, FAST, method="learned", learner=_Quadratic(), folds=1)


def test_learned_search_takes_a_scikit_learn_regressor():
    ensemble = pytest.importorskip("sklearn.ensemble")
    forest = ensemble.RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=0)
    data = Dataset.from_records(curved_frame(), covariates=["X1", "X2", "X3"])
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional")
    found = search(ONE_CMT, data, pop, FAST, method="learned", learner=forest, parameters=["CL"])
    assert found.effects == [("CL", "score_CL")]
    assert found.result.parameters["omega_CL"] < 0.7 * found.start.parameters["omega_CL"]
