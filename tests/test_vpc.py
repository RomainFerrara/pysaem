"""The visual predictive check of continuous outputs: censored records read alike in the data and
the replicates, and the prediction correction of Bergstrand et al. (2011)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import pysaem
from pysaem import Dataset, ErrorModel, Model, Population, Settings, post
from pysaem.engine import Engine
from pysaem.plots import _vpc

ONE_CMT = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"},
                aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A"}})
TIMES = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0)


def two_doses(error=None, n=40, lloq=None):
    """Engine of a bolus study whose data are one replicate of the model, half the subjects at a
    dose ten times the other half's; values below lloq are censored at it."""
    rows = []
    for i in range(n):
        rows.append({"ID": i, "TIME": 0.0, "DV": ".", "AMT": 10.0 if i % 2 else 100.0})
        rows += [{"ID": i, "TIME": t, "DV": 1.0, "AMT": "."} for t in TIMES]
    pop = Population.build(ONE_CMT, init={"CL": 2.0, "V": 10.0}, omega=0.2,
                           error=error or ErrorModel("proportional", b=0.1))
    eng = Engine(ONE_CMT, Dataset.from_records(pd.DataFrame(rows)), pop, seed=7)
    y = eng.from_scale(post.simulate(eng, 1, stream=11).y[0], eng.data.obs_out)
    df = pd.DataFrame(rows)
    on = df["AMT"] == "."
    df.loc[on, "DV"] = y
    if lloq is not None:
        df["CENS"] = 0
        df.loc[on & (pd.to_numeric(df["DV"], errors="coerce") < lloq), ["DV", "CENS"]] = [lloq, 1]
    return Engine(ONE_CMT, Dataset.from_records(df), pop, seed=7)


def population_prediction(eng):
    return eng.cumulative(eng.predict(eng.mean()[None])[0])[0]


def test_the_correction_takes_the_doses_out_of_the_percentiles():
    eng = two_doses()
    sim = post.simulate(eng, 200)
    plain, _ = post.vpc(eng, sim)
    corrected, read = post.vpc(eng, sim, pop_pred=population_prediction(eng))
    assert (plain["correction"] == "none").all() and (corrected["correction"] == "multiplicative").all()
    # same bins, same records
    assert corrected[["bin", "t_mid", "n"]].equals(plain[["bin", "t_mid", "n"]])
    assert np.isfinite(read).all()
    # uncorrected, the tenfold dose spreads the 10th to 90th percentiles over a decade or more;
    # corrected, only the variability between and within subjects remains
    width = lambda v: np.log(v["sim_p90"] / v["sim_p10"])  # noqa: E731
    assert (width(plain) > np.log(10)).all() and (width(corrected) < 0.5 * width(plain)).all()
    # the model is true: the observed median stays in its band
    inside = (corrected["obs_p50"] >= corrected["sim_p50_low"]) & (
        corrected["obs_p50"] <= corrected["sim_p50_high"])
    assert inside.mean() >= 0.8


@pytest.mark.parametrize("error, form", [(ErrorModel("proportional", b=0.1), "multiplicative"),
                                         (ErrorModel("exponential", a=0.1), "multiplicative"),
                                         (ErrorModel("constant", a=0.05), "additive")])
def test_each_record_is_corrected_by_the_median_prediction_of_its_bin(error, form):
    eng = two_doses(error, n=10)
    sim = post.simulate(eng, 30)
    p = population_prediction(eng)
    table, read = post.vpc(eng, sim, pop_pred=p)
    assert (table["correction"] == form).all()
    d = eng.data
    edges = post.time_bins(d.obs_time, 10)
    which = np.clip(np.searchsorted(edges, d.obs_time, side="right") - 1, 0, len(edges) - 2)
    ref = np.array([np.median(p[which == b]) for b in which])
    expected = d.obs_y * ref / p if form == "multiplicative" else d.obs_y + ref - p
    assert read == pytest.approx(expected, rel=1e-12)
    for _, row in table.iterrows():
        assert row["obs_p50"] == pytest.approx(np.percentile(expected[which == row["bin"]], 50))


def test_a_record_predicted_near_zero_is_left_out():
    eng = two_doses(n=10)
    sim = post.simulate(eng, 30)
    p = population_prediction(eng)
    p[0], p[1], p[2] = 0.0, p[1] * 1e-4, p[2] * 1e-2
    table, read = post.vpc(eng, sim, pop_pred=p)
    # at 0 and below 1/1000 of its bin's median: out; a hundredth of it: kept
    assert np.isnan(read[:2]).all() and np.isfinite(read[2:]).all()
    assert table["n"].sum() == eng.data.n_obs - 2


def test_censored_records_and_replicates_are_read_at_the_limit():
    lloq = 0.5
    eng = two_doses(lloq=lloq)
    assert 0.1 < eng.data.obs_cens.mean() < 0.5
    sim = post.simulate(eng, 100)
    table, read = post.vpc(eng, sim)
    assert (read >= lloq).all() and read == pytest.approx(np.maximum(eng.data.obs_y, lloq))
    # the late bins, mostly censored, have their low percentiles at the limit in both
    late = table[table["t_mid"] >= 8]
    assert (late["obs_p10"] == lloq).all() and (late["sim_p10"] == lloq).all()
    assert (table[[c for c in table if c.startswith("sim_p")]] >= lloq).all().all()


def test_a_fit_reports_both_checks_and_the_corrected_reading_of_the_data(tmp_path, monkeypatch):
    eng = two_doses(n=20)
    df = pd.DataFrame({"ID": np.repeat(eng.data.ids, 7), "TIME": np.tile((0.0,) + TIMES, 20)})
    df["AMT"] = pd.Series(".", index=df.index, dtype=object)
    df.loc[df["TIME"] == 0, "AMT"] = np.where(np.arange(20) % 2, 10.0, 100.0)
    df["DV"] = pd.Series(".", index=df.index, dtype=object)
    df.loc[df["TIME"] > 0, "DV"] = eng.data.obs_y
    pop = Population.build(ONE_CMT, init={"CL": 3.0, "V": 8.0}, omega=0.3,
                           error=ErrorModel("proportional", b=0.2))
    fast = Settings(n_exploratory=200, n_smoothing=100, exploratory_autostop=False,
                    smoothing_autostop=False, seed=1, fim=False, n_simulations=100)
    res = pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, fast)
    assert (res.vpc["correction"] == "none").all()
    assert (res.pcvpc["correction"] == "multiplicative").all()
    assert res.pcvpc[["bin", "t_mid", "n"]].equals(res.vpc[["bin", "t_mid", "n"]])
    assert np.isfinite(res.predictions["pcvpc_value"]).all()
    assert not np.allclose(res.predictions["pcvpc_value"], res.predictions["value"])
    back = pysaem.FitResult.from_json(res.to_json())
    assert back.pcvpc.equals(res.pcvpc)
    # a result saved before the corrected check loads without it
    d = res.to_dict()
    del d["pcvpc"]
    assert pysaem.FitResult.from_dict(d).pcvpc is None
    # each check has its page in the plots
    pytest.importorskip("matplotlib")
    from pysaem import plots
    drawn = []

    def page(plt, p, v, output):
        drawn.append(v["correction"].iloc[0])
        return _vpc(plt, p, v, output)
    monkeypatch.setattr(plots, "_vpc", page)
    plots.diagnostics(res, tmp_path / "fit.pdf")
    assert drawn == ["none", "multiplicative"] and (tmp_path / "fit.pdf").stat().st_size > 0
    quiet = pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, replace(fast, simulations=False))
    assert quiet.pcvpc is None and quiet.predictions["pcvpc_value"].isna().all()


def test_the_plot_says_the_check_is_corrected_and_what_it_left_out():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    t = np.tile(np.arange(0.0, 100.0, 10.0), 5)
    value = np.exp(rng.normal(1.0, 1.0, len(t)))
    p = pd.DataFrame(dict(output="y", id=np.repeat(np.arange(5), 10), time=t, value=value,
                          pcvpc_value=value * 0.5))
    p.loc[p.index[:3], "pcvpc_value"] = np.nan
    v = pd.DataFrame(dict(output="y", t_mid=np.arange(5.0, 100.0, 10.0), correction="multiplicative"))
    for q, level in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        v[f"obs_{q}"] = np.quantile(value * 0.5, level)
        v[f"sim_{q}_low"], v[f"sim_{q}_high"] = v[f"obs_{q}"] / 2, v[f"obs_{q}"] * 2
    a = _vpc(plt, p, v, "y").axes[0]
    assert a.get_title().startswith("Prediction-corrected visual predictive check (multiplicative)")
    assert "3 observations left out" in a.get_title() and a.get_ylabel() == "y, prediction-corrected"
    # framed on the observations as the check reads them
    assert a.get_ylim()[1] == pytest.approx(3 * 0.5 * value[3:].max())
    # the uncorrected check shows the data as they are
    a = _vpc(plt, p, v.drop(columns="correction"), "y").axes[0]
    assert a.get_title().startswith("Visual predictive check: y") and a.get_ylabel() == "y"
    assert a.get_ylim()[1] == pytest.approx(3 * value.max())
    plt.close("all")
