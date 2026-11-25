"""Bayesian individualisation, forecasting and the command line."""

import json

import numpy as np
import pandas as pd
import pytest

import exact
import pysaem
from pysaem import Dataset, Model, Population, Settings
from pysaem.cli import main
from pysaem.individual import individualize

INTERCEPT = Model(states={}, odes={}, parameters=["m"], aux={"y": "m"}, outputs=["y"])
ONE_CMT = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"},
                aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A"}})


def test_new_subject_gets_the_conjugate_posterior():
    mu, omega, a = 10.0, 2.0, 1.0
    pop = Population.build(INTERCEPT, init={"m": mu}, distribution="normal", omega=omega,
                           error=pysaem.ErrorModel("constant", a=a))
    y = np.array([12.1, 13.4, 12.8, 11.9])
    df = pd.DataFrame({"ID": "new", "TIME": np.arange(4.0), "DV": y})
    ind = individualize(INTERCEPT, pop, Dataset.from_records(df),
                        Settings(n_conditional=2000, n_conditional_burn=200))
    precision = 1 / omega**2 + len(y) / a**2
    post_mean = (mu / omega**2 + y.sum() / a**2) / precision
    assert ind.mode["m"][0] == pytest.approx(post_mean, abs=2e-3)
    assert ind.mean["m"][0] == pytest.approx(post_mean, abs=0.05)
    assert ind.samples[:, 0, 0].std() == pytest.approx(precision**-0.5, rel=0.1)


def test_forecast_brackets_future_observations():
    df, _ = exact.simulate(n=300, seed=21)
    pop = Population.build(ONE_CMT, init={"CL": 2.0, "V": 10.0}, omega={"CL": 0.3, "V": 0.2},
                           error=pysaem.ErrorModel("proportional", b=0.1))
    early = df[df["TIME"] <= 4]
    late = df[df["TIME"] > 4]
    ind = individualize(ONE_CMT, pop, Dataset.from_records(early))
    fc = ind.forecast(sorted(late["TIME"].unique()), level=0.9)
    merged = fc.merge(late.assign(id=late["ID"], time=late["TIME"], y=late["DV"].astype(float)),
                      on=["id", "time"])
    inside = ((merged["y"] >= merged["low"]) & (merged["y"] <= merged["high"])).mean()
    assert 0.86 <= inside <= 0.94
    assert np.all(merged["low"] < merged["median"]) and np.all(merged["median"] < merged["high"])


def test_a_forecast_starts_each_subject_where_its_records_start():
    model = Model(states={"X": "X0"}, odes={"X": "-k*X"}, parameters=["X0", "k"], outputs=["X"])
    pop = Population.build(model, init={"X0": 10.0, "k": 0.2}, omega=0.2,
                           error=pysaem.ErrorModel("proportional", b=0.1))
    df = pd.DataFrame({"ID": ["a"] * 3 + ["b"] * 3, "TIME": [2.0, 3.0, 4.0, 0.0, 1.0, 2.0],
                       "DV": [9.8, 8.1, 6.9, 10.3, 8.0, 6.6]})
    ind = individualize(model, pop, Dataset.from_records(df), Settings(seed=1))
    times = [0.0, 1.0, 2.0, 5.0]
    fc = ind.forecast(times, noise=False).set_index(["id", "time"])
    values = ["mode", "median", "mean", "low", "high"]
    # nothing before the start of the solve, which stays at the subject's first record
    assert fc.loc["a"].loc[[0.0, 1.0], values].isna().all().all()
    assert fc[values].drop(index=[("a", 0.0), ("a", 1.0)]).notna().all().all()
    mode = ind.mode.set_index("id")
    for i, start in (("a", 2.0), ("b", 0.0)):
        t = np.array([x for x in times if x >= start])
        np.testing.assert_allclose(fc.loc[i].loc[t, "mode"],
                                   mode.loc[i, "X0"] * np.exp(-mode.loc[i, "k"] * (t - start)),
                                   rtol=1e-9)


def test_hazard_outputs_are_forecast_given_the_data_from_their_last_record():
    # a terminal event whose hazard follows the marker, and repeated events; the subject "new"
    # has no marker record
    model = Model(states={"X": "X0"}, odes={"X": "g*X"}, parameters=["X0", "g", "h0", "beta", "r0"],
                  aux={"h": "h0*exp(beta*X)", "r": "r0*X"}, outputs=["X", "h", "r"],
                  hazards={"h": "terminal", "r": "repeated"})
    pop = Population.build(model, init={"X0": 5.0, "g": 0.1, "h0": 0.01, "beta": 0.15, "r0": 0.02},
                           omega={"X0": 0.3, "g": 0.3}, no_iiv=["h0", "beta", "r0"],
                           distribution={"beta": "normal"},
                           error={"X": pysaem.ErrorModel("proportional", b=0.1)})

    def rows(i, out, *records):
        return [{"ID": i, "TIME": float(t), "DV": y, "YTYPE": out} for t, y in records]
    df = pd.DataFrame(rows("new", "h", (0, 0), (6, 0)) + rows("new", "r", (0, 0), (2, 1), (5, 0))
                      + rows("dead", "h", (0, 0), (4, 1)) + rows("dead", "r", (0, 0), (4, 0))
                      + rows("dead", "X", (0, 5.1), (2, 6.0)))
    data = Dataset.from_records(df, output="YTYPE", output_labels=model.outputs)
    ind = individualize(model, pop, data, Settings(seed=3))
    times = [2.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    fc = ind.forecast(times)
    values = ["mode", "median", "mean", "low", "high"]
    x0, g, h0, beta, r0 = ind.samples[:, 0].T
    for out, t_ref, expected in (
            ("h", 6.0, lambda t: np.exp(exact.joint_cumulative(x0, g, h0, beta, 6.0, "exp")
                                        - exact.joint_cumulative(x0, g, h0, beta, t, "exp"))),
            ("r", 5.0, lambda t: r0 * x0 * (np.exp(g * t) - np.exp(g * 5.0)) / g)):
        got = fc[(fc["id"] == "new") & (fc["output"] == out)].set_index("time")
        before = [t for t in times if t < t_ref]
        assert got.loc[before, values].isna().all().all(), out
        # the mean over the posterior draws, without residual error
        after = [t for t in times if t >= t_ref]
        np.testing.assert_allclose(got.loc[after, "mean"], [expected(t).mean() for t in after],
                                   rtol=1e-6, err_msg=out)
        assert (got.loc[after, "low"] <= got.loc[after, "median"]).all()
    # after a terminal event, nothing is forecast for the hazard outputs; the marker still is
    dead = fc[fc["id"] == "dead"]
    assert dead.loc[dead["output"] != "X", values].isna().all().all()
    assert dead.loc[dead["output"] == "X", "mean"].notna().all()
    assert fc.columns.tolist() == ["id", "time", "output", "mode", "median", "mean", "low", "high"]


def test_command_line_fits_and_writes_its_output(tmp_path, capsys):
    df, _ = exact.simulate(n=20, seed=2)
    (tmp_path / "model.json").write_text(ONE_CMT.to_json())
    df.to_csv(tmp_path / "data.csv", index=False)
    code = main(["fit", str(tmp_path / "model.json"), str(tmp_path / "data.csv"),
                 "--init", "CL=1,V=5", "--omega", "0.5", "--error", "proportional",
                 "--exploratory", "150", "--smoothing", "50", "--no-autostop", "--quiet",
                 "--out", str(tmp_path / "res.json")])
    assert code == 0 and "CL_pop" in capsys.readouterr().out
    res = json.loads((tmp_path / "res.json").read_text())
    assert res["parameters"]["CL_pop"] == pytest.approx(2.0, rel=0.25)


def test_command_line_reports_a_bad_model(tmp_path, capsys):
    (tmp_path / "model.json").write_text(json.dumps({"states": {"A": "0"}, "odes": {"A": "-k*A"},
                                                     "parameters": ["CL"], "outputs": ["A"]}))
    (tmp_path / "data.csv").write_text("ID,TIME,DV\n1,0,1\n")
    assert main(["fit", str(tmp_path / "model.json"), str(tmp_path / "data.csv")]) == 1
    assert "unknown symbol 'k'" in capsys.readouterr().err
