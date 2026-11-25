"""Data container: records to per-subject arrays."""

import io

import numpy as np
import pandas as pd
import pytest

from pysaem.data import DataError, Dataset
from pysaem.model import Model

CSV = """ID,TIME,DV,AMT,RATE,WT
a,0,.,100,50,70
a,1,4.0,.,.,70
a,2,3.0,.,.,70
b,0,.,200,.,82
b,0.5,9.0,.,.,82
b,2,6.5,.,.,82
b,24,.,200,.,82
b,25,8.0,.,.,82
"""


def frame(text=CSV):
    return pd.read_csv(io.StringIO(text))


def test_records_split_into_observations_and_events():
    d = Dataset.from_records(frame(), covariates=["WT"])
    assert d.n_subjects == 2 and d.n_obs == 5
    assert list(d.ids) == ["a", "b"]
    assert d.obs_ptr.tolist() == [0, 2, 5] and d.ev_ptr.tolist() == [0, 1, 3]
    assert d.obs_y.tolist() == [4.0, 3.0, 9.0, 6.5, 8.0]
    assert d.ev_amt.tolist() == [100.0, 200.0, 200.0]
    assert d.ev_rate.tolist() == [50.0, 0.0, 0.0]
    assert d.covariates["WT"].tolist() == [70.0, 82.0]
    assert d.t_first.tolist() == [0.0, 0.0]


def test_reset_records_and_their_additional_doses():
    df = pd.DataFrame({"ID": [1] * 4, "TIME": [0, 1, 5, 6], "DV": [".", 2.0, ".", 1.0],
                       "AMT": [100, ".", 50, "."], "EVID": [1, 0, 4, 0], "ADDL": [0, 0, 2, 0],
                       "II": [0, 0, 12, 0]})
    d = Dataset.from_records(df)
    assert d.ev_time.tolist() == [0.0, 5.0, 17.0, 29.0]
    assert d.ev_reset.tolist() == [0, 1, 0, 0]
    only = Dataset.from_records(df.assign(EVID=[1, 0, 3, 0], ADDL=0))
    assert only.ev_reset.tolist() == [0, 1] and only.ev_amt.tolist() == [100.0, 0.0]
    # a reset row (EVID 3) gives no input: its rate is dropped with its amount
    rated = Dataset.from_records(df.assign(EVID=[1, 0, 3, 0], ADDL=0, RATE=[50, ".", 40, "."]))
    assert rated.ev_amt.tolist() == [100.0, 0.0] and rated.ev_rate.tolist() == [50.0, 0.0]


def test_steady_state_records():
    df = pd.DataFrame({"ID": [1] * 3, "TIME": [0, 1, 2], "DV": [".", 2.0, 1.0], "AMT": [100, ".", "."],
                       "SS": [1, 0, 0], "II": [12, 0, 0], "ADDL": [2, 0, 0]})
    d = Dataset.from_records(df)
    assert d.ev_time.tolist() == [0.0, 12.0, 24.0]
    assert d.ev_ss.tolist() == [12.0, 0.0, 0.0] and d.ev_reset.tolist() == [1, 0, 0]
    with pytest.raises(DataError, match="positive interval"):
        Dataset.from_records(df.assign(II=0, ADDL=0))
    with pytest.raises(DataError, match="SS = 0 or 1"):
        Dataset.from_records(df.assign(SS=[2, 0, 0]))
    with pytest.raises(DataError, match="longer than its interval"):
        Dataset.from_records(df.assign(TINF=[13, ".", "."]))
    assert Dataset.from_records(df.assign(TINF=[11, ".", "."])).ev_rate[0] == pytest.approx(100 / 11)


def test_observation_types_map_to_outputs_in_order():
    df = pd.DataFrame({
        "ID": [1, 1, 1, 1, 2, 2],
        "TIME": [0, 0, 5, 5, 0, 3],
        "DV": [10, 1, 12, 2, 9, 3],
        "YTYPE": [1, 2, 1, 2, 1, 2],
    })
    d = Dataset.from_records(df, output="YTYPE")
    assert d.output_labels == ["1", "2"]
    assert d.obs_out.tolist() == [0, 1, 0, 1, 0, 1]
    # two records at the same time share one grid point
    assert d.grid_ptr.tolist() == [0, 2, 4]
    assert d.grid_time.tolist() == [0.0, 5.0, 0.0, 3.0]
    assert d.obs_grid.tolist() == [0, 0, 1, 1, 2, 3]
    assert d.n_obs_by_output.tolist() == [3, 3]


def test_output_order_can_be_imposed():
    df = pd.DataFrame({"ID": [1, 1], "TIME": [0, 1], "DV": [1, 2], "YTYPE": ["T", "D"]})
    d = Dataset.from_records(df, output="YTYPE", output_labels=["T", "D"])
    assert d.obs_out.tolist() == [0, 1]


def test_regressor_is_carried_forward():
    df = pd.DataFrame({
        "ID": [1, 1, 1],
        "TIME": [0, 1, 2],
        "DV": [5.0, 4.0, 3.0],
        "CONC": [2.0, np.nan, 0.5],
    })
    d = Dataset.from_records(df, regressors=["CONC"])
    assert d.reg_time.tolist() == [0.0, 1.0, 2.0]
    assert d.reg_value[:, 0].tolist() == [2.0, 2.0, 0.5]


def test_censoring_columns_are_read():
    df = pd.DataFrame({
        "ID": [1, 1], "TIME": [0, 1], "DV": [0.1, 2.0], "CENS": [1, 0], "LIMIT": [0.0, np.nan],
    })
    d = Dataset.from_records(df)
    assert d.obs_cens.tolist() == [1, 0]
    assert d.obs_limit[0] == 0.0 and np.isnan(d.obs_limit[1])


def test_event_flag_overrides_the_amount_rule():
    df = pd.DataFrame({
        "ID": [1, 1, 1], "TIME": [0, 1, 2], "DV": [np.nan, 3.0, 9.0],
        "AMT": [10, 0, 0], "EVID": [1, 0, 2],
    })
    d = Dataset.from_records(df)
    assert d.n_obs == 1 and d.ev_amt.tolist() == [10.0]


def test_additional_events_are_repeated_at_the_interval():
    df = pd.DataFrame({
        "ID": [1, 1, 1, 1], "TIME": [0, 5, 30, 40], "DV": [np.nan, 3.0, np.nan, 2.0],
        "AMT": [10, 0, 20, 0], "EVID": [1, 0, 1, 0], "ADDL": [2, 0, 0, 0],
        "II": [12, 0, 0, 0],
    })
    d = Dataset.from_records(df)
    assert d.ev_time.tolist() == [0.0, 12.0, 24.0, 30.0]
    assert d.ev_amt.tolist() == [10.0, 10.0, 10.0, 20.0]
    df.loc[0, "II"] = 0
    with pytest.raises(DataError, match="positive interval"):
        Dataset.from_records(df)


def test_two_tables_give_the_same_dataset_as_records():
    obs = pd.DataFrame({"id": ["a", "a", "b"], "time": [1, 2, 0.5], "value": [4.0, 3.0, 9.0]})
    ev = pd.DataFrame({"id": ["a", "b"], "time": [0, 0], "amount": [100, 200],
                       "rate": [50, 0]})
    d = Dataset.from_tables(obs, events=ev)
    assert d.obs_y.tolist() == [4.0, 3.0, 9.0]
    assert d.ev_rate.tolist() == [50.0, 0.0] and d.t_first.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("mutate, message", [
    (lambda df: df.drop(columns="TIME"), "TIME"),
    (lambda df: df.assign(DV="."), "no observation"),
    (lambda df: df.assign(DV=["x"] * len(df)), "DV"),
    (lambda df: df.assign(WT=[70, 71] + [82] * 6), "WT"),
])
def test_invalid_data_fail_with_the_cause(mutate, message):
    with pytest.raises(DataError, match=message):
        Dataset.from_records(mutate(frame()), covariates=["WT"])


def test_text_covariates_are_numbers_labels_or_refused():
    def read(values):
        df = pd.DataFrame({"ID": [1, 1, 2, 2, 3, 3], "TIME": [0, 1] * 3, "DV": 1.0,
                           "X": np.repeat(values, 2)})
        return Dataset.from_records(df, covariates=["X"])

    assert read(["1.5", "2", "3"]).covariates["X"].tolist() == [1.5, 2.0, 3.0]
    assert read(["m", "f", "m"]).covariate_levels == {"X": ["f", "m"]}
    with pytest.raises(DataError, match="missing"):
        read(["1.5", ".", "3"])
    with pytest.raises(DataError, match="missing"):
        read(["m", None, "f"])
    with pytest.raises(DataError, match="among numbers"):
        read(["1.5", "high", "3"])
    with pytest.raises(DataError, match=r"infinite for subject\(s\) \[2\]"):
        read([1.5, -np.inf, 3.0])


def test_subject_without_observation_is_dropped_and_reported():
    df = frame().assign(DV=lambda d: d["DV"].where(d["ID"] == "b", "."))
    d = Dataset.from_records(df)
    assert list(d.ids) == ["b"] and d.dropped_ids == ["a"]


def test_a_row_read_as_input_and_observation_is_flagged_only_without_event_flag():
    df = pd.DataFrame({"ID": [2, 1, 1, 2], "TIME": [0.0, 3.0, 0.0, 5.0],
                       "DV": [1.0, 2.0, 0.0, 4.0], "AMT": [np.nan, np.nan, 50.0, 20.0]})
    d = Dataset.from_records(df)
    # in kernel order: subject 2 at 0 and 5, subject 1 at 0 and 3
    assert d.obs_time.tolist() == [0.0, 5.0, 0.0, 3.0] and d.ev_time.tolist() == [5.0, 0.0]
    assert d.obs_input.tolist() == [False, True, True, False]
    flagged = Dataset.from_records(df.assign(EVID=[0, 0, 1, 1]))
    assert flagged.n_obs == 2 and not flagged.obs_input.any()
    assert Dataset.from_records(df.assign(MDV=[0, 0, 1, 1])).obs_input.tolist() == [False, False]
    tables = Dataset.from_tables(pd.DataFrame({"id": 1, "time": [0.0], "value": [1.0]}))
    assert tables.obs_input.tolist() == [False]


def test_each_subject_keeps_the_start_of_its_solve_at_its_own_times():
    from pysaem import Model
    from pysaem.solve import Problem

    m = Model(states={"X": "X0"}, parameters=["X0", "k"], odes={"X": "-k*X"}, outputs=["X"])
    d = Dataset.from_records(pd.DataFrame({"ID": [1, 1, 2, 2], "TIME": [2.0, 4.0, 1.0, 3.0],
                                           "DV": 1.0}))

    def solve(data):
        prob = Problem(m, data)
        return prob.gather(prob.predict(np.array([[5.0, 0.3], [5.0, 0.3]]))[0])[0]

    grid = d.at_times([[6.0, 2.0, 4.0, 2.0], [1.0, 3.0]])
    assert grid.t_first.tolist() == d.t_first.tolist() == [2.0, 1.0]
    assert grid.grid_time.tolist() == [2.0, 4.0, 6.0, 1.0, 3.0]
    assert grid.grid_ptr.tolist() == [0, 3, 5] and grid.obs_input.tolist() == [False] * 5
    exact = 5.0 * np.exp(-0.3 * np.array([0.0, 2.0, 4.0, 0.0, 2.0]))
    np.testing.assert_allclose(solve(grid), exact, rtol=1e-12)
    # one grid for all moves the start of each solve to its first time: the states move
    common = d.on_grid([1.0, 2.0])
    assert common.t_first.tolist() == [1.0, 1.0]
    np.testing.assert_allclose(solve(common), 5.0 * np.exp(-0.3 * np.array([0.0, 1.0, 0.0, 1.0])),
                               rtol=1e-12)
    with pytest.raises(ValueError, match="1 subject.*the data have 2"):
        d.at_times([[1.0]])


def test_a_blank_cell_of_an_interpolated_regressor_is_not_a_record():
    from pysaem.model import Model
    from pysaem.solve import Problem
    obs_t = np.array([1.0, 2.0, 2.9, 3.5, 5.0, 7.0])
    df = pd.DataFrame([{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": 1.0, "EVID": 1, "C": 0.0},
                       {"ID": 1, "TIME": 2.0, "DV": ".", "AMT": 1.0, "EVID": 1, "C": "."},
                       {"ID": 1, "TIME": 4.0, "DV": ".", "AMT": ".", "EVID": 2, "C": 4.0},
                       {"ID": 1, "TIME": 8.0, "DV": ".", "AMT": ".", "EVID": 2, "C": 0.0}]
                      + [{"ID": 1, "TIME": x, "DV": 1.0, "AMT": ".", "EVID": 0, "C": "."} for x in obs_t])
    d = Dataset.from_records(df.sort_values("TIME", kind="stable"), regressors=["C"])
    spec = dict(states={"X": "0"}, parameters=["k"], regressors=["C"], odes={"X": "-k*X"},
                aux={"Cout": "C"}, outputs=["Cout"], inputs={"dose": {"target": "X"}})
    got = {}
    for name, extra in (("line", {"interpolated": ["C"]}), ("steps", {})):
        prob = Problem(Model(**spec, **extra), d)
        pred, status = prob.predict(np.array([[0.1]]))
        assert status.ravel().tolist() == [0]
        got[name] = prob.gather(pred)[0]
    # until 1 November 2026 the blanks were filled forward first: 0, 0, 0, 0, 4, 4 on the line
    np.testing.assert_allclose(got["line"], np.interp(obs_t, [0.0, 4.0, 8.0], [0.0, 4.0, 0.0]), rtol=1e-12)
    # piecewise constant: the last value carried forward
    assert got["steps"].tolist() == [0.0, 0.0, 0.0, 0.0, 4.0, 4.0]
    assert d.reg_given[:, 0].sum() == 3


TWO_OUT = Model(states={}, odes={}, parameters=["a", "b"], aux={"prey": "a", "predator": "b"},
                outputs=["prey", "predator"])


def labelled(types):
    return pd.DataFrame({"ID": [1] * len(types), "TIME": np.arange(len(types), dtype=float),
                         "DV": np.arange(len(types), dtype=float), "TYPE": types})


def test_observation_types_go_to_the_outputs_they_name():
    d = Dataset.from_records(labelled(["predator", "prey", "predator"]), output="TYPE")
    m = d.matched_to(TWO_OUT.outputs, TWO_OUT.inputs)
    assert m.output_labels == ["prey", "predator"] and m.obs_out.tolist() == [1, 0, 1]
    only = Dataset.from_records(labelled(["predator"]), output="TYPE")
    assert only.matched_to(TWO_OUT.outputs, {}).obs_out.tolist() == [1]


def test_observation_types_without_output_name_keep_their_order_or_are_refused():
    numbered = Dataset.from_records(labelled([2, 1]), output="TYPE")
    assert numbered.matched_to(TWO_OUT.outputs, {}).obs_out.tolist() == [1, 0]
    given = Dataset.from_records(labelled(["x", "w"]), output="TYPE", output_labels=["x", "w"])
    assert given.matched_to(TWO_OUT.outputs, {}).obs_out.tolist() == [0, 1]
    text = Dataset.from_records(labelled(["x", "w"]), output="TYPE")
    with pytest.raises(DataError, match="give output_labels"):
        text.matched_to(TWO_OUT.outputs, {})
    one = Model(states={}, odes={}, parameters=["a"], aux={"y": "a"}, outputs=["y"])
    assert Dataset.from_records(labelled(["conc"]), output="TYPE").matched_to(
        one.outputs, {}).obs_out.tolist() == [0]


def test_inputs_go_to_the_inputs_they_name():
    df = pd.DataFrame({"ID": [1, 1, 1], "TIME": [0.0, 0.0, 1.0], "DV": [np.nan, np.nan, 1.0],
                       "AMT": [5.0, 7.0, np.nan], "ROUTE": ["oral", "iv", None]})
    d = Dataset.from_records(df, input="ROUTE")
    assert d.input_labels == ["iv", "oral"]
    m = d.matched_to(["y"], {"oral": {"target": "G"}, "iv": {"target": "C"}})
    assert m.input_labels == ["oral", "iv"]
    assert dict(zip(m.ev_amt.tolist(), m.ev_input.tolist())) == {5.0: 0, 7.0: 1}


def test_a_text_covariate_with_more_than_two_levels_becomes_indicator_columns():
    df = pd.DataFrame({"ID": [1, 1, 2, 3, 4], "TIME": [0.0, 1.0, 0.0, 0.0, 0.0],
                       "DV": [1.0, 2.0, 3.0, 4.0, 5.0], "G": ["b", "b", "a", "c", "a"],
                       "S": ["m", "m", "f", "m", "f"]})
    d = Dataset.from_records(df, covariates=["G", "S"])
    assert d.covariate_levels == {"G": ["a", "b", "c"], "S": ["f", "m"]}
    assert d.covariate_groups == {"G": ["G_b", "G_c"]}
    assert sorted(d.covariates) == ["G_b", "G_c", "S"]
    assert d.covariates["G_b"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert d.covariates["G_c"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert d.covariates["S"].tolist() == [1.0, 0.0, 1.0, 0.0]
