"""Hazard outputs: their error model, the names of the error parameters, the likelihood of
their records, the checks of those records, and their slopes and chains in the estimation."""

import math
import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.integrate import quad

import exact
from pysaem import DataError, Dataset, ErrorModel, FitError, Model, Population, Settings, saem
from pysaem.engine import _G_MIN, _HALF_LOG_2PI, Engine, _hazard_term

JOINT = Model(states={"X": "X0"}, parameters=["X0", "g", "h0", "beta"], odes={"X": "g*X"},
              aux={"h": "h0*exp(beta*X)"}, outputs=["X", "h"], hazards={"h": "terminal"})
# constant hazards, solved in closed form: two competing terminal causes, a single event and
# repeated events
CONSTANT = Model(states={}, odes={}, parameters=["a", "b", "c", "d"],
                 aux={"T1": "a", "T2": "b", "S": "c", "R": "d"}, outputs=["T1", "T2", "S", "R"],
                 hazards={"T1": "terminal", "T2": "terminal", "S": "single", "R": "repeated"})
RATES = {"a": 0.3, "b": 0.2, "c": 0.25, "d": 0.4}
# a Weibull baseline, integrated numerically
WEIBULL = Model(states={}, odes={}, parameters=["lam", "kw"],
                aux={"S": "lam*kw*t**(kw - 1)", "R": "lam*kw*t**(kw - 1)"}, outputs=["S", "R"],
                hazards={"S": "single", "R": "repeated"})
# a marker after an input, and repeated events whose hazard rises with it
DECAY = Model(states={"A": "0"}, parameters=["k", "V", "lam", "beta"], odes={"A": "-k*A"},
              aux={"C": "A/V", "h": "lam*exp(beta*C)"}, outputs=["C", "h"],
              hazards={"h": "repeated"}, inputs={"u": {"target": "A"}})


def rows(i, out, *records, **columns):
    """Rows of subject i for output out, one per record (time, value[, censoring flag])."""
    return [{"ID": i, "TIME": float(r[0]), "DV": r[1], "CENS": r[2] if len(r) > 2 else 0,
             "YTYPE": out, **columns} for r in records]


def engine(model, records, values, iiv=(), **population):
    """Engine for the records, whose YTYPE names the output, random effects on iiv only."""
    pop = Population.build(model, init=values, distribution="normal", omega=0.3,
                           no_iiv=[p for p in model.parameters if p not in iiv], **population)
    data = Dataset.from_records(pd.DataFrame(records), output="YTYPE",
                                output_labels=model.outputs)
    return Engine(model, data, pop)


def nll(eng):
    pred, status = eng.predict(eng.mean()[None])
    assert (status == 0).all()
    return eng.nll(pred)[0]


def test_a_hazard_output_takes_the_error_model_hazard_whatever_the_global_one():
    # the command line gives one ErrorModel to every output
    for error in ("proportional", ErrorModel("proportional", b=0.1), {"X": "proportional"},
                  {"X": "proportional", "h": "hazard"}):
        pop = Population.build(JOINT, error=error)
        assert pop.error["X"].kind == "proportional"
        assert pop.error["h"].kind == "hazard" and pop.error["h"].names == ()
    assert Population.build(JOINT).error["X"].kind == "combined1"
    again = Population.from_dict(JOINT, Population.build(JOINT).to_dict())
    assert again.error["h"].kind == "hazard"


def test_the_error_model_hazard_is_for_hazard_outputs_only():
    params = {p: {"init": 1.0} for p in JOINT.parameters}
    with pytest.raises(ValueError, match="output h is a hazard output: its error model is "
                                         "'hazard', not 'proportional'"):
        Population(JOINT, params, {"X": ErrorModel("constant"), "h": ErrorModel("proportional")})
    with pytest.raises(ValueError, match="output X: the error model 'hazard' is for hazard "
                                         "outputs, and the model's hazards do not name X"):
        Population(JOINT, params, {"X": ErrorModel("hazard"), "h": ErrorModel("hazard")})
    # an error model named for a hazard output is not replaced
    with pytest.raises(ValueError, match="output h is a hazard output"):
        Population.build(JOINT, error={"h": "proportional"})
    plain = Model(states={"X": "X0"}, parameters=["X0", "g"], odes={"X": "g*X"}, outputs=["X"])
    with pytest.raises(ValueError, match="hazards do not name X"):
        Population.build(plain, error="hazard")


def test_error_parameters_are_named_by_output_only_when_several_outputs_have_them():
    pop = Population.build(JOINT)
    assert [pop.error_name("X", w) for w in pop.error["X"].names] == ["a", "b"]
    two = dict(states={"X": "X0"}, parameters=["X0", "g", "h0"], odes={"X": "g*X"},
               aux={"Y": "2*X", "h": "h0*X"}, outputs=["X", "Y", "h"])
    pop = Population.build(Model(**two, hazards={"h": "repeated"}),
                           error={"X": "constant", "Y": "proportional"})
    assert (pop.error_name("X", "a"), pop.error_name("Y", "b")) == ("a_X", "b_Y")
    # as without hazards
    pop = Population.build(Model(**dict(two, aux={"Y": "2*X"}, outputs=["X", "Y"])))
    assert pop.error_name("X", "a") == "a_X"


def test_each_record_adds_the_probability_of_what_happened_since_the_previous_one():
    a, b, c, d = RATES.values()
    cases = [
        # an event at 3.2; no event by 5
        (rows(1, "T1", (0, 0), (3.2, 1)), 3.2 * a - math.log(a)),
        (rows(2, "T1", (0, 0), (5, 0)), 5 * a),
        # visits without event leave the time of the event as the only information
        (rows(3, "S", (0, 0), (1, 0), (2, 0), (3, 1)), 3 * c - math.log(c)),
        # S entered at 2 while the solve starts at 0: without random effect, exact
        (rows(4, "R", (0, 0), (6, 0)) + rows(4, "S", (2, 0), (6, 1)), 6 * d + 4 * c - math.log(c)),
        # one event in (2, 5]
        (rows(5, "S", (0, 0), (2, 0), (5, 1, 1)), 2 * c - math.log(-math.expm1(-3 * c))),
        # events at 1, 2.5 (two) and 4, then no more by 6
        (rows(6, "R", (0, 0), (1, 1), (2.5, 1), (2.5, 1), (4, 1), (6, 0)), 6 * d - 4 * math.log(d)),
        # 2, 0 and 3 events in three intervals of 3: Poisson counts
        (rows(7, "R", (0, 0), (3, 2, 1), (6, 0), (9, 3, 1)),
         -sum(stats.poisson.logpmf(k, 3 * d) for k in (2, 0, 3))),
        # competing terminal causes: T1 at 4, which ends the follow-up of T2
        (rows(8, "T1", (0, 0), (4, 1)) + rows(8, "T2", (0, 0), (4, 0)), 4 * a - math.log(a) + 4 * b),
    ]
    eng = engine(CONSTANT, [r for records, _ in cases for r in records], RATES)
    got = nll(eng)
    assert eng.prob.last_stats["exact"] == len(cases)
    np.testing.assert_allclose(got, [ref for _, ref in cases], rtol=0, atol=1e-12)


def test_a_weibull_hazard_integrated_by_the_solver_gives_the_same_terms():
    lam, kw = 0.3, 1.7

    def H(t):
        return lam * t ** kw

    def log_h(t):
        return math.log(lam * kw * t ** (kw - 1))
    cases = [
        (rows(1, "S", (0, 0), (2.2, 1)), H(2.2) - log_h(2.2)),
        (rows(2, "R", (0, 0), (3, 0)) + rows(2, "S", (1, 0), (3, 0)), 2 * H(3) - H(1)),
        (rows(3, "R", (0, 0), (0.5, 1), (1.5, 1), (2.5, 0)), H(2.5) - log_h(0.5) - log_h(1.5)),
        (rows(4, "S", (0, 0), (1, 0), (2, 1, 1)), H(1) - math.log(-math.expm1(H(1) - H(2)))),
    ]
    eng = engine(WEIBULL, [r for records, _ in cases for r in records], {"lam": lam, "kw": kw})
    got = nll(eng)
    assert eng.prob.last_stats == {"implicit": 0, "exact": 0, "failed": 0, "events": 0,
                                   "unlocated": 0}
    np.testing.assert_allclose(got, [ref for _, ref in cases], rtol=1e-6)


def test_a_joint_subject_adds_its_gaussian_records_and_its_events():
    k, V, lam, beta, sd = 0.2, 10.0, 0.3, 0.4, 0.1
    obs = [(1.0, 8.0), (4.0, 4.6), (9.0, 1.7)]
    records = ([{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": 100.0, "EVID": 1, "YTYPE": "C"}]
               + rows(1, "C", *obs, EVID=0)
               + rows(1, "h", (0, 0), (2, 1), (3, 0), (6.5, 1), (12, 0), EVID=0))
    eng = engine(DECAY, records, {"k": k, "V": V, "lam": lam, "beta": beta},
                 error={"C": ErrorModel("constant", a=sd)})

    def C(t):
        return 100 / V * math.exp(-k * t)

    def h(t):
        return lam * math.exp(beta * C(t))
    ref = sum(0.5 * ((y - C(t)) / sd) ** 2 + math.log(sd) + _HALF_LOG_2PI for t, y in obs)
    ref += quad(h, 0, 12, epsabs=0, epsrel=1e-12)[0] - math.log(h(2.0)) - math.log(h(6.5))
    assert nll(eng)[0] == pytest.approx(ref, abs=1e-5)

    # records of h at 0, 2, 3, 6.5 and 12, each linked to the previous one and to the first
    pred = eng.predict(eng.mean()[None])[0]
    on, t = eng.on_hazard, eng.data.obs_time
    assert t[on].tolist() == [0.0, 2.0, 3.0, 6.5, 12.0] and t[~on].tolist() == [1.0, 4.0, 9.0]
    assert eng.hcol.tolist() == [-1, 2] and eng.hrep.tolist() == [False, True]
    assert eng.err_free.tolist() == [True, False]
    grid = eng.data.grid_time
    assert eng.obs_prev[on][0] == -1 and grid[eng.obs_prev[on][1:]].tolist() == [0, 2, 3, 6.5]
    assert (eng.obs_open[on] == 0).all()
    assert (eng.obs_prev[~on] == -1).all() and (eng.obs_open[~on] == -1).all()
    t = t[on]
    expected = np.array([quad(h, 0, x, epsabs=0, epsrel=1e-12)[0] for x in t])
    np.testing.assert_allclose(eng.cumulative(pred)[0, on], expected, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(eng.cumulative(pred)[0, ~on], [C(x) for x, _ in obs], rtol=1e-6)
    dH, hz = eng.hazard_parts(pred)
    np.testing.assert_allclose(dH[0, on], np.diff(expected, prepend=0.0), rtol=1e-5, atol=1e-12)
    # read before the input at 0, as every output is
    np.testing.assert_allclose(hz[0, on], [h(x) if x > 0 else lam for x in t], rtol=1e-6)
    assert np.isnan(dH[0, ~on]).all() and np.isnan(hz[0, ~on]).all()
    sd_rec = eng.residual_sd(eng.gather(pred), eng.data.obs_out)[0]
    assert np.isnan(sd_rec[on]).all() and (sd_rec[~on] == sd).all()
    # the E-step fills the hazards' columns too
    uy, after, _, _ = eng.estep(np.zeros((2, 1, 4)), eng.rng_states(2, 0), eng.omega,
                                np.zeros(0, np.int64), (2, 2, 2, 1), 2, np.ones(4), np.ones(4))
    assert after.shape == (2, grid.size, eng.prob.n_columns) == (2, grid.size, 3)
    assert (uy == eng.nll(after)).all() and uy[0, 0] == pytest.approx(ref, abs=1e-5)


def test_the_layout_of_the_records_does_not_change_the_likelihood():
    a, b, c, d = RATES.values()
    layouts = [
        rows(1, "S", (0, 0), (5, 1)) + rows(1, "R", (0, 0), (2, 1), (2, 1), (4, 1), (6, 0)),
        # visits without events
        rows(2, "S", (0, 0), (1, 0), (3, 0), (5, 1))
        + rows(2, "R", (0, 0), (1, 0), (2, 1), (2, 1), (3, 0), (4, 1), (5, 0), (6, 0)),
        # rows written in another order, the same at equal times
        (rows(3, "R", (0, 0), (2, 0), (2, 1), (2, 1), (4, 1), (4, 0), (6, 0))
         + rows(3, "S", (0, 0), (5, 1)))[::-1],
        # one row of two events in place of two rows of one
        rows(4, "S", (0, 0), (5, 1)) + rows(4, "R", (0, 0), (2, 2), (4, 1), (6, 0)),
    ]
    eng = engine(CONSTANT, [r for records in layouts for r in records], RATES)
    got = nll(eng)
    assert eng.prob.last_stats["exact"] == len(layouts)
    np.testing.assert_allclose(got, 5 * c - math.log(c) + 6 * d - 3 * math.log(d), rtol=0,
                               atol=1e-12)


def test_a_negative_hazard_or_a_hazard_lost_at_an_event_rejects_the_draw():
    eng = engine(CONSTANT, rows(1, "S", (0, 0), (2, 0), (5, 1)), RATES)
    pred = eng.predict(eng.mean()[None])[0]
    col, grid = eng.hcol[CONSTANT.outputs.index("S")], eng.data.grid_time.tolist()

    def nll_with(value, at):
        p = pred.copy()
        p[0, grid.index(at), col] = value
        return eng.nll(p)[0, 0]
    base = eng.nll(pred)[0, 0]
    # negative where a record closes an interval; not a number, or infinite, at an event
    assert nll_with(-1e-12, 2.0) == nll_with(np.nan, 5.0) == nll_with(np.inf, 5.0) == np.inf
    # zero at an event costs log 1e-300; the hazard is not read elsewhere
    assert nll_with(0.0, 5.0) == pytest.approx(base + math.log(RATES["c"]) - math.log(1e-300))
    assert nll_with(np.nan, 2.0) == nll_with(-1.0, 0.0) == base
    # the floors of the terms
    assert _hazard_term(2.0, 0.0, 1.0, 0, False) == math.log(1e-300) - 2.0
    assert _hazard_term(2.0, np.nan, 1.0, 0, True) == -np.inf
    assert _hazard_term(2.0, np.nan, 0.0, 0, True) == -2.0
    assert _hazard_term(0.0, 1.0, 1.0, 1, False) == math.log(1e-300)
    assert _hazard_term(0.0, 1.0, 2.0, 1, True) == 2 * math.log(1e-300) - math.log(2.0)
    assert _hazard_term(1e-20, 1.0, 1.0, 1, False) == pytest.approx(math.log(1e-20), rel=1e-12)


def test_the_initial_values_must_give_a_hazard_never_negative_and_positive_at_events():
    for expression, values, records, message in (
            ("lam - b*t", {"lam": 0.1, "b": 0.1}, rows(7, "h", (0, 0), (3, 0)),
             "the hazard h of subject 7 is negative (-0.2) at time 3"),
            ("lam*(t - b)**2", {"lam": 0.1, "b": 2.0}, rows(7, "h", (0, 0), (2, 1)),
             "the hazard h of subject 7 is 0 at the event at time 2, which needs a positive"),
            # positive at the records, negative between them
            ("lam*(t - 2)**2 - b", {"lam": 0.1, "b": 0.2}, rows(7, "h", (0, 0), (4, 0)),
             "the hazard h of subject 7 has an integral falling from 0 to -0.266667 between "
             "times 0 and 4: it is negative in between")):
        model = Model(states={}, odes={}, parameters=["lam", "b"], aux={"h": expression},
                      outputs=["h"], hazards={"h": "single"})
        eng = engine(model, rows(1, "h", (0, 0), (0.5, 0)) + records, values)
        with pytest.raises(FitError, match=re.escape(f"at the initial values, {message}")):
            eng.check_initial()
    # infinite at an event, written into the predictions the check reads
    eng = engine(CONSTANT, rows(1, "S", (0, 0), (2, 0)) + rows(7, "S", (0, 0), (5, 1)), RATES)
    eng.check_initial()
    pred, status = eng.predict(eng.mean()[None])
    pred[0, eng.data.grid_ptr[1] + 1, eng.hcol[CONSTANT.outputs.index("S")]] = np.inf
    with pytest.raises(FitError, match="hazard S of subject 7 is inf at the event at time 5"):
        eng._check_hazards(pred[0], status[0], {})


def valid(model):
    """Records of a subject 1 that are accepted, written before those refused."""
    if "S" in model.outputs:
        return rows(1, "S", (1, 0), (5, 0))
    first = [{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": 100.0, "YTYPE": "C"}] if model.inputs else []
    return first + rows(1, "h", (0, 0), (5, 0))


@pytest.mark.parametrize("model, records, message, iiv", [
    # 1. a start row marked missing is not read: the first record read has the value 1
    pytest.param(CONSTANT, rows(7, "S", (0, 0), MDV=1) + rows(7, "S", (5, 1)),
                 "the first record of hazard output S opens its follow-up, with the value 0 and "
                 "the censoring flag 0 (a row marked missing is not read)", (), id="opening"),
    # 2.
    pytest.param(CONSTANT, rows(7, "S", (0, 0)),
                 "hazard output S has only the record that opens its follow-up", (),
                 id="opening only"),
    # 3.
    pytest.param(CONSTANT, rows(7, "R", (0, 0), (2, 1.5)),
                 "a record of hazard output R has the value 1.5; it counts events, a whole "
                 "number from 0", (), id="not a count"),
    pytest.param(CONSTANT, rows(7, "S", (0, 0), (2, 2)),
                 "a record of hazard output S has the value 2; a single output has one event at "
                 "most", (), id="two events on a single output"),
    # 4.
    pytest.param(CONSTANT, rows(7, "S", (0, 0), (2, 1, -1)),
                 "a record of hazard output S has the censoring flag -1", (), id="flag"),
    pytest.param(CONSTANT, rows(7, "S", (0, 0), (2, 0, 1)),
                 "the record of hazard output S at time 2 has the censoring flag 1 and the value "
                 "0", (), id="flag 1 without event"),
    pytest.param(CONSTANT, rows(7, "S", (0, 0), (2, 0), (2, 1, 1)),
                 "the record of hazard output S at time 2 counts events since the previous "
                 "record (censoring flag 1), which is at the same time", (),
                 id="flag 1 on an empty interval"),
    pytest.param(CONSTANT, rows(7, "S", (0, 0)) + rows(7, "S", (2, 0), LIMIT=1.0),
                 "a record of hazard output S has a limit", (), id="limit"),
    # 5.
    pytest.param(CONSTANT, rows(7, "R", (0, 0), (0, 1), (3, 0)),
                 "hazard output R has an event at time 0, when its follow-up opens", (),
                 id="event at the opening"),
    # 6.
    pytest.param(CONSTANT, rows(7, "S", (0, 0), (2, 1), (3, 1)),
                 "hazard output S is single and has a second event at time 3", (),
                 id="second event"),
    pytest.param(CONSTANT, rows(7, "S", (0, 0), (2, 1), (3, 0)),
                 "hazard output S has a record at time 3, after its event at time 2", (),
                 id="record after the event"),
    pytest.param(CONSTANT, rows(7, "T1", (0, 0), (2, 1)) + rows(7, "R", (0, 0), (3, 0)),
                 "output R has a record at time 3, after the terminal event of T1 at time 2", (),
                 id="record after a terminal event"),
    pytest.param(CONSTANT, rows(7, "T1", (0, 0), (2, 1)) + rows(7, "T2", (0, 0), (2, 1)),
                 "hazard outputs T1 and T2 have their terminal event at the same time 2", (),
                 id="two terminal events at once"),
    # 7.
    pytest.param(CONSTANT, rows(7, "T1", (0, 0), (2, 0), (4, 1, 1)),
                 "the event of hazard output T1 lies between two records (censoring flag 1), and "
                 "the terminal outputs ['T1', 'T2'] compete", (), id="interval, competing causes"),
    pytest.param(JOINT, rows(7, "h", (0, 0), (8, 1, 1)) + rows(7, "X", (0, 5.0), (4, 7.0)),
                 "output X has a record at time 4, inside the interval (0, 8] where the terminal "
                 "event of h happened; a record of h with the value 0 at time 4 puts the event "
                 "after it", (), id="record inside the interval"),
    # 8.
    pytest.param(Model(**CONSTANT.to_dict() | {"t0": 1.0}), rows(7, "S", (0, 0), (3, 0)),
                 "the follow-up of hazard output S opens at time 0, before the solve starts at "
                 "time 1", (), id="opening before the start"),
    pytest.param(CONSTANT, rows(7, "R", (0, 0), (6, 0)) + rows(7, "S", (2, 0), (6, 1)),
                 "the follow-up of hazard output S opens at time 2, after the solve starts at "
                 "time 0, and the random effects of ['c'] reach the hazard", ("c",),
                 id="late entry with a random effect"),
    # 9. without an event flag column, a row with an amount and a value is read twice
    pytest.param(DECAY, rows(7, "h", (0, 0), AMT=100.0) + rows(7, "h", (5, 0)),
                 "the row at time 0 is read both as an input, for its amount, and as a record of "
                 "hazard output h, for its value", (), id="row read twice"),
])
def test_records_that_do_not_read_as_a_follow_up_are_refused_naming_the_subject(model, records,
                                                                                message, iiv):
    values = {p: 0.5 for p in model.parameters}
    with pytest.raises(DataError, match=re.escape(f"subject 7: {message}")):
        engine(model, valid(model) + records, values, iiv)
    # subject 1 alone is accepted
    engine(model, valid(model), values, iiv)


def test_the_example_opens_each_follow_up_at_the_start_and_links_each_record_to_it():
    # an event at 5.3, no event by 12, an event in (4, 8] after a visit at 4; the random
    # effects of X0 and g reach the hazard, every follow-up opens at the start of the solve
    records = (rows(1, "h", (0, 0), (5.3, 1)) + rows(1, "X", (0, 4.9), (2, 6.0), (4, 7.2))
               + rows(2, "h", (0, 0), (12, 0)) + rows(2, "X", (0, 5.2), (4, 7.1), (8, 9.8),
                                                      (12, 13.6))
               + rows(3, "h", (0, 0), (4, 0), (8, 1, 1)) + rows(3, "X", (0, 4.4), (4, 6.4)))
    eng = engine(JOINT, records, {"X0": 5.0, "g": 0.1, "h0": 0.01, "beta": 0.1}, ("X0", "g"),
                 error={"X": "proportional"})
    eng.check_initial()
    d, on = eng.data, eng.on_hazard
    assert d.obs_time[on].tolist() == [0.0, 5.3, 0.0, 12.0, 0.0, 4.0, 8.0]
    local = d.grid_ptr[d.obs_subject[on]]
    prev = np.where(eng.obs_prev[on] >= 0, d.grid_time[eng.obs_prev[on] + local], np.nan)
    np.testing.assert_array_equal(prev, [np.nan, 0.0, np.nan, 0.0, np.nan, 0.0, 4.0])
    assert (d.grid_time[eng.obs_open[on] + local] == 0.0).all()


def test_models_without_hazards_keep_their_likelihood_and_e_step():
    # a linear model in closed form and a saturable one by the integrator
    one = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"},
                aux={"C": "A/V"}, outputs=["C"], inputs={"u": {"target": "A"}})
    saturable = Model(states={"A": "0"}, parameters=["Vm", "Km", "V"],
                      odes={"A": "-Vm*(A/V)/(Km + A/V)"}, aux={"C": "A/V"}, outputs=["C"],
                      inputs={"u": {"target": "A"}})
    df, _ = exact.simulate(n=20)
    data = Dataset.from_records(df)
    for model, init, before in (
            (one, {"CL": 2.0, "V": 10.0},
             (6887.3801946632775, 1305.5755850396797, -0.5216206584249311)),
            (saturable, {"Vm": 40.0, "Km": 20.0, "V": 10.0},
             (1489.6121652518427, 254.2266984068416, 2.721757087497903))):
        eng = Engine(model, data, Population.build(model, init=init, omega=0.3,
                                                   error="proportional"))
        eta = np.random.default_rng(0).normal(0.0, 0.05, (2, eng.N, eng.P))
        pred, _ = eng.predict(eng.mean()[None] + eta)
        got = eng.nll(pred)
        # the Gaussian density in the order of the kernel, bit for bit
        f, y, b = eng.gather(pred), data.obs_y, eng.eb[0]
        for c in range(2):
            for i in range(eng.N):
                total = 0.0
                for j in range(data.obs_ptr[i], data.obs_ptr[i + 1]):
                    g = max(b * abs(f[c, j]), _G_MIN)
                    z = (y[j] - f[c, j]) / g
                    total += 0.5 * z * z + math.log(g) + _HALF_LOG_2PI
                assert got[c, i] == total
        uy, after, _, _ = eng.estep(eta, eng.rng_states(2, 0), eng.omega, np.arange(eng.P),
                                    (2, 2, 2, 1), 2, np.full(eng.P, 0.1), np.full(eng.P, 0.1))
        assert after.shape[2] == 1 and (uy == eng.nll(after)).all()
        # the values of the code before hazard outputs
        assert (got.sum(), uy.sum(), eta.sum()) == pytest.approx(before, rel=1e-12)


def test_the_scoring_step_takes_the_slopes_of_each_record_by_the_chain_rule():
    a, b, c, d = RATES.values()
    records = (rows(1, "T1", (0, 0), (3.2, 1)) + rows(1, "T2", (0, 0), (3.2, 0))
               + rows(2, "S", (0, 0), (2, 0), (5, 1, 1))
               + rows(2, "R", (0, 0), (1, 1), (2.5, 2), (4, 0), (6, 3, 1)))
    eng = engine(CONSTANT, records, RATES)
    # the rates are on their natural scale: the forward differences are exact
    _, f, D, owners, raw = saem.sensitivities(eng, eng.mean()[None], np.arange(4))
    assert owners == [(0, 0), (1, 0), (2, 0), (3, 0)] and (raw == D).all()
    expected = {
        # opening; an event at 3.2: log a - 3.2 a; no event of T2 by 3.2: -3.2 b
        "T1": [0.0, 1 / a - 3.2], "T2": [0.0, -3.2],
        # no event by 2; the event in (2, 5]: log(1 - exp(-3 c))
        "S": [0.0, -2.0, 3 / math.expm1(3 * c)],
        # an event, a tie of two, none, then a Poisson count of three in (4, 6]
        "R": [0.0, 1 / d - 1, 2 / d - 1.5, -1.5, 3 / d - 2],
    }
    for q, name in enumerate(CONSTANT.outputs):
        mine = eng.data.obs_out == q
        np.testing.assert_allclose(D[0, mine, q], expected[name], rtol=1e-9, atol=1e-12)
        assert (D[0, ~mine, q] == 0).all()
    # the slopes sum to those of the likelihood of each subject
    for q in range(4):
        step = 1e-6 * np.eye(4)[q]
        up, down = (eng.nll(eng.predict(eng.mean()[None] + s)[0])[0] for s in (step, -step))
        slope = -(up - down) / 2e-6
        per_subject = np.add.reduceat(D[0, :, q], eng.data.obs_ptr[:-1])
        np.testing.assert_allclose(per_subject, slope, rtol=1e-6, atol=1e-8)
    # their information: the outer products of the slopes, censored records included
    c1, w = saem._score_terms(eng, f)
    assert (c1 == 1).all() and (w == 1).all()


def test_the_slopes_skip_a_hazard_infinite_at_the_opening_and_drop_a_negative_one():
    # a Weibull shape below 1 is infinite at the origin, where the likelihood does not read it
    records = rows(1, "S", (0, 0), (2, 1)) + rows(1, "R", (0, 0), (1, 1), (3, 0))
    eng = engine(WEIBULL, records, {"lam": 0.3, "kw": 0.5})
    pred = eng.predict(eng.mean()[None])[0]
    assert np.isinf(eng.hazard_parts(pred)[1][0, eng.obs_prev < 0]).all()
    _, _, D, _, _ = saem.sensitivities(eng, eng.mean()[None], np.arange(2))
    assert np.isfinite(D).all() and (D[0, eng.obs_prev < 0] == 0).all()
    # a hazard negative at a record: the record's likelihood is lost, its slopes are not numbers
    model = Model(states={}, odes={}, parameters=["lam", "b"], aux={"h": "lam - b*t"},
                  outputs=["h"], hazards={"h": "single"})
    eng = engine(model, rows(1, "h", (0, 0), (1, 0), (3, 0)), {"lam": 0.1, "b": 0.1})
    _, _, D, _, _ = saem.sensitivities(eng, eng.mean()[None], np.arange(2))
    assert np.isfinite(D[0, :2]).all() and np.isnan(D[0, 2]).all()


def test_a_marker_keeps_its_slopes_and_weights_beside_the_events():
    # a hazard linear in the marker: the joint model is solved in closed form, as the marker alone
    marker = Model(states={"X": "X0"}, parameters=["X0", "g"], odes={"X": "g*X"}, outputs=["X"])
    joint = Model(states={"X": "X0"}, parameters=["X0", "g", "h0"], odes={"X": "g*X"},
                  aux={"h": "h0*X"}, outputs=["X", "h"], hazards={"h": "terminal"})
    # the second subject's last value is below 2 (censoring flag 1), its event in (4, 6]
    X = rows(1, "X", (0, 4.9), (2, 6.0), (4, 7.2)) + rows(2, "X", (0, 5.2), (4, 2.0, 1))
    h = rows(1, "h", (0, 0), (5, 1)) + rows(2, "h", (0, 0), (4, 0), (6, 1, 1))
    values, error = {"X0": 5.0, "g": 0.1, "h0": 0.01}, {"X": "proportional"}
    both = engine(joint, X + h, values, ("X0", "g"), error=error)
    alone = engine(marker, X, {k: values[k] for k in marker.parameters}, ("X0", "g"), error=error)
    _, f, _, _, raw = saem.sensitivities(both, both.mean()[None], np.arange(3))
    _, f_alone, _, _, raw_alone = saem.sensitivities(alone, alone.mean()[None], np.arange(2))
    on = both.on_hazard
    # the two subjects solved once with their sensitivities, still in closed form
    sens = next(iter(both._sensitivity.values()))
    assert sens.last_stats == {"implicit": 0, "exact": 2, "failed": 0, "events": 0,
                               "unlocated": 0} and on.sum() == 5
    np.testing.assert_allclose(f[0, ~on], f_alone[0], rtol=1e-13)
    np.testing.assert_allclose(raw[0, ~on, :2], raw_alone[0], rtol=1e-9)
    assert (raw[0, ~on, 2] == 0).all()
    c1, w = saem._score_terms(both, f)
    c1_alone, w_alone = saem._score_terms(alone, f_alone)
    np.testing.assert_allclose(c1[0, ~on], c1_alone[0], rtol=1e-9)
    np.testing.assert_allclose(w[0, ~on], w_alone[0], rtol=1e-9)
    assert (c1[0, on] == 1).all() and (w[0, on] == 1).all()


def test_a_frailty_gets_ten_chains_unless_the_chains_are_set():
    def chains(model, records, iiv=(), n_chains=None, **population):
        eng = engine(model, [r for i in range(40) for r in records(i)],
                     {p: 0.5 for p in model.parameters}, iiv, **population)
        return saem.run(eng, Settings(n_exploratory=0, n_smoothing=0, n_chains=n_chains)).n_chains

    def events(i):
        return rows(i, "T1", (0, 0), (5, 0)) + rows(i, "R", (0, 0), (5, 0))

    def occasions(i):
        return rows(i, "T1", (0, 0), (5, 0), OCC=1) + rows(i, "T1", (8, 0), OCC=2)

    def with_marker(i):
        return rows(i, "h", (0, 0), (5, 0)) + rows(i, "X", (0, 5.0), (2, 6.0))
    # by default, 200 chain-subjects: 5 chains for 40 subjects
    assert chains(CONSTANT, events) == chains(CONSTANT, events, ("d",)) == 5
    # a random effect, of the subject or of each occasion, that reaches a terminal or single
    # output and no output without hazard
    assert chains(CONSTANT, events, ("a",)) == chains(CONSTANT, events, ("a", "d")) == 10
    assert chains(CONSTANT, occasions, iov={"a": 0.3}) == 10
    assert chains(CONSTANT, events, ("a",), n_chains=3) == 3
    # the records of the marker inform the random effects that reach it
    assert chains(JOINT, with_marker, ("X0", "g")) == 5
    assert chains(JOINT, with_marker, ("X0", "h0")) == 10
