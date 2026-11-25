"""Simulation of hazard outputs and the checks built on it: event times, their laws, Kaplan-Meier
and mean numbers of events, the time-to-event NPDE, dropout in the visual predictive checks, and
what a fit with hazard outputs reports."""

import json
import math
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.optimize import brentq

import exact
import pysaem
from pysaem import Dataset, ErrorModel, Model, Population, Settings, post
from pysaem.engine import Engine, _rand
from pysaem.result import FitResult

JOINT = Model(states={"X": "X0"}, odes={"X": "g*X"}, parameters=["X0", "g", "h0", "beta"],
              aux={"h": "h0*exp(beta*X)"}, outputs=["X", "h"], hazards={"h": "terminal"})
WEIBULL = "lam*kw*t**(kw - 1)"


def follow_up(model, records, values, iiv=(), seed=123456, **population):
    """Engine for rows (ID, TIME, DV[, CENS], YTYPE naming the output), random effects on iiv."""
    pop = Population.build(model, init=values, no_iiv=[p for p in model.parameters if p not in iiv],
                           **population)
    data = Dataset.from_records(pd.DataFrame(records), output="YTYPE", output_labels=model.outputs)
    return Engine(model, data, pop, seed=seed)


def draws(eng, n, stream=3):
    """The Exp(1) draw of each replicate subject's first event, replayed from its random state."""
    states = np.random.SeedSequence([eng.seed, stream, 1]).generate_state(n * eng.N, np.uint64)
    return np.array([-math.log(_rand(s)[0]) for s in states]).reshape(n, eng.N)


def test_the_crossing_is_exact_on_a_cubic_and_linear_without_a_finite_slope():
    # H(t) = t**3 on [1, 2]: values 1 and 8, slopes 3 and 12
    assert post._crossing(1.0, 2.0, 1.0, 8.0, 3.0, 12.0, 3.375) == pytest.approx(1.5, abs=1e-14)
    assert post._crossing(0.0, 1.0, 0.0, 2.0, np.inf, 1.0, 0.5) == 0.25
    assert post._crossing(0.0, 1.0, 0.0, 2.0, np.nan, 1.0, 2.0) == 1.0


@pytest.mark.parametrize("expression, values, H", [
    ("h0*exp(beta*X)", {"X0": 5.0, "g": 0.1, "h0": 0.05, "beta": 0.15},
     lambda t: exact.joint_cumulative(5.0, 0.1, 0.05, 0.15, t, "exp")),
    # infinite at the origin: the first interval is interpolated linearly
    (WEIBULL, {"lam": 0.3, "kw": 0.5}, lambda t: 0.3 * t**0.5),
], ids=["exp link", "weibull 0.5"])
def test_event_times_invert_the_integral_of_the_hazard(expression, values, H):
    model = Model(states={"X": "X0"} if "X" in expression else {},
                  odes={"X": "g*X"} if "X" in expression else {}, parameters=list(values),
                  aux={"h": expression}, outputs=["h"], hazards={"h": "single"})
    end, n = 12.0, 40
    eng = follow_up(model, [{"ID": i, "TIME": t, "DV": 0, "YTYPE": "h"} for i in range(50)
                            for t in (0.0, end)], values)
    sim = post.simulate(eng, n)
    T, E = sim.first()[:, :, 0], draws(eng, n)
    assert sim.kept.all() and not np.isnan(T).any()
    # an event within the follow-up exactly when its draw is below the integral at its end
    inside = E < H(end)
    assert (np.isfinite(T) == inside).all() and (sim.count[:, :, 0] == inside).all()
    ref = np.array([brentq(lambda t: H(t) - e, 0.0, end, xtol=1e-14, rtol=1e-15)
                    for e in E[inside]])
    assert np.abs(T[inside] - ref).max() <= 1e-4 * end


def test_simulated_events_follow_the_laws_of_their_records():
    lam, kw, end, n = 0.05, 1.6, 12.0, 2000
    model = Model(states={}, odes={}, parameters=["lam", "kw"], aux={o: WEIBULL for o in "SRQ"},
                  outputs=["S", "R", "Q"],
                  hazards={"S": "single", "R": "repeated", "Q": "repeated"})
    records = []
    for i in range(n):
        records += [{"ID": i, "TIME": t, "DV": 0, "CENS": 0, "YTYPE": o} for o in "SR"
                    for t in (0.0, end)]
        # Q counts its events by interval: one record of the data carries the censoring flag 1
        records += [{"ID": i, "TIME": t, "DV": int(t == 3.0), "CENS": int(t == 3.0), "YTYPE": "Q"}
                    for t in (0.0, 3.0, 6.0, 9.0, end)]
    eng = follow_up(model, records, {"lam": lam, "kw": kw})
    sim = post.simulate(eng, 1)

    def H(t):
        return lam * t**kw
    # the first event of a single output: a Weibull time truncated at the end of follow-up
    first = sim.first()[0, :, 0]
    seen = first[np.isfinite(first)]
    assert stats.kstest(seen, lambda t: -np.expm1(-H(t)) / -np.expm1(-H(end))).pvalue > 0.01
    # repeated events: H(end) events on average
    assert abs(sim.count[0, :, 1].mean() - H(end)) < 3 * math.sqrt(H(end) / n)
    # counts by interval, at the records that close them: Poisson
    _, who, when = sim.events(2)
    assert set(np.unique(when)) <= {3.0, 6.0, 9.0, end}
    for a, b in ((0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, end)):
        k = np.bincount(who[when == b], minlength=n)
        mu = H(b) - H(a)
        top = int(stats.poisson.ppf(1 - 5 / n, mu)) + 1
        seen = np.bincount(np.minimum(k, top), minlength=top + 1)
        expected = n * np.append(stats.poisson.pmf(np.arange(top), mu),
                                 stats.poisson.sf(top - 1, mu))
        assert stats.chi2.sf(((seen - expected) ** 2 / expected).sum(), top) > 0.001, (a, b)


def test_competing_terminal_causes_end_each_other_follow_up():
    a, b, end, n = 0.05, 0.1, 10.0, 400
    model = Model(states={}, odes={}, parameters=["a", "b"], aux={"A": "a", "B": "b"},
                  outputs=["A", "B"], hazards={"A": "terminal", "B": "terminal"})
    eng = follow_up(model, [{"ID": i, "TIME": t, "DV": 0, "YTYPE": o} for i in range(n)
                            for o in "AB" for t in (0.0, end)], {"a": a, "b": b})
    sim = post.simulate(eng, 2)
    # each cause draws in turn from the subject's stream: T_A, then T_B
    states = np.random.SeedSequence([eng.seed, 3, 1]).generate_state(2 * n, np.uint64)
    T = np.empty((2 * n, 2))
    for q, s in enumerate(states):
        u, s = _rand(s)
        v, _ = _rand(np.uint64(s))
        T[q] = -math.log(u) / a, -math.log(v) / b
    T = T.reshape(2, n, 2)
    death = T.min(2)
    dead = death <= end
    np.testing.assert_allclose(sim.end[dead], death[dead], rtol=1e-12)
    assert np.isposinf(sim.end[~dead]).all()
    # the first cause is the event of its output; the other is censored at it
    first = sim.first()
    cause = T.argmin(2)
    for k in range(2):
        mine = dead & (cause == k)
        np.testing.assert_allclose(first[..., k][mine], death[mine], rtol=1e-12)
        assert np.isposinf(first[..., k][~mine]).all()
        np.testing.assert_allclose(sim.exit[..., k], np.where(dead, death, end), rtol=1e-12)
    # events of the second cause after the first one are drawn but do not count
    assert (sim.count[dead].sum(1) == 1 + (T[dead].max(1) <= end)).all()
    share = (cause[dead] == 0).mean()
    assert abs(share - a / (a + b)) < 3 * math.sqrt(a * b / (a + b) ** 2 / dead.sum())


def test_a_follow_up_ended_by_an_event_draws_its_end_among_the_censoring_times():
    model = Model(states={}, odes={}, parameters=["s", "r", "d"],
                  aux={"S": "s", "R": "r", "D": "d"}, outputs=["S", "R", "D"],
                  hazards={"S": "single", "R": "repeated", "D": "terminal"})
    # time, events and censoring flag of each record after the opening at 0
    table = {0: {"S": [(3, 1)], "R": [(2, 1), (6, 0)], "D": [(6, 0)]},
             # ends with an exact event of R
             1: {"S": [(8, 0)], "R": [(4, 1)], "D": [(8, 0)]},
             # R counts its events by interval up to the last visit
             2: {"S": [(5, 0)], "R": [(3, 0), (5, 2, 1)], "D": [(5, 0)]},
             # a terminal event at 2 ends every follow-up
             3: {"S": [(2, 0)], "R": [(2, 0)], "D": [(2, 1)]}}
    records = [{"ID": i, "TIME": t, "DV": y, "CENS": c[0] if c else 0, "YTYPE": o}
               for i, outputs in table.items() for o, rows in outputs.items()
               for t, y, *c in [(0, 0)] + rows]
    eng = follow_up(model, records, {"s": 0.1, "r": 0.1, "d": 0.1})
    first, last, window, interval, ended = post._follow_ups(eng)
    assert (first == 0).all()
    np.testing.assert_array_equal(last, [[3, 6, 6], [8, 4, 8], [5, 5, 5], [2, 2, 2]])
    np.testing.assert_array_equal(ended, [[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]])
    # the grid of a follow-up ended by an event runs to the latest end of its output
    np.testing.assert_array_equal(window, [[8, 6, 6], [8, 6, 8], [5, 5, 5], [8, 6, 8]])
    # only subject 2 records the events of R by interval; subject 3, without events of R, as
    # the two subjects with exact ones
    np.testing.assert_array_equal(interval, [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]])

    # reverse Kaplan-Meier by hand. S: censored at 5 (2 followed) and 8 (1); after an event at
    # 3 or 2, P(end 5) = 1/2. R: censored at 5 (2 followed) and 6 (1). D: censored at 5 (3
    # followed), 6 (2) and 8 (1): P(end 5) = 1/3, P(end 6) = 1/3, P(end 8) = 1/3.
    u = np.array([1.0, 0.7, 0.5, 0.4, 0.2])
    stop = post._planned_ends(first, last, window, ended, np.repeat(u[:, None], 4, axis=1))
    for i in range(3):
        np.testing.assert_array_equal(stop[:, i, :][:, ~ended[i]],
                                      np.broadcast_to(last[i, ~ended[i]], (5, (~ended[i]).sum())))
    np.testing.assert_array_equal(stop[:, 0, 0], [5, 5, 5, 8, 8])
    np.testing.assert_array_equal(stop[:, 1, 1], [5, 5, 5, 6, 6])
    np.testing.assert_array_equal(stop[:, 3], [[5, 5, 5], [5, 5, 5], [5, 5, 6], [8, 6, 6],
                                               [8, 6, 8]])
    # an event after the last censoring time, or an output without censoring: the grid's end
    later = last.copy()
    later[3] = [9, 9, 9]
    every = ended.copy()
    every[:, 2] = True
    stop = post._planned_ends(first, later, np.full_like(last, 9.0), every,
                              np.repeat(u[:, None], 4, axis=1))
    assert (stop[:, 3] == 9).all() and (stop[:, :, 2] == 9).all()

    # the simulation follows a single output to those ends: S of subject 0 runs to 5 or 8
    single = Model(states={}, odes={}, parameters=["s"], aux={"S": "s"}, outputs=["S"],
                   hazards={"S": "single"})
    eng = follow_up(single, [r for r in records if r["YTYPE"] == "S"], {"s": 0.1})
    sim = post.simulate(eng, 400)
    u = 1.0 - np.random.default_rng([eng.seed, 3, 2]).random((400, eng.N))
    np.testing.assert_array_equal(sim.exit[:, 0, 0], np.where(u[:, 0] >= 0.5, 5.0, 8.0))
    np.testing.assert_array_equal(sim.exit[:, 1:3, 0], np.broadcast_to([8.0, 5.0], (400, 2)))
    # an event after that end does not count
    np.testing.assert_array_equal(sim.count[:, 0, 0], draws(eng, 400)[:, 0] / 0.1
                                  <= sim.exit[:, 0, 0])
    assert abs((sim.exit[:, 0, 0] == 5).mean() - 0.5) < 3 * math.sqrt(0.25 / 400)


def test_the_kaplan_meier_band_keeps_its_level_when_ends_of_follow_up_differ():
    # datasets drawn from the model: late entry, censoring after entry; a follow-up ended by an
    # event simulated to the latest end of follow-up left 71% inside a 90% band late on
    lam, n, datasets = 0.15, 400, 60
    model = Model(states={}, odes={}, parameters=["lam"], aux={"h": "lam"}, outputs=["h"],
                  hazards={"h": "terminal"})
    pop = Population.build(model, init={"lam": lam}, no_iiv=["lam"])
    inside = []
    for r in range(datasets):
        rng = np.random.default_rng(100 + r)
        entry = np.r_[0.0, rng.uniform(0, 3, n - 1)]
        C = entry + rng.uniform(2, 8, n)
        T = entry + rng.exponential(1 / lam, n)
        rows = [{"ID": i, "TIME": t, "DV": y} for i in range(n)
                for t, y in ((entry[i], 0), (min(T[i], C[i]), int(T[i] <= C[i])))]
        eng = Engine(model, Dataset.from_records(pd.DataFrame(rows)), pop, seed=r + 1)
        table = post.vpc_hazards(eng, post.simulate(eng, 200))
        inside.append(((table["obs"] >= table["sim_p5"]) & (table["obs"] <= table["sim_p95"]))
                      .to_numpy())
        t = table["time"].to_numpy()
    cover = np.array(inside).mean(0)
    for a, b in ((0, 3), (3, 6), (6, 9), (9, 12)):
        assert cover[(t >= a) & (t < b)].mean() >= 0.8, (a, b)


def test_exact_and_interval_subjects_share_an_output():
    lam = 0.2
    model = Model(states={}, odes={}, parameters=["lam"], aux={"h": "lam"}, outputs=["h"],
                  hazards={"h": "single"})
    rng = np.random.default_rng(1)
    T, C = rng.exponential(1 / lam, 400), rng.uniform(2, 10, 400)
    rows = [{"ID": i, "TIME": t, "DV": y, "CENS": 0} for i in range(400)
            for t, y in ((0.0, 0), (min(T[i], C[i]), int(T[i] <= C[i])))]
    # one subject whose event was found at a visit
    rows += [{"ID": 999, "TIME": t, "DV": y, "CENS": y} for t, y in ((0, 0), (4, 0), (8, 1))]
    pop = Population.build(model, init={"lam": lam}, no_iiv=["lam"])
    eng = Engine(model, Dataset.from_records(pd.DataFrame(rows)), pop)
    interval = post._follow_ups(eng)[3][:, 0]
    assert interval[-1] and not interval[:-1].any()
    sim = post.simulate(eng, 200)
    # the events of the other subjects keep their exact times
    first = sim.first()[:, :-1, 0]
    d = eng.data
    on_record = np.isin(first, d.obs_time[d.obs_subject < 400]) & np.isfinite(first)
    assert not on_record.any()
    table = post.vpc_hazards(eng, sim)
    assert np.abs(table["sim_p50"] - np.exp(-lam * table["time"])).max() < 0.03
    assert ((table["obs"] >= table["sim_p5"]) & (table["obs"] <= table["sim_p95"])).mean() >= 0.9
    # without exact events, a subject without event records them by interval, as the others
    time, event = rng.uniform(2, 12, 100), np.zeros(100, bool)
    event[:80] = True
    eng = Engine(model, Dataset.from_records(visits(time, event)), pop)
    assert post._follow_ups(eng)[3].all()


def test_residuals_weigh_the_replicates_by_the_likelihood_of_the_hazard_records():
    df, _ = exact.joint_simulate(n=6)
    eng = Engine(JOINT, Dataset.from_records(df, output="YTYPE"),
                 joint_population([5.0, 0.1, 0.3, 0.3, 0.01, 0.15, 0.1]))
    sim = post.simulate(eng, 40)
    d, marker = eng.data, ~eng.on_hazard
    # the weights: minus the log-likelihood of each subject's event or censoring under each
    # replicate's random effects, in closed form
    rng = np.random.default_rng([eng.seed, 3])
    _, chol, _ = eng.prior(eng.omega, np.flatnonzero(eng.iiv))
    x0, g, h0, beta = np.moveaxis(
        eng.psi(eng.mean() + rng.standard_normal((40, eng.N, eng.P)) @ chol.T), -1, 0)
    hazard = df[df["YTYPE"] == 2].groupby("ID")
    end, event = hazard["TIME"].max().to_numpy(), hazard["DV"].max().to_numpy()
    np.testing.assert_allclose(sim.history, exact.joint_cumulative(x0, g, h0, beta, end, "exp")
                               - event * (np.log(h0) + beta * x0 * np.exp(g * end)), rtol=1e-5)
    # weight 0 outside 15 replicates for subject 0, outside 25 for subject 1; a common offset
    # does not count
    history = np.full_like(sim.history, 7.0)
    history[15:, 0] = np.inf
    history[25:, 1] = np.inf
    pwres, npde = post.population_residuals(eng, replace(sim, history=history,
                                                         kept=np.ones(40, bool)))
    on = [np.flatnonzero(marker & (d.obs_subject == i)) for i in range(eng.N)]
    assert np.isnan(pwres[on[0]]).all() and np.isnan(npde[on[0]]).all()
    alone = replace(sim, y=sim.y[:25], kept=np.ones(25, bool), history=np.zeros((25, eng.N)))
    ref = post.population_residuals(eng, alone)
    for got, want in zip((pwres, npde), ref):
        np.testing.assert_allclose(got[on[1]], want[on[1]], rtol=1e-12, atol=1e-12)
    assert np.isfinite(pwres[np.concatenate(on[1:])]).all()
    assert np.isnan(pwres[eng.on_hazard]).all() and np.isnan(npde[eng.on_hazard]).all()


def test_the_marker_npde_of_subjects_with_an_event_is_centred_under_the_true_model():
    # replicates still followed at the last visit gave subjects with an event a mean of +0.3
    df, _ = exact.joint_simulate(n=300, seed=10)
    eng = Engine(JOINT, Dataset.from_records(df, output="YTYPE"),
                 joint_population([5.0, 0.1, 0.3, 0.3, 0.01, 0.15, 0.1]), seed=10)
    _, npde = post.population_residuals(eng, post.simulate(eng, 500))
    d, marker = eng.data, ~eng.on_hazard
    event = np.zeros(eng.N, bool)
    event[d.obs_subject[eng.on_hazard & (d.obs_y > 0)]] = True
    assert np.isfinite(npde[marker]).all()
    for group in (event, ~event):
        x, who = npde[marker & group[d.obs_subject]], d.obs_subject[marker & group[d.obs_subject]]
        # standard error from the subjects' sums
        se = np.sqrt((np.bincount(who, x - x.mean()) ** 2).sum()) / x.size
        assert abs(x.mean()) < 3 * se


# six subjects: entry, end of follow-up, event of a single output at the end, events of a
# repeated one
TABLE = [(0.0, 2.0, 1, [1.5, 2.0]), (0.0, 3.0, 0, [1.0]), (0.0, 4.0, 1, [2.5, 3.0, 4.0]),
         (1.0, 4.0, 1, [2.0]), (2.0, 5.0, 0, [4.5]), (3.0, 6.0, 1, [5.0, 6.0])]


def by_hand(t, steps, values, before):
    return np.select([t < s for s in steps], [before] + values[:-1], values[-1])


def test_kaplan_meier_and_mean_events_with_late_entry_by_hand():
    entry, leave, event, events = (np.array(x, dtype=object) for x in zip(*TABLE))
    entry, leave, event = entry.astype(float), leave.astype(float), event.astype(bool)
    t = np.array([0.0, 1.9, 2.0, 3.5, 4.0, 5.9, 6.0])
    # at risk at 2: subjects 1 to 4; at 4: 3 to 6; at 6: subject 6
    np.testing.assert_array_equal(post.kaplan_meier(entry, leave, event, t),
                                  [1, 1, 0.75, 0.75, 0.375, 0.375, 0])
    times = np.concatenate(events)
    t = np.array([0.5, 1.0, 2.0, 3.9, 6.0])
    np.testing.assert_allclose(post.mean_events(entry, leave, times, t),
                               [0, 1 / 3, 13 / 12, 19 / 12, 23 / 6], rtol=1e-15)

    # the same table read from records: each follow-up opens at its first record
    model = Model(states={}, odes={}, parameters=["a", "d"], aux={"S": "a", "R": "d"},
                  outputs=["S", "R"], hazards={"S": "single", "R": "repeated"})
    records = []
    for i, (lo, hi, ev, times_i) in enumerate(TABLE):
        records += [{"ID": i, "TIME": lo, "DV": 0, "YTYPE": "S"},
                    {"ID": i, "TIME": hi, "DV": ev, "YTYPE": "S"},
                    {"ID": i, "TIME": lo, "DV": 0, "YTYPE": "R"}]
        records += [{"ID": i, "TIME": x, "DV": 1, "YTYPE": "R"} for x in times_i]
        records.append({"ID": i, "TIME": hi, "DV": 0, "YTYPE": "R"})
    eng = follow_up(model, records, {"a": 0.2, "d": 0.5})
    table = post.vpc_hazards(eng, post.simulate(eng, 20))
    s, r = table[table["output"] == "S"], table[table["output"] == "R"]
    assert s["measure"].unique().tolist() == ["survival"] and r["measure"].unique().tolist() == [
        "mean events"]
    t = s["time"].to_numpy()
    np.testing.assert_array_equal(t, np.linspace(0, 6, 100))
    np.testing.assert_array_equal(s["obs"], by_hand(t, [2, 4, 6], [0.75, 0.375, 0.0], 1.0))
    np.testing.assert_allclose(
        r["obs"], by_hand(t, [1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 6],
                          [1 / 3, 7 / 12, 13 / 12, 4 / 3, 19 / 12, 11 / 6, 7 / 3, 17 / 6, 23 / 6],
                          0.0), rtol=1e-15)
    # followed at t: entered by t and not out before it
    assert s["at_risk"].iloc[0] == 3 and s["at_risk"].iloc[-1] == 1
    assert (s["sim_p5"] <= s["sim_p50"]).all() and (s["sim_p50"] <= s["sim_p95"]).all()


def visits(time, event, every=2.0, end=12.0):
    """The events of exact.weibull found at visits: an event in (v - every, v] is recorded at v
    with the censoring flag 1."""
    rows = []
    grid = np.arange(0.0, end + 1e-9, every)
    for i, (t, d) in enumerate(zip(time, event)):
        if d:
            v = grid[grid >= t][0]
            rows += [{"ID": i, "TIME": x, "DV": 0, "CENS": 0} for x in grid[grid < v]]
            rows.append({"ID": i, "TIME": v, "DV": 1, "CENS": 1})
        else:
            rows += [{"ID": i, "TIME": x, "DV": 0, "CENS": 0} for x in grid[grid <= t]]
    return pd.DataFrame(rows)


def test_the_time_to_event_npde_is_standard_normal_under_the_true_model():
    df, time, event = exact.weibull(n=200)
    model = Model(states={}, odes={}, parameters=["lam", "kw"], aux={"h": WEIBULL},
                  outputs=["h"], hazards={"h": "single"})
    pop = Population.build(model, init={"lam": 0.05, "kw": 1.6}, omega={"lam": 0.5},
                           no_iiv=["kw"])
    for frame in (df, visits(time, event)):
        eng = Engine(model, Dataset.from_records(frame), pop)
        npde = post.hazard_npde(eng, post.simulate(eng, 500))
        # on the record that closes each subject's period: its event, or its last record
        assert np.isfinite(npde).sum() == 200
        x = npde[np.isfinite(npde)]
        assert abs(x.mean()) < 3 / math.sqrt(x.size)
        assert abs(x.var(ddof=1) - 1) < 3 * math.sqrt(2 / (x.size - 1))


def joint_population(theta):
    """The population of JOINT at values in the order of exact.JOINT_NAMES."""
    X0, g, w0, wg, h0, beta, b = theta
    return Population.build(JOINT, init={"X0": X0, "g": g, "h0": h0, "beta": beta},
                            omega={"X0": w0, "g": wg}, no_iiv=["h0", "beta"],
                            distribution={"beta": "normal"},
                            error={"X": ErrorModel("proportional", b=b)})


def test_the_checks_of_a_marker_and_its_terminal_event_hold_at_the_quadrature_mle():
    df, subjects = exact.joint_simulate()
    best, _ = exact.joint_mle([5.0, 0.1, 0.3, 0.3, 0.01, 0.15, 0.1], subjects)
    eng = Engine(JOINT, Dataset.from_records(df, output="YTYPE"), joint_population(best))
    sim = post.simulate(eng, 500)
    km = post.vpc_hazards(eng, sim)
    inside = (km["obs"] >= km["sim_p5"] - 1e-12) & (km["obs"] <= km["sim_p95"] + 1e-12)
    assert inside.mean() >= 0.9
    # the replicates lose the marker after their event as the data do
    marker, _ = post.vpc(eng, sim)
    assert marker["t_mid"].tolist() == [0, 2, 4, 6, 8, 10, 12]
    assert ((marker["obs_p50"] >= marker["sim_p50_low"])
            & (marker["obs_p50"] <= marker["sim_p50_high"])).all()
    # without dropout, the survivors' marker looks too low at the last visits
    kept, _ = post.vpc(eng, replace(sim, end=np.full_like(sim.end, np.inf)))
    assert (kept["obs_p50"] < kept["sim_p50_low"]).to_numpy()[-2:].all()


def test_replicates_do_not_depend_on_their_batches(monkeypatch):
    df, subjects = exact.joint_simulate(n=30)
    eng = Engine(JOINT, Dataset.from_records(df, output="YTYPE"),
                 joint_population([5.0, 0.1, 0.3, 0.3, 0.01, 0.15, 0.1]))
    one = post.simulate(eng, 12)
    monkeypatch.setattr(post, "_BATCH_VALUES", 1)
    many = post.simulate(eng, 12)
    for name in ("y", "kept", "end", "exit", "count", "time"):
        np.testing.assert_array_equal(getattr(one, name), getattr(many, name), err_msg=name)
    # a model without hazard outputs draws its replicates as it always did
    model = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"},
                  aux={"C": "A/V"}, outputs=["C"], inputs={"u": {"target": "A"}})
    df, _ = exact.simulate(n=15)
    eng = Engine(model, Dataset.from_records(df),
                 Population.build(model, init={"CL": 2.0, "V": 10.0}, omega=0.3,
                                  error="proportional"))
    sim = post.simulate(eng, 30)
    rng = np.random.default_rng([eng.seed, 3])
    _, chol, _ = eng.prior(eng.omega, np.flatnonzero(eng.iiv))
    f = eng.gather(eng.predict(eng.mean() + rng.standard_normal((30, eng.N, eng.P)) @ chol.T)[0])
    np.testing.assert_array_equal(sim.y, f + eng.residual_sd(f, eng.data.obs_out)
                                  * rng.standard_normal(f.shape))
    assert sim.kept.all() and np.isposinf(sim.end).all() and sim.count.shape == (30, 15, 0)


@pytest.fixture(scope="module")
def joint_fit():
    df, subjects = exact.joint_simulate()
    pop = joint_population([4.0, 0.08, 0.5, 0.5, 0.02, 0.1, 0.2])
    data = Dataset.from_records(df, output="YTYPE")
    return pysaem.fit(JOINT, data, pop, Settings(seed=1, fim=False)), data


def test_a_joint_fit_reports_its_residuals_its_events_and_its_hazard_check(joint_fit):
    res, data = joint_fit
    p = res.predictions
    on = (p["variable"] == "h").to_numpy()
    assert np.isfinite(p.loc[~on, ["pwres", "npde"]].to_numpy()).all()
    assert p.loc[on, ["iwres_mode", "pwres"]].isna().all().all()
    # one NPDE per subject on its hazard records: at its event or at the end of its follow-up
    closing = p[on].groupby("id")["time"].transform("max") == p.loc[on, "time"]
    npde = p.loc[on, "npde"]
    assert npde[closing].notna().all() and npde[~closing].isna().all() and closing.sum() == 150
    # the predictions of h count the events expected since the follow-up opened
    assert (p.loc[on, "pop_pred"] >= 0).all()
    assert (p.loc[on & (p["time"] == 0), "pop_pred"] == 0).all()
    assert res.hazard_events == {"h": 75} and "events   h 75" in res.summary()
    # the marker's VPC only; the hazard's against its replicates
    assert set(res.vpc["output"]) == {"1"}
    assert res.vpc_hazards["output"].unique().tolist() == ["2"] and len(res.vpc_hazards) == 100
    # log(n) of the BICc counts the marker's records and the events
    n_info = int((~on).sum()) + 75
    assert res.bicc == pytest.approx(res.ofv + 4 * math.log(150) + 3 * math.log(n_info), abs=1e-9)
    assert res.tests["2"]["npde"]["normality_pvalue"] > 0.0


def test_a_fit_with_hazard_outputs_survives_json_and_older_results_load(joint_fit, tmp_path):
    res, _ = joint_fit
    again = FitResult.from_json(res.to_json())
    pd.testing.assert_frame_equal(again.vpc_hazards, res.vpc_hazards, check_dtype=False)
    assert again.hazard_events == res.hazard_events
    older = json.loads(res.to_json())
    del older["vpc_hazards"], older["hazard_events"]
    old = FitResult.from_dict(older)
    assert old.vpc_hazards is None and old.hazard_events == {}
    pytest.importorskip("matplotlib")
    from pysaem.plots import diagnostics
    pages = diagnostics(again, tmp_path / "fit.pdf").read_bytes()
    # the marker: goodness of fit, VPC and prediction-corrected VPC; the hazard: Kaplan-Meier;
    # 8 pages of individual fits; the iterations
    assert len(re.findall(rb"/Type\s*/Page\b", pages)) == 3 + 1 + 8 + 1


def test_a_result_records_the_version_that_made_it(joint_fit):
    import pysaem
    res, _ = joint_fit
    assert res.version == pysaem.__version__
    saved = json.loads(res.to_json())
    assert FitResult.from_dict(saved).version == pysaem.__version__
    del saved["version"]  # saved before the version was recorded
    assert FitResult.from_dict(saved).version == "unknown"
