"""Typical values that run off during the iterations, alone or with a partner whose ratio or
product the data fix, and correlated estimates: the messages and the convergence flag of a fit."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pysaem.api import CORRELATED, LOCKSTEP, RUNAWAY, _correlated, _runaway
from pysaem.engine import _TKIND

LOG, NORMAL = _TKIND["lognormal"], _TKIND["normal"]


def trace(n, **columns):
    """Trace of n iterations whose second half moves each column by the given log change."""
    k = np.arange(n) / (n // 2)
    return SimpleNamespace(names=[f"{c}_pop" for c in columns],
                           values=np.column_stack([start * np.exp(move * k)
                                                   for start, move in columns.values()]))


def engine(*names, kinds=None):
    return SimpleNamespace(names=list(names), p0=len(names),
                           tkind=np.array(kinds or [LOG] * len(names)))


def test_two_values_that_run_off_with_their_ratio_held_are_one_pair():
    up = 2 * RUNAWAY
    messages, pairs = _runaway(engine("ka", "Vmax", "Km"),
                               trace(200, ka=(0.6, 0.0), Vmax=(3.0, up), Km=(22.0, up)))
    assert pairs == {frozenset(("Vmax_pop", "Km_pop"))} and len(messages) == 1
    assert f"Vmax/Km stays at {3.0 / 22.0:.4g}" in messages[0] and "not identifiable" in messages[0]


def test_a_product_held_and_a_value_that_runs_off_alone():
    up = 2 * RUNAWAY
    messages, pairs = _runaway(engine("A", "B", "C"),
                               trace(100, A=(1.0, up), B=(5.0, -up), C=(1.0, 1.5 * up)))
    assert pairs == {frozenset(("A_pop", "B_pop"))} and len(messages) == 2
    assert "A*B stays at 5" in messages[0] and messages[1].startswith("C_pop runs off")


def test_values_below_the_thresholds_are_not_reported():
    below = 0.9 * RUNAWAY
    # a log-normal value just under the runaway move; a normal one is not read on the log scale
    messages, pairs = _runaway(engine("CL", "E0", "V", kinds=[LOG, NORMAL, LOG]),
                               trace(300, CL=(1.0, below), E0=(1.0, 2 * RUNAWAY), V=(8.0, 0.0)))
    assert messages == [] and pairs == set()
    # two that run off but drift apart by more than LOCKSTEP are two values, not a pair
    up = 2 * RUNAWAY
    messages, pairs = _runaway(engine("X", "Y"),
                               trace(100, X=(1.0, up), Y=(1.0, up + 2 * LOCKSTEP)))
    assert pairs == set() and len(messages) == 2


def test_correlated_estimates_are_named_once():
    names = ["Vmax_pop", "Km_pop", "omega_Vmax", "omega_Km", "a"]
    above, below = (CORRELATED + 1) / 2, CORRELATED - 0.01
    r = np.eye(5)
    r[0, 1] = r[1, 0] = above
    r[2, 3] = r[3, 2] = -above
    r[0, 4] = r[4, 0] = below
    corr = pd.DataFrame(r, index=names, columns=names)
    # a pair already reported as running off together is not named a second time
    messages = _correlated(corr, {frozenset(("Vmax_pop", "Km_pop"))})
    assert len(messages) == 1 and "omega_Vmax and omega_Km" in messages[0]
    assert f"({-above:+.2f})" in messages[0]
    assert len(_correlated(corr, set())) == 2 and _correlated(None, set()) == []
