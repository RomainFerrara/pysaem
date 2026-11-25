"""Typical values that run off during the iterations, alone or with a partner whose ratio or
product the data fix, and estimates correlated beyond 0.95: the messages and the convergence
flag of a fit."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pysaem.api import _correlated, _runaway
from pysaem.engine import _TKIND

LOG, NORMAL = _TKIND["lognormal"], _TKIND["normal"]


def trace(**columns):
    names = list(columns)
    return SimpleNamespace(names=names, values=np.column_stack([columns[n] for n in names]))


def engine(names, kinds):
    return SimpleNamespace(names=names, p0=len(names), tkind=np.array(kinds))


def test_two_values_that_run_off_with_their_ratio_held_are_reported_together():
    k = np.arange(200.0)
    tr = trace(ka_pop=np.full(200, 0.6), Vmax_pop=3.0 * np.exp(0.2 * k),
               Km_pop=22.0 * np.exp(0.2 * k) * (1 + 0.01 * np.sin(k)))
    messages, pairs = _runaway(engine(["ka", "Vmax", "Km"], [LOG, LOG, LOG]), tr)
    assert pairs == {frozenset(("Vmax_pop", "Km_pop"))} and len(messages) == 1
    assert "Vmax_pop and Km_pop run off together" in messages[0]
    assert "Vmax/Km stays at 0.13" in messages[0] and "not identifiable" in messages[0]


def test_a_product_held_and_a_value_that_runs_off_alone():
    k = np.arange(100.0)
    tr = trace(A_pop=np.exp(0.1 * k), B_pop=5 * np.exp(-0.1 * k), C_pop=np.exp(0.15 * k))
    messages, pairs = _runaway(engine(["A", "B", "C"], [LOG, LOG, LOG]), tr)
    assert pairs == {frozenset(("A_pop", "B_pop"))}
    assert any("A*B stays at 5" in m for m in messages)
    assert any(m.startswith("C_pop runs off (3 orders of magnitude over the last 50") for m in messages)


def test_values_that_settle_or_are_not_lognormal_are_not_reported():
    k = np.arange(300.0)
    # from a poor start, a factor 1000 in the first half then settled; a normal parameter moving
    # by 100 units is not read on the log scale
    settle = 0.01 + 10 * (1 - np.exp(-k / 10))
    tr = trace(CL_pop=settle, E0_pop=100 * k / 300, V_pop=np.full(300, 8.0))
    messages, pairs = _runaway(engine(["CL", "E0", "V"], [LOG, NORMAL, LOG]), tr)
    assert messages == [] and pairs == set()


def test_correlated_estimates_are_reported_once_and_not_twice():
    names = ["Vmax_pop", "Km_pop", "omega_Vmax", "omega_Km", "a"]
    r = np.eye(5)
    r[0, 1] = r[1, 0] = 0.97
    r[2, 3] = r[3, 2] = -0.96
    r[0, 4] = r[4, 0] = 0.94
    corr = pd.DataFrame(r, index=names, columns=names)
    messages = _correlated(corr, {frozenset(("Vmax_pop", "Km_pop"))})
    assert messages == ["the estimates of omega_Vmax and omega_Km are correlated (-0.96): the data "
                        "hardly separate them"]
    assert len(_correlated(corr, set())) == 2 and _correlated(None, set()) == []
