"""The frame of a diagnostic plot comes from the observations, never from a prediction or a simulation."""
import numpy as np
import pandas as pd
import pytest

from pysaem.plots import _frame, _goodness_of_fit, _in_frame, _vpc


def test_the_frame_follows_the_observations():
    assert _frame([1.0, 5.0, 20.0]) == ("log", (1.0 / 3, 60.0))
    scale, lim = _frame([2.0, 3.0, 4.0])
    assert scale == "linear" and lim == pytest.approx((1.84, 4.16))
    # mostly zeros: a log scale would hide them, so the frame stays linear
    assert _frame([0.0, 0.0, 0.0, 1.0, 100.0])[0] == "linear"
    assert _frame([np.nan]) == ("linear", None)
    assert _in_frame([0.0, 5.0], "log", (0.1, 10.0)).tolist() == [0.1, 5.0]
    assert _in_frame([0.0, 5.0], "linear", (-1.0, 10.0)).tolist() == [0.0, 5.0]


def test_a_runaway_simulation_does_not_set_the_frame():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    t = np.tile(np.arange(0.0, 100.0, 10.0), 5)
    value = np.exp(rng.normal(1.0, 1.0, len(t)))  # observations over several decades
    p = pd.DataFrame(dict(output="y", id=np.repeat(np.arange(5), 10), time=t, value=value, pop_pred=1.1 * value,
                          ind_pred_mode=value, iwres_mode=rng.normal(size=len(t)), npde=rng.normal(size=len(t))))
    v = pd.DataFrame(dict(output="y", t_mid=np.arange(5.0, 100.0, 10.0)))
    for q, level in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        v[f"obs_{q}"] = np.quantile(value, level)
        v[f"sim_{q}_low"], v[f"sim_{q}_high"] = v[f"obs_{q}"] / 2, v[f"obs_{q}"] * 2
    v.loc[v.index[-1], "sim_p90_high"] = 3e8  # one replicate ran away
    p.loc[p.index[-1], "value"] = 0.0  # one observation at zero

    a = _vpc(plt, p, v, "y").axes[0]
    assert a.get_yscale() == "log" and a.get_ylim()[1] == pytest.approx(3 * value[:-1].max())
    assert "leaves the frame" in a.get_title() and "1 observations at or below 0" in a.get_title()

    p["pop_pred"] = p["pop_pred"].where(p.index > 1, 0.0)  # two predictions at zero
    a = _goodness_of_fit(plt, p, "y").axes[0]
    assert a.get_xscale() == a.get_yscale() == "log" and a.get_xlim() == a.get_ylim()
    assert any("2 predictions at or below 0" in text.get_text() for text in a.texts)
    plt.close("all")
