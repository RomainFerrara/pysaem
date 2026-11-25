"""Bayesian individualisation: a subject's parameters given a fixed population."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import post, saem
from .data import Dataset
from .engine import Engine
from .model import Model
from .population import Population
from .saem import Settings


@dataclass
class Individualization:
    """Posterior of the individual parameters of each subject, natural scale."""

    model: Model
    data: Dataset
    names: list
    mode: pd.DataFrame          # one row per subject
    mean: pd.DataFrame
    samples: np.ndarray         # (draws, subjects, parameters)
    predictions: pd.DataFrame   # at the observation records, from the mode
    _engine: Engine

    def forecast(self, times, level: float = 0.9, noise: bool = True, seed: int = 0):
        """Prediction at the mode, predictive median, mean and interval of every output at
        times, per subject; each subject's solve starts where it does on its own records, and
        nothing is predicted before that start (NaN).

        The interval carries the uncertainty on the individual parameters and, with noise, the
        residual error. A hazard output is predicted from the subject's last record of it,
        t_ref, given its data (Rizopoulos 2011): the probability of no event by t,
        exp(-(H(t) - H(t_ref))), on terminal and single outputs, the expected number of events
        H(t) - H(t_ref) on repeated ones; without residual error, and NaN before t_ref, after a
        terminal event and after the event of a single output.
        """
        eng, d, model = self._engine, self.data, self.model
        times = np.unique(np.asarray(times, float))
        ref, stop = _references(eng)
        grid = d.at_times([np.union1d(times, r[np.isfinite(r)]) for r in ref])
        prob = post.problem(eng, grid)
        draws = prob.predict(eng.rows(eng.phi(self.samples), grid))[0]
        best = prob.predict(eng.rows(eng.phi(self.mode[self.names].to_numpy()[None]), grid))[0]
        f, mode = prob.gather(draws), prob.gather(best)[0]
        col = np.array([model.outputs.index(o) for o in model.hazards], np.int64)
        on = np.isin(grid.obs_out, col)
        if noise:
            rng = np.random.default_rng(seed)
            v = eng.to_scale(f, grid.obs_out)
            noisy = eng.from_scale(
                v + eng.residual_sd(v, grid.obs_out) * rng.standard_normal(f.shape), grid.obs_out)
            f = np.where(on, f, noisy) if on.any() else noisy
        if on.any():
            j = np.flatnonzero(on)
            s = grid.obs_subject[j]
            k_of = np.full(len(model.outputs), -1)
            k_of[col] = np.arange(col.size)
            k = k_of[grid.obs_out[j]]
            g_ref = np.zeros(ref.shape, np.int64)
            for i in range(d.n_subjects):
                a, b = grid.grid_ptr[i], grid.grid_ptr[i + 1]
                known = np.isfinite(ref[i])
                g_ref[i, known] = a + np.searchsorted(grid.grid_time[a:b], ref[i, known])
            t = grid.obs_time[j]
            valid = (t >= ref[s, k]) & (t <= stop[s, k])
            repeated = np.array([model.hazards[o] == "repeated" for o in model.hazards])[k]

            def given(pred):
                x = pred[:, grid.obs_grid[j], col[k]] - pred[:, g_ref[s, k], col[k]]
                return np.where(valid, np.where(repeated, x, np.exp(-x)), np.nan)
            f[:, j], mode[j] = given(draws), given(best)[0]
        early = grid.obs_time < prob._t_start[grid.obs_subject]
        f[:, early], mode[early] = np.nan, np.nan
        keep = np.isin(grid.obs_time, times)
        with warnings.catch_warnings():
            # nothing is predicted before the start of the solve, nor a hazard output before its
            # last record
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(f[:, keep], [50 * (1 - level), 50 * (1 + level)], axis=0)
            median, mean = np.nanmedian(f[:, keep], axis=0), np.nanmean(f[:, keep], axis=0)
        out = grid.observations()[["id", "time", "output"]][keep].reset_index(drop=True)
        out["mode"], out["median"], out["mean"] = mode[keep], median, mean
        out["low"], out["high"] = lo, hi
        return out


def _references(eng: Engine):
    """Per subject and hazard output (model.hazards order): the last record, NaN without one,
    and the last time the output is predicted: its terminal event or that of another output,
    -inf after the event of a single output, inf otherwise."""
    d, model = eng.data, eng.model
    hz = model.hazards
    sub, t, out = d.obs_subject, d.obs_time, d.obs_out
    ref, stop = np.full((eng.N, len(hz)), np.nan), np.full((eng.N, len(hz)), np.inf)
    event = eng.on_hazard & (d.obs_y > 0)
    terminal = event & np.array([hz.get(o) == "terminal" for o in model.outputs])[out]
    dead = np.full(eng.N, np.inf)
    np.minimum.at(dead, sub[terminal], t[terminal])
    for k, (o, kind) in enumerate(hz.items()):
        on = out == model.outputs.index(o)
        last = np.full(eng.N, -np.inf)
        np.maximum.at(last, sub[on], t[on])
        ref[:, k] = np.where(np.isfinite(last), last, np.nan)
        stop[:, k] = dead
        if kind != "repeated":
            stop[sub[on & event], k] = -np.inf
    return ref, stop


def individualize(model: Model, population: Population, data: Dataset,
                  settings: Settings | None = None) -> Individualization:
    """Condition the random effects of the subjects in data on a fixed population.

    population carries the population values, for instance
    Population.from_dict(model, result.population) after a fit. Nothing is
    estimated at the population level; subjects may be new.
    """
    st = settings or Settings()
    frozen = Population.from_dict(model, copy.deepcopy(population.to_dict()))
    for p in frozen.parameters.values():
        p.fixed = p.omega_fixed = True
    for e in frozen.error.values():
        e.fixed = True
    eng = Engine(model, data, frozen, rtol=st.rtol, atol=st.atol, dynamic_range=st.dynamic_range,
                 method=st.method, sensitivities=st.sensitivities,
                 seed=st.seed)
    eng.check_initial()
    trace = saem.run(eng, st)
    cond = post.conditional(eng, st, trace, keep=st.n_conditional)
    mean = eng.mean()
    eta_mode = post.modes(eng, cond.best)
    names = list(eng.names)
    f_mode = eng.cumulative(eng.predict((mean + eta_mode)[None])[0])[0]
    pred = data.observations()
    pred["ind_pred_mode"] = f_mode
    v = eng.to_scale(f_mode, data.obs_out)
    pred["iwres_mode"] = (eng.obs_y - v) / eng.residual_sd(v, data.obs_out)

    def table(phi):
        t = pd.DataFrame(eng.psi(phi), columns=names)
        t.insert(0, "id", data.ids)
        return t
    return Individualization(model=model, data=data, names=names, mode=table(mean + eta_mode),
                             mean=table(mean + cond.mean), samples=eng.psi(mean + cond.samples),
                             predictions=pred, _engine=eng)
