"""Checks that refit the model: convergence from several starts, likelihood profile, bootstrap."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from .api import _perturbed, fit
from .data import Dataset
from .engine import FitError
from .model import Model
from .population import Population
from .saem import Settings


def _quick(st: Settings | None, **changes) -> Settings:
    """The settings of a refit: no standard errors, no simulations, one start."""
    return replace(st or Settings(), fim=False, simulations=False, n_starts=1, **changes)


def _row(result, names) -> dict:
    if result is None:
        return {"converged": False, "ofv": np.nan, **{n: np.nan for n in names}}
    return {"converged": result.converged, "ofv": result.ofv,
            **{n: result.parameters.get(n, np.nan) for n in names}}


def _names(model, data, population, st) -> list:
    """The names of the estimates, from the engine, without a fit."""
    from .api import _engine
    from .saem import estimates
    return list(estimates(_engine(model, data, population, st)))


@dataclass
class Convergence:
    """Fits from several seeds and starting values, one row per run."""

    table: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Per estimate, the lowest and highest values over the converged runs and their
        range as a percentage of the mean; the -2LL is listed as a range in points."""
        ok = self.table[self.table["converged"]]
        rows = []
        for name in [c for c in self.table.columns if c not in ("run", "seed", "converged")]:
            v = ok[name].to_numpy(float)
            if name == "ofv":
                rows.append({"estimate": name, "lowest": v.min(), "highest": v.max(),
                             "range": v.max() - v.min()})
            else:
                mean = np.abs(v).mean()
                rows.append({"estimate": name, "lowest": v.min(), "highest": v.max(),
                             "range": 100 * (v.max() - v.min()) / mean if mean > 0 else np.nan})
        return pd.DataFrame(rows)


def convergence(model: Model, data: Dataset, population: Population, settings: Settings | None = None,
                *, n: int = 5, spread: float = 0.5, seed: int | None = None, on_fit=None) -> Convergence:
    """Fit n times, each from a different seed and, with spread > 0, from starting values moved
    on their Gaussian scale by spread times a standard normal draw (the first run starts as
    given). Runs that end at the same -2LL and estimates show that the fit does not depend on
    the seed or the start; one that ends higher shows a local maximum.

    on_fit(k, result) is called after each run, with None when it failed.
    """
    st = _quick(settings)
    base = seed if seed is not None else st.seed
    rng = np.random.default_rng([base, 11])
    names = _names(model, data, population, st)
    rows = []
    for k in range(n):
        pop = population if k == 0 or spread == 0 else _perturbed(population, spread, rng)
        try:
            result = fit(model, data, pop, replace(st, seed=base + k), progress=False)
        except FitError:
            result = None
        if on_fit:
            on_fit(k, result)
        rows.append({"run": k + 1, "seed": base + k, **_row(result, names)})
    return Convergence(pd.DataFrame(rows))


@dataclass
class Profile:
    """The -2LL against the values held for one estimate."""

    parameter: str
    table: pd.DataFrame

    @property
    def minimum(self) -> float:
        return float(np.nanmin(self.table["ofv"]))

    def interval(self, level: float = 0.95) -> tuple:
        """The values where the profile rises by the chi-square quantile of level above its
        lowest point (Venzon and Moolgavkar 1988), by linear interpolation; nan on a side the
        grid does not reach."""
        t = self.table.dropna(subset=["ofv"]).sort_values("value")
        v, f = t["value"].to_numpy(float), t["ofv"].to_numpy(float)
        bar = self.minimum + float(stats.chi2.isf(1 - level, 1))
        best = int(np.argmin(f))
        lo = hi = np.nan
        for k in range(best, 0, -1):
            if f[k - 1] >= bar:
                lo = v[k] + (v[k - 1] - v[k]) * (bar - f[k]) / (f[k - 1] - f[k])
                break
        for k in range(best, len(f) - 1):
            if f[k + 1] >= bar:
                hi = v[k] + (v[k + 1] - v[k]) * (bar - f[k]) / (f[k + 1] - f[k])
                break
        return lo, hi


def profile(model: Model, data: Dataset, population: Population, settings: Settings | None = None,
            *, parameter: str, values=None, se: float | None = None, n_points: int = 9,
            span: float = 3.0, on_fit=None) -> Profile:
    """Likelihood profile of one estimate: a typical value, "X_pop", or a standard deviation,
    "omega_X". The estimate is held at each value in turn, the others are refitted, and the
    -2LL by importance sampling is recorded with its Monte Carlo standard error.

    values gives the grid. Without it, se (the standard error of the estimate, from a fit) sets
    a grid of n_points from minus to plus span standard errors around the value in population,
    on the log scale for a lognormal typical value or an omega.
    """
    st = _quick(settings)
    name, kind = _target(population, parameter)
    p = population.parameters[name]
    centre = p.omega if kind == "omega" else p.init
    if values is None:
        if se is None:
            raise ValueError("give values, or se to build the grid around the estimate")
        if kind == "omega" or p.distribution == "lognormal":
            values = centre * np.exp(np.linspace(-span, span, n_points) * se / centre)
        else:
            values = centre + np.linspace(-span, span, n_points) * se
            if p.distribution == "logitnormal":
                values = np.clip(values, *p.bounds)
    rows = []
    for k, v in enumerate(np.asarray(values, float)):
        held = Population.from_dict(model, population.to_dict())
        q = held.parameters[name]
        if kind == "omega":
            q.omega, q.omega_fixed = float(v), True
        else:
            q.init, q.fixed = float(v), True
        try:
            result = fit(model, data, held, st, progress=False)
            row = {"value": v, "ofv": result.ofv, "ofv_se": 2 * result.ll_se,
                   "converged": result.converged}
        except FitError:
            result, row = None, {"value": v, "ofv": np.nan, "ofv_se": np.nan, "converged": False}
        if on_fit:
            on_fit(k, result)
        rows.append(row)
    return Profile(parameter, pd.DataFrame(rows))


def _target(population: Population, parameter: str) -> tuple:
    for name in population.parameters:
        if parameter == f"{name}_pop":
            return name, "typical"
        if parameter == f"omega_{name}" and population.parameters[name].iiv:
            return name, "omega"
    raise ValueError(f"{parameter!r} is not a typical value (X_pop) or an omega (omega_X) "
                     f"of the parameters {list(population.parameters)}")


@dataclass
class Bootstrap:
    """Estimates of the replicates, one row each, and the subjects each replicate drew."""

    estimates: pd.DataFrame
    samples: list
    failed: int

    def summary(self, level: float = 0.95) -> pd.DataFrame:
        """Per estimate, the mean and sd over the converged replicates and the percentile
        interval of the level given."""
        ok = self.estimates[self.estimates["converged"]]
        q = 100 * (1 - level) / 2
        rows = []
        for name in [c for c in ok.columns if c not in ("replicate", "converged", "ofv")]:
            v = ok[name].to_numpy(float)
            rows.append({"estimate": name, "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                         "lower": np.percentile(v, q), "upper": np.percentile(v, 100 - q)})
        return pd.DataFrame(rows)


def bootstrap(model: Model, data: Dataset, population: Population, settings: Settings | None = None,
              *, n: int = 200, seed: int = 0, strata: str | None = None, loglik: bool = False,
              on_fit=None) -> Bootstrap:
    """Case bootstrap: n datasets of the same size, each drawing the subjects with replacement
    (Efron 1979), refitted from population. The sd and the percentiles of the estimates over
    the replicates estimate their uncertainty without the information matrix (Thai et al.
    2014). With strata, a covariate of the data, subjects are drawn within each of its levels,
    so that every replicate keeps the levels' sizes.

    The replicates are fitted without standard errors and, unless loglik is set, without
    likelihood. on_fit(k, result) is called after each one, with None when it failed.
    """
    st = _quick(settings, loglik=loglik)
    rng = np.random.default_rng([seed, 13])
    N = data.n_subjects
    if strata is None:
        groups = [np.arange(N)]
    else:
        if strata not in data.covariates:
            raise ValueError(f"strata {strata!r} is not a covariate of the data")
        x = np.asarray(data.covariates[strata])
        groups = [np.flatnonzero(x == level) for level in np.unique(x)]
    names = _names(model, data, population, st)
    rows, samples, failed = [], [], 0
    for k in range(n):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        samples.append(idx)
        try:
            result = fit(model, data.take(idx), population, replace(st, seed=st.seed + k),
                         progress=False)
        except FitError:
            result = None
            failed += 1
        if on_fit:
            on_fit(k, result)
        rows.append({"replicate": k + 1, **_row(result, names)})
    return Bootstrap(pd.DataFrame(rows), samples, failed)
