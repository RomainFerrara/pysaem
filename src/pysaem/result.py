"""Result of a fit: plain data, serialisable to JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

from ._version import __version__

_TABLES = ("trace", "individual", "predictions", "vpc", "correlation", "covariate_screen",
           "pcvpc", "vpc_hazards", "draws", "curves")


@dataclass
class FitResult:
    """Everything a fit returns, as plain data: `summary()` prints it, `to_json()` and
    `from_json()` save and read it."""

    #: The model, as `Model.to_dict()` gives it.
    model: dict
    #: The population model at the estimates, as `Population.to_dict()` gives it.
    population: dict
    #: The settings of the fit, as a dict.
    settings: dict
    #: Estimates by name: typical values (`CL_pop`), covariate effects, standard deviations of
    #: the random effects (`omega_CL`), correlations and error parameters.
    parameters: dict
    #: Standard error of each estimate.
    se: dict
    #: Relative standard error of each estimate, in percent.
    rse: dict
    #: 95 % confidence interval of each estimate, built on the scale on which it is
    #: unconstrained.
    ci: dict
    #: "ok", or why some standard errors are missing or imprecise.
    fim_status: str
    #: Log-likelihood estimated by importance sampling.
    ll: float
    #: Monte Carlo standard error of `ll`.
    ll_se: float
    #: -2 log-likelihood.
    ofv: float
    #: Akaike information criterion.
    aic: float
    #: Bayesian information criterion, with the number of subjects.
    bic: float
    #: BICc of Delattre, Lavielle and Poursat (2014).
    bicc: float
    #: Number of subjects.
    n_subjects: int
    #: Number of observations.
    n_obs: int
    #: Number of estimated parameters.
    n_parameters: int
    #: The stopping rules of the iterations were met, every estimate is finite and no typical
    #: value runs off (see `messages`). This describes the trajectory of the iterations and does
    #: not prove convergence.
    converged: bool
    #: Messages of the fit: parameters that look unidentifiable, imprecise standard errors,
    #: correlated random effects, dropped subjects.
    messages: list
    #: Shrinkage of each random effect, in percent.
    shrinkage: dict
    #: Tests of the residuals of each output.
    tests: dict
    #: Seconds spent in each part of the fit.
    timings: dict
    #: Iterations of SAEM: one row per iteration, with the phase and the convergence indicator.
    trace: pd.DataFrame
    #: Individual parameters: conditional modes and means, and random effects, one row per
    #: subject.
    individual: pd.DataFrame
    #: One row per observation: population and individual predictions, residuals, NPDE, and the
    #: observation as the prediction-corrected check reads it (`pcvpc_value`).
    predictions: pd.DataFrame
    #: Visual predictive check: percentiles of the observations and of the simulated replicates
    #: by time bin.
    vpc: pd.DataFrame
    #: Correlation matrix of the estimates.
    correlation: pd.DataFrame | None = None
    #: Association of each random effect with each covariate of the data.
    covariate_screen: pd.DataFrame | None = None
    #: The iterations were stopped by the time budget (`Settings.on_budget`).
    provisional: bool = False
    #: One row per start when `Settings.n_starts` > 1.
    starts: list = field(default_factory=list)
    #: Subjects solved by the fit: in all, in closed form, implicitly, failed.
    solver: dict = field(default_factory=dict)
    #: Prediction-corrected visual predictive check (Bergstrand et al. 2011): the table of `vpc`
    #: with observations and replicates corrected for the population prediction of each record.
    pcvpc: pd.DataFrame | None = None
    #: Hazard outputs: survival or mean number of events against the simulated replicates.
    vpc_hazards: pd.DataFrame | None = None
    #: Hazard outputs: the events of the data.
    hazard_events: dict = field(default_factory=dict)
    #: Individual parameters drawn from their conditional distribution.
    draws: pd.DataFrame | None = None
    #: Individual predictions at the conditional modes on a fine time grid, for plots.
    curves: pd.DataFrame | None = None
    #: Version of PySAEM that made the fit.
    version: str = field(default=__version__)

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in _TABLES:
                v = None if v is None else {"columns": [str(c) for c in v.columns],
                                            "index": [str(i) for i in v.index]
                                            if f.name == "correlation" else None,
                                            "data": _plain(v.to_numpy().tolist())}
            out[f.name] = _plain(v)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "FitResult":
        kw = dict(d)
        # a result saved before its version was recorded does not take the current one
        kw.setdefault("version", "unknown")
        for name in _TABLES:
            t = kw.get(name)
            if t is not None:
                kw[name] = pd.DataFrame(t["data"], columns=t["columns"], index=t["index"])
        kw["ci"] = {k: tuple(v) for k, v in kw["ci"].items()}
        return cls(**kw)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "FitResult":
        return cls.from_dict(json.loads(text))

    def summary(self) -> str:
        lines = [f"{'parameter':<24}{'estimate':>12}{'se':>12}{'rse %':>9}"]
        for name, value in self.parameters.items():
            lines.append(f"{name:<24}{value:>12.5g}{self.se[name]:>12.3g}{self.rse[name]:>9.1f}")
        lines += ["", f"-2LL {self.ofv:.2f}   AIC {self.aic:.2f}   BIC {self.bic:.2f}   "
                      f"BICc {self.bicc:.2f}",
                  f"subjects {self.n_subjects}   observations {self.n_obs}   "
                  f"converged {self.converged}"]
        if self.hazard_events:
            lines.append("events   " + "   ".join(f"{k} {v}"
                                                   for k, v in self.hazard_events.items()))
        return "\n".join(lines + self.messages)


def _plain(v):
    """NumPy scalars and non-finite floats made JSON-safe (NaN becomes None)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v
