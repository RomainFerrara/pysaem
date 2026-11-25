"""Population model: distribution of each parameter and residual error per output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .model import Model

DISTRIBUTIONS = ("normal", "lognormal", "logitnormal")
# exponential: constant error on the log of the observations and predictions; hazard: the
# records of a hazard output count events, without error parameter
ERROR_MODELS = ("constant", "proportional", "combined1", "combined2", "exponential", "hazard")


@dataclass
class Parameter:
    """Typical value, distribution and variability of one individual parameter.

    covariates maps a covariate name to the initial value of its coefficient on
    the transformed scale. With iov, the parameter also varies between the occasions
    of a subject: a random effect of sd gamma per occasion, on the transformed scale.
    prior, as (m, s), puts a normal prior on the typical value on its transformed scale,
    centred on the transform of m with sd s: a lognormal prior of median m for a
    lognormal parameter. The estimate is then the mode of the posterior.
    """

    init: float = 1.0
    distribution: str = "lognormal"
    iiv: bool = True
    omega: float = 1.0
    fixed: bool = False
    omega_fixed: bool = False
    bounds: tuple = (0.0, 1.0)
    covariates: dict = field(default_factory=dict)
    iov: bool = False
    gamma: float = 0.3
    gamma_fixed: bool = False
    prior: tuple | None = None


@dataclass
class ErrorModel:
    """Residual error sd: a, b*|f|, a + b*|f| or sqrt(a^2 + b^2 f^2); none for a hazard output."""

    kind: str = "combined1"
    a: float = 1.0
    b: float = 0.3
    fixed: bool = False

    @property
    def names(self) -> tuple:
        return {"constant": ("a",), "proportional": ("b",), "exponential": ("a",),
                "hazard": ()}.get(self.kind, ("a", "b"))


class Population:
    """Statistical model on top of a structural Model."""

    def __init__(self, model: Model, parameters: dict, error: dict, correlations=None,
                 correlation_values=None):
        """correlations lists blocks of parameters whose random effects are correlated,
        for instance [["CL", "V"]]; correlation_values gives initial correlations by
        name ("corr_CL_V"), zero when absent."""
        self.model = model
        self.correlations = [list(block) for block in (correlations or [])]
        self.correlation_values = dict(correlation_values or {})
        self.parameters = {k: v if isinstance(v, Parameter) else Parameter(**v)
                           for k, v in parameters.items()}
        self.error = {k: v if isinstance(v, ErrorModel) else ErrorModel(**v)
                      for k, v in error.items()}
        if list(self.parameters) != model.parameters:
            raise ValueError(f"population parameters {list(self.parameters)} do not match "
                             f"the model parameters {model.parameters}")
        if list(self.error) != model.outputs:
            raise ValueError(f"error models {list(self.error)} do not match the model "
                             f"outputs {model.outputs}")
        for name, p in self.parameters.items():
            if p.distribution not in DISTRIBUTIONS:
                raise ValueError(f"{name}: distribution must be one of {DISTRIBUTIONS}")
            lo, hi = p.bounds
            if p.distribution == "lognormal" and not p.init > 0:
                raise ValueError(f"{name}: a lognormal parameter needs a positive initial value")
            if p.distribution == "logitnormal" and not lo <= p.init <= hi:
                raise ValueError(f"{name}: initial value outside its bounds {p.bounds}")
            if p.distribution == "logitnormal":
                # a reported estimate can sit on a bound: the logit saturates beyond double precision
                p.init, _ = inside(p.distribution, p.bounds, float(p.init))
            if p.iiv and not p.omega > 0:
                raise ValueError(f"{name}: omega must be positive")
            if p.prior is not None:
                try:
                    m, s = (float(v) for v in p.prior)
                except (TypeError, ValueError):
                    raise ValueError(f"{name}: prior must be (m, s), a centre and a positive sd")
                if not s > 0:
                    raise ValueError(f"{name}: the sd of the prior must be positive")
                if p.distribution == "lognormal" and not m > 0:
                    raise ValueError(f"{name}: the centre of a lognormal prior must be positive")
                if p.distribution == "logitnormal" and not lo < m < hi:
                    raise ValueError(f"{name}: the centre of the prior must be inside {p.bounds}")
                p.prior = (m, s)
            if p.iov and not p.gamma > 0:
                raise ValueError(f"{name}: gamma must be positive")
        seen = set()
        for block in self.correlations:
            for name in block:
                if name not in self.parameters or not self.parameters[name].iiv:
                    raise ValueError(f"correlation block {block}: {name} has no random effect")
                if name in seen:
                    raise ValueError(f"{name} appears in two correlation blocks")
                seen.add(name)
            if len(block) < 2:
                raise ValueError(f"correlation block {block} needs at least two parameters")
        for name, e in self.error.items():
            if e.kind not in ERROR_MODELS:
                raise ValueError(f"output {name}: error model must be one of {ERROR_MODELS}")
            if any(not getattr(e, k) > 0 for k in e.names):
                raise ValueError(f"output {name}: error parameters must be positive")
            if name in model.hazards and e.kind != "hazard":
                raise ValueError(f"output {name} is a hazard output: its error model is "
                                 f"'hazard', not {e.kind!r}")
            if e.kind == "hazard" and name not in model.hazards:
                raise ValueError(f"output {name}: the error model 'hazard' is for hazard "
                                 f"outputs, and the model's hazards do not name {name}")

    @classmethod
    def build(cls, model: Model, *, init=None, distribution="lognormal", no_iiv=(), omega=1.0,
              error="combined1", covariates=None, correlations=None, iov=None,
              fixed=(), prior=None) -> "Population":
        """Population with one setting for all parameters, overridable by name.

        init, omega and distribution accept a value or a dict by parameter name;
        error accepts a kind, an ErrorModel or a dict by output name, and a hazard output
        takes 'hazard' unless the dict names its error model; covariates maps a parameter
        name to a list of covariate names; iov maps a parameter name to the initial sd of
        its inter-occasion random effect; no_iiv lists the parameters without random effect,
        fixed those whose typical value stays at init; prior maps a parameter name to the
        (m, s) of a prior on its typical value. A name that is not a parameter or an output of
        the model is refused.
        """
        def pick(value, name, default):
            return value.get(name, default) if isinstance(value, dict) else value

        no_iiv, fixed = _names(no_iiv), _names(fixed)
        for what, given in (("init", init), ("omega", omega), ("distribution", distribution),
                            ("covariates", covariates), ("iov", iov), ("no_iiv", no_iiv),
                            ("fixed", fixed), ("prior", prior)):
            _known(given if isinstance(given, (dict, list)) else (), model.parameters, what,
                   "parameter")
        if isinstance(error, dict):
            _known(error, model.outputs, "error", "output")

        params = {}
        for name in model.parameters:
            params[name] = Parameter(
                init=float(pick(init or {}, name, 1.0)),
                distribution=pick(distribution, name, "lognormal"),
                iiv=name not in no_iiv,
                omega=float(pick(omega, name, 1.0)),
                covariates={c: 0.0 for c in (covariates or {}).get(name, ())},
                iov=name in (iov or {}), gamma=float((iov or {}).get(name, 0.3)),
                fixed=name in fixed, prior=(prior or {}).get(name))
        errors = {}
        for out in model.outputs:
            if out in model.hazards:
                # an error given to every output is meant for the others
                e = error.get(out, "hazard") if isinstance(error, dict) else "hazard"
            else:
                e = pick(error, out, "combined1")
            errors[out] = ErrorModel(kind=e) if isinstance(e, str) else e
        return cls(model, params, errors, correlations)

    def to_dict(self) -> dict:
        return {"parameters": {k: asdict(v) for k, v in self.parameters.items()},
                "error": {k: asdict(v) for k, v in self.error.items()},
                "correlations": self.correlations,
                "correlation_values": self.correlation_values}

    @classmethod
    def from_dict(cls, model: Model, d: dict) -> "Population":
        return cls(model, d["parameters"], d["error"], d.get("correlations"),
                   d.get("correlation_values"))

    def error_name(self, output: str, which: str) -> str:
        # suffixed by the output when several outputs have error parameters
        return which if sum(bool(e.names) for e in self.error.values()) == 1 \
            else f"{which}_{output}"


def _names(names) -> list:
    return [names] if isinstance(names, str) else list(names)


def _known(names, known, what, kind):
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"{what}: {unknown} not among the {kind}s of the model {list(known)}")


def inside(distribution: str, bounds: tuple, value: float) -> tuple:
    """The value kept strictly inside the support of its distribution, and whether it had
    reached the edge: a typical value rounded onto the edge is a boundary estimate."""
    lo, hi = bounds
    if distribution == "logitnormal":
        kept = min(max(value, np.nextafter(lo, hi)), np.nextafter(hi, lo))
    elif distribution == "lognormal":
        kept = max(value, np.finfo(float).tiny)
    else:
        kept = value
    return float(kept), kept != value


def transform(kind: int, lo: float, hi: float, psi):
    """Natural scale to the Gaussian scale: 0 identity, 1 log, 2 logit on (lo, hi)."""
    psi = np.asarray(psi, float)
    if kind == 0:
        return psi
    if kind == 1:
        return np.log(psi)
    return np.log((psi - lo) / (hi - psi))


def inverse(kind: int, lo: float, hi: float, phi):
    phi = np.asarray(phi, float)
    if kind == 0:
        return phi
    if kind == 1:
        return np.exp(phi)
    return lo + (hi - lo) / (1.0 + np.exp(-phi))


def jacobian(kind: int, lo: float, hi: float, psi):
    """d psi / d phi, used to carry standard errors to the natural scale."""
    psi = np.asarray(psi, float)
    if kind == 0:
        return np.ones_like(psi)
    if kind == 1:
        return psi
    return (psi - lo) * (hi - psi) / (hi - lo)
