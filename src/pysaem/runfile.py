"""A whole fit described in one TOML file: data, model, population and settings."""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd

from .data import Dataset
from .model import Model
from .population import ErrorModel, Parameter, Population
from .saem import Settings

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_PARAMETER = {f.name for f in fields(Parameter)}
_SETTINGS = {f.name for f in fields(Settings)}


def load(path) -> dict:
    """Read a run file; returns model, data, population, settings and the result path.

    Paths in the file are relative to it. A parameter given as a number is its initial
    value; covariates may be listed by name, their coefficients then start at zero.
    """
    path = Path(path)
    spec = tomllib.loads(path.read_text())
    unknown = set(spec) - {"data", "model", "population", "settings", "out"}
    if unknown:
        raise ValueError(f"{path.name}: unknown section(s) {sorted(unknown)}")
    pop = dict(spec.get("population", {}))
    error = pop.pop("error", "combined1")
    correlations = pop.pop("correlations", None)

    m = dict(spec.get("model", {}))
    m.setdefault("parameters", list(pop))
    if isinstance(m.get("python"), str):
        # a model written as Python code: inline, or a .py file next to the run file; the
        # initial values of the population guide the check of what pysaem reads
        if m["python"].strip().endswith(".py") and "\n" not in m["python"].strip():
            m["python"] = (path.parent / m["python"].strip()).read_text()
        m.setdefault("values", {k: (v if isinstance(v, (int, float)) else v.get("init", 1.0))
                                for k, v in pop.items()
                                if k in m["parameters"] and isinstance(v, (int, float, dict))})
    model = Model.from_dict(m, python=True)

    params = {}
    for name in model.parameters:
        p = pop.get(name, {})
        p = {"init": p} if isinstance(p, (int, float)) else dict(p)
        bad = set(p) - _PARAMETER
        if bad:
            raise ValueError(f"{path.name}: parameter {name}: unknown key(s) {sorted(bad)}")
        if isinstance(p.get("covariates"), list):
            p["covariates"] = {c: 0.0 for c in p["covariates"]}
        if "bounds" in p:
            p["bounds"] = tuple(p["bounds"])
        if p.get("prior") is not None:
            p["prior"] = tuple(p["prior"])
        params[name] = Parameter(**p)
    by_output = isinstance(error, dict) and not {"kind", "a", "b"} & set(error)
    errors = {}
    for o in model.outputs:
        # a hazard output takes 'hazard' unless the file names its error model
        if by_output:
            e = error.get(o, "hazard" if o in model.hazards else "combined1")
        else:
            e = "hazard" if o in model.hazards else error
        errors[o] = _error(e)
    population = Population(model, params, errors, correlations)

    s = spec.get("settings", {})
    bad = set(s) - _SETTINGS
    if bad:
        raise ValueError(f"{path.name}: unknown setting(s) {sorted(bad)}")
    # the data last: a mistake in the file is reported before a long read
    d = spec.get("data")
    d = {"file": d} if isinstance(d, str) else dict(d or {})
    if "file" not in d:
        raise ValueError(f"{path.name}: the data file is missing")
    frame = pd.read_csv(path.parent / d.pop("file"), sep=None, engine="python")
    data = Dataset.from_records(frame, **d)
    out = path.parent / spec.get("out", path.with_suffix(".json").name)
    return {"model": model, "data": data, "population": population, "settings": Settings(**s),
            "out": out}


def _error(e) -> ErrorModel:
    return ErrorModel(kind=e) if isinstance(e, str) else ErrorModel(**e)
