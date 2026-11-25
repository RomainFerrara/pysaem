"""Command line: pysaem fit RUN.toml, or pysaem fit MODEL.json DATA.csv [options];
pysaem plot RESULT.json for the diagnostic plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .api import fit
from . import runfile
from .data import DataError, Dataset
from .engine import FitError
from .model import Model, ModelError
from .population import ErrorModel, Population
from .saem import Settings


def _pairs(text: str) -> dict:
    return {k.strip(): float(v) for k, v in (item.split("=") for item in text.split(",") if item)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pysaem", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit", help="estimate a population model by SAEM")
    f.add_argument("model", help="run file (TOML), or model specification (JSON)")
    f.add_argument("data", nargs="?", help="data, one row per record (ID, TIME, DV, AMT, ...)")
    f.add_argument("--population", help="population model, JSON (overrides the options below)")
    f.add_argument("--init", default="", help="initial typical values, k=0.1,V=40")
    f.add_argument("--omega", type=float, default=1.0, help="initial omega of every parameter")
    f.add_argument("--no-iiv", default="", help="parameters without variability, comma-separated")
    f.add_argument("--error", default="combined1",
                   help="constant, proportional, combined1, combined2 or exponential; a "
                        "hazard output always takes hazard")
    f.add_argument("--output-column", help="column naming the observation type")
    f.add_argument("--regressors", default="", help="regressor columns, comma-separated")
    f.add_argument("--covariates", default="", help="covariate columns, comma-separated")
    f.add_argument("--interpolate", default="", help="regressors read linearly between records")
    f.add_argument("--exploratory", type=int, default=500)
    f.add_argument("--smoothing", type=int, default=200)
    f.add_argument("--no-autostop", action="store_true")
    f.add_argument("--seed", type=int, default=123456)
    f.add_argument("--out", help="result file, JSON (default: pysaem_result.json, or next to the run file)")
    f.add_argument("--quiet", action="store_true")
    g = sub.add_parser("plot", help="diagnostic plots of a fit, in one PDF")
    g.add_argument("result", help="result file written by pysaem fit (JSON)")
    g.add_argument("--out", help="PDF file (default: next to the result, same name)")
    args = parser.parse_args(argv)
    if args.command == "plot":
        return _plot(args)

    try:
        if args.model.endswith(".toml"):
            run = runfile.load(args.model)
            model, data, population, settings = (run[k] for k in ("model", "data", "population",
                                                                  "settings"))
            args.out = args.out or run["out"]
        else:
            model, data, population, settings = _from_options(args)
        result = fit(model, data, population, settings, progress=not args.quiet)
    except (ModelError, FitError, DataError, ValueError, OSError) as e:
        print(f"pysaem: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    Path(args.out).write_text(result.to_json())
    print(result.summary())
    return 0


def _plot(args) -> int:
    from .plots import diagnostics
    out = args.out or str(Path(args.result).with_suffix(".pdf"))
    try:
        diagnostics(args.result, out)
    except (ImportError, OSError, ValueError, KeyError) as e:
        print(f"pysaem: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"plots written to {out}")
    return 0


def _from_options(args):
    """Model JSON, data file and the population and settings given as options."""
    if args.data is None:
        raise ValueError("a model JSON needs a data file")
    # the model file is the user's: Python code in it is read
    model = Model.from_json(Path(args.model).read_text(), python=True)
    if args.interpolate:
        model = Model.from_dict(model.to_dict() | {"interpolated": args.interpolate.split(",")},
                                python=True)
    frame = pd.read_csv(args.data, sep=None, engine="python")
    data = Dataset.from_records(
        frame, output=args.output_column,
        regressors=[c for c in args.regressors.split(",") if c],
        covariates=[c for c in args.covariates.split(",") if c])
    if args.population:
        population = Population.from_dict(model, json.loads(Path(args.population).read_text()))
    else:
        population = Population.build(
            model, init=_pairs(args.init), omega=args.omega,
            no_iiv=[p for p in args.no_iiv.split(",") if p], error=ErrorModel(args.error))
    settings = Settings(n_exploratory=args.exploratory, n_smoothing=args.smoothing,
                        exploratory_autostop=not args.no_autostop,
                        smoothing_autostop=not args.no_autostop, seed=args.seed)
    args.out = args.out or "pysaem_result.json"
    return model, data, population, settings


if __name__ == "__main__":
    sys.exit(main())
