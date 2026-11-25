"""Diagnostic plots of a fit, in one PDF: goodness of fit, visual predictive check, individual
fits and iterations; for a hazard output, its survival or mean number of events against the
replicates. Needs matplotlib (pip install "pysaem[plot]")."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .result import FitResult

PER_PAGE = 20     # subjects per page of individual fits


def diagnostics(result: FitResult | str | Path, out: str | Path) -> Path:
    """Write the diagnostic plots of a fit (a FitResult or its JSON file) to the PDF out."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as e:
        raise ImportError('the plots need matplotlib: pip install "pysaem[plot]"') from e
    if not isinstance(result, FitResult):
        result = FitResult.from_json(Path(result).read_text())
    out = Path(out)
    pred = result.predictions
    labels = list(dict.fromkeys(pred["output"]))
    outputs = (result.model or {}).get("outputs", [])
    if "variable" in pred:
        name = dict(zip(pred["output"], pred["variable"]))
    else:
        # results written before the variable column: a single observation type is the single output
        name = {labels[0]: outputs[0]} if len(labels) == 1 and len(outputs) == 1 else {}
    hazards = (result.model or {}).get("hazards", {})
    # records of hazard outputs have their own page: no residual, no individual curve
    on = pred["output"].map(lambda o: name.get(o, o) in hazards).astype(bool)
    vh = result.vpc_hazards
    with PdfPages(out) as pdf:
        for output, p in pred[~on].groupby("output", sort=False):
            pdf.savefig(_goodness_of_fit(plt, p, name.get(output, output)))
            for table in (result.vpc, result.pcvpc):
                v = table[table["output"] == output] if table is not None and len(table) else None
                if v is not None and len(v):
                    pdf.savefig(_vpc(plt, p, v, name.get(output, output)))
        if vh is not None and len(vh):
            for output, v in vh.groupby("output", sort=False):
                pdf.savefig(_vpc_hazard(plt, v, name.get(output, output)))
        ids = list(dict.fromkeys(pred.loc[~on, "id"]))
        for start in range(0, len(ids), PER_PAGE):
            pdf.savefig(_individual(plt, pred[~on], ids[start:start + PER_PAGE]))
        if len(result.trace):
            pdf.savefig(_trace(plt, result.trace))
        plt.close("all")
    return out


def _observed(p):
    """Observations above the limit of quantification, and censored ones."""
    cens = p["cens"].fillna(0) != 0 if "cens" in p else pd.Series(False, index=p.index)
    return p[~cens], p[cens]


def _frame(values):
    """Scale and limits of an axis from the observations alone, so that no prediction or simulation
    sets the frame: log when the positive observations span at least a decade (and are at least half
    of them), linear otherwise. (scale, None) without a finite observation."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    pos = v[v > 0]
    if len(pos) and pos.max() / pos.min() >= 10 and 2 * len(pos) >= len(v):
        return "log", (float(pos.min() / 3), float(pos.max() * 3))
    if not len(v):
        return "linear", None
    span = float(v.max() - v.min()) or max(abs(float(v.max())), 1.0)
    return "linear", (float(v.min() - 0.08 * span), float(v.max() + 0.08 * span))


def _in_frame(x, scale, lim):
    """Values as drawn in a frame: on a log scale, those at or below zero sit at its lower edge."""
    x = np.asarray(x, float)
    return np.where(x > 0, x, lim[0]) if scale == "log" and lim else x


def _outside(scale, lim, **series):
    """Says which values (named series) the frame leaves out or pushes to its lower edge; empty if none."""
    parts = []
    for name, x in series.items():
        x = np.asarray(x, float)
        edge = (x <= 0) if scale == "log" else np.zeros(len(x), bool)
        beyond = ((x < lim[0]) | (x > lim[1])) & ~edge
        if beyond.any():
            parts.append(f"{int(beyond.sum())} {name} beyond the frame")
        if edge.any():
            parts.append(f"{int(edge.sum())} {name} at or below 0, drawn at the lower edge")
    return "; ".join(parts)


def _goodness_of_fit(plt, p, output):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    obs, cens = _observed(p)
    scale, lim = _frame(p["value"])
    for a, col, label in ((ax[0, 0], "pop_pred", "population prediction"),
                          (ax[0, 1], "ind_pred_mode", "individual prediction (mode)")):
        a.scatter(_in_frame(obs[col], scale, lim), _in_frame(obs["value"], scale, lim), s=8, alpha=0.6)
        if len(cens):
            a.scatter(_in_frame(cens[col], scale, lim), _in_frame(cens["value"], scale, lim), s=8, marker="x",
                      color="tab:red", label="censored")
            a.legend(fontsize=8)
        if lim:
            a.plot(lim, lim, color="k", lw=0.8)
            a.set_xscale(scale)
            a.set_yscale(scale)
            a.set_xlim(lim)
            a.set_ylim(lim)
            note = _outside(scale, lim, predictions=p[col], observations=p["value"])
            if note:
                a.text(0.02, 0.98, note, transform=a.transAxes, va="top", fontsize=7)
        a.set_xlabel(label)
        a.set_ylabel("observation")
    for a, col, label in ((ax[1, 0], "iwres_mode", "IWRES"), (ax[1, 1], "npde", "NPDE")):
        y = obs[col]
        if y.notna().any():
            a.scatter(obs["time"], y, s=8, alpha=0.6)
            for level, style in ((0, "-"), (-1.96, "--"), (1.96, "--")):
                a.axhline(level, color="k", lw=0.8, ls=style)
        a.set_xlabel("time")
        a.set_ylabel(label)
    fig.suptitle(f"Goodness of fit: {output}")
    fig.tight_layout()
    return fig


def _vpc(plt, p, v, output):
    fig, a = plt.subplots(figsize=(10, 6))
    obs, _ = _observed(p)
    form = v["correction"].iloc[0] if "correction" in v else "none"
    # a corrected check shows the observations as it read them
    read = p["pcvpc_value"] if form != "none" and "pcvpc_value" in p else p["value"]
    scale, lim = _frame(read)
    a.scatter(obs["time"], _in_frame(read[obs.index], scale, lim), s=6, color="0.6", alpha=0.6,
              label="observations")
    for q, color in (("p10", "tab:blue"), ("p50", "tab:red"), ("p90", "tab:blue")):
        a.fill_between(v["t_mid"], _in_frame(v[f"sim_{q}_low"], scale, lim), _in_frame(v[f"sim_{q}_high"], scale, lim),
                       color=color, alpha=0.2, lw=0)
        a.plot(v["t_mid"], _in_frame(v[f"obs_{q}"], scale, lim), color=color, lw=1.5,
               label=f"observed {q[1:]}th percentile")
    notes = []
    left = int((read.isna() & p["value"].notna()).sum()) if form != "none" else 0
    if left:
        notes.append(f"{left} observations left out: predicted at 0 or below 1/1000 of their bin")
    if lim:
        a.set_yscale(scale)
        a.set_ylim(lim)
        if (v[[f"sim_{q}_high" for q in ("p10", "p50", "p90")]].to_numpy(float) > lim[1]).any():
            notes.append("the simulated band leaves the frame of the observations")
        notes += [s for s in [_outside(scale, lim, observations=read)] if s]
    a.set_xlabel("time")
    a.set_ylabel(output if form == "none" else f"{output}, prediction-corrected")
    a.legend(fontsize=8)
    kind = "Visual predictive check" if form == "none" else \
        f"Prediction-corrected visual predictive check ({form})"
    a.set_title(f"{kind}: {output} (bands: 90% of the simulated percentiles)"
                + "".join(f"\n{s}" for s in notes), fontsize=10)
    fig.tight_layout()
    return fig


def _vpc_hazard(plt, v, output):
    fig, a = plt.subplots(figsize=(10, 6))
    survival = v["measure"].iloc[0] == "survival"
    a.fill_between(v["time"], v["sim_p5"], v["sim_p95"], step="post", color="tab:blue", alpha=0.2,
                   lw=0, label="replicates, 5th to 95th percentile")
    a.step(v["time"], v["sim_p50"], where="post", color="tab:blue", lw=1, ls="--",
           label="replicates, median")
    a.step(v["time"], v["obs"], where="post", color="k", lw=1.5, label="observed")
    a.set_xlabel("time")
    a.set_ylabel("probability of no event" if survival else "mean number of events")
    a.legend(fontsize=8)
    a.set_title(f"{'Kaplan-Meier' if survival else 'Mean number of events'}: {output} "
                "(band: 90% of the simulated curves)")
    fig.tight_layout()
    return fig


def _individual(plt, pred, ids):
    n = len(ids)
    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig, ax = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.1 * rows), squeeze=False, sharex=True)
    for k, i in enumerate(ids):
        a = ax.flat[k]
        subject = pred[pred["id"] == i]
        scale, lim = _frame(subject["value"])
        for output, p in subject.groupby("output", sort=False):
            p = p.sort_values("time")
            obs, cens = _observed(p)
            line, = a.plot(p["time"], _in_frame(p["ind_pred_mode"], scale, lim), lw=1)
            a.plot(p["time"], _in_frame(p["pop_pred"], scale, lim), lw=0.8, ls="--", color=line.get_color())
            a.scatter(obs["time"], _in_frame(obs["value"], scale, lim), s=6, color=line.get_color())
            a.scatter(cens["time"], _in_frame(cens["value"], scale, lim), s=6, marker="x", color="tab:red")
        if lim:
            a.set_yscale(scale)
            a.set_ylim(lim)
        a.set_title(f"subject {i}", fontsize=8)
        a.tick_params(labelsize=6)
    for a in ax.flat[n:]:
        a.axis("off")
    fig.suptitle("Individual fits: observations, individual (solid) and population (dashed) "
                 "predictions at the observation times", fontsize=9)
    fig.tight_layout()
    return fig


def _trace(plt, trace):
    names = [c for c in trace.columns if c not in ("iteration", "phase")]
    cols = min(4, len(names))
    rows = math.ceil(len(names) / cols)
    fig, ax = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.0 * rows), squeeze=False, sharex=True)
    end = trace.loc[trace["phase"] == trace["phase"].iloc[0], "iteration"].max()
    for a, name in zip(ax.flat, names):
        a.plot(trace["iteration"], trace[name], lw=0.8)
        a.axvline(end, color="k", lw=0.6, ls="--")
        a.set_title(name, fontsize=8)
        a.tick_params(labelsize=6)
    for a in ax.flat[len(names):]:
        a.axis("off")
    fig.suptitle("Iterations (dashed: end of the exploratory phase)", fontsize=9)
    fig.tight_layout()
    return fig
