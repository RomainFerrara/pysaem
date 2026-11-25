"""Search of the covariate model: which covariate acts on which parameter.

Two searches, with effects linear on the Gaussian scale of the parameter:
- samba, the default: the covariate step of SAMBA (Prague and Lavielle 2022), which chooses
  the covariates of each parameter on draws of the individual parameters, then refits;
- stepwise: additions and removals led by the association between random effects and
  covariates, in the manner of COSSAC (Ayral et al. 2021), each one decided by a fit;
- learned: the loop of samba with any regressor in place of the linear regression; its
  cross-fitted prediction enters the model as one covariate per parameter.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

from .api import fit
from .data import Dataset
from .engine import _TKIND
from .model import Model
from .population import Population, inverse, transform
from .result import FitResult
from .saem import Settings

_NOISE = 2.0  # a BICc gap within two Monte Carlo standard errors of the difference is no gap
_POSITIVE = 2.0  # BICc gap of positive evidence (Kass and Raftery 1995)


@dataclass
class Search:
    """Outcome of a covariate search."""

    result: FitResult               # fit of the final model
    population: Population          # its population model, at the estimates
    effects: list                   # (parameter, covariate) pairs kept
    history: pd.DataFrame           # one row per model fitted
    notes: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)  # kept covariates swapped for correlated ones
    retention_removed: list = field(default_factory=list)  # effects that did not hold on their own
    start: FitResult | None = None   # fit of the model given (samba)
    near: pd.DataFrame | None = None  # samba: each covariate allowed, in or out, and the change of
    # the criterion of its parameter's regression if it went out or came in, at the last iteration;
    # effects the elimination removed are left out
    learners: dict = field(default_factory=dict)  # learned: fitted learner of each score kept
    scores: pd.DataFrame | None = None  # learned: id and the centred score of each parameter
    centres: dict = field(default_factory=dict)  # learned: mean subtracted from each score
    features: dict = field(default_factory=dict)  # learned: covariate columns each learner reads
    data: Dataset | None = None  # learned: the data with the score columns


def _with(model: Model, base: dict, effects: list, groups: dict | None = None) -> Population:
    """The population model of base carrying exactly these effects; a covariate group (several
    levels) stands for its indicator columns."""
    d = {**base, "parameters": {k: dict(v) for k, v in base["parameters"].items()}}
    for name, p in d["parameters"].items():
        p["covariates"] = {col: p["covariates"].get(col, 0.0) for q, c in effects if q == name
                           for col in (groups or {}).get(c, [c])}
    return Population.from_dict(model, d)


def search(model: Model, data: Dataset, population: Population, settings: Settings | None = None,
           *, method: str = "samba", **options) -> Search:
    """Covariate model by `samba` (default), `stepwise` or `learned`; options go to the method."""
    methods = {"samba": samba, "stepwise": stepwise, "learned": learned}
    if method not in methods:
        raise ValueError("method must be 'samba', 'stepwise' or 'learned'")
    return methods[method](model, data, population, settings, **options)


def supported(current: tuple, other: tuple, size_change: int) -> bool:
    """Whether the data support moving from the current model to the other, each given as (BICc,
    Monte Carlo standard error of its log-likelihood), the other having size_change more
    parameters. A model chosen after looking at the data needs more than a lower BICc: a larger
    or an equal one must lower it by _POSITIVE beyond two standard errors of the difference, a
    smaller one is taken unless it raises it by as much."""
    delta = other[0] - current[0]
    return delta < margin(current[1], other[1]) if size_change < 0 else delta < -margin(current[1], other[1])


def margin(se: float, other_se: float) -> float:
    """The BICc gap below which the data do not tell two models apart: _POSITIVE beyond two Monte
    Carlo standard errors of the difference, from the errors of their log-likelihoods."""
    return _POSITIVE + 2 * _NOISE * float(np.hypot(se, other_se))


def _draws(result: FitResult, n_obs: int) -> dict:
    """Draws of each parameter that kept its random effect, on its Gaussian scale, shape (L, subjects).

    L = min(10, ceil(max(500 / N, 5000 / n_obs))) draws per subject (Ayral et al. 2021, Appendix
    S1), taken evenly among the draws of the fit.
    """
    table = result.draws
    N = result.n_subjects
    n = int(table["draw"].max()) + 1
    L = min(n, 10, math.ceil(max(500 / N, 5000 / max(n_obs, 1))))
    keep = np.isin(table["draw"].to_numpy(), np.unique(np.linspace(0, n - 1, L).round()))
    out = {}
    for name, p in result.population["parameters"].items():
        if p["iiv"]:
            psi = table[name].to_numpy()[keep].reshape(-1, N)
            out[name] = transform(_TKIND[p["distribution"]], *p["bounds"], psi)
    return out


def samba(model: Model, data: Dataset, population: Population, settings: Settings | None = None,
          *, candidates: dict | None = None, max_iterations: int = 20, max_fits: int = 200,
          draws: str = "stack", eliminate: bool = True, on_step=None) -> Search:
    """Covariate model by the covariate step of SAMBA (Prague and Lavielle 2022).

    Each iteration fits the current model and takes L draws per subject of its individual
    parameters (`_draws`). For each parameter with a random effect separately, the covariates
    of a linear regression of its draws on the Gaussian scale minimise the mean over the draws of
    -2LL + log(N) k, k the number of covariates: over every subset of the covariates allowed when
    there are fewer than 11, by additions and removals from the current ones otherwise. The union
    of these choices is the next model, started from the estimates of the regressions. The search
    stops when the model does not change, comes back, or after max_iterations; the model of
    lowest BICc among those fitted is returned. candidates maps a parameter to the covariates it
    may take (by default every one). draws="mean" regresses instead the mean of the draws of each
    subject, with -2LL + log(N) k (the Remark of Prague and Lavielle).

    With eliminate, each effect of that model is then fitted out (a set of effects already fitted
    is not fitted again), and the removal the data allow (`supported`, from a fit that
    converged) of lowest BICc is made, until none is: the search keeps no effect without
    positive evidence, whatever its parameter.
    """
    if draws not in ("stack", "mean"):
        raise ValueError("draws must be 'stack' or 'mean'")
    st = settings or Settings()
    rows, fitted = [], {}

    def run(effects, base, action):
        key = frozenset(effects)
        if key not in fitted:
            if len(fitted) >= max_fits:
                raise RuntimeError(f"the covariate search reached its limit of {max_fits} fits")
            fitted[key] = fit(model, data, _with(model, base, sorted(effects), groups), st)
        r = fitted[key]
        rows.append({"action": action, "parameter": "", "covariate": "", "effects": _label(effects),
                     "ofv": r.ofv, "bicc": r.bicc, "decision": ""})
        if on_step:
            on_step(rows[-1])
        return r

    groups = data.covariate_groups
    # the columns of each covariate: a group has one indicator per level after the first
    columns = {c: [np.asarray(data.covariates[col], float) for col in groups.get(c, [c])]
               for c in [*(c for c in data.covariates if c not in _grouped(groups)), *groups]}
    notes = []
    usable = [c for c in columns if all(np.std(col) > 0 for col in columns[c])]
    N = data.n_subjects
    effects = frozenset((p, c) for p, spec in population.parameters.items() for c in spec.covariates)
    current = run(effects, population.to_dict(), "start")
    seen, start, near = {effects}, current, []
    for _ in range(max_iterations):
        z, near = _draws(current, data.n_obs), []
        proposal = {(p, c) for p, c in effects if p not in z}  # no random effect: not searched
        base = {**current.population, "parameters": {k: dict(v) for k, v in
                                                     current.population["parameters"].items()}}
        for name, y in z.items():
            L = len(y)
            y, L = (y.mean(0), 1) if draws == "mean" else (y.reshape(-1), L)

            def score(chosen):
                X = np.column_stack([np.ones(len(y))] + [np.tile(col, L) for c in chosen
                                                         for col in columns[c]])
                b, *_ = np.linalg.lstsq(X, y, rcond=None)
                rss = max(float(((y - X @ b) ** 2).sum()), 1e-300)
                m2ll = len(y) * (math.log(2 * math.pi * rss / len(y)) + 1)
                return m2ll / L + math.log(N) * sum(len(columns[c]) for c in chosen), b

            chosen = sorted(c for p, c in effects if p == name)
            allowed = sorted({c for c in usable if candidates is None or c in candidates.get(name, ())}
                             | set(chosen))
            if len(allowed) < 11:
                subsets = [list(k) for n in range(len(allowed) + 1)
                           for k in itertools.combinations(allowed, n)]
                (best, b), chosen = min(((score(m), m) for m in subsets), key=lambda s: s[0][0])
            else:
                best, b = score(chosen)
                while True:
                    moves = [sorted([*chosen, c]) for c in allowed if c not in chosen]
                    moves += [[c for c in chosen if c != out] for out in chosen]
                    (mb, mcoef), m = min(((score(m), m) for m in moves), key=lambda s: s[0][0])
                    if mb >= best:
                        break
                    best, b, chosen = mb, mcoef, m
            proposal |= {(name, c) for c in chosen}
            near += [{"parameter": name, "covariate": c, "in_model": c in chosen,
                      "change": score([k for k in chosen if k != c] if c in chosen
                                      else sorted([*chosen, c]))[0] - best} for c in allowed]
            spec = base["parameters"][name]
            spec["init"] = float(inverse(_TKIND[spec["distribution"]], *spec["bounds"], b[0]))
            names = [col for c in chosen for col in groups.get(c, [c])]
            spec["covariates"] = {col: float(v) for col, v in zip(names, b[1:])}
        proposal = frozenset(proposal)
        if proposal == effects:
            break
        if proposal in seen:
            notes.append(f"the search came back to a model already fitted: {_label(proposal) or 'none'}")
            break
        seen.add(proposal)
        effects, current = proposal, run(proposal, base, "proposal")
    else:
        notes.append(f"the search stopped after {max_iterations} iterations, the model still changing")
    kept = min(fitted, key=lambda k: fitted[k].bicc)
    final, kept = fitted[kept], sorted(kept)
    while eliminate and kept:
        trials = {e: run([x for x in kept if x != e], final.population, "remove") for e in kept}
        ok = {e: r for e, r in trials.items()
              if r.converged and supported((final.bicc, final.ll_se), (r.bicc, r.ll_se), -1)}
        if not ok:
            break
        out = min(ok, key=lambda e: ok[e].bicc)
        final, kept = ok[out], [x for x in kept if x != out]
    # a set of effects fitted twice is listed twice; the last row of the model returned is kept
    last = max(k for k, row in enumerate(rows) if row["effects"] == _label(kept))
    for k, row in enumerate(rows):
        row["decision"] = "kept" if k == last else ""
    near = [r for r in near if not r["in_model"] or (r["parameter"], r["covariate"]) in kept]
    return Search(result=final, population=Population.from_dict(model, final.population),
                  effects=kept, history=pd.DataFrame(rows), notes=notes, start=start,
                  near=pd.DataFrame(near, columns=["parameter", "covariate", "in_model", "change"]))


def learned(model: Model, data: Dataset, population: Population, settings: Settings | None = None,
            *, learner=None, parameters: list | None = None, candidates: dict | None = None,
            folds: int = 5, max_iterations: int = 5, draws: str = "mean", eliminate: bool = True,
            on_step=None) -> Search:
    """Covariate model learned from the covariates by any regressor.

    The loop of `samba` with `learner` in place of the linear regression: an object with
    `fit(X, y)` and `predict(X)`, a scikit-learn regressor for instance, copied unfitted before
    each use. Each iteration fits the current model and takes draws of its individual parameters
    (`_draws`). For each parameter with a random effect (`parameters`, by default all of them),
    the learner predicts the mean of the draws on the Gaussian scale (draws="stack": every draw)
    from the covariates (`candidates` maps a parameter to the covariates it may read, by default
    every one; a group of levels is read as its indicator columns). The prediction for a subject
    comes from a learner trained on the other subjects, in `folds` folds, so that a learner that
    only memorises its subjects scores nothing. The predictions, centred, enter the model as the
    covariate `score_<parameter>` with one coefficient estimated like any other effect; the
    effects of the population model given stay. The iterations stop when every score correlates
    above 0.995 with the previous one, or after max_iterations; the model of lowest BICc among
    those fitted is returned. With eliminate, each score is then fitted out and removed when the
    data allow it (`supported`), as in `samba`.

    `Search.learners` holds the learner of each score kept, fitted on every subject,
    `Search.features` the columns it reads, `Search.centres` the mean subtracted, `Search.scores`
    the centred scores and `Search.data` the data with the score columns: the typical value of
    a new subject is the transform of intercept + coefficient x (learner.predict(x) - centre).
    The uncertainty of the learner is not in the standard errors; `pysaem.assess.bootstrap`
    includes it.

    Run it after `samba`, on the population model it returns: the lines keep what they explain
    and the learner looks for what they left (a product, a step, a curve), and its score is
    removed when nothing is left. On the development half of PMX-CovEval the learner alone lost
    an effect a line on log(CLCR) finds, because the score learned on 425 subjects with omega 0.43
    was too noisy; after the lines it kept that effect and added nothing.
    """
    if learner is None or not (hasattr(learner, "fit") and hasattr(learner, "predict")):
        raise ValueError("learner must be an object with fit(X, y) and predict(X)")
    if draws not in ("stack", "mean"):
        raise ValueError("draws must be 'stack' or 'mean'")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    unknown = set(parameters or []) - set(population.parameters)
    if unknown:
        raise ValueError(f"parameters not in the population model: {sorted(unknown)}")
    st = settings or Settings()
    rows, notes = [], []
    groups = data.covariate_groups
    columns = {c: [np.asarray(data.covariates[col], float) for col in groups.get(c, [c])]
               for c in [*(c for c in data.covariates if c not in _grouped(groups)), *groups]}
    usable = [c for c in columns if all(np.std(col) > 0 for col in columns[c])]
    N = data.n_subjects
    given = frozenset((p, c) for p, spec in population.parameters.items() for c in spec.covariates)

    def run(effects, base, table, action):
        r = fit(model, table, _with(model, base, sorted(effects), groups), st)
        rows.append({"action": action, "parameter": "", "covariate": "", "effects": _label(effects),
                     "ofv": r.ofv, "bicc": r.bicc, "decision": ""})
        if on_step:
            on_step(rows[-1])
        return r

    def rows_of(X, y, subjects):
        """Training rows of a learner: one per subject (mean of the draws) or one per draw."""
        if draws == "mean":
            return X[subjects], y[:, subjects].mean(0)
        return np.tile(X[subjects], (len(y), 1)), y[:, subjects].reshape(-1)

    start = current = run(given, population.to_dict(), data, "start")
    best = {"fit": start, "data": data, "effects": given, "scores": {}, "learners": {},
            "centres": {}, "features": {}}
    scores = {}
    fold = np.random.default_rng(st.seed).permutation(N) % folds
    for _ in range(max_iterations):
        z = _draws(current, data.n_obs)
        base = {**current.population, "parameters": {k: dict(v) for k, v in
                                                     current.population["parameters"].items()}}
        previous, scores, learners, centres, features = scores, {}, {}, {}, {}
        effects = set(given)
        for name, y in z.items():
            allowed = [c for c in usable if not c.startswith("score_")
                       and (parameters is None or name in parameters)
                       and (candidates is None or c in candidates.get(name, ()))]
            if not allowed:
                continue
            X = np.column_stack([col for c in allowed for col in columns[c]])
            score = np.zeros(N)
            for k in range(folds):
                est = copy.deepcopy(learner)
                est.fit(*rows_of(X, y, fold != k))
                score[fold == k] = np.asarray(est.predict(X[fold == k]), float).reshape(-1)
            if not np.isfinite(score).all() or np.std(score) == 0:
                notes.append(f"the learner gave no usable score for {name}")
                continue
            est = copy.deepcopy(learner)
            est.fit(*rows_of(X, y, np.ones(N, bool)))
            col, centre = f"score_{name}", float(score.mean())
            scores[name], learners[name], centres[name] = score - centre, est, centre
            features[name] = [c2 for c in allowed for c2 in groups.get(c, [c])]
            spec = base["parameters"][name]
            spec["init"] = float(inverse(_TKIND[spec["distribution"]], *spec["bounds"], float(y.mean())))
            spec["covariates"] = {**spec["covariates"], col: spec["covariates"].get(col, 1.0)}
            effects.add((name, col))
        if not scores:
            notes.append("no score was learned")
            break
        table = replace(data, covariates={**data.covariates,
                                          **{f"score_{p}": s for p, s in scores.items()}})
        current = run(frozenset(effects), base, table, "proposal")
        if current.bicc < best["fit"].bicc:
            best = {"fit": current, "data": table, "effects": frozenset(effects), "scores": scores,
                    "learners": learners, "centres": centres, "features": features}
        if previous and all(p in previous and np.corrcoef(previous[p], s)[0, 1] > 0.995
                            for p, s in scores.items()):
            break
    else:
        notes.append(f"the search stopped after {max_iterations} iterations, the scores still changing")
    final, table, kept = best["fit"], best["data"], sorted(best["effects"])
    while eliminate:
        removable = [e for e in kept if e not in given]
        if not removable:
            break
        trials = {e: run([x for x in kept if x != e], final.population, table, "remove")
                  for e in removable}
        ok = {e: r for e, r in trials.items()
              if r.converged and supported((final.bicc, final.ll_se), (r.bicc, r.ll_se), -1)}
        if not ok:
            break
        out = min(ok, key=lambda e: ok[e].bicc)
        final, kept = ok[out], [x for x in kept if x != out]
    last = max(k for k, row in enumerate(rows) if row["bicc"] == final.bicc)
    rows[last]["decision"] = "kept"
    names = [p for p in best["scores"] if (p, f"score_{p}") in kept]
    return Search(result=final, population=Population.from_dict(model, final.population),
                  effects=kept, history=pd.DataFrame(rows), notes=notes, start=start,
                  learners={p: best["learners"][p] for p in names},
                  scores=pd.DataFrame({"id": data.ids, **{f"score_{p}": best["scores"][p] for p in names}}),
                  centres={p: best["centres"][p] for p in names},
                  features={p: best["features"][p] for p in names}, data=table)


def _grouped(groups: dict) -> set:
    """The indicator columns that belong to a covariate group."""
    return {col for cols in groups.values() for col in cols}


def _label(effects) -> str:
    return ", ".join(f"{c} on {p}" for p, c in sorted(effects))


def stepwise(model: Model, data: Dataset, population: Population, settings: Settings | None = None,
             *, candidates: dict | None = None, criterion: str = "bicc", p_forward: float = 0.01,
             p_backward: float = 0.01, p_screen: float | None = 0.05, max_fits: int = 200,
             correlated: float = 0.7, equivalent: float = 6.0, p_retain: float | None = 0.001,
             on_step=None) -> Search:
    """Covariate model by forward additions and backward removals.

    candidates maps a parameter to the covariates it may take; by default every
    parameter with a random effect may take every covariate of the data. A pair is
    tried in the order of the screen of the current model, while its screen p-value
    is below p_screen (None: every pair). With criterion "bicc" (Delattre, Lavielle and
    Poursat 2014), an effect is added or removed when the BICc decreases by more than two
    Monte Carlo standard errors of the difference; with "lrt", it is
    added when the likelihood ratio test has a p-value below p_forward and removed
    when the test of its removal has a p-value above p_backward (0.01 both, the
    defaults of COSSAC). A pair the screen leaves above p_screen is not fitted: the
    screen is close to the score test of the same effect, so few such pairs would pass.

    Once no pair passes, each kept covariate is swapped, on its parameter, for every
    covariate correlated with it beyond `correlated` in absolute value, and refitted:
    a swap that lowers the BICc beyond that noise is adopted and the search goes on. A swap
    within `equivalent` points of BICc is reported in `alternatives` as not
    distinguished by the data (6: the threshold of strong evidence of Kass and
    Raftery, 1995). Covariates that go together otherwise compete only through the
    order of the screen.

    With p_retain set, each kept effect must last hold on its own: the effect whose
    removal raises the -2LL least is removed while that rise is below the chi-square
    quantile of p_retain (0.001: 10.83, the backward threshold of SCM, Jonsson and
    Karlsson 1998), and the swaps are checked again on what remains. None disables it.
    """
    if criterion not in ("bicc", "lrt"):
        raise ValueError("criterion must be 'bicc' or 'lrt'")
    st = settings or Settings()
    notes, rows, fitted = [], [], {}
    start = [(p, c) for p, spec in population.parameters.items() for c in spec.covariates]

    def run(effects, base, action, pair):
        key = frozenset(effects)
        if key not in fitted:
            if len(fitted) >= max_fits:
                raise RuntimeError(f"the covariate search reached its limit of {max_fits} fits")
            fitted[key] = fit(model, data, _with(model, base, effects, groups), st)
        r = fitted[key]
        rows.append({"action": action, "parameter": pair[0], "covariate": pair[1],
                     "effects": ", ".join(f"{c} on {p}" for p, c in sorted(effects)),
                     "ofv": r.ofv, "bicc": r.bicc, "decision": ""})
        if on_step:
            on_step(rows[-1])
        return r

    def allowed(parameter, covariate):
        return candidates is None or covariate in candidates.get(parameter, ())

    def noise(a: FitResult, b: FitResult) -> float:
        return _NOISE * 2 * float(np.hypot(a.ll_se, b.ll_se))

    def worth(without: FitResult, with_: FitResult, threshold: float) -> bool:
        """Whether the effect that separates the two fits earns its place."""
        if criterion == "bicc":
            return with_.bicc < without.bicc - noise(without, with_)
        return float(stats.chi2.sf(max(without.ofv - with_.ofv, 0.0), 1)) < threshold

    groups = data.covariate_groups
    # swaps compare covariates by their correlation, which a group of levels does not have
    values = {c: np.asarray(v, float) for c, v in data.covariates.items() if c not in _grouped(groups)}
    alternatives = []

    def swap(effects, current):
        """The first swap that lowers the BICc, as (effects, fit); None when none does."""
        alternatives.clear()
        for k, (parameter, kept) in enumerate(effects):
            for other in values:
                if (kept not in values or other == kept or (parameter, other) in effects
                        or not allowed(parameter, other)
                        or np.std(values[other]) == 0 or np.std(values[kept]) == 0):
                    continue
                r = float(np.corrcoef(values[kept], values[other])[0, 1])
                if abs(r) <= correlated:
                    continue
                trial_effects = [*effects[:k], *effects[k + 1:], (parameter, other)]
                trial = run(trial_effects, current.population, "swap", (parameter, other))
                delta = trial.bicc - current.bicc
                if delta < -noise(trial, current):
                    rows[-1]["decision"] = "kept"
                    return trial_effects, trial
                rows[-1]["decision"] = "equivalent" if delta <= equivalent else "dropped"
                if delta <= equivalent:
                    alternatives.append({"parameter": parameter, "kept": kept, "alternative": other,
                                         "correlation": round(r, 2), "delta_bicc": float(delta)})
        return None

    effects = list(start)
    current = run(effects, population.to_dict(), "start", ("", ""))
    rows[-1]["decision"] = "kept"
    tried = set()
    while True:
        screen = current.covariate_screen
        if p_screen is not None:
            screen = screen[screen["p_value"] < p_screen]
        queue = [(r.parameter, r.covariate) for r in screen.itertuples()
                 if (r.parameter, r.covariate) not in effects and allowed(r.parameter, r.covariate)
                 and (frozenset(effects), (r.parameter, r.covariate)) not in tried]
        added = None
        for pair in queue:
            tried.add((frozenset(effects), pair))
            trial = run([*effects, pair], current.population, "add", pair)
            if worth(current, trial, p_forward):
                rows[-1]["decision"] = "kept"
                effects, current, added = [*effects, pair], trial, pair
                break
            rows[-1]["decision"] = "dropped"
        if added is None:
            added = swap(effects, current)
            if added is None:
                break
            effects, current = added
            added = effects[-1]
        # an effect whose Wald test no longer passes is challenged by a fit without it
        while True:
            removed = None
            for pair in [e for e in effects if e != added]:
                # a group has several coefficients and is always challenged by a fit without it
                name = f"beta_{pair[0]}_{pair[1]}"
                se = current.se.get(name)
                if se and se == se and 2 * stats.norm.sf(abs(current.parameters[name]) / se) < p_backward:
                    continue
                rest = [e for e in effects if e != pair]
                trial = run(rest, current.population, "remove", pair)
                if not worth(trial, current, p_backward):
                    rows[-1]["decision"] = "removed"
                    effects, current, removed = rest, trial, pair
                    break
                rows[-1]["decision"] = "kept"
            if removed is None:
                break
    removed_at_retention = []
    if p_retain is not None and effects:
        bar = float(stats.chi2.isf(p_retain, 1))
        while effects:
            rises = {}
            for pair in effects:
                trial = run([e for e in effects if e != pair], current.population, "retain", pair)
                rises[pair] = (trial.ofv - current.ofv, trial)
                rows[-1]["decision"] = "kept"
            weakest = min(rises, key=lambda e: rises[e][0])
            rise, trial = rises[weakest]
            if rise >= bar:
                break
            rows[[r["action"] == "retain" and (r["parameter"], r["covariate"]) == weakest
                  for r in rows].index(True, len(rows) - len(effects))]["decision"] = "removed"
            removed_at_retention.append({"parameter": weakest[0], "covariate": weakest[1],
                                         "rise_of_minus_2ll": float(rise)})
            effects, current = [e for e in effects if e != weakest], trial
        while (swapped := swap(effects, current)) is not None:
            effects, current = swapped
    return Search(result=current, population=Population.from_dict(model, current.population),
                  effects=sorted(effects), history=pd.DataFrame(rows), notes=notes,
                  alternatives=list(alternatives), retention_removed=removed_at_retention)
