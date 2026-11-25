"""Entry point: fit a population model."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from . import post, saem
from .display import Progress
from .data import Dataset
from .engine import _TKIND, Engine, FitError
from .model import Model
from .population import Population, inside, inverse, transform
from .result import FitResult
from .saem import Settings


def fit(model: Model, data: Dataset, population: Population, settings: Settings | None = None,
        on_progress=None, progress=None) -> FitResult:
    """Estimate the population parameters by SAEM, then everything read from a fit.

    on_progress receives a dict (iteration, n_iterations, phase, parameters) at
    each iteration; returning False cancels the fit. progress shows the run on
    stderr: None in a terminal or a notebook only, True always, False never.
    Raises FitError when the model cannot be evaluated or the estimation diverges.
    """
    st = settings or Settings()
    show = Progress.create(progress)
    try:
        result = _fit(model, data, population, st, _relay(show, on_progress), show)
    except BaseException:
        if show:
            show.abort()
        raise
    if show:
        show.close(result)
    return result


def _relay(show, on_progress):
    if not show:
        return on_progress

    def step(event):
        show.iteration(event)
        return on_progress(event) if on_progress else None
    return step


def _fit(model, data, population, st, on_progress, show):
    if st.n_starts > 1:
        return _multistart(model, data, population, st, on_progress, show)
    clock = _Clock(show)
    clock.begin("compiling and checking the model")
    eng = _engine(model, data, population, st)
    clock.lap("setup")
    if show:
        show.header(eng.N, eng.data.n_obs)
    trace = saem.run(eng, st, on_progress)
    clock.lap("saem")
    if trace.provisional:
        # a provisional fit gets its likelihood in minutes: a tenth of the draws, no FIM
        st = replace(st, n_conditional=30, n_conditional_burn=20, n_importance=1000, fim=False,
                     simulations=False)
    return finish(eng, st, trace, clock)


def _engine(model, data, population, st):
    eng = Engine(model, data, population, rtol=st.rtol, atol=st.atol, dynamic_range=st.dynamic_range,
                 method=st.method, sensitivities=st.sensitivities, seed=st.seed)
    eng.check_initial()
    return eng


def _multistart(model, data, population, st, on_progress, show=None) -> FitResult:
    """SAEM from n_starts initial values, the first one as given; the best quick -2LL gets the
    full post-estimation. Each start keeps its own seed."""
    quick = replace(st, n_conditional=30, n_conditional_burn=20, n_importance=1000)
    rng = np.random.default_rng(st.seed)
    clock = _Clock(show)
    runs = []
    for k in range(st.n_starts):
        pop = population if k == 0 else _perturbed(population, st.start_spread, rng)
        eng = _engine(model, data, pop, replace(st, seed=st.seed + k))
        if show:
            if k == 0:
                show.header(eng.N, eng.data.n_obs)
            show.start(f"start {k + 1}/{st.n_starts}")
        try:
            t0 = time.perf_counter()
            trace = saem.run(eng, replace(st, seed=st.seed + k), on_progress)
            if show:
                show.end("saem", time.perf_counter() - t0)
            cond = post.conditional(eng, quick, trace)
            ll = post.loglik(eng, quick, cond)[0]
        except FitError as e:
            runs.append({"start": k, "ofv": float("nan"), "parameters": None, "error": str(e)})
            continue
        runs.append({"start": k, "ofv": -2 * ll, "parameters": saem.estimates(eng), "engine": eng,
                     "trace": trace})
    clock.lap("starts")
    done = [r for r in runs if r["parameters"] is not None and math.isfinite(r["ofv"])]
    if not done:
        raise FitError("no start reached a finite log-likelihood")
    best = min(done, key=lambda r: r["ofv"])
    result = finish(best["engine"], st, best["trace"], clock)
    spread = max(r["ofv"] for r in done) - best["ofv"]
    if spread > 10:
        result.messages.append(f"the {len(done)} starts end up to {spread:.1f} apart in -2LL: "
                               f"start {best['start']} kept, the likelihood has several modes")
    result.starts = [{k: v for k, v in r.items() if k not in ("engine", "trace")} for r in runs]
    return result


def _perturbed(population: Population, spread: float, rng) -> Population:
    """The population with each free typical value moved on its Gaussian scale."""
    d = population.to_dict()
    for name, p in d["parameters"].items():
        if p["fixed"]:
            continue
        kind, (lo, hi) = _TKIND[p["distribution"]], p["bounds"]
        phi = float(transform(kind, lo, hi, p["init"]))
        step = spread * rng.standard_normal() * (abs(phi) if kind == 0 and phi != 0 else 1.0)
        p["init"] = float(inverse(kind, lo, hi, phi + step))
    return Population.from_dict(population.model, d)


def finish(eng: Engine, st: Settings, trace, clock=None) -> FitResult:
    """Every post-estimation task at the engine's current population values."""
    clock = clock or _Clock()
    values = saem.estimates(eng)
    names = list(values)
    iiv = np.flatnonzero(eng.iiv)
    mean = eng.mean()
    singular = _singular_blocks(eng)
    eng.corr_boundary = [pair for _, _, pairs in singular for pair in pairs]

    clock.begin("conditional distribution")
    cond = post.conditional(eng, st, trace)
    clock.lap("conditional distribution")
    clock.begin("conditional mode")
    eta_mode = post.modes(eng, cond.best)
    clock.lap("conditional mode")
    clock.begin("log-likelihood")
    ll, ll_se, sample = post.loglik(eng, st, cond) if st.loglik or st.fim else (
        float("nan"), float("nan"), None)
    if st.loglik and not math.isfinite(ll):
        raise FitError("the log-likelihood is not finite at the estimates "
                       f"{ {k: round(v, 6) for k, v in values.items()} }")
    clock.lap("log-likelihood")
    clock.begin("information")
    info, index, halves = post.information(eng, sample, eta_mode) if st.fim else (None, [], None)
    se, se_u, corr, fim_status = post.standard_errors(eng, info, index, names, halves) if st.fim else (
        {n: float("nan") for n in names}, {}, None, "not computed")
    rse = {n: 100 * se[n] / abs(values[n]) if values[n] != 0 else float("nan") for n in names}
    ci = post.intervals(eng, index, values, se_u)
    clock.lap("information")
    clock.begin("predictions and simulations")

    d = eng.data
    out = d.obs_out
    # on a record of a hazard output: the events expected since its follow-up opened
    f_pop = eng.cumulative(eng.predict(mean[None])[0])[0]
    f_mode = eng.cumulative(eng.predict((mean + eta_mode)[None])[0])[0]
    f_mean = eng.cumulative(eng.predict((mean + cond.mean)[None])[0])[0]
    pred = d.observations()
    pred["variable"] = [eng.model.outputs[o] for o in out]
    pred["pop_pred"] = f_pop
    pred["ind_pred_mode"] = f_mode
    pred["ind_pred_mean"] = f_mean
    for name, f in (("iwres_mode", f_mode), ("iwres_mean", f_mean)):
        f = eng.to_scale(f, out)
        pred[name] = (eng.obs_y - f) / eng.residual_sd(f, out)
    pred["pwres"] = np.nan
    pred["npde"] = np.nan
    pred["pcvpc_value"] = np.nan
    vpc, pcvpc, vpc_hazards = pd.DataFrame(), None, None
    if st.simulations and st.n_simulations > 0:
        sim = post.simulate(eng, st.n_simulations)
        pred["pwres"], npde = post.population_residuals(eng, sim)
        pred["npde"] = np.where(eng.on_hazard, post.hazard_npde(eng, sim), npde) \
            if eng.on_hazard.any() else npde
        vpc, _ = post.vpc(eng, sim)
        pcvpc, pred["pcvpc_value"] = post.vpc(eng, sim, pop_pred=f_pop)
        if eng.on_hazard.any():
            vpc_hazards = post.vpc_hazards(eng, sim)
    tests = {}
    for o, label in enumerate(d.output_labels):
        sel = (out == o).to_numpy() if hasattr(out, "to_numpy") else out == o
        tests[label] = {k: post.residual_tests(pred[k].to_numpy()[sel])
                        for k in ("iwres_mode", "pwres", "npde")}
    clock.lap("predictions and simulations")

    ind = pd.DataFrame({"id": d.ids})
    psi_mode, psi_mean = eng.psi(mean + eta_mode), eng.psi(mean + cond.mean)
    shrinkage = {}
    for j, name in enumerate(eng.names):
        ind[f"{name}_mode"] = psi_mode[:, j]
        ind[f"{name}_mean"] = psi_mean[:, j]
        if j in iiv:
            ind[f"eta_{name}_mode"] = eta_mode[:, j]
            ind[f"eta_{name}_mean"] = cond.mean[:, j]
            ind[f"eta_{name}_sd"] = cond.sd[:, j]
            # eta-shrinkage on the sd scale (Savic and Karlsson 2009)
            w = eng.omega[j]
            ddof = 1 if eng.N > 1 else 0
            shrinkage[name] = {
                "mode": float(1 - eta_mode[:, j].std(ddof=ddof) / w),
                "mean": float(1 - cond.mean[:, j].std(ddof=ddof) / w),
                "samples": float(1 - cond.samples[:, :, j].std(ddof=ddof) / w),
            }
    # draws of each subject's parameters, spread evenly over the kept iterations
    n_draws = len(cond.samples)
    draws = pd.DataFrame({"id": np.tile(np.asarray(d.ids), n_draws),
                          "draw": np.repeat(np.arange(n_draws), eng.N)})
    psi_draws = eng.psi(mean[None] + cond.samples)
    for j, name in enumerate(eng.names[:eng.p0]):
        draws[name] = psi_draws[:, :, j].reshape(-1)

    # BICc (Delattre, Lavielle and Poursat 2014): log(N) for the variances and correlations of
    # the random effects and for the typical value and covariate coefficients of a parameter
    # with a random effect, log(n_obs) for the other fixed effects and the error parameters
    n_random = int((eng.omega_free & (eng.tie == np.arange(eng.P))).sum()) + len(eng.corr_pairs)
    n_between = int(sum(f.sum() for j, f in enumerate(eng.coef_free) if eng.iiv[j]))
    n_fixed = int(sum(f.sum() for f in eng.coef_free)
                  + sum(len(eng.err_names[o]) for o in np.flatnonzero(eng.err_free))) - n_between
    n_free = n_random + n_between + n_fixed
    events = {o: int(d.obs_y[out == eng.model.outputs.index(o)].sum()) for o in eng.model.hazards}
    # with hazard outputs, log(n_obs) counts the other records and the events: information
    # grows with events, not with records without one (Volinsky and Raftery 2000)
    n_info = max(int((~eng.on_hazard).sum()) + sum(events.values()), 1) \
        if eng.on_hazard.any() else d.n_obs
    ofv = -2 * ll
    messages = list(trace.messages)
    runaway, pairs = _runaway(eng, trace)
    messages += runaway
    if st.fim and fim_status != "ok" and n_free:
        messages.append(f"standard errors: {fim_status}")
    messages += _correlated(corr, pairs)
    for j in eng.collapsed:
        if eng.tie[j] != j:
            continue
        what = "inter-individual" if j < eng.p0 else "inter-occasion"
        on = eng.names[j] if j < eng.p0 else eng.iov_names[(j - eng.p0) % eng.n_iov]
        messages.append(f"{eng.omega_name(j)} was estimated at zero: the data support no "
                        f"{what} variability on {on}")
    if len(iiv) > 1 and eng.N > 3:
        # one draw per subject is free of shrinkage, unlike the modes (Lavielle and Ribba 2016)
        r = np.mean([np.corrcoef(draw[:, iiv].T) for draw in cond.samples], axis=0)
        estimated = set(eng.corr_pairs)
        for a, j in enumerate(iiv):
            for b in range(a + 1, len(iiv)):
                k = iiv[b]
                if (j, k) not in estimated and abs(r[a, b]) >= 0.5:
                    messages.append(
                        f"the random effects of {eng.names[j]} and {eng.names[k]} look "
                        f"correlated ({r[a, b]:+.2f}): consider estimating their correlation")
    for j in iiv:
        if j < eng.p0 and eng.tkind[j] != 0 and eng.omega[j] > 1.5:
            messages.append(f"omega_{eng.names[j]} is very large ({eng.omega[j]:.2f} on the "
                            f"Gaussian scale): {eng.names[j]} may not be identifiable")
    typical = eng.typical()
    for j, name in enumerate(eng.names[:eng.p0]):
        p = eng.population.parameters[name]
        if inside(p.distribution, p.bounds, float(typical[j]))[1]:
            messages.append(f"{name}_pop is at the edge of its support ({typical[j]:g}): "
                            f"{name} is not identifiable from these data")
    for low, dependent, _ in singular:
        messages.append(f"the estimated correlation matrix of the random effects is singular "
                        f"(smallest eigenvalue {low:.1e}), mostly along {', '.join(dependent)}: "
                        f"the correlations of the block get no standard error")
    if d.dropped_ids:
        messages.append(f"{len(d.dropped_ids)} subject(s) without observation were dropped")
    with_prior = [eng.names[j] for j in np.flatnonzero(eng.prior_prec[:eng.p0] > 0)]
    if with_prior:
        messages.append(f"prior on the typical value of {', '.join(with_prior)}: the estimates "
                        "are posterior modes and the standard errors posterior sds")

    tr = pd.DataFrame(trace.values, columns=trace.names)
    tr.insert(0, "iteration", np.arange(1, len(tr) + 1))
    tr.insert(1, "phase", np.where(tr["iteration"] <= trace.n_exploratory, "exploratory",
                                   "smoothing"))
    tr["convergence"] = trace.indicator
    return FitResult(
        model=eng.model.to_dict(), population=_estimated(eng).to_dict(), settings=asdict(st),
        parameters=values, se=se, rse=rse, ci=ci, fim_status=fim_status, ll=ll, ll_se=ll_se,
        ofv=ofv, aic=ofv + 2 * n_free, bic=ofv + n_free * math.log(eng.N),
        bicc=ofv + (n_random + n_between) * math.log(eng.N) + n_fixed * math.log(n_info),
        n_subjects=eng.N, n_obs=d.n_obs, n_parameters=n_free,
        converged=bool(trace.stabilized and not runaway
                       and all(math.isfinite(v) for v in values.values())),
        provisional=bool(getattr(trace, "provisional", False)),
        messages=messages, shrinkage=shrinkage, tests=tests, timings=clock.laps, trace=tr,
        individual=ind, draws=draws, predictions=pred, vpc=vpc, pcvpc=pcvpc, correlation=corr,
        covariate_screen=post.covariate_screen(eng, cond), solver=dict(eng.prob.totals),
        vpc_hazards=vpc_hazards, hazard_events=events, curves=post.curves(eng, mean + eta_mode))


# a log-normal typical value still moving by a factor 100 over the second half of the iterations
# runs off; two that run off with their ratio or product held within 20% run off together
RUNAWAY = math.log(100.0)
LOCKSTEP = math.log(1.2)
CORRELATED = 0.95


def _runaway(eng: Engine, trace) -> tuple:
    """Messages on the log-normal typical values that run off over the second half of the
    iterations, alone or in pairs whose ratio or product holds, and the pairs found."""
    cols = {n: i for i, n in enumerate(trace.names)}
    half = np.asarray(trace.values, float)[len(trace.values) // 2:]
    logs = {}
    for j, name in enumerate(eng.names[:eng.p0]):
        c = cols.get(f"{name}_pop")
        if eng.tkind[j] == _TKIND["lognormal"] and c is not None and len(half) > 1:
            x = half[:, c]
            if np.isfinite(x).all() and (x > 0).all():
                logs[name] = np.log(x)
    off = [n for n, v in logs.items() if abs(v[-1] - v[0]) > RUNAWAY]
    orders = {n: abs(logs[n][-1] - logs[n][0]) / math.log(10) for n in off}
    messages, pairs, paired = [], set(), set()
    for a, n in enumerate(off):
        for m in off[a + 1:]:
            if n in paired or m in paired:
                continue
            same = (logs[n][-1] - logs[n][0]) * (logs[m][-1] - logs[m][0]) > 0
            held = logs[n] - logs[m] if same else logs[n] + logs[m]
            if np.ptp(held) < LOCKSTEP:
                paired |= {n, m}
                pairs.add(frozenset((f"{n}_pop", f"{m}_pop")))
                what = f"{n}/{m}" if same else f"{n}*{m}"
                messages.append(
                    f"{n}_pop and {m}_pop run off together ({orders[n]:.0f} and {orders[m]:.0f} "
                    f"orders of magnitude over the last {len(half)} iterations) while {what} stays "
                    f"at {math.exp(held[-1]):.4g}: the data determine {what}, not {n} and {m}, "
                    f"which are not identifiable")
    for n in off:
        if n not in paired:
            messages.append(
                f"{n}_pop runs off ({orders[n]:.0f} orders of magnitude over the last {len(half)} "
                f"iterations): the data do not limit {n} in that direction, and {n} is not "
                f"identifiable")
    return messages, pairs


def _correlated(corr, pairs) -> list:
    """Messages on the pairs of estimates correlated beyond CORRELATED, except the pairs already
    reported as running off together."""
    if corr is None:
        return []
    names, r = list(corr.columns), corr.to_numpy()
    return [f"the estimates of {a} and {b} are correlated ({r[i, k]:+.2f}): the data hardly "
            f"separate them" for i, a in enumerate(names) for k, b in enumerate(names)
            if k > i and abs(r[i, k]) > CORRELATED and frozenset((a, b)) not in pairs]


def _singular_blocks(eng: Engine) -> list:
    """Correlation blocks whose matrix ends at the floor CORR_FLOOR: a maximum on the
    boundary of the positive definite matrices, as a variance at zero is. For each, the
    smallest eigenvalue, the parameters that weigh most in its direction, and the pairs."""
    out = []
    for block in eng.population.correlations:
        members = [eng.names.index(n) for n in block if eng.iiv[eng.names.index(n)]]
        pairs = [(j, k) for j, k in eng.corr_pairs if j in members and k in members]
        if len(members) < 2 or not pairs:
            continue
        low, vectors = np.linalg.eigh(eng.corr[np.ix_(members, members)])
        # at the clip of one correlation, a pair's matrix has an eigenvalue of 1 - CORR_CLIP;
        # a larger block is held at CORR_FLOOR
        if low[0] <= 1 - saem.CORR_CLIP + 1e-9:
            weight = np.abs(vectors[:, 0])
            dependent = [eng.names[j] for j, w in zip(members, weight) if w >= 0.5 * weight.max()]
            out.append((float(low[0]), dependent, pairs))
    return out


def _estimated(eng: Engine) -> Population:
    """The population model carrying the current estimates as its values."""
    d = eng.population.to_dict()
    pop = eng.typical()
    for j, name in enumerate(eng.names[:eng.p0]):
        p = d["parameters"][name]
        p["init"], _ = inside(p["distribution"], tuple(p["bounds"]), float(pop[j]))
        p["iiv"] = bool(eng.iiv[j])
        if eng.iiv[j]:
            p["omega"] = float(eng.omega[j])
        p["covariates"] = {c: float(eng.coef[j][1 + k]) for k, c in enumerate(eng.cov_names[j])}
    for jj, name in enumerate(eng.iov_names):
        lead = eng.p0 + jj
        p = d["parameters"][name]
        p["iov"] = bool(eng.iiv[lead])
        if eng.iiv[lead]:
            p["gamma"] = float(eng.omega[lead])
    d["correlation_values"] = {f"corr_{eng.names[j]}_{eng.names[k]}": float(eng.corr[j, k])
                               for j, k in eng.corr_pairs}
    d["correlations"] = [[n for n in block if eng.iiv[eng.names.index(n)]]
                         for block in eng.population.correlations]
    d["correlations"] = [b for b in d["correlations"] if len(b) > 1]
    for o, output in enumerate(eng.model.outputs):
        d["error"][output]["a"] = float(eng.ea[o]) if "a" in eng.err_names[o] else 1.0
        d["error"][output]["b"] = float(eng.eb[o]) if "b" in eng.err_names[o] else 0.3
    return Population.from_dict(eng.model, d)


class _Clock:
    def __init__(self, show=None):
        self.laps = {}
        self.show = show
        self._t = time.perf_counter()

    def begin(self, name: str):
        if self.show:
            self.show.begin(name)

    def lap(self, name: str):
        now = time.perf_counter()
        self.laps[name] = round(now - self._t, 4)
        self._t = now
        if self.show and name != "starts":
            self.show.end(name, self.laps[name])
