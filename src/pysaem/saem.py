"""SAEM: stochastic approximation EM with an exploratory and a smoothing phase."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .engine import _G_MIN, Engine, FitError

_SETTLE = 0.02        # largest move over the second half of the smoothing phase: 2%
_SETTLE_WINDOW = 10   # smoothing iterations below twice this fall back on the exploratory test
_MC_SHARE = 0.1       # the smoothing phase also ends when the Monte Carlo error of the averaged
_MC_MIN = 200         # estimates is below this share of their statistical error, from this iteration
JUMP_WINDOW = 20      # iterations of the fifth kernel after each refresh of the modes
JUMP_GAIN = 2.0       # log posterior density by which the mode from the mean must beat the chain's
CORR_CLIP = 0.995     # a correlation of random effects is held within this
CORR_FLOOR = 1e-3     # smallest eigenvalue a correlation matrix of random effects is held at
_ARTIFICIAL_OMEGA = 0.3  # initial sd lent to parameters without variability, on their scale


@dataclass
class Settings:
    """Settings of the estimation and of what is computed after it. The defaults suit most fits;
    the iteration numbers are the usual 1000 and 500."""

    #: Largest number of iterations of the exploratory phase (step size 1).
    n_exploratory: int = 1000
    #: Largest number of iterations of the smoothing phase (decreasing step size).
    n_smoothing: int = 500
    #: End the exploratory phase when the two halves of its last 100 iterations agree.
    exploratory_autostop: bool = True
    #: End the smoothing phase when the averaged estimates have settled.
    smoothing_autostop: bool = True
    #: Iterations of the exploratory phase before it may stop.
    min_exploratory: int = 150
    #: Iterations of the smoothing phase before it may stop.
    min_smoothing: int = 50
    #: Iterations of the Markov chains alone before the first update of the parameters.
    n_burn: int = 5
    #: Markov chains per subject; by default 200 chain-subjects in all, at least 10 with a
    #: frailty.
    n_chains: int | None = None
    #: Iterations of the four Metropolis-Hastings kernels at each SAEM iteration: draws from the
    #: population distribution, random walk by component, random walk by block, and draw of a
    #: group of components given the others.
    mcmc: tuple = (2, 2, 2, 1)
    #: Largest factor by which a variance may decrease in one iteration (simulated annealing).
    annealing: float = 0.95
    #: Iterations under annealing; by default the whole exploratory phase.
    n_annealing: int | None = None
    #: Step size 1 / k**exponent in the smoothing phase; convergence needs 0.5 < exponent <= 1
    #: (Delyon et al. 1999), and a smaller exponent forgets faster where the exploratory phase
    #: ended.
    smoothing_exponent: float = 0.7
    #: Also update the fixed effects with the random effects held fixed (interweaving).
    interweave: bool = True
    #: Derivatives of the predictions at the conditional modes: "exact" (sensitivity equations,
    #: finite differences where they do not apply) or "fd". The scoring step always uses "fd".
    sensitivities: str = "exact"
    #: Seed of the random numbers; a fit is reproducible for a given seed and machine.
    seed: int = 123456
    #: Relative tolerance of the ODE solver.
    rtol: float = 1e-6
    #: Absolute floor of the solver tolerance.
    atol: float = 0.0
    #: Tolerance relative to the largest value of a state, lowered per subject down to the data
    #: on an output with a relative error.
    dynamic_range: float = 1e-12
    #: ODE solver: "auto" (explicit, implicit when the system becomes stiff), "explicit" or
    #: "implicit".
    method: str = "auto"
    #: Kept iterations of the conditional distribution of the random effects after SAEM.
    n_conditional: int = 300
    #: Discarded iterations before those.
    n_conditional_burn: int = 50
    #: Draws per subject for the log-likelihood by importance sampling.
    n_importance: int = 10000
    #: Draws per subject for the information matrix.
    n_information: int = 2000
    #: Simulated replicates of the study for the NPDE and the visual predictive check.
    n_simulations: int = 500
    #: Compute the standard errors.
    fim: bool = True
    #: Compute the log-likelihood and the information criteria.
    loglik: bool = True
    #: Simulate the study for the NPDE and the visual predictive check.
    simulations: bool = True
    #: Time budget of the SAEM iterations in seconds; None for no budget.
    max_seconds: float | None = None
    #: When the budget is exceeded: "raise" an error, or "estimate", which stops the iterations
    #: and reports the fit as provisional, with a quick likelihood and no standard error.
    on_budget: str = "raise"
    #: An omega below this value (relative to the parameter for a normal distribution) has
    #: reached the boundary: it is set to zero and the parameter is estimated without
    #: variability from then on; 0 disables.
    min_omega: float = 0.01
    #: Number of starts; the first is the one given, the best quick log-likelihood is kept.
    n_starts: int = 1
    #: Starts beyond the first perturb the typical values on the Gaussian scale by this many
    #: standard normal draws.
    start_spread: float = 0.5


@dataclass
class Trace:
    """What the SAEM iterations leave behind."""

    names: list
    values: np.ndarray               # (iterations, parameters), burn-in excluded
    indicator: np.ndarray            # complete log-likelihood per iteration
    n_exploratory: int
    n_smoothing: int
    n_chains: int
    stabilized: bool
    phi: np.ndarray                  # last simulated individual parameters
    scale2: np.ndarray
    scale3: np.ndarray
    messages: list = field(default_factory=list)
    provisional: bool = False        # the time budget stopped the iterations


def estimates(eng: Engine) -> dict:
    """Current population values by reporting name."""
    out = {}
    pop = eng.typical()
    for j, name in enumerate(eng.names[:eng.p0]):
        out[f"{name}_pop"] = float(pop[j])
        for c, cov in enumerate(eng.cov_names[j]):
            out[f"beta_{name}_{cov}"] = float(eng.coef[j][1 + c])
    for j in range(eng.P):
        # one sd per parameter with occasions: reported once, as gamma
        if (eng.iiv[j] or j in eng.collapsed) and eng.tie[j] == j:
            out[eng.omega_name(j)] = float(eng.omega[j])
    for j, k in eng.corr_declared:
        out[f"corr_{eng.names[j]}_{eng.names[k]}"] = float(eng.corr[j, k])
    for o, output in enumerate(eng.model.outputs):
        for which in eng.err_names[o]:
            value = eng.ea[o] if which == "a" else eng.eb[o]
            out[eng.population.error_name(output, which)] = float(value)
    return out


def run(eng: Engine, st: Settings, on_progress=None) -> Trace:
    N, P = eng.N, eng.P
    # 200 chain-subjects, 400 with correlations: 1 chain took a wrong mode in 18 of 20 fits with six
    # correlated effects (4 chains: 1) and in 22 of 200 with saturable elimination (4 chains: 1).
    # A frailty with one event per subject: 1 chain ended 0.64 below the maximum, 10 chains 0.014
    C = st.n_chains or max(1, math.ceil((400 if eng.corr_pairs else 200) / N),
                           10 if _frailty(eng) else 1)
    free_fixed = np.array([f.any() for f in eng.coef_free])
    art_idx = np.flatnonzero(~eng.iiv & free_fixed)
    nothing = not (free_fixed.any() or eng.omega_free.any() or eng.err_free.any())
    K1 = 0 if nothing else st.n_exploratory
    K2 = 0 if nothing else st.n_smoothing
    K_sa = min(K1, st.n_annealing if st.n_annealing is not None else K1)
    # parameters without variability keep a lent one for at most 50 iterations, so that the
    # automatic stop can end the exploration from 150 iterations
    K_art = min(K_sa, K1 // 2, 50)

    omega_art = np.zeros(P)
    for j in art_idx:
        scale = abs(eng.coef[j][0]) if eng.tkind[j] == 0 and eng.coef[j][0] != 0 else 1.0
        omega_art[j] = _ARTIFICIAL_OMEGA * scale
    phi = np.repeat(eng.mean()[None], C, 0)
    rng = eng.rng_states(C, stream=0)
    jump_rng, jump_until = np.random.default_rng([eng.seed, 5]), 0
    scale2 = 0.5 * np.where(eng.iiv, eng.omega, omega_art)
    scale3 = np.repeat(scale2[:, None], P + 1, 1)
    s1 = np.zeros((N, P))
    s2 = np.zeros((P, P))
    s_err = np.zeros(eng.model.n_outputs)
    names = list(estimates(eng))
    values, indicator, series = [], [], []
    held = []  # per iteration, the variances the annealing floor held up
    stabilized = K1 == 0
    messages = []
    follow = _Follow()
    provisional = False
    # with a step 1/k^a, a < 1, the estimate reported is the average of the iterates of the
    # second half of the smoothing phase (Polyak and Juditsky 1992)
    averaging = st.smoothing_exponent < 1
    snapshots = []

    k = 1 - (st.n_burn if K1 + K2 > 0 else 0)
    k_end1 = K1
    total = K1 + K2
    started = time.perf_counter()
    while k <= total:
        iiv_idx = np.flatnonzero(eng.iiv)
        if st.max_seconds is not None and time.perf_counter() - started > st.max_seconds:
            if st.on_budget != "estimate":
                raise FitError(
                    f"the fit exceeded its time budget of {st.max_seconds:.0f} s at iteration "
                    f"{max(k, 1)} of {total}; current estimates "
                    f"{ {n: float(f'{v:.4g}') for n, v in estimates(eng).items()} }")
            messages.append(
                f"the time budget of {st.max_seconds:.0f} s ran out at iteration {max(k, 1)} of "
                f"{total}: the estimates are provisional, and the -2LL, computed with fewer "
                "draws, is an upper bound (up to its Monte Carlo error) of the -2LL at the "
                "maximum of the likelihood")
            provisional, total, k_end1 = True, max(k - 1, 0), min(k_end1, max(k - 1, 0))
            break
        annealing = 1 <= k <= K_sa
        with_art = k <= K_art
        omega = eng.omega.copy()
        if with_art:
            omega[art_idx] = omega_art[art_idx]
        sampled = np.sort(np.concatenate([iiv_idx, art_idx])) if with_art else iiv_idx
        ns = len(sampled)
        block = (k % (ns - 1)) + 2 if ns > 1 else 0
        mean = eng.mean()
        eta = phi - mean
        keep = np.zeros(P, bool)
        keep[sampled] = True
        eta[:, :, ~keep] = 0.0
        # a chain left in a secondary mode is offered the mode reached from the population
        # mean, for JUMP_WINDOW iterations from mid-exploration and from the first smoothing one
        if not with_art and ns and k in (max(k_end1 // 2, K_art + 1), k_end1 + 1):
            d, jump_until = jump_vectors(eng, eta), k + JUMP_WINDOW
        if not with_art and ns and k < jump_until:
            eta = jump(eng, eta, omega, sampled, d, jump_rng)
        uy, pred, acc2, acc3 = eng.estep(eta, rng, omega, sampled, st.mcmc, block, scale2,
                                         scale3[:, block])
        phi = mean + eta
        scale2[sampled] *= 1 + 0.4 * (acc2[sampled] - 0.4)
        moved = sampled[np.isfinite(acc3[sampled])]
        scale3[moved, block] *= 1 + 0.4 * (acc3[moved] - 0.4)
        # a flat likelihood would let the scales grow without bound
        scale2[sampled] = np.minimum(scale2[sampled], 3 * omega[sampled])
        scale3[sampled] = np.minimum(scale3[sampled], 3 * omega[sampled, None])

        ok = np.isfinite(uy)
        if ok.mean() < 0.5:
            raise FitError(f"iteration {k}: the model fails to evaluate for "
                           f"{(~ok).sum()} of {ok.size} chain-subjects")
        if k < 1:
            k += 1
            if on_progress and on_progress({"iteration": k - 1, "n_iterations": total,
                                            "phase": "burn-in",
                                            "parameters": estimates(eng)}) is False:
                raise FitError("cancelled by the progress callback")
            continue

        gamma = 1.0 if k <= k_end1 else 1.0 / (k - k_end1) ** st.smoothing_exponent
        s1 += gamma * (phi.mean(0) - s1)
        s2 += gamma * (np.einsum("cna,cnb->ab", phi, phi) / C - s2)
        _update_fixed(eng, s1, sampled, omega)
        m = eng.mean()
        cov = (s2 - s1.T @ m - m.T @ s1 + m.T @ m) / N
        held_now = np.zeros(P, bool)
        for j in sampled:
            if eng.omega_free[j]:
                w2 = max(cov[j, j], 1e-12)
                if annealing:
                    held_now[j] = w2 < st.annealing * eng.omega[j] ** 2
                    w2 = max(w2, st.annealing * eng.omega[j] ** 2)
                eng.omega[j] = math.sqrt(w2)
        held.append(held_now)
        # the random effects of one parameter on each occasion share their variance
        for group in eng.tie_groups:
            live = group[eng.iiv[group] & eng.omega_free[group]]
            if live.size:
                eng.omega[live] = math.sqrt(float(np.mean(eng.omega[live] ** 2)))
        _update_correlations(eng, cov)
        if with_art:
            omega_art *= math.sqrt(st.annealing)
        elif st.min_omega > 0:
            for j in iiv_idx:
                scale = max(abs(eng.coef[j][0]), 1e-12) if eng.tkind[j] == 0 else 1.0
                if eng.omega_free[j] and eng.omega[j] < st.min_omega * scale:
                    # boundary reached: the parameter loses its random effect for good
                    eng.iiv[j], eng.omega[j] = False, 0.0
                    eng.collapsed.append(j)
                    eng.corr_pairs = [p for p in eng.corr_pairs if j not in p]
                    eng.corr[j, :] = eng.corr[:, j] = 0.0
                    eng.corr[j, j] = 1.0
                    phi[:, :, j] = eng.design[j] @ eng.coef[j]
                    free_fixed[j] = eng.coef_free[j].any()
            iiv_idx = np.flatnonzero(eng.iiv)
            art_idx = np.flatnonzero(~eng.iiv & free_fixed)
        moving = np.sort(np.concatenate([iiv_idx, art_idx])) if st.interweave else (
            art_idx if not with_art else art_idx[:0])
        moving = moving[[eng.coef_free[j].any() for j in moving]]
        if moving.size:
            # interweaving: the centred update above is fast when the data say
            # little about an individual, this one when they say a lot
            deltas, moved = _scoring(eng, phi, moving, follow if not with_art else None)
            shift = gamma * moved
            for j, delta in deltas.items():
                shift[:, j] += gamma * (eng.design[j] @ delta)
                eng.coef[j] += gamma * delta
            phi += shift
            # the statistics follow the shift
            s2 += shift.T @ s1 + s1.T @ shift + shift.T @ shift
            s1 += shift
        _update_error(eng, eng.gather(pred), gamma, s_err, annealing, st.annealing)

        current = estimates(eng)
        if not all(np.isfinite(v) for v in current.values()):
            bad = [n for n, v in current.items() if not np.isfinite(v)]
            raise FitError(f"iteration {k}: non-finite estimate for {bad}")
        log_prior = eng.log_prior(phi - eng.mean(), omega, sampled)
        indicator.append(float(np.where(ok, log_prior - uy, np.nan).sum(1).mean()))
        values.append([current[n] for n in names])
        series.append(_series(eng))
        if averaging and k > k_end1:
            snapshots.append(([c.copy() for c in eng.coef], eng.omega ** 2, eng.corr.copy(),
                              eng.ea.copy(), eng.eb.copy(), len(eng.collapsed)))

        phase = "exploratory" if k <= k_end1 else "smoothing"
        if on_progress and on_progress({"iteration": k, "n_iterations": total, "phase": phase,
                                        "parameters": current,
                                        "convergence": indicator[-1]}) is False:
            raise FitError("cancelled by the progress callback")

        if phase == "exploratory":
            done = k == k_end1
            # a variance the annealing floor held up in a fifth of the last 100 iterations or
            # more drifts by the floor itself (it cuts its Monte Carlo noise below): its
            # drift does not keep the phase open
            ratchet = np.mean(held[-100:], axis=0) >= 0.2 if held else np.zeros(P, bool)
            if (st.exploratory_autostop and k >= max(st.min_exploratory, K_art + 100)
                    and _stable(series, 100, 0.5, skip=_omega_columns(eng, ratchet))):
                done = stabilized = True
                total -= k_end1 - k
                k_end1 = k
            if done and not stabilized:
                stabilized = len(series) < 100 or _stable(series, 100, 1.5)
        elif st.smoothing_autostop and k - k_end1 >= st.min_smoothing and (
                np.ptp(np.array(series[-50:]), axis=0).max() < 3e-3
                or (k - k_end1 >= _MC_MIN and (k - k_end1) % 10 == 0
                    and _precise(eng, series[-((k - k_end1) - (k - k_end1) // 2):]
                                 if averaging else series[-50:]))):
            total = k
        k += 1

    smoothed = total - k_end1
    window = snapshots[len(snapshots) // 2:]
    # a random effect set to zero inside the window starts it again after that point
    window = window[max((i for i, w in enumerate(window) if w[5] != window[-1][5]), default=-1) + 1:]
    if averaging and window and not provisional:
        for j in range(P):
            eng.coef[j][:] = np.mean([w[0][j] for w in window], 0)
        eng.omega[:] = np.sqrt(np.mean([w[1] for w in window], 0))
        eng.corr[:] = np.mean([w[2] for w in window], 0)
        eng.ea[:] = np.mean([w[3] for w in window], 0)
        eng.eb[:] = np.mean([w[4] for w in window], 0)
    if provisional:
        stabilized = False
    elif smoothed >= 2 * _SETTLE_WINDOW:
        # the smoothed estimates are judged on what they still do at the end; averaged ones on
        # how their running average moves over the second half of its window
        half = np.array(series[-smoothed // 2:])
        if averaging:
            half = np.cumsum(half, 0) / np.arange(1, len(half) + 1)[:, None]
            half = half[len(half) // 2:]
        stabilized = bool((np.abs(half[-1] - half[0]) <= _SETTLE).all())
    if not stabilized and not provisional:
        messages.append("the estimates were still moving when the iterations ended")
    return Trace(names=names, values=np.array(values).reshape(-1, len(names)),
                 indicator=np.array(indicator), n_exploratory=k_end1,
                 n_smoothing=total - k_end1, n_chains=C, stabilized=stabilized, phi=phi,
                 scale2=scale2, scale3=scale3, messages=messages, provisional=provisional)


def _frailty(eng: Engine) -> bool:
    """A random effect reaches a terminal or single hazard output and no output without hazard
    (a frailty)."""
    model = eng.model
    once = [o for o, kind in model.hazards.items() if kind != "repeated"]
    if not once:
        return False
    plain = [o for o in model.outputs if o not in model.hazards]
    reach = {o: set(model.dependencies(o)) for o in once + plain}
    for j, name in enumerate(eng.names[:eng.p0]):
        if (eng.iiv[j] or eng.iov_of[j] >= 0) and any(name in reach[o] for o in once) and not any(
                name in reach[o] for o in plain):
            return True
    return False


def _update_fixed(eng: Engine, s1, sampled, omega):
    """Fixed effects of the sampled parameters: least squares of the simulated individual
    parameters on their designs, generalised within a correlation block. A prior on a typical
    value adds its quadratic penalty, so the solution is the mode of the posterior."""
    sampled = [int(j) for j in sampled]
    group = {j: {j} for j in sampled}
    for j, k in eng.corr_pairs:
        if j in group and k in group:
            merged = group[j] | group[k]
            for q in merged:
                group[q] = merged
    for members in {frozenset(g) for g in group.values()}:
        members = sorted(members)
        if len(members) == 1:
            j = members[0]
            X, free = eng.design[j], eng.coef_free[j]
            if free.any():
                target = s1[:, j] - X[:, ~free] @ eng.coef[j][~free]
                rows = X[:, free]
                if eng.prior_prec[j] > 0 and free[0]:
                    # one more row: the prior as an observation of the intercept, weighted so
                    # that the squares are on the scale of the squares of the subjects
                    w = math.sqrt(eng.prior_prec[j]) * omega[j]
                    row = np.zeros(int(free.sum()))
                    row[0] = w
                    rows = np.vstack([rows, row])
                    target = np.append(target, w * eng.prior_mean[j])
                eng.coef[j][free] = np.linalg.lstsq(rows, target, rcond=None)[0]
            continue
        _, _, prec = eng.prior(omega, np.array(members))
        target = {j: s1[:, j] - eng.design[j][:, ~eng.coef_free[j]] @ eng.coef[j][~eng.coef_free[j]]
                  for j in members}
        unknowns = [(j, c) for j in members for c in np.flatnonzero(eng.coef_free[j])]
        if not unknowns:
            continue
        A = np.empty((len(unknowns), len(unknowns)))
        rhs = np.empty(len(unknowns))
        for a, (j, c) in enumerate(unknowns):
            x = eng.design[j][:, c]
            rhs[a] = sum(prec[j, k] * (x @ target[k]) for k in members)
            for b, (k, c2) in enumerate(unknowns):
                A[a, b] = prec[j, k] * (x @ eng.design[k][:, c2])
            if c == 0 and eng.prior_prec[j] > 0:
                A[a, a] += eng.prior_prec[j]
                rhs[a] += eng.prior_prec[j] * eng.prior_mean[j]
        solution = np.linalg.lstsq(A, rhs, rcond=None)[0]
        for value, (j, c) in zip(solution, unknowns):
            eng.coef[j][c] = value


def jump_vectors(eng: Engine, eta):
    """Per chain and subject, the move from the mode of the chain's own basin to the mode reached
    from the population mean, kept only where that mode is more probable by JUMP_GAIN or more:
    along a flat ridge two starts end at points of similar density, not in two modes."""
    from .post import modes  # post imports this module
    sampled = np.flatnonzero(eng.iiv)

    def density(e):
        return -eng.nll(eng.predict(eng.mean() + e)[0]) + eng.log_prior(e, eng.omega, sampled)
    main = modes(eng, np.zeros((eng.N, eng.P)))
    local = np.stack([modes(eng, eta[c]) for c in range(eta.shape[0])])
    gain = density(main[None])[0] - density(local)
    return np.where((gain >= JUMP_GAIN)[..., None], main[None] - local, 0.0)


def jump(eng: Engine, eta, omega, sampled, d, rng):
    """Mode jumping (Tjelmeland and Hegstad 2001), the fifth kernel: the state
    moves by +d or -d with equal probability, a symmetric proposal accepted with the ratio of
    the posterior densities. Current and proposed states are solved together."""
    C = eta.shape[0]
    sign = np.where(rng.random((C, eng.N)) < 0.5, 1.0, -1.0)
    new = eta + sign[..., None] * d
    both = np.concatenate([eta, new])
    lp = -eng.nll(eng.predict(eng.mean() + both)[0]) + eng.log_prior(both, omega, sampled)
    with np.errstate(invalid="ignore"):
        log_a = np.nan_to_num(lp[C:] - lp[:C], nan=-np.inf, posinf=np.inf)
    accept = (np.log(rng.random((C, eng.N))) < log_a) & (np.abs(d).max(-1) > 0)
    eta[accept] = new[accept]
    return eta


def _update_correlations(eng: Engine, cov):
    for j, k in eng.corr_pairs:
        if cov[j, j] > 0 and cov[k, k] > 0:
            rho = cov[j, k] / math.sqrt(cov[j, j] * cov[k, k])
            eng.corr[j, k] = eng.corr[k, j] = float(np.clip(rho, -CORR_CLIP, CORR_CLIP))
    # clipped pair by pair, the matrix can lose positive definiteness: shrink it towards the
    # identity just enough to bring its smallest eigenvalue back to the floor
    low = float(np.linalg.eigvalsh(eng.corr)[0]) if eng.corr_pairs else 1.0
    if low < CORR_FLOOR:
        lam = (CORR_FLOOR - low) / (1 - low)
        eng.corr[:] = (1 - lam) * eng.corr + lam * np.eye(eng.P)


def _series(eng: Engine) -> np.ndarray:
    """Estimates scaled so that a difference reads as a relative change."""
    scale = np.array([max(abs(eng.coef[j][0]), 1e-8) if eng.tkind[j] == 0 else 1.0
                      for j in range(eng.P)])
    with_omega = eng.iiv.copy()
    with_omega[eng.collapsed] = True
    parts = [np.concatenate([c / scale[j] for j, c in enumerate(eng.coef)]),
             (eng.omega / scale)[with_omega],
             np.array([eng.corr[j, k] for j, k in eng.corr_declared])]
    y = np.abs(eng.data.obs_y)
    for o, which in enumerate(eng.err_names):
        # an exponential error is already relative
        level = 1.0 if eng.elog[o] else max(
            float(np.median(y[eng.data.obs_out == o])) if (eng.data.obs_out == o).any() else 1.0, 1e-12)
        parts.append(np.array([eng.ea[o] / level if w == "a" else eng.eb[o] for w in which]))
    return np.concatenate(parts)


def _precision(eng: Engine) -> np.ndarray:
    """Rough standard error of each quantity of _series, as if every subject informed every
    parameter: omega / sqrt(N) for a typical value or a covariate coefficient (over the spread of
    the covariate), omega / sqrt(2 N) for an omega, (1 - rho^2) / sqrt(N) for a correlation, the
    value / sqrt(2 n) for an error parameter; 1% for a parameter without variability."""
    N = eng.N
    scale = np.array([max(abs(eng.coef[j][0]), 1e-8) if eng.tkind[j] == 0 else 1.0
                      for j in range(eng.P)])
    with_omega = eng.iiv.copy()
    with_omega[eng.collapsed] = True
    parts = []
    for j, c in enumerate(eng.coef):
        w = eng.omega[j] / scale[j] if eng.iiv[j] else 0.01 * math.sqrt(N)
        spread = np.ones(len(c))
        spread[1:] = np.maximum(eng.design[j][:, 1:].std(0), 1e-12)
        parts.append(w / math.sqrt(N) / spread)
    parts.append((eng.omega / scale)[with_omega] / math.sqrt(2 * N))
    parts.append(np.array([1 - eng.corr[j, k] ** 2 for j, k in eng.corr_declared]) / math.sqrt(N))
    y = np.abs(eng.data.obs_y)
    for o, which in enumerate(eng.err_names):
        n = max(int((eng.data.obs_out == o).sum()), 1)
        level = 1.0 if eng.elog[o] else max(float(np.median(y[eng.data.obs_out == o])) if n > 1 else 1.0, 1e-12)
        parts.append(np.array([eng.ea[o] / level if w == "a" else eng.eb[o] for w in which])
                     / math.sqrt(2 * n))
    return np.concatenate(parts)


def _precise(eng: Engine, window, batches: int = 5) -> bool:
    """The average over window is known to within _MC_SHARE of the rough standard error of each
    quantity: Monte Carlo error by batch means (batches of 10 iterations at least)."""
    x = np.array(window)
    if len(x) < 10 * batches:
        return False
    means = x[: len(x) // batches * batches].reshape(batches, -1, x.shape[1]).mean(1)
    mcse = means.std(0, ddof=1) / math.sqrt(batches)
    return bool((mcse <= _MC_SHARE * _precision(eng)).all())


def _omega_columns(eng: Engine, which) -> np.ndarray:
    """Columns of _series holding the omegas of the random effects in which."""
    with_omega = eng.iiv.copy()
    with_omega[eng.collapsed] = True
    n_coef = sum(len(c) for c in eng.coef)
    mask = np.zeros(len(_series(eng)), bool)
    mask[n_coef:n_coef + int(with_omega.sum())] = np.asarray(which, bool)[with_omega]
    return mask


def _stable(series, window, tolerance, floor=2e-3, skip=None) -> bool:
    """No drift over the last window iterations, columns in skip aside.

    The shift between the means of the two halves is compared with the spread of
    the window: a linear trend gives a ratio of 1.7, stationary noise well under 1.
    Shifts below floor (0.2% in relative terms) never count as drift.
    """
    if len(series) < window:
        return False
    x = np.array(series[-window:])
    if skip is not None:
        x = x[:, ~skip]
    shift = np.abs(x[: window // 2].mean(0) - x[window // 2:].mean(0))
    return bool((shift <= np.maximum(tolerance * x.std(0), floor)).all())


def sd_slope(eng: Engine, f, out) -> np.ndarray:
    """d g / d f of the residual sd."""
    k, a, b = eng.ek[out], eng.ea[out], eng.eb[out]
    g = eng.residual_sd(f, out)
    return np.select([k == 0, k == 3], [0.0 * f, b * b * f / g], b * np.sign(f))


def _hazard_slopes(eng: Engine, dH, h):
    """Slopes of the log-likelihood of each record of a hazard output (columns of eng.on_hazard)
    in dH, the events expected since the previous record, and in h, the hazard at the record:
    zero where a floor holds the term. The slope in dH is NaN where
    the term is not finite, which drops the record."""
    d, on = eng.data, eng.on_hazard
    k, cens, repeated = eng.obs_y[on], d.obs_cens[on], eng.hrep[d.obs_out[on]]
    later = eng.obs_prev[on] >= 0
    exact = later & (k > 0) & (cens == 0)
    box = later & (k > 0) & (cens == 1)
    tiny = 1e-300
    with np.errstate(divide="ignore", invalid="ignore"):
        once = -np.expm1(-dH)
        a = np.select([box & repeated, box, later],
                      [np.where(dH > tiny, k / dH, 0.0) - 1.0,
                       np.where(once > tiny, np.exp(-dH) / once, 0.0), -1.0], 0.0)
        b = np.where(exact & (h > tiny), k / h, 0.0)
    lost = later & ((h < 0) | (exact & ~np.isfinite(h)))
    return np.where(lost, np.nan, a), b


def sensitivities(eng: Engine, phi, idx, pred0=None, h=1e-3, exact=None):
    """d f / d coefficient of the parameters idx: exact (Engine.derivatives) or, with
    sensitivities "fd", by forward differences in one batched solve.

    Returns the predictions at phi (chains, grid, outputs), their values at the
    observations (chains, n_obs), the derivatives (chains, n_obs, q), the
    (parameter, coefficient) owner of each column and the derivatives with respect
    to each parameter of idx itself (chains, n_obs, len(idx)).
    On a record of a hazard output, the derivatives are those of its log-likelihood.
    """
    if eng.sensitivities == "exact" if exact is None else exact:
        return _exact_sensitivities(eng, phi, idx, h)
    C = phi.shape[0]
    stack = np.concatenate([phi] * (len(idx) + (pred0 is None)))
    first = int(pred0 is None)
    for k, j in enumerate(idx):
        stack[(first + k) * C:(first + k + 1) * C, :, j] += h
    pred = eng.predict(stack)[0]
    if pred0 is None:
        pred0 = pred[:C]
    f0 = eng.gather(pred0)
    on = eng.on_hazard
    hazards = bool(on.any())
    if hazards:
        # chain rule: the slopes of the record's log-likelihood at phi times the differences of
        # its two inputs; differencing the log-likelihood itself biases the estimates
        dH0, hz0 = (x[:, on] for x in eng.hazard_parts(pred0))
        a, b = _hazard_slopes(eng, dH0, hz0)
        live = b != 0
    cols, owners, raw = [], [], []
    for k, j in enumerate(idx):
        shifted = pred[(first + k) * C:(first + k + 1) * C]
        s = (eng.gather(shifted) - f0) / h
        if hazards:
            dH1, hz1 = (x[:, on] for x in eng.hazard_parts(shifted))
            # the hazard is read at events only: elsewhere it may be infinite
            s_on = a * (dH1 - dH0)
            s_on[live] += b[live] * (hz1[live] - hz0[live])
            s[:, on] = s_on / h
        raw.append(s)
        for c in np.flatnonzero(eng.coef_free[j]):
            cols.append(s * eng.design[j][eng.obs_subject, c])
            owners.append((j, c))
    D = np.stack(cols, -1) if cols else np.zeros(f0.shape + (0,))
    return pred0, f0, D, owners, np.stack(raw, -1) if raw else np.zeros(f0.shape + (0,))


def _exact_sensitivities(eng: Engine, phi, idx, h):
    """sensitivities from the derivatives of the predictions, by the chain rule on the scale of
    the error model and, on a hazard record, through dH and h."""
    pred0, dpred = eng.derivatives(phi, idx, h)
    d = eng.data
    f0 = eng.gather(pred0)
    raw = dpred[:, d.obs_grid, d.obs_out]
    on_log = eng.elog[d.obs_out]
    if on_log.any():
        f = pred0[:, d.obs_grid, d.obs_out]
        with np.errstate(divide="ignore", invalid="ignore"):
            raw[:, on_log] /= np.where(f > 1e-300, f, np.inf)[:, on_log, None]
    on = eng.on_hazard
    if on.any():
        dH0, hz0 = (x[:, on] for x in eng.hazard_parts(pred0))
        a, b = _hazard_slopes(eng, dH0, hz0)
        g, o = d.obs_grid[on], d.obs_out[on]
        gp = np.where(eng.obs_prev[on] >= 0, eng.obs_prev[on] + d.grid_ptr[d.obs_subject[on]], g)
        H = pred0[:, g, o] - pred0[:, gp, o]
        dH = np.where((H > 0)[..., None], dpred[:, g, o] - dpred[:, gp, o], 0.0)
        # the hazard is read at events only: elsewhere it may be infinite
        live = (b != 0)[..., None]
        dh = np.where(live, dpred[:, g, eng.hcol[o]], 0.0)
        raw[:, on] = a[..., None] * dH + np.where(live, b[..., None] * dh, 0.0)
    cols, owners = [], []
    for k, j in enumerate(idx):
        for c in np.flatnonzero(eng.coef_free[j]):
            cols.append(raw[..., k] * eng.design[j][eng.obs_subject, c])
            owners.append((j, c))
    D = np.stack(cols, -1) if cols else np.zeros(f0.shape + (0,))
    return pred0, f0, D, owners, raw


def _score_terms(eng: Engine, f):
    """Per observation: d loglik / d f and its expected negative curvature. On a record of a
    hazard output both are 1: sensitivities already gives the slopes of its log-likelihood."""
    d = eng.data
    y, out = eng.obs_y, d.obs_out
    g = eng.residual_sd(f, out)
    gf = sd_slope(eng, f, out)
    r = y - f
    c1 = r / g**2 + gf / g * (r**2 / g**2 - 1)
    w = 1 / g**2 + 2 * gf**2 / g**2
    floor = g <= _G_MIN
    c1, w = np.where(floor, 0.0, c1), np.where(floor, 0.0, w)
    on = eng.on_hazard
    cens = (d.obs_cens != 0) & ~on
    if cens.any():
        # censored records: slope of their log-probability by central differences
        step = 1e-5 * np.maximum(np.abs(f[:, cens]), 1e-8)
        up = _censored_logp(eng, f[:, cens] + step, cens)
        dn = _censored_logp(eng, f[:, cens] - step, cens)
        c1[:, cens] = (up - dn) / (2 * step)
        w[:, cens] = c1[:, cens] ** 2
    if on.any():
        # the information is the sum of the outer products of the records' slopes (Berndt et
        # al. 1974), as for censored records
        c1[:, on], w[:, on] = 1.0, 1.0
    return c1, w


def _censored_logp(eng: Engine, f, sel):
    from scipy.special import ndtr
    d = eng.data
    y, lim, cens, out = eng.obs_y[sel], eng.obs_limit[sel], d.obs_cens[sel], d.obs_out[sel]
    g = eng.residual_sd(f, out)
    at_y = ndtr((y - f) / g)
    at_l = np.where(np.isfinite(lim), ndtr((lim - f) / g), np.where(cens > 0, 0.0, 1.0))
    return np.log(np.maximum(np.where(cens > 0, at_y - at_l, at_l - at_y), 1e-300))


class _Follow:
    """Shift of each subject's random effects per unit move of the fixed effects that
    act through the model alone: -(J' W J + Omega^-1)^-1 J' W K.

    J' W J and J' W K are running means with their own gain 1/n: the SAEM gain is 1
    in the exploratory phase, which would tie the shift to the previous draw.
    """

    def __init__(self):
        self.key = None
        self.jwj = self.jwk = None
        self.n = 0

    def update(self, key, jwj, jwk):
        if key != self.key:
            self.key, self.jwj, self.jwk, self.n = key, jwj, jwk, 1
        else:
            self.n += 1
            self.jwj += (jwj - self.jwj) / self.n
            self.jwk += (jwk - self.jwk) / self.n

    def track(self, key, prec):
        """(subjects, random effects, fixed effects), None before the first update."""
        if key != self.key:
            return None
        return -np.linalg.solve(self.jwj + prec, self.jwk)


def _scoring(eng: Engine, phi, idx, follow=None):
    """One Fisher-scoring step on the fixed effects idx.

    With follow, a fixed effect that acts through the model alone moves with the
    random effects that offset it. Returns the coefficient increments by parameter
    and the shift of the random effects (subjects, parameters), zero when no step
    lowers the misfit.
    """
    none = {j: np.zeros_like(eng.coef[j]) for j in idx}, np.zeros((eng.N, eng.P))
    iiv = np.flatnonzero(eng.iiv)
    wanted = follow is not None and iiv.size > 0 and bool((~eng.iiv[idx]).any())
    every = np.union1d(idx, iiv).astype(int) if wanted else np.asarray(idx, int)
    # differences: with exact ones, 2 of 5 seeds of a saturable replicate stopped 0.9 to 1.1 higher
    pred0, f0, D, owners, raw = sensitivities(eng, phi, every, exact=False)
    owned = set(int(v) for v in idx)
    keep = [q for q, (j, _) in enumerate(owners) if j in owned]
    D, owners = D[..., keep], [owners[q] for q in keep]
    if not owners:
        return none
    c1, w = _score_terms(eng, f0)
    valid = np.isfinite(f0) & np.isfinite(D).all(-1) & np.isfinite(raw).all(-1)
    c1, w, D = np.where(valid, c1, 0.0), np.where(valid, w, 0.0), np.where(valid[..., None], D, 0)
    C = phi.shape[0]
    track = np.zeros((eng.N, eng.P, len(owners)))
    grad_prior = np.zeros(len(owners))
    info_prior = np.zeros((len(owners), len(owners)))
    eta = phi - eng.mean()
    moving = False
    if wanted:
        ptr = eng.data.obs_ptr[:-1]
        J = np.where(valid[..., None], raw[..., np.searchsorted(every, iiv)], 0.0)
        alone = [q for q, (j, _) in enumerate(owners) if not eng.iiv[j]]
        key = (tuple(iiv), tuple(owners[q] for q in alone))
        prec = eng.prior(eng.omega, iiv)[2][np.ix_(iiv, iiv)]
        G = follow.track(key, prec)
        follow.update(
            key, np.add.reduceat(np.einsum("co,coa,cob->oab", w, J, J), ptr, 0) / C,
            np.add.reduceat(np.einsum("co,coa,coq->oaq", w, J, D[..., alone]), ptr, 0) / C)
        if G is not None and np.isfinite(G).all():
            moving = True
            track[np.ix_(range(eng.N), iiv, alone)] = G
            # the predictions respond to the fixed effect net of the shift it brings
            D = D + np.einsum("coa,oaq->coq", J, track[eng.obs_subject][:, iiv])
            pull = np.einsum("ab,nbq->naq", prec, track[:, iiv])
            grad_prior = -np.einsum("cna,naq->q", eta[:, :, iiv], pull)
            info_prior = C * np.einsum("nap,naq->pq", track[:, iiv], pull)
    # a prior on a typical value, counted once per chain as the data are
    with_prior = [(q, j) for q, (j, c) in enumerate(owners) if c == 0 and eng.prior_prec[j] > 0]
    for q, j in with_prior:
        grad_prior[q] -= C * eng.prior_prec[j] * (eng.coef[j][0] - eng.prior_mean[j])
        info_prior[q, q] += C * eng.prior_prec[j]
    grad = np.einsum("co,coq->q", c1, D) + grad_prior
    info = np.einsum("co,coq,cor->qr", w, D, D) + info_prior
    if not (np.isfinite(grad).all() and np.isfinite(info).all()):
        return none
    step = np.linalg.lstsq(info + 1e-10 * np.eye(len(grad)) * np.trace(info), grad,
                           rcond=None)[0]
    biggest = np.abs(step).max()
    if not np.isfinite(biggest) or biggest == 0.0:
        return none
    step *= min(1.0, 1.0 / biggest)
    # the four step lengths of the line search in one batched solve
    alphas = (1.0, 0.5, 0.25, 0.125)
    deltas, moves = [], []
    stack = np.concatenate([phi] * len(alphas))
    for a, alpha in enumerate(alphas):
        delta = {j: np.zeros_like(eng.coef[j]) for j in idx}
        for q, (j, c) in enumerate(owners):
            delta[j][c] = alpha * step[q]
        moves.append(alpha * (track @ step))
        stack[a * C:(a + 1) * C] += moves[-1]
        for j in idx:
            stack[a * C:(a + 1) * C, :, j] += eng.design[j] @ delta[j]
        deltas.append(delta)

    def misfit(nll, moved, delta=None):
        # the prior of the random effects changes only when they move with the step
        total = nll.sum()
        if moving:
            total -= eng.log_prior(eta + moved, eng.omega, iiv).sum()
        for q, j in with_prior:
            b0 = eng.coef[j][0] + (delta[j][0] if delta else 0.0)
            total += 0.5 * C * eng.prior_prec[j] * (b0 - eng.prior_mean[j]) ** 2
        return total

    base = misfit(eng.nll(pred0), 0.0)
    trial = eng.nll(eng.predict(stack)[0]).reshape(len(alphas), C, -1)
    for a in range(len(alphas)):
        if misfit(trial[a], moves[a], deltas[a]) < base:
            return deltas[a], moves[a]
    return none


def _update_error(eng: Engine, f, gamma, s_err, annealing, rate):
    d = eng.data
    # an error parameter is annealed as the variances are: it loses at most
    # 1 - rate per iteration
    floor = rate
    for o in range(eng.model.n_outputs):
        sel = d.obs_out == o
        n_o = int(sel.sum())
        if not eng.err_free[o] or n_o == 0:
            continue
        fo, yo = f[:, sel], eng.obs_y[sel]
        rows = np.isfinite(fo).all(1)
        if not rows.any():
            continue
        fo = fo[rows]
        kind = eng.ek[o]
        old_a, old_b = eng.ea[o], eng.eb[o]
        if kind in (0, 1) and not d.obs_cens[sel].any():
            if kind == 0:
                stat = ((yo - fo) ** 2).mean(1).mean()
            else:
                # a prediction of zero leaves the sd at its floor: no information on b
                live = old_b * np.abs(fo) > _G_MIN
                res = np.where(live, (yo - fo) / np.where(live, fo, 1.0), 0.0)
                stat = ((res**2).sum(1) / np.maximum(live.sum(1), 1)).mean()
            s_err[o] += gamma * (stat - s_err[o])
            value = math.sqrt(s_err[o])
            if kind == 0:
                eng.ea[o] = value
            else:
                eng.eb[o] = value
        else:
            # the floors bound the maximisation itself: with a held up, b gets the
            # value that goes with it, not the one of a free a
            a, b = _maximise_error(eng, o, yo, fo, sel, floor if annealing else 0.0)
            eng.ea[o] += gamma * (a - old_a)
            eng.eb[o] += gamma * (b - old_b)
        if annealing:
            eng.ea[o] = max(eng.ea[o], floor * old_a)
            eng.eb[o] = max(eng.eb[o], floor * old_b)


def _maximise_error(eng: Engine, o, y, f, sel, floor=0.0):
    """Error parameters maximising the likelihood of the simulated residuals, none
    below floor times its current value."""
    kind = eng.ek[o]
    cens, lim = eng.data.obs_cens[sel], eng.obs_limit[sel]
    has_a, has_b = "a" in eng.err_names[o], "b" in eng.err_names[o]

    def unpack(x):
        x = np.exp(x)
        a = x[0] if has_a else 0.0
        b = x[-1] if has_b else 0.0
        return a, b

    def objective(x):
        a, b = unpack(x)
        g = np.maximum(np.select([kind == 0, kind == 1, kind == 2],
                                 [a + 0 * f, b * np.abs(f), a + b * np.abs(f)],
                                 np.sqrt(a * a + b * b * f * f)), _G_MIN)
        z = (y - f) / g
        nll = 0.5 * z * z + np.log(g)
        if cens.any():
            from scipy.special import ndtr
            at_l = np.where(np.isfinite(lim), ndtr((lim - f) / g), np.where(cens > 0, 0.0, 1.0))
            p = np.where(cens > 0, ndtr(z) - at_l, at_l - ndtr(z))
            nll = np.where(cens != 0, -np.log(np.maximum(p, 1e-300)), nll)
        return float(nll.sum())

    start = np.log([v for v, has in ((eng.ea[o], has_a), (eng.eb[o], has_b)) if has])
    bounds = [(v + math.log(floor), None) for v in start] if floor > 0 else None
    res = minimize(objective, start, method="Nelder-Mead", bounds=bounds,
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 200})
    return unpack(res.x)
