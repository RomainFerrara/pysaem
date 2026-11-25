"""After the estimation: conditional distributions, modes, likelihood, Fisher
information, predictions, residuals and simulations."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats

from .data import Dataset
from .engine import Engine, _hazard_nll, _links, _rand
from .population import inverse, jacobian, transform
from .saem import Settings, Trace, _score_terms, jump, jump_vectors, sensitivities
from .solve import _METHODS, Problem


@dataclass
class Conditional:
    """Conditional distribution of the random effects, on the Gaussian scale."""

    mean: np.ndarray        # (subjects, P) posterior mean of eta
    sd: np.ndarray          # (subjects, P)
    cov: np.ndarray         # (subjects, P, P) posterior covariance of eta
    best: np.ndarray        # (subjects, P) most probable sampled eta
    samples: np.ndarray     # (draws, subjects, P)


def theta_index(eng: Engine) -> list:
    """Estimated quantities in Fisher-information order."""
    idx = []
    for j in np.flatnonzero(eng.iiv):
        idx += [("coef", j, c) for c in np.flatnonzero(eng.coef_free[j])]
    idx += [("omega", j, 0) for j in np.flatnonzero(eng.iiv & eng.omega_free) if eng.tie[j] == j]
    idx += [("corr", j, k) for j, k in eng.corr_pairs if (j, k) not in eng.corr_boundary]
    for j in np.flatnonzero(~eng.iiv):
        idx += [("coef0", j, c) for c in np.flatnonzero(eng.coef_free[j])]
    for o in np.flatnonzero(eng.err_free):
        idx += [("err", o, w) for w in eng.err_names[o]]
    return idx


def theta_name(eng: Engine, entry) -> str:
    kind, j, c = entry
    if kind == "omega":
        return eng.omega_name(j)
    if kind == "corr":
        return f"corr_{eng.names[j]}_{eng.names[c]}"
    if kind == "err":
        return eng.population.error_name(eng.model.outputs[j], c)
    return f"{eng.names[j]}_pop" if c == 0 else f"beta_{eng.names[j]}_{eng.cov_names[j][c - 1]}"


class _Marginal:
    """The importance-sampling log-likelihood as a function of the estimated values.

    Draws and proposal are fixed, so the function is smooth and its Hessian is the
    observed information. Values live on the scale they are unconstrained on:
    Gaussian scale for fixed effects, log for omegas and error parameters, atanh
    for correlations. Predictions are solved again only when a fixed effect
    without variability moves; the draws are then translated by track.
    """

    def __init__(self, eng: Engine, eta, log_q, pred):
        self.eng, self.log_q = eng, log_q
        self.phi = eng.mean() + eta
        self.index = theta_index(eng)
        self.data_cols = [q for q, e in enumerate(self.index) if e[0] in ("coef0", "err")]
        self.solve_cols = [q for q, e in enumerate(self.index) if e[0] == "coef0"]
        self.prior_cols = [q for q, e in enumerate(self.index) if e[0] != "err"]
        self.u0 = np.array([self._get(e) for e in self.index])
        self.track = np.zeros((eng.N, eng.P, len(self.solve_cols)))
        self._pred = {tuple(self.u0[self.solve_cols]): pred}
        self._data, self._prior = {}, {}

    def follow(self, eta_mode, q, step):
        """Set the track of column q: the move of each subject's conditional mode per
        unit of that effect, by a forward difference over step."""
        eng = self.eng
        if not eng.iiv.any():
            return
        _, j, c = self.index[q]
        eng.coef[j][c] += step
        try:
            move = (modes(eng, eta_mode) - eta_mode) / step
        finally:
            eng.coef[j][c] = self.u0[q]
        self.track[:, :, self.solve_cols.index(q)] = np.where(np.isfinite(move), move, 0.0)

    def forget(self, u):
        """Drop what is cached at u."""
        for cache, cols in ((self._pred, self.solve_cols), (self._data, self.data_cols),
                            (self._prior, self.prior_cols)):
            cache.pop(tuple(u[cols]), None)

    def _phi(self, u):
        """Individual parameters of the draws when the effects that act through the
        model alone sit at u."""
        eng = self.eng
        phi = self.phi + self.track @ (u[self.solve_cols] - self.u0[self.solve_cols])
        for j in {e[1] for e in self.index if e[0] == "coef0"}:
            phi[:, :, j] = eng.design[j] @ eng.coef[j]
        return phi

    def _get(self, entry):
        kind, j, c = entry
        eng = self.eng
        if kind in ("coef", "coef0"):
            return eng.coef[j][c]
        if kind == "omega":
            return math.log(eng.omega[j])
        if kind == "corr":
            return math.atanh(eng.corr[j, c])
        return math.log(eng.ea[j] if c == "a" else eng.eb[j])

    def _set(self, entry, u):
        kind, j, c = entry
        eng = self.eng
        if kind in ("coef", "coef0"):
            eng.coef[j][c] = u
        elif kind == "omega":
            # a gamma moves the sd of every occasion of its parameter
            eng.omega[eng.tie == j] = math.exp(u)
        elif kind == "corr":
            eng.corr[j, c] = eng.corr[c, j] = math.tanh(u)
        elif c == "a":
            eng.ea[j] = math.exp(u)
        else:
            eng.eb[j] = math.exp(u)

    def steps(self) -> np.ndarray:
        eng = self.eng
        h = np.empty(len(self.index))
        for q, (kind, j, c) in enumerate(self.index):
            if kind in ("omega", "corr"):
                h[q] = 0.02
            elif kind == "err":
                h[q] = 0.01
            else:
                spread = 1.0 if c == 0 else max(float(eng.design[j][:, c].std()), 1e-12)
                if kind == "coef":
                    h[q] = 0.02 * max(eng.omega[j], 1e-3) / spread
                else:
                    level = max(abs(eng.coef[j][0]), 1e-6) if eng.tkind[j] == 0 else 1.0
                    h[q] = 5e-3 * level / spread
        return h

    def value(self, u) -> np.ndarray:
        """Log-likelihood over all draws, then over the even and the odd draws."""
        eng = self.eng
        key_d = tuple(u[self.data_cols])
        key_p = tuple(u[self.prior_cols])
        if key_d not in self._data or key_p not in self._prior:
            for entry, v in zip(self.index, u):
                self._set(entry, v)
            try:
                if key_d not in self._data:
                    key_s = tuple(u[self.solve_cols])
                    if key_s not in self._pred:
                        self._pred[key_s] = eng.predict(self._phi(u))[0]
                    self._data[key_d] = -eng.nll(self._pred[key_s])
                if key_p not in self._prior:
                    iiv = np.flatnonzero(eng.iiv)
                    try:
                        self._prior[key_p] = eng.log_prior(self._phi(u) - eng.mean(), eng.omega, iiv)
                    except np.linalg.LinAlgError:
                        # a step on a correlation can leave the positive definite matrices
                        self._prior[key_p] = -np.inf
            finally:
                for entry, v in zip(self.index, self.u0):
                    self._set(entry, v)
        lw = self._data[key_d] + self._prior[key_p] - self.log_q
        # a prior on a typical value: its log density is part of what is maximised
        penalty = sum(-0.5 * eng.prior_prec[j] * (v - eng.prior_mean[j]) ** 2
                      for (kind, j, c), v in zip(self.index, u)
                      if c == 0 and kind in ("coef", "coef0") and eng.prior_prec[j] > 0)
        if len(lw) < 2:
            return np.full(3, _logsumexp(lw).sum() + penalty)
        return np.array([_logsumexp(x).sum() + penalty for x in (lw, lw[0::2], lw[1::2])])


def information(eng: Engine, sample, eta_mode) -> tuple:
    """Observed information on the unconstrained scale, by central differences of the
    importance-sampling log-likelihood, with the index of its rows and the same matrix
    from the even and from the odd draws. eta_mode holds the conditional modes, which
    the draws follow."""
    f = _Marginal(eng, *sample)
    T = len(f.index)
    if T == 0:
        return None, [], None
    u0, h = f.u0, f.steps()
    f0 = f.value(u0)

    def point(*moves):
        u = u0.copy()
        for q, sign in moves:
            u[q] = u0[q] + sign * h[q]
        return u

    def at(*moves):
        return f.value(point(*moves))

    for q in f.solve_cols:
        # the scale of such an effect is not known in advance: aim for a second
        # difference of 0.2 log-likelihood units; a step not measured in
        # [0.02, 1] is not kept
        start = h[q]
        for _ in range(8):
            f.follow(eta_mode, q, h[q])
            bend = (2 * f0 - at((q, 1)) - at((q, -1)))[0]
            if 0.02 <= bend <= 1.0:
                break
            f.forget(point((q, 1)))
            f.forget(point((q, -1)))
            h[q] *= float(np.clip(math.sqrt(0.2 / bend), 0.01, 100.0)) if bend > 0 else 10.0
        else:
            h[q] = start
            f.follow(eta_mode, q, start)

    H = np.empty((3, T, T))
    side = np.array([[at((a, 1)), at((a, -1))] for a in range(T)])
    # a step that leaves the parameter space (a correlation block at its boundary) is halved,
    # six times at most; a quantity still outside carries no measured information
    for a in range(T):
        for _ in range(6):
            if np.isfinite(side[a]).all():
                break
            f.forget(point((a, 1)))
            f.forget(point((a, -1)))
            h[a] *= 0.5
            side[a] = [at((a, 1)), at((a, -1))]
    lost = ~np.isfinite(side).all((1, 2))
    for a in range(T):
        bend = side[a].sum(0) - 2 * f0
        # a second difference at rounding level is no information
        H[:, a, a] = np.where(np.abs(bend) > 1e-10 * np.maximum(1.0, np.abs(f0)), bend / h[a] ** 2, 0.0)
        for b in range(a + 1, T):
            if lost[a] or lost[b]:
                H[:, a, b] = H[:, b, a] = 0.0
                continue
            # second order with two new points instead of four; a pair of steps that leaves
            # the parameter space together gives no cross-information
            cross = (at((a, 1), (b, 1)) + at((a, -1), (b, -1)) - side[a].sum(0) - side[b].sum(0)
                     + 2 * f0) / (2 * h[a] * h[b])
            H[:, a, b] = H[:, b, a] = np.where(np.isfinite(cross), cross, 0.0)
    H[:, lost, :] = 0.0
    H[:, :, lost] = 0.0
    return -H[0], f.index, -H[1:]


def conditional(eng: Engine, st: Settings, trace: Trace, keep: int = 10) -> Conditional:
    """Sample each subject's random effects with the population fixed.

    keep is the number of iterations whose draws are returned in samples.
    """
    N, P = eng.N, eng.P
    sampled = np.flatnonzero(eng.iiv)
    ns = len(sampled)
    mean = eng.mean()
    if ns == 0:
        zero = np.zeros((N, P))
        return Conditional(zero, zero, np.zeros((N, P, P)), zero, np.zeros((1, N, P)))
    # enough chains for CONDITIONAL_DRAWS per subject: the draws only centre and scale the proposal
    C = min(trace.n_chains, max(1, math.ceil(CONDITIONAL_DRAWS / max(st.n_conditional, 1))))
    eta = trace.phi[:C] - mean
    eta[:, :, ~eng.iiv] = 0.0
    rng = eng.rng_states(C, stream=1)
    scale2, scale3 = trace.scale2.copy(), trace.scale3.copy()
    # during the burn-in, every chain is offered the mode reached from the population mean
    d, jump_rng = jump_vectors(eng, eta), np.random.default_rng([eng.seed, 6])
    total = np.zeros((N, P))
    total2 = np.zeros((N, P, P))
    count = np.zeros((N, 1))
    best = eta[0].copy()
    best_u = np.full(N, np.inf)
    samples = []
    every = max(1, st.n_conditional // max(1, keep))
    for it in range(st.n_conditional_burn + st.n_conditional):
        if it < st.n_conditional_burn:
            eta = jump(eng, eta, eng.omega, sampled, d, jump_rng)
        block = (it % (ns - 1)) + 2 if ns > 1 else 0
        uy, pred, acc2, acc3 = eng.estep(eta, rng, eng.omega, sampled, st.mcmc, block, scale2,
                                         scale3[:, block])
        if it < st.n_conditional_burn:
            scale2[sampled] *= 1 + 0.4 * (acc2[sampled] - 0.4)
            moved = sampled[np.isfinite(acc3[sampled])]
            scale3[moved, block] *= 1 + 0.4 * (acc3[moved] - 0.4)
            continue
        ok = np.isfinite(uy)
        kept = np.where(ok[..., None], eta, 0.0)
        total += kept.sum(0)
        total2 += np.einsum("cna,cnb->nab", kept, kept)
        count += ok.sum(0)[:, None]
        post = uy - eng.log_prior(eta, eng.omega, sampled)
        c_best = post.argmin(0)
        u_best = post[c_best, np.arange(N)]
        better = u_best < best_u
        best[better] = eta[c_best[better], np.flatnonzero(better)]
        best_u[better] = u_best[better]
        if (it - st.n_conditional_burn) % every == 0:
            samples.append(eta[0].copy())
    count = np.maximum(count, 1)
    m = total / count
    cov = total2 / count[:, :, None] - m[:, :, None] * m[:, None, :]
    sd = np.sqrt(np.maximum(np.einsum("naa->na", cov), 0.0))
    return Conditional(m, sd, cov, best, np.array(samples))


def modes(eng: Engine, start: np.ndarray) -> np.ndarray:
    """Conditional modes of the random effects by a batched BFGS, one problem per subject."""
    sampled = np.flatnonzero(eng.iiv)
    ns, N = len(sampled), eng.N
    eta = np.zeros((N, eng.P))
    if ns == 0:
        return eta
    mean = eng.mean()
    w = eng.omega[sampled]
    cov, _, prec = eng.prior(eng.omega, sampled)
    cov, prec = cov[np.ix_(sampled, sampled)], prec[np.ix_(sampled, sampled)]

    def at(x, wanted):
        # a subject whose search has ended is not solved again: a row that is not finite ends
        # its solve at once, where a derivative solve that crawls or fails is paid in full at
        # every step of the search
        e = np.zeros((N, eng.P))
        e[:, sampled] = x
        phi = mean + e
        phi[~wanted] = np.nan
        return phi[None]

    def objective(x, wanted):
        u = eng.nll(eng.predict(at(x, wanted))[0])[0]
        return u + 0.5 * np.einsum("na,ab,nb->n", x, prec, x)

    def gradient(x, wanted):
        if eng.sensitivities == "exact":
            _, f, _, _, raw = sensitivities(eng, at(x, wanted), sampled)
            c1 = _score_terms(eng, f)[0][0]
            g = np.zeros((N, ns))
            np.add.at(g, eng.obs_subject, -c1[:, None] * raw[0])
            return g + x @ prec
        g = np.empty_like(x)
        for a in range(ns):
            h = np.zeros(ns)
            h[a] = 1e-3 * w[a]
            g[:, a] = (objective(x + h, wanted) - objective(x - h, wanted)) / (2 * h[a])
        return g

    everyone = np.ones(N, bool)
    x = start[:, sampled].copy()
    f = objective(x, everyone)
    bad = ~np.isfinite(f)
    x[bad] = 0.0
    f = objective(x, everyone)
    g = gradient(x, everyone)
    H = np.repeat(cov[None], N, 0)
    # without a derivative a subject keeps its start
    active = np.isfinite(f) & np.isfinite(g).all(1)
    for _ in range(60):
        if not active.any():
            break
        p = -np.einsum("nab,nb->na", H, g)
        slope = (g * p).sum(1)
        # a first step of at most one omega per effect: far from the mode of informative data the
        # gradient is huge, and twelve halvings of a full step did not reach a decrease
        alpha = np.minimum(1.0, 1.0 / np.maximum(np.abs(p / w).max(1), 1e-12))
        x_new, f_new = x.copy(), f.copy()
        todo = active.copy()
        for _ in range(12):
            trial = x + alpha[:, None] * p
            ft = objective(trial, todo)
            good = todo & np.isfinite(ft) & (ft <= f + 1e-4 * alpha * slope)
            x_new[good], f_new[good] = trial[good], ft[good]
            todo &= ~good
            if not todo.any():
                break
            alpha[todo] *= 0.5
        s = x_new - x
        g_new = gradient(x_new, active)
        g_new[~active] = g[~active]
        yv = g_new - g
        sy = (s * yv).sum(1)
        upd = active & (sy > 1e-12)
        if upd.any():
            rho = 1.0 / sy[upd]
            I = np.eye(ns)[None]
            A = I - rho[:, None, None] * s[upd][:, :, None] * yv[upd][:, None, :]
            H[upd] = (A @ H[upd] @ A.transpose(0, 2, 1)
                      + rho[:, None, None] * s[upd][:, :, None] * s[upd][:, None, :])
        x, f, g = x_new, f_new, np.where(np.isfinite(g_new), g_new, 0.0)
        active = active & ~todo & (np.abs(g * w).max(1) > 1e-5) & (np.abs(s / w).max(1) > 1e-7)
    eta[:, sampled] = x
    return eta


def loglik(eng: Engine, st: Settings, cond: Conditional):
    """Log-likelihood by importance sampling.

    The proposal of a subject is a multivariate Student distribution with 5 degrees
    of freedom, centred on its conditional mean with its conditional covariance.
    Returns the value, its standard error and the first draws (eta, log proposal
    density, predictions) for the information matrix.
    """
    sampled = np.flatnonzero(eng.iiv)
    ns, N = len(sampled), eng.N
    mean = eng.mean()
    if ns == 0:
        pred = eng.predict(mean[None])[0]
        return (-float(eng.nll(pred)[0].sum()), 0.0,
                (np.zeros((1, N, eng.P)), np.zeros((1, N)), pred))
    w = eng.omega[sampled]
    m = cond.mean[:, sampled]
    cov = cond.cov[np.ix_(range(N), sampled, sampled)] + np.diag((1e-3 * w) ** 2)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.zeros_like(cov)
        L[:, range(ns), range(ns)] = np.maximum(cond.sd[:, sampled], 1e-3 * w)
    log_det = np.log(np.einsum("naa->na", L)).sum(1)
    rng = np.random.default_rng([eng.seed, 2])
    df = 5
    const = (math.lgamma((df + ns) / 2) - math.lgamma(df / 2) - 0.5 * ns * math.log(df * math.pi))
    acc = np.full(N, -np.inf)
    acc2 = np.full(N, -np.inf)
    M = max(1, st.n_importance)
    batch = max(1, min(M, 60000 // max(N, 1)))
    keep = min(M, st.n_information, max(200, 20_000_000 // max(eng.data.n_obs, 1)))
    kept = []
    done = 0
    while done < M:
        B = min(batch, M - done)
        t = rng.standard_normal((B, N, ns)) / np.sqrt(rng.chisquare(df, (B, N, 1)) / df)
        eta = np.zeros((B, N, eng.P))
        eta[:, :, sampled] = m + np.einsum("nab,mnb->mna", L, t)
        pred = eng.predict(mean + eta)[0]
        log_q = const - log_det - 0.5 * (df + ns) * np.log1p((t**2).sum(-1) / df)
        lw = -eng.nll(pred) + eng.log_prior(eta, eng.omega, sampled) - log_q
        if sum(len(k[0]) for k in kept) < keep:
            n = keep - sum(len(k[0]) for k in kept)
            kept.append((eta[:n], log_q[:n], pred[:n]))
        acc = np.logaddexp(acc, _logsumexp(lw))
        acc2 = np.logaddexp(acc2, _logsumexp(2 * lw))
        done += B
    ll_i = acc - math.log(M)
    rel_var = np.maximum(np.exp(acc2 - 2 * acc) - 1.0 / M, 0.0)
    sample = tuple(np.concatenate([k[q] for k in kept]) for q in range(3))
    return float(ll_i.sum()), float(np.sqrt(rel_var.sum())), sample


def _logsumexp(a):
    top = a.max(0)
    top = np.where(np.isfinite(top), top, 0.0)
    return top + np.log(np.exp(a - top).sum(0))


# draws per subject of the conditional distributions after the iterations
CONDITIONAL_DRAWS = 1000

# a standard error that unresolved directions of the information change by more than this
# share is reported as imprecise
MC_ERROR_MAX = 0.2


def _report_scale(eng: Engine, entry) -> float:
    """d (reported value) / d (unconstrained value) of one estimated quantity."""
    kind, j, c = entry
    if kind in ("coef", "coef0"):
        return float(jacobian(eng.tkind[j], eng.tlo[j], eng.thi[j], eng.typical()[j])) \
            if c == 0 else 1.0
    if kind == "omega":
        return float(eng.omega[j])
    if kind == "corr":
        return float(1 - eng.corr[j, c] ** 2)
    return float(eng.ea[j] if c == "a" else eng.eb[j])


def _covariance(info, error=None):
    """Covariance of the estimates on the unconstrained scale, the rows without any
    information and the rows whose variance is imprecise.

    On the matrix scaled to unit diagonal, an eigenvalue is resolved when it exceeds twice
    its Monte Carlo error, v' E v for its eigenvector v and the scaled error matrix E, and
    1e-8. A positive eigenvalue is kept as it is; one at 1e-8 or below is replaced by that
    floor. A row whose variance exceeds the one from the resolved directions alone by more
    than (1 + MC_ERROR_MAX)^2 is imprecise.
    """
    info = 0.5 * (info + info.T)
    none = np.diag(info) == 0
    rows = np.flatnonzero(~none)
    cov = np.full(info.shape, np.nan)
    low = np.zeros(len(info), bool)
    if len(rows):
        sub = info[np.ix_(rows, rows)]
        d = np.sqrt(np.abs(np.diag(sub)))
        lam, V = np.linalg.eigh(sub / np.outer(d, d))
        floor = np.full(len(lam), 1e-8)
        if error is not None:
            E = 0.5 * (error + error.T)[np.ix_(rows, rows)] / np.outer(d, d)
            floor = np.maximum(floor, 2 * np.abs(np.einsum("ik,ij,jk->k", V, E, V)))
        res = lam > floor
        var_res = (V[:, res] ** 2 / lam[res]).sum(1)
        c = (V / np.where(lam > 1e-8, lam, floor)) @ V.T
        low[rows] = ~(np.diag(c) <= (1 + MC_ERROR_MAX) ** 2 * var_res)
        cov[np.ix_(rows, rows)] = c / np.outer(d, d)
    return cov, none, low


def standard_errors(eng: Engine, info, index: list, names: list, halves=None):
    """Standard errors on the reporting scale, on the unconstrained scale, the
    correlation matrix of the estimates and a status.

    halves holds the information from the even and from the odd draws; half their
    difference estimates the Monte Carlo error of the information. A quantity without
    information gets no standard error; one that depends on directions this error does
    not resolve is named in the status.
    """
    se = {n: float("nan") for n in names}
    se_u = {}
    if info is None or not index:
        return se, se_u, None, "no estimated parameter" if not index else "not computed"
    labels = [theta_name(eng, e) for e in index]
    if not np.isfinite(info).all():
        return se, se_u, None, "the Fisher information matrix is not finite"
    error = 0.5 * (halves[0] - halves[1]) if halves is not None and np.isfinite(halves).all() else None
    cov, none, low = _covariance(info, error)
    keep = list(np.flatnonzero(~none))
    sd = np.sqrt(np.diag(cov)[keep])
    for pos, q in enumerate(keep):
        se_u[labels[q]] = float(sd[pos])
        se[labels[q]] = float(sd[pos] * abs(_report_scale(eng, index[q])))
    kept = [labels[q] for q in keep]
    corr = pd.DataFrame(cov[np.ix_(keep, keep)] / np.outer(sd, sd), index=kept, columns=kept) if keep else None
    parts = []
    if none.any():
        parts.append("no information on " + ", ".join(labels[q] for q in np.flatnonzero(none)))
    if low.any():
        parts.append("imprecise for " + ", ".join(labels[q] for q in np.flatnonzero(low))
                     + " (directions not resolved beyond the Monte Carlo error)")
    return se, se_u, corr, "; ".join(parts) or "ok"


def intervals(eng: Engine, index: list, values: dict, se_u: dict) -> dict:
    """95% intervals built on the scale each estimate is unconstrained on."""
    ci = {n: (float("nan"), float("nan")) for n in values}
    for entry in index:
        name = theta_name(eng, entry)
        if name not in se_u:
            continue
        kind, j, c = entry
        v, half = values[name], 1.959964 * se_u[name]
        if kind in ("coef", "coef0") and c == 0:
            k, lo, hi = eng.tkind[j], eng.tlo[j], eng.thi[j]
            mid = float(transform(k, lo, hi, v))
            ci[name] = (float(inverse(k, lo, hi, mid - half)), float(inverse(k, lo, hi, mid + half)))
        elif kind in ("omega", "err"):
            # exp overflows past 709
            ci[name] = (v * math.exp(-half), v * math.exp(min(half, 700.0)))
        elif kind == "corr":
            ci[name] = (math.tanh(math.atanh(v) - half), math.tanh(math.atanh(v) + half))
        else:
            ci[name] = (v - half, v + half)
    return ci


# points over the simulated follow-up of a hazard output whose events have exact times: the cubic
# Hermite interpolant of the integral of the hazard places them within about 1e-6 of it; the
# first interval is halved REFINE times, since a hazard such as t**(k - 1) is not smooth at 0
SIM_POINTS = 100
REFINE = 10
# predictions held at once while simulating: replicates run in batches of this many values
_BATCH_VALUES = 20_000_000


@dataclass
class Simulation:
    """Replicates of the dataset under the model (simulate). A hazard output k is the k-th of
    model.hazards; its events are those of its own follow-up, terminal events of other outputs
    aside, at their exact times or at the record that closes their interval."""

    y: np.ndarray           # (replicates, records) on the error model's scale; NaN: hazards
    kept: np.ndarray        # (replicates,) every subject solved
    end: np.ndarray         # (replicates, subjects) earliest terminal event, inf without one
    exit: np.ndarray        # (replicates, subjects, hazards) end of the follow-up, NaN without one
    count: np.ndarray       # (replicates, subjects, hazards) events of each follow-up
    time: np.ndarray        # their times, by replicate, subject, hazard output and time
    # (replicates, subjects) minus log-likelihood of the subject's records of hazard outputs
    # under the replicate's random effects; 0 without such records
    history: np.ndarray

    def first(self) -> np.ndarray:
        """Time of the first event of each follow-up (replicates, subjects, hazards), inf
        without one; an event after the follow-up ended, by a terminal event of another output,
        does not count."""
        start = np.cumsum(self.count) - self.count.ravel()
        at = np.minimum(start, max(self.time.size - 1, 0)).reshape(self.count.shape)
        t = np.where(self.count > 0, self.time[at] if self.time.size else np.inf, np.inf)
        return np.where(t <= self.exit, t, np.inf)

    def events(self, k: int):
        """Replicate, subject and time of each event of hazard output k."""
        R, N, K = self.count.shape
        n = self.count.ravel()
        start = np.cumsum(n) - n
        n, start = n[k::K], start[k::K]
        q = np.repeat(np.arange(R * N), n)
        within = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        return q // N, q % N, self.time[np.repeat(start, n) + within]

    def followed(self, data: Dataset) -> np.ndarray:
        """(replicates, records): the record comes before the earliest terminal event, or at its
        time; later records are removed from the replicate (Friberg et al. 2009)."""
        return data.obs_time <= self.end[:, data.obs_subject]


CURVE_POINTS = 100


def curves(eng: Engine, phi) -> pd.DataFrame:
    """Each subject's prediction at Gaussian-scale phi (subjects, P), for plots: at its records
    and CURVE_POINTS times spread from its first to its last (to the study's last for a subject
    seen once), for every output that is not a hazard."""
    d = eng.data
    end = float(d.grid_time.max()) if d.grid_time.size else 0.0
    own = []
    for i in range(d.n_subjects):
        t = d.grid_time[d.grid_ptr[i]:d.grid_ptr[i + 1]]
        last = t[-1] if t.size and t[-1] > t[0] else end
        own.append(np.union1d(t, np.linspace(t[0], last, CURVE_POINTS)) if t.size else t)
    grid = d.at_times(own)
    pred = problem(eng, grid).predict(eng.rows(np.asarray(phi)[None], grid))[0][0]
    hazards = [eng.model.outputs.index(o) for o in eng.model.hazards]
    keep = ~np.isin(grid.obs_out, hazards)
    return pd.DataFrame({
        "id": np.asarray(d.ids, dtype=object)[grid.obs_subject[keep]],
        "time": grid.obs_time[keep],
        "output": np.asarray(d.output_labels, dtype=object)[grid.obs_out[keep]],
        "ind_pred_mode": pred[grid.obs_grid[keep], grid.obs_out[keep]]})


def problem(eng: Engine, data: Dataset) -> Problem:
    """The model on another design of the same subjects, solved with the settings of the fit."""
    method = {v: k for k, v in _METHODS.items()}[eng.prob.method]
    return Problem(eng.model, data, rtol=eng.prob.rtol, atol=eng.prob.atol,
                   dynamic_range=eng.prob.dyn if data.n_subjects == eng.prob.dyn.size
                   else eng.prob.dynamic_range, method=method,
                   max_steps=eng.prob.max_steps)


@njit(cache=True, error_model="numpy")
def _crossing(t0, t1, H0, H1, h0, h1, target):
    """Time in (t0, t1] where H, known at both ends with its slopes h, reaches target, with
    H0 < target <= H1: bisection on the cubic Hermite interpolant, linear interpolation when a
    slope is not finite (a Weibull hazard of shape below 1 at the origin)."""
    d = t1 - t0
    if not (np.isfinite(h0) and np.isfinite(h1)):
        return t0 + (target - H0) / (H1 - H0) * d
    lo = 0.0
    hi = 1.0
    for _ in range(52):
        u = 0.5 * (lo + hi)
        v = u * u
        p = (((2.0 * u - 3.0) * v + 1.0) * H0 + ((u - 2.0) * u + 1.0) * u * d * h0
             + (3.0 - 2.0 * u) * v * H1 + (u - 1.0) * v * d * h1)
        if p < target:
            lo = u
        else:
            hi = u
    return t0 + 0.5 * (lo + hi) * d


@njit(cache=True, parallel=True, error_model="numpy")
def _events(pred, status, grid_time, g_open, g_last, g_end, stop, rec_ptr, rec_g, col, hcol,
            repeated, terminal, interval, states, fill, count, start, time, end, exit):
    """Events of each hazard output of each replicate subject by time rescaling: the integral
    of the hazard since the opening reaches successive sums of Exp(1) draws (Brown et al. 2002),
    one draw for a terminal or single output. g_open, g_last and g_end index the grid at the
    opening, the last record and the end of the grid (-1 without records); stop (replicates,
    subjects, hazards) ends each follow-up, at or before the end of the grid; interval
    (subjects, hazards) marks the subjects that record events by interval.

    Without fill, only count is written. With fill, the times go to time from start, and end and
    exit get the earliest terminal event and the end of each follow-up it leaves.
    """
    B, N = status.shape
    K = col.size
    for q in prange(B * N):
        b = q // N
        i = q % N
        state = states[b, i]
        cut = np.inf
        cause = -1
        for k in range(K):
            n_ev = 0
            g0 = g_open[i, k]
            if g0 >= 0 and status[b, i] == 0:
                c = col[k]
                g1 = g_end[i, k]
                target = pred[b, g0, c]
                # the draw is positive, so the first crossing lies after the opening
                pos = g0 + 1
                while True:
                    u, state = _rand(state)
                    target -= np.log(u)
                    while pos <= g1 and not pred[b, pos, c] >= target:
                        pos += 1
                    if pos > g1:
                        break
                    t = min(_crossing(grid_time[pos - 1], grid_time[pos], pred[b, pos - 1, c],
                                      pred[b, pos, c], pred[b, pos - 1, hcol[k]],
                                      pred[b, pos, hcol[k]], target), grid_time[pos])
                    if t > stop[b, i, k]:
                        break
                    if n_ev == 0 and terminal[k] and t < cut:
                        cut = t
                        cause = k
                    if fill:
                        if interval[i, k] and pos <= g_last[i, k]:
                            # known at the record that closes its interval
                            j = rec_ptr[i * K + k]
                            while rec_g[j] < pos:
                                j += 1
                            t = grid_time[rec_g[j]]
                        time[start[b, i, k] + n_ev] = t
                    n_ev += 1
                    if not repeated[k]:
                        break
            count[b, i, k] = n_ev
        if not fill:
            continue
        end[b, i] = cut if status[b, i] == 0 else np.nan
        for k in range(K):
            if g_open[i, k] < 0 or status[b, i] != 0:
                exit[b, i, k] = np.nan
            elif k == cause:
                exit[b, i, k] = time[start[b, i, k]]
            else:
                # followed to the earliest terminal event; by interval, to the last record before it
                e = min(stop[b, i, k], cut)
                x = max(e, grid_time[g_open[i, k]])
                if interval[i, k] and e < grid_time[g_last[i, k]]:
                    x = grid_time[g_open[i, k]]
                    for j in range(rec_ptr[i * K + k], rec_ptr[i * K + k + 1]):
                        if grid_time[rec_g[j]] <= e:
                            x = grid_time[rec_g[j]]
                exit[b, i, k] = x


@njit(cache=True, parallel=True, error_model="numpy")
def _history(pred, grid_ptr, obs_ptr, y, gl, out, cens, prev, hcol, hrep, nll):
    """Minus log-likelihood of each subject's records of hazard outputs (Engine.nll's terms) under
    the predictions of each replicate, on the simulation grid (replicates, subjects)."""
    B, N = nll.shape
    for q in prange(B * N):
        b = q // N
        i = q % N
        o0, o1 = obs_ptr[i], obs_ptr[i + 1]
        nll[b, i] = _hazard_nll(pred[b, grid_ptr[i]:grid_ptr[i + 1]], y[o0:o1], gl[o0:o1],
                                out[o0:o1], cens[o0:o1], prev[o0:o1], hcol, hrep)


def _follow_ups(eng: Engine):
    """Follow-up of each subject and hazard output in the data (subjects, hazards): the first
    and last records and the end of the simulation grid, NaN without records; whether the
    subject records events by interval; whether an event ended its follow-up."""
    d, model = eng.data, eng.model
    N, K = eng.N, len(model.hazards)
    sub, t, out = d.obs_subject, d.obs_time, d.obs_out
    event = eng.on_hazard & (d.obs_y > 0)
    terminal = event & np.array([model.hazards.get(o) == "terminal" for o in model.outputs])[out]
    dead = np.full(N, np.inf)
    np.minimum.at(dead, sub[terminal], t[terminal])
    first, last, window = (np.full((N, K), np.nan) for _ in range(3))
    interval, ended = np.zeros((N, K), bool), np.zeros((N, K), bool)
    for k, (o, kind) in enumerate(model.hazards.items()):
        on = out == model.outputs.index(o)
        if not on.any():
            continue
        lo, hi = np.full(N, np.inf), np.full(N, -np.inf)
        np.minimum.at(lo, sub[on], t[on])
        np.maximum.at(hi, sub[on], t[on])
        has = np.isfinite(lo)
        # ended by an event: its own (a count by interval on a repeated output closes a visit),
        # or a terminal one at its last record
        own = np.zeros(N, bool)
        own[sub[on & event & (t == hi[sub]) & ((d.obs_cens == 0) | (kind != "repeated"))]] = True
        ended[:, k] = has & (own | (hi == dead))
        first[has, k], last[has, k] = lo[has], hi[has]
        # the grid of a follow-up ended by an event runs on to the latest end of the output
        window[has, k] = np.where(ended[has, k], hi[has].max(), hi[has])
        # by interval: a subject with a record of censoring flag 1; a subject without events of
        # the output as most subjects with events of it
        flagged, seen = np.zeros(N, bool), np.zeros(N, bool)
        flagged[sub[on & (d.obs_cens == 1)]] = True
        seen[sub[on & event]] = True
        interval[:, k] = np.where(seen, flagged, (seen & flagged).sum() > (seen & ~flagged).sum())
    return first, last, window, interval, ended


def _planned_ends(first, last, window, ended, u):
    """End of the simulated follow-up of each replicate subject (replicates, subjects, hazards).
    A follow-up ended by censoring keeps its end. One ended by an event draws its censoring time
    given that it came no earlier, from the reverse Kaplan-Meier estimate of the output's
    censoring times (Schemper and Smith 1996), the end of the grid beyond the last of them. u
    holds one uniform draw in (0, 1] per replicate subject, shared by its outputs."""
    stop = np.repeat(last[None], len(u), axis=0)
    for k in range(last.shape[1]):
        e = np.flatnonzero(ended[:, k])
        if e.size == 0:
            continue
        has = np.isfinite(first[:, k])
        cens = has & ~ended[:, k] & (last[:, k] > first[:, k])
        c, n_c = np.unique(last[cens, k], return_counts=True)
        if c.size == 0:
            stop[:, e, k] = window[e, k]
            continue
        # censoring free past c: the product of 1 - (censored at c) / (followed at c), logs
        # summed apart from the factors at 0
        f = 1.0 - n_c / _at_risk(first[has, k], last[has, k], c)
        zero = f <= 0.0
        Z = np.cumsum(zero)
        L = np.cumsum(np.log(np.where(zero, 1.0, f)))
        s = np.searchsorted(c, last[e, k], side="left")
        Z0 = np.where(s > 0, Z[s - 1], 0)
        L0 = np.where(s > 0, L[s - 1], 0.0)
        # the first censoring time from the event on where that probability falls to u or below
        j = np.maximum(np.searchsorted(-L, -(L0 + np.log(u[:, e])), side="left"), s)
        j = np.minimum(j, np.searchsorted(Z, Z0 + 1, side="left"))
        stop[:, e, k] = np.where(j < c.size, c[np.minimum(j, c.size - 1)], window[e, k])
    return stop


def _design(eng: Engine, first, last, window, interval):
    """Each subject's times for the simulation: its records and SIM_POINTS over the follow-up of
    each hazard output whose events it records at their times, REFINE more toward its opening,
    or over the extension after its last record when it records them by interval. Returns
    the design, the grid index of each record, and per subject and hazard output those of the
    opening, the last record and the end of the grid (-1 without records)."""
    d = eng.data
    N, K = first.shape
    own = []
    for i in range(N):
        pts = [d.grid_time[d.grid_ptr[i]:d.grid_ptr[i + 1]]]
        for k in range(K):
            if np.isfinite(first[i, k]):
                lo = last[i, k] if interval[i, k] else first[i, k]
                if window[i, k] > lo:
                    pts.append(np.linspace(lo, window[i, k], SIM_POINTS))
                    if not interval[i, k]:
                        step = (window[i, k] - lo) / (SIM_POINTS - 1)
                        pts.append(lo + step * 0.5 ** np.arange(1, REFINE + 1))
        own.append(np.concatenate(pts))
    design = d.at_times(own)
    record = np.empty(d.n_obs, np.int64)
    marks = np.full((3, N, K), -1, np.int64)
    for i in range(N):
        g0, g1 = design.grid_ptr[i], design.grid_ptr[i + 1]
        times = design.grid_time[g0:g1]
        a, b = d.obs_ptr[i], d.obs_ptr[i + 1]
        record[a:b] = g0 + np.searchsorted(times, d.obs_time[a:b])
        has = np.isfinite(first[i])
        for m, x in enumerate((first, last, window)):
            marks[m, i, has] = g0 + np.searchsorted(times, x[i, has])
    return design, record, marks


def simulate(eng: Engine, n: int, stream: int = 3) -> Simulation:
    """Replicates of the dataset under the model at the current population values: random effects
    drawn once per replicate for every output, the predictions, the residual errors and the
    events of hazard outputs. Their follow-up ends where the data's ended by censoring, at a
    drawn censoring time where an event ended it (_planned_ends), and at the earliest terminal
    event.

    Valid only when no hazard depends on past events.
    """
    d, model = eng.data, eng.model
    rng = np.random.default_rng([eng.seed, stream])
    _, chol, _ = eng.prior(eng.omega, np.flatnonzero(eng.iiv))
    eta = rng.standard_normal((n, eng.N, eng.P)) @ chol.T
    phi = eng.mean() + eta
    K = len(model.hazards) if eng.on_hazard.any() else 0
    count = np.zeros((n, eng.N, K), np.int64)
    end = np.full((n, eng.N), np.inf)
    exit = np.full((n, eng.N, K), np.nan)
    history = np.zeros((n, eng.N))
    times = []
    if K:
        first, last, window, interval, ended = _follow_ups(eng)
        stop = _planned_ends(first, last, window, ended,
                             1.0 - np.random.default_rng([eng.seed, stream, 2]).random((n, eng.N)))
        design, record, (g_open, g_last, g_end) = _design(eng, first, last, window, interval)
        prob = problem(eng, design)
        # each record and the previous one of its output, on the subject's simulation grid
        ptr = design.grid_ptr[d.obs_subject]
        prev, _ = _links(d)
        prev = np.where(eng.on_hazard & (prev >= 0), record[prev] - ptr, -1)
        local = record - ptr
        hz = list(model.hazards)
        col = np.array([model.outputs.index(o) for o in hz], np.int64)
        hcol = eng.hcol[col]
        kinds = [model.hazards[o] for o in hz]
        repeated = np.array([k == "repeated" for k in kinds])
        terminal = np.array([k == "terminal" for k in kinds])
        # the records of each subject and hazard output, in time order
        k_of = np.full(len(model.outputs), -1, np.int64)
        k_of[col] = np.arange(K)
        on = np.flatnonzero(eng.on_hazard)
        key = d.obs_subject[on] * K + k_of[d.obs_out[on]]
        rec_g = record[on][np.argsort(key, kind="stable")]
        rec_ptr = np.concatenate([[0], np.cumsum(np.bincount(key, minlength=eng.N * K))])
        states = np.random.SeedSequence([eng.seed, stream, 1]).generate_state(
            n * eng.N, np.uint64).reshape(n, eng.N)
    else:
        design, record, prob = d, d.obs_grid, eng.prob
    f = np.empty((n, d.n_obs))
    solved = np.empty(n, bool)
    batch = max(1, min(n, _BATCH_VALUES // max(1, len(design.grid_time) * prob.n_columns)))
    for r0 in range(0, n, batch):
        r1 = min(n, r0 + batch)
        pred, status = prob.predict(eng.rows(phi[r0:r1], design))
        f[r0:r1] = pred[:, record, d.obs_out]
        solved[r0:r1] = (status == 0).all(1)
        if not K:
            continue
        _history(pred, design.grid_ptr, d.obs_ptr, eng.obs_y, local, d.obs_out, d.obs_cens, prev,
                 eng.hcol, eng.hrep, history[r0:r1])
        args = (pred, status, design.grid_time, g_open, g_last, g_end, stop[r0:r1], rec_ptr,
                rec_g, col, hcol, repeated, terminal, interval)
        c = count[r0:r1]
        # a first pass counts the events, the second one, from the same draws, places them
        _events(*args, states[r0:r1], False, c, c, np.zeros(0), end[r0:r1], exit[r0:r1])
        start = (np.cumsum(c) - c.ravel()).reshape(c.shape)
        times.append(np.empty(int(c.sum())))
        _events(*args, states[r0:r1], True, c, start, times[-1], end[r0:r1], exit[r0:r1])
    if K:
        for name in eng.prob.totals:
            eng.prob.totals[name] += prob.totals[name]
        f[:, eng.on_hazard] = np.nan
    # on the scale of the error model (logs for an exponential error)
    f = eng.to_scale(f, d.obs_out)
    g = eng.residual_sd(f, d.obs_out)
    y = f + g * rng.standard_normal(f.shape)
    # kept: every subject solved; records lost to a terminal event do not drop a replicate
    kept = np.isfinite(y[:, ~eng.on_hazard]).all(1) & (solved if K else True)
    return Simulation(y=y, kept=kept, end=end, exit=exit, count=count,
                      time=np.concatenate(times) if times else np.zeros(0), history=history)


def population_residuals(eng: Engine, sim: Simulation):
    """PWRES and NPDE per observation, decorrelated within subject (Comets et al. 2008); NaN on
    records of hazard outputs. For a subject with records of hazard outputs, each replicate
    counts in proportion to the likelihood of those records under its random effects, so the
    reference is the distribution of its other records given its events and follow-up."""
    d = eng.data
    y = eng.obs_y
    pwres = np.full(y.shape, np.nan)
    npde = np.full(y.shape, np.nan)
    sims = sim.y[sim.kept]
    history = sim.history[sim.kept]
    if len(sims) < 20:
        return pwres, npde
    continuous = ~eng.on_hazard
    for i in range(eng.N):
        a, b = d.obs_ptr[i], d.obs_ptr[i + 1]
        rec = a + np.flatnonzero(continuous[a:b])
        if rec.size == 0:
            continue
        s = sims[:, a:b] if rec.size == b - a else sims[:, rec]
        w, R = None, len(s)
        if rec.size < b - a:
            lw = -history[:, i]
            if not np.isfinite(lw.max()):
                continue
            w = np.exp(lw - lw.max())
            s, w = s[w > 0], w[w > 0]
            # effective number of replicates (Kish 1965)
            R = w.sum() ** 2 / (w @ w)
        if R < 20:
            continue
        mu = s.mean(0) if w is None else w @ s / w.sum()
        cov = np.atleast_2d(np.cov(s, rowvar=False, aweights=w))
        try:
            L = np.linalg.cholesky(cov + 1e-12 * np.trace(cov) * np.eye(rec.size))
        except np.linalg.LinAlgError:
            continue
        ys = np.linalg.solve(L, y[rec] - mu)
        ss = np.linalg.solve(L, (s - mu).T).T
        pwres[rec] = ys
        pde = (ss < ys).mean(0) if w is None else w @ (ss < ys) / w.sum()
        npde[rec] = stats.norm.ppf(np.clip(pde, 0.5 / R, 1 - 0.5 / R))
    return pwres, npde


def hazard_npde(eng: Engine, sim: Simulation, stream: int = 7) -> np.ndarray:
    """Time-to-event NPDE of the first event of each hazard output (Cerou et al. 2018), on the
    record that closes its period: the event, or the last record without one; NaN elsewhere.

    pd is the share of replicates whose first event comes by the observed time, those without
    one in their follow-up counting as later. Censored at C, pd is drawn in [pd(C), 1]; an event
    in (L, R], in [pd(L), pd(R)].
    """
    d, model = eng.data, eng.model
    out = np.full(d.n_obs, np.nan)
    first = sim.first()[sim.kept]
    R = len(first)
    if R < 20 or not first.shape[2]:
        return out
    rng = np.random.default_rng([eng.seed, stream])
    for k, o in enumerate(model.hazards):
        on = np.flatnonzero(d.obs_out == model.outputs.index(o))
        if on.size == 0:
            continue
        sub = d.obs_subject[on]
        # records are in time order within a subject: its first event, else its last record
        close = np.full(eng.N, -1)
        np.maximum.at(close, sub, on)
        ev = on[d.obs_y[on] > 0]
        first_ev = np.full(eng.N, d.n_obs)
        np.minimum.at(first_ev, d.obs_subject[ev], ev)
        close = np.where(first_ev < d.n_obs, first_ev, close)
        i = np.flatnonzero(close >= 0)
        j = close[i]
        T = first[:, i, k]
        hi = (T <= d.obs_time[j]).mean(0)
        prev = d.grid_time[d.grid_ptr[i] + np.maximum(eng.obs_prev[j], 0)]
        lo = np.where(d.obs_cens[j] == 1, (T <= prev).mean(0), hi)
        lo, hi = np.where(d.obs_y[j] > 0, lo, hi), np.where(d.obs_y[j] > 0, hi, 1.0)
        pd_ = lo + (hi - lo) * rng.uniform(size=i.size)
        out[j] = stats.norm.ppf(np.clip(pd_, 0.5 / R, 1 - 0.5 / R))
    return out


def covariate_screen(eng: Engine, cond: Conditional) -> pd.DataFrame:
    """Association between the random effect of each parameter and each covariate of
    the data, strongest first.

    Computed on the conditional means of the random effects: the test keeps its
    level for a covariate unrelated to the data, and shrinkage attenuates the
    association. The p-values are not corrected for the number of pairs.
    """
    rows = []
    N = eng.N
    for name, x, codes, levels in screened_covariates(eng.data):
        values = np.unique(x)
        if N < 4 or len(values) < 2 or levels > N - 2:
            continue
        for j in np.flatnonzero(eng.iiv[:eng.p0]):
            e = cond.mean[:, j]
            if levels > 2:
                f, pv = stats.f_oneway(*[e[codes == k] for k in range(levels)])
                # share of variance between levels, corrected for their number
                value = np.sqrt(max(0.0, 1 - (N - 1) / (f * (levels - 1) + N - levels)))
            else:
                xc, ec = x - x.mean(), e - e.mean()
                value = ec @ xc / np.sqrt((ec**2).sum() * (xc**2).sum())
                pv = 2 * stats.t.sf(abs(value) * np.sqrt((N - 2) / (1 - value**2)), N - 2)
            if np.isfinite(value) and np.isfinite(pv):
                rows.append({"parameter": eng.names[j], "covariate": name,
                             "measure": "correlation ratio" if levels > 2 else "correlation",
                             "association": float(value), "p_value": float(pv)})
    table = pd.DataFrame(rows, columns=["parameter", "covariate", "measure", "association",
                                        "p_value"])
    return table.sort_values("p_value", kind="stable").reset_index(drop=True)


def screened_covariates(data: Dataset) -> list:
    """Each covariate as the screen and the searches see it: name, values, level codes and the
    number of levels (0 when compared as a number). A group of indicator columns counts as one
    covariate with its levels; labels, or a number with at most 5 values, are compared by level."""
    out = []
    grouped = {col for cols in data.covariate_groups.values() for col in cols}
    for name, x in data.covariates.items():
        if name in grouped:
            continue
        values, codes = np.unique(x, return_inverse=True)
        by_level = name in data.covariate_levels or len(values) <= 5
        out.append((name, x, codes, len(values) if by_level else 0))
    for name, cols in data.covariate_groups.items():
        ind = np.column_stack([data.covariates[c] for c in cols])
        codes = np.where(ind.any(1), ind.argmax(1) + 1, 0)
        out.append((name, codes.astype(float), codes, len(cols) + 1))
    return out


def residual_tests(values: np.ndarray) -> dict:
    """Shapiro-Wilk normality and Miao-Gel-Gastwirth symmetry tests."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    out = {"normality_statistic": np.nan, "normality_pvalue": np.nan,
           "symmetry_statistic": np.nan, "symmetry_pvalue": np.nan}
    if x.size < 8:
        return out
    w, p = stats.shapiro(x)
    med = np.median(x)
    j = math.sqrt(math.pi / 2) * np.abs(x - med).mean()
    z = math.sqrt(x.size) * (x.mean() - med) / (j * math.sqrt(0.5708)) if j > 0 else 0.0
    out.update(normality_statistic=float(w), normality_pvalue=float(p),
               symmetry_statistic=float(abs(z)), symmetry_pvalue=float(2 * stats.norm.sf(abs(z))))
    return out


# a record predicted below 1/UNCORRECTABLE of its bin's median prediction is left out of a
# prediction-corrected check: dividing by its prediction sends its corrected values far out
UNCORRECTABLE = 1e3


def vpc(eng: Engine, sim: Simulation, n_bins: int = 10, pop_pred=None):
    """Observed and simulated 10th, 50th and 90th percentiles by time bin and output, outputs
    other than hazards; a replicate's records after its earliest terminal event are removed.

    Where an output has censored records, observed and simulated values are both read as
    max(value, limit) below and min(value, limit) above, the limits being the highest of the
    left-censored records and the lowest of the right-censored ones, so both are censored alike.
    Given pop_pred, the population prediction at each record, both are then corrected for it
    (Bergstrand et al. 2011): multiplied by the median prediction of the record's bin over its
    own, or shifted by their difference where the residual error is constant. A record predicted
    at or below 0, or below 1/UNCORRECTABLE of that median, is left out.

    Returns the table and, per record, the observation as the check reads it (NaN when left
    out or on a hazard output).
    """
    d = eng.data
    rows = []
    read = np.full(d.n_obs, np.nan)
    y = sim.y if np.isposinf(sim.end).all() else np.where(sim.followed(d), sim.y, np.nan)
    sims = eng.from_scale(y[sim.kept], d.obs_out)
    for o, label in enumerate(d.output_labels):
        sel = np.flatnonzero(d.obs_out == o)
        if sel.size == 0 or eng.hcol[o] >= 0:
            continue
        cens = d.obs_cens[sel]
        lo = float(d.obs_y[sel][cens == 1].max()) if (cens == 1).any() else -np.inf
        hi = float(d.obs_y[sel][cens == -1].min()) if (cens == -1).any() else np.inf
        obs, sims_o = np.clip(d.obs_y[sel], lo, hi), np.clip(sims[:, sel], lo, hi)
        form = "none" if pop_pred is None else (
            "additive" if eng.ek[o] == 0 and not eng.elog[o] else "multiplicative")
        t = d.obs_time[sel]
        edges = time_bins(t, n_bins)
        which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            k = np.flatnonzero(which == b)
            if k.size == 0:
                continue
            ob, block = obs[k], sims_o[:, k]
            if form != "none":
                p = np.asarray(pop_pred, float)[sel[k]]
                ref = np.nanmedian(p) if np.isfinite(p).any() else np.nan
                if form == "additive":
                    ob, block = ob + (ref - p), block + (ref - p)
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        f = ref / p
                    f = np.where((p > 0) & (f > 0) & (f <= UNCORRECTABLE), f, np.nan)
                    ob, block = ob * f, block * f
                keep = np.isfinite(ob)
                k, ob, block = k[keep], ob[keep], block[:, keep]
                if k.size == 0:
                    continue
            read[sel[k]] = ob
            row = {"output": label, "bin": b, "t_low": edges[b], "t_high": edges[b + 1],
                   "t_mid": float(np.median(t[k])), "n": int(k.size), "correction": form}
            # records lost to a terminal event are NaN; a replicate that lost the whole bin has no
            # percentile there
            pct, med = (np.nanpercentile, np.nanmedian) if np.isnan(block).any() else (
                np.percentile, np.median)
            for q in (10, 50, 90):
                row[f"obs_p{q}"] = float(np.percentile(ob, q))
                if len(sims):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        per = pct(block, q, axis=1)
                        row[f"sim_p{q}"] = float(med(per))
                        row[f"sim_p{q}_low"] = float(pct(per, 5))
                        row[f"sim_p{q}_high"] = float(pct(per, 95))
            rows.append(row)
    return pd.DataFrame(rows), read


def kaplan_meier(entry, exit, event, times) -> np.ndarray:
    """Probability of no event by each of times (Kaplan and Meier 1958): the product, over the
    event times u, of 1 - d(u) / n(u), n(u) counting the subjects at risk, entry < u <= exit
    (delayed entry). A subject has its event at exit where event is true."""
    u, d = np.unique(exit[event], return_counts=True)
    n = _at_risk(entry, exit, u)
    s = np.cumprod(1.0 - d / np.maximum(n, 1))
    return np.concatenate([[1.0], s])[np.searchsorted(u, times, side="right")]


def mean_events(entry, exit, events, times) -> np.ndarray:
    """Mean number of events by each of times (Nelson 2003): the sum, over the event times u, of
    d(u) / n(u), n(u) counting the subjects at risk, entry < u <= exit; events holds the time of
    each event."""
    u, d = np.unique(events, return_counts=True)
    m = np.cumsum(d / np.maximum(_at_risk(entry, exit, u), 1))
    return np.concatenate([[0.0], m])[np.searchsorted(u, times, side="right")]


def _at_risk(entry, exit, u):
    # entry < u <= exit: entered before u, less those already out
    return (np.searchsorted(np.sort(entry), u, side="left")
            - np.searchsorted(np.sort(exit), u, side="left"))


def vpc_hazards(eng: Engine, sim: Simulation, n_times: int = 100) -> pd.DataFrame:
    """Per hazard output, on n_times common times: the probability of no event yet by
    Kaplan-Meier (terminal and single outputs) or the mean number of events (repeated), observed
    and the 5th, 50th and 95th percentiles of the replicates' curves (Holford 2013), with the
    observed subjects at risk. A follow-up starts at its first record; an event known by
    interval sits at the record that closes it, in observed and simulated data alike."""
    d, model = eng.data, eng.model
    rows = []
    kept = np.flatnonzero(sim.kept)
    first = sim.first()
    for k, (o, kind) in enumerate(model.hazards.items()):
        c = model.outputs.index(o)
        on = d.obs_out == c
        if not on.any() or k >= sim.count.shape[2]:
            continue
        sub, t, y = d.obs_subject[on], d.obs_time[on], d.obs_y[on]
        entry, leave = np.full(eng.N, np.inf), np.full(eng.N, -np.inf)
        np.minimum.at(entry, sub, t)
        np.maximum.at(leave, sub, t)
        i = np.flatnonzero(np.isfinite(entry))
        entry, leave = entry[i], leave[i]
        times = np.linspace(entry.min(), leave.max(), n_times)
        curves = np.full((kept.size, n_times), np.nan)
        if kind == "repeated":
            obs = mean_events(entry, leave, np.repeat(t, y.astype(np.int64)), times)
            rep, who, when = sim.events(k)
            keep = when <= sim.exit[rep, who, k]
            rep, who, when = rep[keep], who[keep], when[keep]
            cuts = np.searchsorted(rep, kept), np.searchsorted(rep, kept, side="right")
            for a, r in enumerate(kept):
                curves[a] = mean_events(entry, sim.exit[r, i, k], when[cuts[0][a]:cuts[1][a]],
                                        times)
        else:
            # the event of a terminal or single output is its last record
            event = np.zeros(eng.N, bool)
            event[sub[y > 0]] = True
            obs = kaplan_meier(entry, leave, event[i], times)
            for a, r in enumerate(kept):
                x, T = sim.exit[r, i, k], first[r, i, k]
                curves[a] = kaplan_meier(entry, np.minimum(T, x), T <= x, times)
        band = np.percentile(curves, [5, 50, 95], axis=0) if kept.size else np.full(
            (3, n_times), np.nan)
        # followed at t: entry <= t <= exit
        at_risk = (np.searchsorted(np.sort(entry), times, side="right")
                   - np.searchsorted(np.sort(leave), times, side="left"))
        rows.append(pd.DataFrame({
            "output": d.output_labels[c], "measure": "mean events" if kind == "repeated"
            else "survival", "time": times, "at_risk": at_risk, "obs": obs, "sim_p5": band[0],
            "sim_p50": band[1], "sim_p95": band[2]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def time_bins(t: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin edges: one bin per distinct time when few, equal-count bins otherwise."""
    distinct = np.unique(t)
    if distinct.size <= n_bins:
        mids = (distinct[1:] + distinct[:-1]) / 2
        return np.concatenate([[distinct[0]], mids, [distinct[-1]]]) if distinct.size > 1 \
            else np.array([distinct[0], distinct[0]])
    edges = np.unique(np.quantile(t, np.linspace(0, 1, n_bins + 1)))
    return edges
