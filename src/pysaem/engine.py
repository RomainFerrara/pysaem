"""Shared state of a fit: model, data, population values and the compiled kernels.

The Metropolis kernels follow the three-kernel scheme of the SAEM algorithm
(Kuhn and Lavielle 2005; Comets, Lavenu and Lavielle 2017): prior proposals,
component-wise random walk, block random walk, with adaptive scales.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numba import njit, prange

from .data import DataError, Dataset
from .model import Model
from .population import ERROR_MODELS, Parameter, Population, inverse, jacobian, transform
from .sensitivity import SensitivityModel, exact_parameters
from .solve import OK, Problem, _solve_one, call

_TKIND = {"normal": 0, "lognormal": 1, "logitnormal": 2}
_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
# floor of the residual sd, reached when a proportional error meets f = 0: machine epsilon,
# the smallest value that keeps the likelihood finite without changing it elsewhere
_G_MIN = 2.220446049250313e-16


class FitError(RuntimeError):
    """The fit cannot proceed or has diverged; the message says why."""


@njit(cache=True, error_model="numpy")
def _rand(s):
    # splitmix64: one independent stream per chain and subject, whatever the threads
    s = s + np.uint64(0x9E3779B97F4A7C15)
    z = s
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (np.float64(z >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0), s


@njit(cache=True, error_model="numpy")
def _randn(s):
    u1, s = _rand(s)
    u2, s = _rand(s)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2), s


@njit(cache=True, error_model="numpy")
def _sd(kind, a, b, f):
    if kind == 0:
        g = a
    elif kind == 1:
        g = b * abs(f)
    elif kind == 2:
        g = a + b * abs(f)
    else:
        g = np.sqrt(a * a + b * b * f * f)
    return max(g, _G_MIN)


@njit(cache=True, error_model="numpy")
def _hazard_term(dH, h, k, cens, repeated):
    """Log-likelihood of a record of a hazard output that closes (t_p, t]: dH events expected
    over it, h the hazard at t, k the events recorded, at t (cens 0) or since t_p (cens 1)."""
    if k == 0.0:
        return -dH
    if cens == 0:
        if np.isnan(h):
            return -np.inf
        return k * np.log(max(h, 1e-300)) - dH
    if repeated:
        # a Poisson count
        return k * np.log(max(dH, 1e-300)) - dH - math.lgamma(k + 1.0)
    # the event of a terminal or single output happened in the interval
    return np.log(max(-math.expm1(-dH), 1e-300))


@njit(cache=True, error_model="numpy")
def _nll(pred, y, gl, out, cens, limit, hcol, ek, ea, eb, elog):
    """Minus log-density of one subject's observations given its predictions, records of hazard
    outputs (hcol >= 0) aside: _hazard_nll reads them.

    On an output with elog, y and limit are already logs and the prediction is taken
    to the log; the density stays that of the observation (Jacobian log y).
    """
    total = 0.0
    for j in range(y.size):
        o = out[j]
        if hcol[o] >= 0:
            continue
        f = pred[gl[j], o]
        if elog[o]:
            f = np.log(max(f, 1e-300))
        g = _sd(ek[o], ea[o], eb[o], f)
        if cens[j] == 0:
            z = (y[j] - f) / g
            total += 0.5 * z * z + np.log(g) + _HALF_LOG_2PI
            if elog[o]:
                total += y[j]
        else:
            # cens 1: true value in (limit, y]; cens -1: in [y, limit)
            at_y = 0.5 * math.erfc(-(y[j] - f) / g * 0.7071067811865476)
            if np.isfinite(limit[j]):
                at_l = 0.5 * math.erfc(-(limit[j] - f) / g * 0.7071067811865476)
            else:
                at_l = 0.0 if cens[j] > 0 else 1.0
            pr = at_y - at_l if cens[j] > 0 else at_l - at_y
            total += -np.log(max(pr, 1e-300))
    return total if np.isfinite(total) else np.inf


@njit(cache=True, error_model="numpy")
def _any_hazard(hcol):
    # a loop: an array expression in a parallel kernel runs as a parallel region of its own,
    # 20 us a call
    for o in range(hcol.size):
        if hcol[o] >= 0:
            return True
    return False


@njit(cache=True, error_model="numpy")
def _hazard_nll(pred, y, gl, out, cens, prev, hcol, hrep):
    """Minus log-likelihood of one subject's records of hazard outputs. The prediction of a hazard
    output is the integral of the hazard, which sits in column hcol; a record closes the interval
    since prev, the grid index of the previous record of that output, the first one (prev -1)
    opens the follow-up. Kept out of _nll: inside its loop, these terms made the likelihood of a
    model without hazards 1.5 times slower."""
    total = 0.0
    for j in range(y.size):
        o = out[j]
        if hcol[o] < 0 or prev[j] < 0:
            continue
        h = pred[gl[j], hcol[o]]
        if h < 0.0:
            return np.inf
        total -= _hazard_term(max(pred[gl[j], o] - pred[prev[j], o], 0.0), h, y[j], cens[j],
                              hrep[o])
    return total if np.isfinite(total) else np.inf


@njit(cache=True, error_model="numpy")
def _rows(mean_i, e, tkind, tlo, thi, iov_of, n_iov, rows):
    """Parameters of each occasion row: the subject's phi plus, for a parameter with
    inter-occasion variability, the random effect of that occasion (stored after the
    structural parameters, occasion by occasion)."""
    p0 = rows.shape[1]
    for k in range(rows.shape[0]):
        for j in range(p0):
            phi = mean_i[j] + e[j]
            if iov_of[j] >= 0:
                phi += e[p0 + k * n_iov + iov_of[j]]
            if tkind[j] == 0:
                rows[k, j] = phi
            elif tkind[j] == 1:
                rows[k, j] = np.exp(phi)
            else:
                rows[k, j] = tlo[j] + (thi[j] - tlo[j]) / (1.0 + np.exp(-phi))


@njit(cache=True, error_model="numpy")
def _quad(prec, e, sampled):
    """0.5 e' Omega^-1 e over the sampled components."""
    total = 0.0
    for a in range(sampled.size):
        j = sampled[a]
        for b in range(sampled.size):
            total += e[j] * prec[j, sampled[b]] * e[sampled[b]]
    return 0.5 * total


@njit(cache=True, error_model="numpy")
def _factor(cov):
    """Cholesky factor and inverse of a covariance matrix, and whether it is positive
    definite. Plain loops: a LAPACK call on a matrix this small takes milliseconds
    when it runs between two parallel kernels."""
    n = cov.shape[0]
    chol = np.zeros((n, n))
    low = np.zeros((n, n))
    prec = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            v = cov[i, j]
            for k in range(j):
                v -= chol[i, k] * chol[j, k]
            if i == j:
                if not v > 0.0:
                    return chol, prec, False
                chol[i, i] = np.sqrt(v)
            else:
                chol[i, j] = v / chol[j, j]
    for j in range(n):
        low[j, j] = 1.0 / chol[j, j]
        for i in range(j + 1, n):
            v = 0.0
            for k in range(j, i):
                v -= chol[i, k] * low[k, j]
            low[i, j] = v / chol[i, i]
    for i in range(n):
        for j in range(n):
            v = 0.0
            for k in range(max(i, j), n):
                v += low[k, i] * low[k, j]
            prec[i, j] = v
    return chol, prec, True


@njit(cache=True, parallel=True, error_model="numpy")
def _estep(rhs, jac, has_jac, out_fn, init, brk, inp, structure, targets, n, n_out, n_brk,
           n_hbrk, swf, n_sw, t_start, grid_ptr, grid_time, ev_ptr, ev_time, ev_input, ev_amt,
           ev_rate, ev_reset, ev_ss, occ_ptr, occ_time, reg_ptr, reg_time, reg_value, rtol, atol,
           dyn, max_steps, method, mmax, n_chunks,
           obs_ptr, obs_y, obs_gl, obs_out, obs_cens, obs_limit, obs_prev, hcol, hrep,
           ek, ea, eb, elog,
           mean, chol, prec, tkind, tlo, thi, p0, iov_of, n_iov, sampled, n1, n2, n3, n4, block, scale2, scale3,
           eta, uy, rng, pred, acc2, acc3, try3):
    """One pass of the Metropolis kernels on every chain and subject: prior draws,
    random walk by component, random walk by block, prior draw of a group of components
    given the others. The last one lets a subject leave a wrong mode of a few parameters.

    eta, uy, rng and pred are updated in place; acc2, acc3 and try3 count
    acceptances and trials per chunk and component.
    """
    C, N, P = eta.shape
    total = C * N
    ns = sampled.size
    hazards = _any_hazard(hcol)
    kmax = 1
    omax = 1
    for i in range(N):
        kmax = max(kmax, grid_ptr[i + 1] - grid_ptr[i])
        omax = max(omax, occ_ptr[i + 1] - occ_ptr[i])
    n_reg = reg_value.shape[1]
    n_inp = targets.size
    for chunk in prange(n_chunks):
        W = np.empty((18, n))
        r = np.empty(n_reg)
        o = np.empty(n_out)
        d = np.empty(2 * n_inp)
        s = np.empty(n_inp)
        b = np.empty(n_brk)
        at = np.empty(mmax)
        av = np.empty(mmax)
        ak = np.empty(mmax, np.int64)
        ai = np.empty(mmax, np.int64)
        ed = np.empty(mmax)
        es = np.empty(mmax)
        edur = np.empty(mmax)
        G = np.empty((4, max(n_sw, 1)))
        buf = np.empty((2, kmax, n_out))
        rows = np.empty((omax, p0))
        ep = np.empty(P)
        z = np.empty(max(ns, 1))
        lb = np.empty((max(ns, 1), max(ns, 1)))
        pos = np.empty(max(ns, 1), np.int64)
        inside = np.empty(max(ns, 1), np.bool_)
        for q in range(chunk * total // n_chunks, (chunk + 1) * total // n_chunks):
            c = q // N
            i = q % N
            g0, g1 = grid_ptr[i], grid_ptr[i + 1]
            e0, e1 = ev_ptr[i], ev_ptr[i + 1]
            r0, r1 = reg_ptr[i], reg_ptr[i + 1]
            o0, o1 = obs_ptr[i], obs_ptr[i + 1]
            k0, k1 = occ_ptr[i], occ_ptr[i + 1]
            ko = k1 - k0
            K = g1 - g0
            e = eta[c, i]
            state = rng[c, i]
            cur = 0

            _rows(mean[i], e, tkind, tlo, thi, iov_of, n_iov, rows[0:ko])
            st, _, _, _ = _solve_one(
                rhs, jac, has_jac, out_fn, init, brk, inp, structure, targets, rows[0:ko],
                occ_time[k0:k1], t_start[i], grid_time[g0:g1], ev_time[e0:e1], ev_input[e0:e1],
                ev_amt[e0:e1], ev_rate[e0:e1], ev_reset[e0:e1], ev_ss[e0:e1], reg_time[r0:r1],
                reg_value[r0:r1], rtol, atol, dyn[i], max_steps, method, buf[cur, 0:K], W, r, o, d,
                s, b, at, ak, ai, av, ed, es, edur, swf, n_sw, G, n_hbrk)
            u_cur = np.inf
            if st == OK:
                u_cur = _nll(buf[cur, 0:K], obs_y[o0:o1], obs_gl[o0:o1], obs_out[o0:o1],
                             obs_cens[o0:o1], obs_limit[o0:o1], hcol, ek, ea, eb, elog)
                if hazards:
                    u_cur += _hazard_nll(buf[cur, 0:K], obs_y[o0:o1], obs_gl[o0:o1],
                                         obs_out[o0:o1], obs_cens[o0:o1], obs_prev[o0:o1], hcol,
                                         hrep)

            q_cur = _quad(prec, e, sampled)
            n_k1 = n1 if ns > 0 else 0
            n_k2 = n2 * ns
            n_blocks = 0
            if ns > 1 and block >= 2:
                n_blocks = ns if block < ns else 1
            n_k3 = n3 * n_blocks
            n_k4 = n4 * ns
            for step in range(n_k1 + n_k2 + n_k3 + n_k4):
                for j in range(P):
                    ep[j] = e[j]
                prior = 0.0
                kernel = 1
                first = 0
                width = 0
                if step < n_k1:
                    # a draw from the prior: eta = L z with L the Cholesky factor of Omega
                    for a in range(ns):
                        z[a], state = _randn(state)
                    for a in range(ns):
                        v = 0.0
                        for c2 in range(a + 1):
                            v += chol[sampled[a], sampled[c2]] * z[c2]
                        ep[sampled[a]] = v
                elif step < n_k1 + n_k2:
                    kernel = 2
                    first = (step - n_k1) % ns
                    width = 1
                elif step < n_k1 + n_k2 + n_k3:
                    kernel = 3
                    first = (step - n_k1 - n_k2) % n_blocks
                    width = block
                else:
                    # w components from their prior given the others, with P the precision:
                    # N(-P_BB^-1 P_BR e_R, P_BB^-1)
                    kernel = 4
                    start = (step - n_k1 - n_k2 - n_k3) % ns
                    w = max(1, min(block - 1, ns - 1))
                    for a in range(ns):
                        inside[a] = False
                    for a in range(w):
                        pos[a] = (start + a) % ns
                        inside[pos[a]] = True
                    for a in range(w):
                        ja = sampled[pos[a]]
                        v = 0.0
                        for c2 in range(ns):
                            if not inside[c2]:
                                v -= prec[ja, sampled[c2]] * e[sampled[c2]]
                        z[a] = v
                        for b2 in range(a + 1):
                            v = prec[ja, sampled[pos[b2]]]
                            for c2 in range(b2):
                                v -= lb[a, c2] * lb[b2, c2]
                            lb[a, b2] = np.sqrt(v) if a == b2 else v / lb[b2, b2]
                    # mean by two triangular solves, then the draw: L' x = normal
                    for a in range(w):
                        for c2 in range(a):
                            z[a] -= lb[a, c2] * z[c2]
                        z[a] /= lb[a, a]
                    for a in range(w):
                        zz, state = _randn(state)
                        z[a] += zz
                    for a in range(w - 1, -1, -1):
                        for c2 in range(a + 1, w):
                            z[a] -= lb[c2, a] * z[c2]
                        z[a] /= lb[a, a]
                        ep[sampled[pos[a]]] = z[a]
                for a in range(width):
                    j = sampled[(first + a) % ns]
                    zz, state = _randn(state)
                    ep[j] = e[j] + (scale2[j] if kernel == 2 else scale3[j]) * zz
                    if kernel == 3:
                        try3[chunk, j] += 1
                q_new = _quad(prec, ep, sampled)
                if kernel == 2 or kernel == 3:
                    prior = q_new - q_cur

                _rows(mean[i], ep, tkind, tlo, thi, iov_of, n_iov, rows[0:ko])
                st, _, _, _ = _solve_one(
                    rhs, jac, has_jac, out_fn, init, brk, inp, structure, targets, rows[0:ko],
                    occ_time[k0:k1], t_start[i], grid_time[g0:g1], ev_time[e0:e1],
                    ev_input[e0:e1], ev_amt[e0:e1], ev_rate[e0:e1], ev_reset[e0:e1],
                    ev_ss[e0:e1], reg_time[r0:r1], reg_value[r0:r1], rtol, atol, dyn[i], max_steps,
                    method, buf[1 - cur, 0:K], W, r, o, d, s, b, at, ak, ai, av, ed, es, edur,
                    swf, n_sw, G, n_hbrk)
                u_new = np.inf
                if st == OK:
                    u_new = _nll(buf[1 - cur, 0:K], obs_y[o0:o1], obs_gl[o0:o1],
                                 obs_out[o0:o1], obs_cens[o0:o1], obs_limit[o0:o1], hcol, ek, ea,
                                 eb, elog)
                    if hazards:
                        u_new += _hazard_nll(buf[1 - cur, 0:K], obs_y[o0:o1], obs_gl[o0:o1],
                                             obs_out[o0:o1], obs_cens[o0:o1], obs_prev[o0:o1],
                                             hcol, hrep)
                lu, state = _rand(state)
                if u_new < np.inf and np.log(lu) < u_cur - u_new - prior:
                    for j in range(P):
                        e[j] = ep[j]
                    u_cur = u_new
                    q_cur = q_new
                    cur = 1 - cur
                    for a in range(width):
                        j = sampled[(first + a) % ns]
                        if kernel == 2:
                            acc2[chunk, j] += 1
                        elif kernel == 3:
                            acc3[chunk, j] += 1
            rng[c, i] = state
            uy[c, i] = u_cur
            for a in range(K):
                for k in range(n_out):
                    pred[c, g0 + a, k] = buf[cur, a, k] if u_cur < np.inf else np.nan


@njit(cache=True, parallel=True, error_model="numpy")
def _nll_batch(pred, grid_ptr, obs_ptr, obs_y, obs_gl, obs_out, obs_cens, obs_limit, obs_prev,
               hcol, hrep, ek, ea, eb, elog, uy):
    C, N = uy.shape
    hazards = _any_hazard(hcol)
    for q in prange(C * N):
        c = q // N
        i = q % N
        g0, g1 = grid_ptr[i], grid_ptr[i + 1]
        o0, o1 = obs_ptr[i], obs_ptr[i + 1]
        ok = True
        # the columns of the outputs: a hazard itself, after them, is checked where it is read
        for a in range(g0, g1):
            for k in range(hcol.size):
                if not np.isfinite(pred[c, a, k]):
                    ok = False
        u = _nll(pred[c, g0:g1], obs_y[o0:o1], obs_gl[o0:o1], obs_out[o0:o1], obs_cens[o0:o1],
                 obs_limit[o0:o1], hcol, ek, ea, eb, elog) if ok else np.inf
        if ok and hazards:
            u += _hazard_nll(pred[c, g0:g1], obs_y[o0:o1], obs_gl[o0:o1], obs_out[o0:o1],
                             obs_cens[o0:o1], obs_prev[o0:o1], hcol, hrep)
        uy[c, i] = u


def _links(data: Dataset):
    """Per record, the index of the previous record of the same subject and output (-1 for the
    first) and that of the first one, in kernel order."""
    n = data.n_obs
    order = np.lexsort((np.arange(n), data.obs_out, data.obs_subject))
    s, o = data.obs_subject[order], data.obs_out[order]
    same = np.zeros(n, bool)
    same[1:] = (s[1:] == s[:-1]) & (o[1:] == o[:-1])
    prev = np.full(n, -1, np.int64)
    prev[order[same]] = order[np.flatnonzero(same) - 1]
    first = np.empty(n, np.int64)
    first[order] = order[np.maximum.accumulate(np.where(same, 0, np.arange(n)))]
    return prev, first


def _check_hazard_records(model: Model, population: Population, data: Dataset, t_start, prev,
                          first):
    """Refuses records of hazard outputs that do not read as a follow-up, naming the first
    subject concerned. prev and first link each record to the previous and first records of
    its subject and output (_links); t_start is the start of each subject's solve."""
    outs, hz = model.outputs, model.hazards
    out, sub, t, y, cens = data.obs_out, data.obs_subject, data.obs_time, data.obs_y, data.obs_cens
    on = np.array([o in hz for o in outs])[out]
    terminal = np.array([hz.get(o) == "terminal" for o in outs])[out]
    repeated = np.array([hz.get(o) == "repeated" for o in outs])[out]
    n, idx = data.n_obs, np.arange(data.n_obs)
    opening, later = on & (prev < 0), on & (prev >= 0)
    t_prev = t[np.maximum(prev, 0)]

    def refuse(bad, why):
        if bad.any():
            j = int(np.flatnonzero(bad)[0])
            raise DataError(f"subject {data.ids[sub[j]]}: {why(j)}")

    def name(j):
        return outs[out[j]]

    refuse(on & ~(np.isfinite(y) & (y >= 0) & (y == np.floor(y))),
           lambda j: f"a record of hazard output {name(j)} has the value {y[j]:g}; it counts "
                     "events, a whole number from 0")
    refuse(on & ~repeated & (y > 1),
           lambda j: f"a record of hazard output {name(j)} has the value {y[j]:g}; a "
                     f"{hz[name(j)]} output has one event at most, its values are 0 and 1")
    refuse(on & (cens != 0) & (cens != 1),
           lambda j: f"a record of hazard output {name(j)} has the censoring flag {cens[j]}; it "
                     "is 0 for events at its time, 1 for events since the previous record")
    refuse(on & np.isfinite(data.obs_limit),
           lambda j: f"a record of hazard output {name(j)} has a limit; such records take none")
    refuse(on & data.obs_input,
           lambda j: f"the row at time {t[j]:g} is read both as an input, for its amount, and as "
                     f"a record of hazard output {name(j)}, for its value; without an event flag "
                     "column, mark the value of an input row missing")
    refuse(opening & ((y != 0) | (cens != 0)),
           lambda j: f"the first record of hazard output {name(j)} opens its follow-up, with the "
                     "value 0 and the censoring flag 0 (a row marked missing is not read)")
    refuse(opening & (np.bincount(first[on], minlength=n) == 1),
           lambda j: f"hazard output {name(j)} has only the record that opens its follow-up; a "
                     "later record closes it, with the value 0 when nothing happened")
    refuse(later & (cens == 1) & (y == 0),
           lambda j: f"the record of hazard output {name(j)} at time {t[j]:g} has the censoring "
                     "flag 1 and the value 0; the flag 1 says that the events the value counts "
                     "happened since the previous record")
    refuse(later & (cens == 1) & (t_prev >= t),
           lambda j: f"the record of hazard output {name(j)} at time {t[j]:g} counts events since "
                     "the previous record (censoring flag 1), which is at the same time")
    event = later & (y > 0)
    refuse(event & (t == t[first]),
           lambda j: f"hazard output {name(j)} has an event at time {t[j]:g}, when its follow-up "
                     "opens; the follow-up opens before the event")

    # terminal and single outputs: one event, which ends the follow-up of that output
    once = event & ~repeated
    first_event = np.full(n, n)
    np.minimum.at(first_event, first[once], idx[once])
    first_event = first_event[first]
    refuse(once & (idx != first_event),
           lambda j: f"hazard output {name(j)} is {hz[name(j)]} and has a second event at time "
                     f"{t[j]:g}; it has one at most")
    t_event = np.where(on & (first_event < n), t[np.minimum(first_event, n - 1)], np.inf)
    refuse(on & ~repeated & (t > t_event),
           lambda j: f"hazard output {name(j)} has a record at time {t[j]:g}, after its event at "
                     f"time {t_event[j]:g}; a {hz[name(j)]} output ends its follow-up there")

    # a terminal event ends the follow-up of every output
    term = once & terminal
    end = np.full(data.n_subjects, np.inf)
    np.minimum.at(end, sub[term], t[term])
    at_end = term & (t == end[sub])

    def causes(i):
        return " and ".join(outs[o] for o in out[at_end & (sub == i)])

    refuse(t > end[sub],
           lambda j: f"output {name(j)} has a record at time {t[j]:g}, after the terminal event "
                     f"of {causes(sub[j])} at time {end[sub[j]]:g}, which ends the follow-up of "
                     "every output")
    refuse(at_end & (np.bincount(sub[at_end], minlength=data.n_subjects)[sub] > 1),
           lambda j: f"hazard outputs {causes(sub[j])} have their terminal event at the same time "
                     f"{t[j]:g}; one of them ends the follow-up")

    # an event known only to lie in an interval: the product of the terms of each output is its
    # probability only when no other terminal output competes and no other record lies inside
    terminals = [o for o, k in hz.items() if k == "terminal"]
    box = term & (cens == 1)
    if len(terminals) > 1:
        refuse(box,
               lambda j: f"the event of hazard output {name(j)} lies between two records "
                         f"(censoring flag 1), and the terminal outputs {terminals} compete; "
                         "competing terminal outputs need the time of each event")
    lo, hi = np.full(data.n_subjects, np.nan), np.full(data.n_subjects, np.nan)
    cause = np.full(data.n_subjects, -1)
    lo[sub[box]], hi[sub[box]], cause[sub[box]] = t_prev[box], t[box], out[box]
    refuse((out != cause[sub]) & (t > lo[sub]) & (t <= hi[sub]),
           lambda j: f"output {name(j)} has a record at time {t[j]:g}, inside the interval "
                     f"({lo[sub[j]]:g}, {hi[sub[j]]:g}] where the terminal event of "
                     f"{outs[cause[sub[j]]]} happened; a record of {outs[cause[sub[j]]]} with the "
                     f"value 0 at time {t[j]:g} puts the event after it")

    # a random effect in the hazard: the subject is at risk from the start of its solve, since
    # entering later selects the random effects
    random = {o: [p for p in model.dependencies(o)
                  if population.parameters[p].iiv or population.parameters[p].iov] for o in hz}
    start = np.asarray(t_start, float)[sub]
    refuse(opening & (t < start),
           lambda j: f"the follow-up of hazard output {name(j)} opens at time {t[j]:g}, before "
                     f"the solve starts at time {start[j]:g}; it opens at that time or later")
    refuse(opening & (t > start) & np.array([bool(random.get(o)) for o in outs])[out],
           lambda j: f"the follow-up of hazard output {name(j)} opens at time {t[j]:g}, after the "
                     f"solve starts at time {start[j]:g}, and the random effects of "
                     f"{random[name(j)]} reach the hazard: open the follow-up at the start of the "
                     "solve and write the hazard as zero before the subject is at risk")


def _observed_range(data: Dataset, relative, dynamic_range):
    """Dynamic range of each subject: an output with a relative error is resolved down to 1e-3 of
    its smallest positive value (observation or limit) relative to its largest, never below
    1e-20. A series observed from 130 down to 3e-12 lost more than 100 points of -2LL at the
    default 1e-12."""
    dyn = np.full(data.n_subjects, dynamic_range)
    on = relative[data.obs_out]
    for i in range(data.n_subjects):
        o = slice(data.obs_ptr[i], data.obs_ptr[i + 1])
        for k in np.unique(data.obs_out[o][on[o]]):
            v = np.abs(np.concatenate([data.obs_y[o][data.obs_out[o] == k],
                                       data.obs_limit[o][data.obs_out[o] == k]]))
            v = v[np.isfinite(v) & (v > 0)]
            if v.size:
                dyn[i] = min(dyn[i], max(1e-3 * v.min() / v.max(), 1e-20))
    return dyn


def expanded(population: Population, data: Dataset) -> Population:
    """The population with each covariate group (a categorical covariate with more than two
    levels) replaced by its indicator columns, all starting at the value given to the group."""
    groups = data.covariate_groups
    if not any(c in groups for p in population.parameters.values() for c in p.covariates):
        return population
    d = population.to_dict()
    for name, p in d["parameters"].items():
        cov = {}
        for c, v in p["covariates"].items():
            for col in groups.get(c, [c]):
                if col in cov:
                    raise ValueError(f"{name}: covariate {col} is given twice, by itself and "
                                     f"through its group")
                cov[col] = v
        p["covariates"] = cov
    return Population.from_dict(population.model, d)


def _warn_late_start(model, data, prob):
    # without t0, a solve starts at the subject's first record, which may not be time 0
    init = getattr(prob._lowered.symbolic, "init", ())
    late = data.t_first > 0
    if model.t0 is None and late.any() and any(e.is_zero is not True for e in init):
        warnings.warn(f"the states take their initial values at each subject's first record, "
                      f"after time 0 for {late.sum()} of {data.n_subjects} subjects (from "
                      f"t = {data.t_first[late].min():g}); give t0 to the model to set them at "
                      f"another time", stacklevel=4)


class Engine:
    """Current population values and the operations every estimation task needs."""

    def __init__(self, model: Model, data: Dataset, population: Population, *, rtol=1e-6,
                 atol=0.0, dynamic_range=1e-12, method="auto", seed=123456,
                 sensitivities="exact"):
        if sensitivities not in ("exact", "fd"):
            raise ValueError("sensitivities must be 'exact' or 'fd'")
        data = data.matched_to(model.outputs, model.inputs)
        population = expanded(population, data)
        self.model, self.data, self.population = model, data, population
        self.sensitivities = sensitivities
        relative = np.array([population.error[o].kind in ("proportional", "exponential")
                             for o in model.outputs], bool)
        self.prob = Problem(model, data, rtol=rtol, atol=atol,
                            dynamic_range=_observed_range(data, relative, dynamic_range),
                            method=method)
        _warn_late_start(model, data, self.prob)
        names = list(model.parameters)
        params = [population.parameters[k] for k in names]
        # inter-occasion variability: one random effect per occasion and parameter, kept
        # after the structural ones (occasion by occasion), mean zero, one sd per parameter
        self.p0 = len(names)
        self.iov_names = [n for n, p in zip(names, params) if p.iov]
        self.n_iov = len(self.iov_names)
        counts = np.diff(data.occ_ptr)
        self.n_occ = int(counts.max()) if self.n_iov and counts.size else 0
        if self.n_iov and self.n_occ < 2:
            raise ValueError(f"inter-occasion variability on {self.iov_names} needs occasions in "
                             "the data (an occasion column with two values or more)")
        self.iov_of = np.array([self.iov_names.index(n) if n in self.iov_names else -1
                                for n in names], np.int64)
        gammas = [population.parameters[n] for n in self.iov_names]
        pseudo = [(f"kappa_{n}_{k + 1}", g) for k in range(self.n_occ)
                  for n, g in zip(self.iov_names, gammas)]
        self.names = names + [x[0] for x in pseudo]
        self.P = len(self.names)
        self.N = data.n_subjects
        self.tkind = np.array([_TKIND[p.distribution] for p in params] + [0] * len(pseudo), np.int64)
        self.tlo = np.array([float(p.bounds[0]) for p in params] + [0.0] * len(pseudo))
        self.thi = np.array([float(p.bounds[1]) for p in params] + [1.0] * len(pseudo))
        self.iiv = np.array([p.iiv for p in params] + [True] * len(pseudo), bool)
        self.omega = np.array([p.omega if p.iiv else 0.0 for p in params]
                              + [g.gamma for _, g in pseudo], float)
        self.omega_free = np.array([p.iiv and not p.omega_fixed for p in params]
                                   + [not g.gamma_fixed for _, g in pseudo], bool)
        # the random effects of one parameter on every occasion share one variance
        self.tie = np.arange(self.P)
        self.tie[self.p0:] = self.p0 + np.arange(len(pseudo)) % max(self.n_iov, 1)
        self.tie_groups = [np.flatnonzero(self.tie == j) for j in range(self.p0, self.p0 + self.n_iov)]
        counts_all = np.diff(data.occ_ptr)
        self.row_subject = np.repeat(np.arange(self.N), counts_all)
        self.row_occasion = np.arange(len(data.occ_time)) - np.repeat(data.occ_ptr[:-1], counts_all)
        self.collapsed = []  # parameters whose omega reached zero during the estimation
        # correlation matrix of the random effects: identity outside the declared blocks
        self.corr = np.eye(self.P)
        self.corr_pairs = []      # pairs still estimated
        self.corr_boundary = []   # pairs of a block found singular at the end, no standard error
        self.corr_declared = []   # every declared pair, for reporting
        for block in population.correlations:
            idx = sorted(names.index(n) for n in block)
            for a, j in enumerate(idx):
                for k in idx[a + 1:]:
                    rho = float(population.correlation_values.get(
                        f"corr_{names[j]}_{names[k]}", 0.0))
                    self.corr[j, k] = self.corr[k, j] = rho
                    self.corr_pairs.append((j, k))
                    self.corr_declared.append((j, k))

        # fixed effects: per parameter, a design (intercept, covariates) and its coefficients
        self.cov_names, self.design, self.coef, self.coef_free = [], [], [], []
        for name, p in zip(self.names, params + [Parameter(init=0.0, distribution="normal",
                                                            fixed=True)] * len(pseudo)):
            missing = [c for c in p.covariates if c not in data.covariates]
            if missing:
                raise ValueError(f"{name}: covariate(s) {missing} not found in the data")
            cols = [np.ones(self.N)] + [np.asarray(data.covariates[c], float)
                                        for c in p.covariates]
            self.cov_names.append(list(p.covariates))
            self.design.append(np.column_stack(cols))
            j = self.names.index(name)
            mu = float(transform(self.tkind[j], self.tlo[j], self.thi[j], p.init))
            self.coef.append(np.array([mu, *[float(v) for v in p.covariates.values()]]))
            self.coef_free.append(np.array([not p.fixed] * (1 + len(p.covariates)), bool))

        # prior of each typical value on its transformed scale: centre and precision, 0 without
        self.prior_mean = np.zeros(self.P)
        self.prior_prec = np.zeros(self.P)
        for j, p in enumerate(params):
            if p.prior is not None:
                m, s = p.prior
                self.prior_mean[j] = float(transform(self.tkind[j], self.tlo[j], self.thi[j], m))
                self.prior_prec[j] = 1.0 / s**2

        outs = model.outputs
        errs = [population.error[o] for o in outs]
        # an exponential error is a constant one on the log scale: observations, limits and
        # predictions of that output are handled as logs throughout the estimation
        self.elog = np.array([e.kind == "exponential" for e in errs], np.bool_)
        self.ek = np.array([0 if e.kind == "exponential" else ERROR_MODELS.index(e.kind)
                            for e in errs], np.int64)
        self.ea = np.array([e.a if "a" in e.names else 0.0 for e in errs], float)
        self.eb = np.array([e.b if "b" in e.names else 0.0 for e in errs], float)
        # a hazard output has no error parameter to estimate
        self.err_free = np.array([not e.fixed and bool(e.names) for e in errs], bool)
        self.err_names = [e.names for e in errs]

        self.obs_gl = (data.obs_grid - data.grid_ptr[data.obs_subject]).astype(np.int64)
        # a hazard output: its column holds the integral of the hazard, the hazard itself
        # follows the outputs (Model.lowered); hrep marks the repeated ones
        self.hcol = np.full(len(outs), -1, np.int64)
        for k, h in enumerate(model.hazards):
            self.hcol[outs.index(h)] = len(outs) + k
        self.hrep = np.array([model.hazards.get(o) == "repeated" for o in outs], np.bool_)
        self.on_hazard = self.hcol[data.obs_out] >= 0
        # on a record of a hazard output, the grid index within the subject of the previous
        # record of that output (-1 at the first, which opens the follow-up) and of the first
        self.obs_prev = np.full(data.n_obs, -1, np.int64)
        self.obs_open = np.full(data.n_obs, -1, np.int64)
        if self.on_hazard.any():
            prev, first = _links(data)
            _check_hazard_records(model, population, data, self.prob._t_start, prev, first)
            later = self.on_hazard & (prev >= 0)
            self.obs_prev[later] = self.obs_gl[prev[later]]
            self.obs_open[self.on_hazard] = self.obs_gl[first[self.on_hazard]]
        on_log = self.elog[data.obs_out]
        bad = on_log & (data.obs_cens == 0) & ~(data.obs_y > 0)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} observation(s) at or below zero on an output with "
                             "an exponential error model")
        with np.errstate(divide="ignore", invalid="ignore"):
            self.obs_y = np.where(on_log, np.log(data.obs_y), data.obs_y)
            self.obs_limit = np.where(on_log & np.isfinite(data.obs_limit),
                                      np.log(np.maximum(data.obs_limit, 0.0)), data.obs_limit)
        self.obs_subject = data.obs_subject
        self.seed = seed
        # per structural parameter: exact sensitivities; none, and the fit uses differences
        self._exact = exact_parameters(self.prob._lowered)
        if not any(self._exact):
            self.sensitivities = "fd"
        self._sensitivity = {}    # problems with the sensitivity equations, by directions

    # population values ----------------------------------------------------

    def mean(self) -> np.ndarray:
        """Individual means on the Gaussian scale, shape (subjects, parameters)."""
        return np.column_stack([x @ c for x, c in zip(self.design, self.coef)])

    def psi(self, phi) -> np.ndarray:
        out = np.empty_like(phi)
        for j in range(phi.shape[-1]):
            out[..., j] = inverse(self.tkind[j], self.tlo[j], self.thi[j], phi[..., j])
        return out

    def phi(self, psi) -> np.ndarray:
        """Inverse of psi: natural-scale parameters to the Gaussian scale."""
        out = np.empty_like(psi)
        for j in range(psi.shape[-1]):
            out[..., j] = transform(self.tkind[j], self.tlo[j], self.thi[j], psi[..., j])
        return out

    def rows(self, phi, data=None) -> np.ndarray:
        """Structural parameters of each occasion row of data (default: the fitted data)
        from Gaussian-scale phi (..., subjects, P): phi plus the occasion's random effect."""
        d = data if data is not None else self.data
        counts = np.diff(d.occ_ptr)
        subject = np.repeat(np.arange(counts.size), counts)
        occasion = np.arange(subject.size) - np.repeat(d.occ_ptr[:-1], counts)
        out = phi[..., subject, :self.p0].copy()
        for jj, name in enumerate(self.iov_names):
            j = self.names.index(name)
            k = np.minimum(occasion, self.n_occ - 1)
            out[..., j] += phi[..., subject, self.p0 + k * self.n_iov + jj]
        return self.psi(out)

    def omega_name(self, j) -> str:
        """Reporting name of the sd of random effect j: omega_X, or gamma_X for an occasion."""
        if j < self.p0:
            return f"omega_{self.names[j]}"
        return f"gamma_{self.iov_names[(j - self.p0) % self.n_iov]}"

    def typical(self) -> np.ndarray:
        """Typical values on the natural scale (covariates at zero)."""
        return np.array([inverse(self.tkind[j], self.tlo[j], self.thi[j], self.coef[j][0])
                         for j in range(self.P)])

    def rng_states(self, n_chains: int, stream: int) -> np.ndarray:
        ss = np.random.SeedSequence([self.seed, stream])
        return ss.generate_state(n_chains * self.N, np.uint64).reshape(n_chains, self.N)

    # kernels --------------------------------------------------------------

    def _obs_args(self):
        d = self.data
        return (d.obs_ptr, self.obs_y, self.obs_gl, d.obs_out, d.obs_cens, self.obs_limit,
                self.obs_prev, self.hcol, self.hrep, self.ek, self.ea, self.eb, self.elog)

    def to_scale(self, f, out) -> np.ndarray:
        """Predictions f of outputs out on the scale of the error model (log where exponential)."""
        on_log = self.elog[out]
        if not on_log.any():
            return f
        return np.where(on_log, np.log(np.maximum(f, 1e-300)), f)

    def from_scale(self, v, out) -> np.ndarray:
        """Inverse of to_scale: values on the scale of the error model back to the data."""
        return np.where(self.elog[out], np.exp(v), v) if self.elog.any() else v

    def gather(self, pred) -> np.ndarray:
        """Predictions at the observation records, on the scale of the error model."""
        return self.to_scale(self.prob.gather(pred), self.data.obs_out)

    def cumulative(self, pred) -> np.ndarray:
        """Predictions at the observation records (chains, n_obs) as the tables show them: on a
        record of a hazard output, the events expected since its follow-up opened."""
        d = self.data
        f = self.prob.gather(pred)
        on = self.on_hazard
        if on.any():
            g = self.obs_open[on] + d.grid_ptr[d.obs_subject[on]]
            f[:, on] = np.maximum(f[:, on] - pred[:, g, d.obs_out[on]], 0.0)
        return f

    def hazard_parts(self, pred):
        """On each record of a hazard output (chains, n_obs; NaN on other records): the events
        expected since the previous record of that output, 0 at the first, and the hazard at
        the record, which may be infinite at the first. The record's log-likelihood depends on
        the predictions through these two only."""
        d = self.data
        dH = np.full((pred.shape[0], d.n_obs), np.nan)
        h = np.full((pred.shape[0], d.n_obs), np.nan)
        on = self.on_hazard
        if on.any():
            g, o = d.obs_grid[on], d.obs_out[on]
            gp = np.where(self.obs_prev[on] >= 0, self.obs_prev[on] + d.grid_ptr[d.obs_subject[on]],
                          g)
            dH[:, on] = np.maximum(pred[:, g, o] - pred[:, gp, o], 0.0)
            h[:, on] = pred[:, g, self.hcol[o]]
        return dH, h

    def predict(self, phi):
        """Predictions for Gaussian-scale parameters phi (chains, subjects, P)."""
        return self.prob.predict(self.rows(phi))

    def derivatives(self, phi, idx, h=1e-3):
        """Predictions at Gaussian-scale phi (chains, subjects, P) and their derivatives in
        phi[..., j] for each j of idx: (chains, grid, columns), (chains, grid, columns, len(idx)).

        Exact (sensitivity.py) where they can be, forward differences of step h for the others;
        the predictions come from the sensitivity solve when no difference needs them.
        """
        idx = [int(j) for j in idx]
        C = phi.shape[0]
        # the structural parameter each column moves, and the occasion of a kappa
        base = [j if j < self.p0 else self.names.index(self.iov_names[(j - self.p0) % self.n_iov])
                for j in idx]
        exact = [k for k, a in enumerate(base) if self._exact[a] and self.sensitivities == "exact"]
        fd = [k for k in range(len(idx)) if k not in exact]
        d = np.empty((C, len(self.data.grid_time), self.prob.n_columns, len(idx)))
        pred0 = None
        if fd:
            stack = np.concatenate([phi] * (len(fd) + 1))
            for m, k in enumerate(fd, 1):
                stack[m * C:(m + 1) * C, :, idx[k]] += h
            pred = self.predict(stack)[0]
            pred0 = pred[:C]
            for m, k in enumerate(fd, 1):
                d[..., k] = (pred[m * C:(m + 1) * C] - pred0) / h
        if exact:
            dirs = tuple(base[k] for k in exact)
            if dirs not in self._sensitivity:
                n_h = len(self.model.hazards)
                aug = SensitivityModel(self.prob._lowered, n_h, dirs)
                self._sensitivity[dirs] = self.prob.with_model(aug, n_h * (1 + len(dirs)))
            prob = self._sensitivity[dirs]
            rows = self.rows(phi)
            occasion = np.minimum(self.row_occasion, max(self.n_occ - 1, 0))
            w = np.empty(rows.shape[:2] + (len(exact),))
            for m, k in enumerate(exact):
                a = base[k]
                w[..., m] = jacobian(self.tkind[a], self.tlo[a], self.thi[a], rows[..., a])
                if idx[k] >= self.p0:
                    w[..., m] *= occasion == (idx[k] - self.p0) // self.n_iov
            values, dv = prob.model.split(prob.predict(np.concatenate([rows, w], -1))[0])
            d[..., exact] = dv
            if pred0 is None:
                pred0 = values
        return pred0, d

    def nll(self, pred) -> np.ndarray:
        """Minus log p(y_i | prediction) per chain and subject; inf where it failed."""
        uy = np.empty((pred.shape[0], self.N))
        call(_nll_batch, self.prob._entries,
             (np.ascontiguousarray(pred), self.data.grid_ptr, *self._obs_args(), uy))
        return uy

    def prior(self, omega, sampled):
        """Covariance of the sampled random effects with its Cholesky factor and its
        inverse, each embedded in a (P, P) array that is zero elsewhere."""
        sampled = np.asarray(sampled, np.int64)
        cov = np.zeros((self.P, self.P))
        chol = np.zeros((self.P, self.P))
        prec = np.zeros((self.P, self.P))
        if sampled.size:
            ix = np.ix_(sampled, sampled)
            sub = np.outer(omega[sampled], omega[sampled]) * self.corr[ix]
            cov[ix] = sub
            chol[ix], prec[ix], ok = _factor(np.ascontiguousarray(sub))
            if not ok:
                raise np.linalg.LinAlgError("the covariance of the random effects is not "
                                            "positive definite")
        return cov, chol, prec

    def log_prior(self, e, omega, sampled):
        """log N(e; 0, Omega) of the sampled components, over the leading axes of e."""
        sampled = np.asarray(sampled, np.int64)
        if not sampled.size:
            return np.zeros(e.shape[:-1])
        _, chol, prec = self.prior(omega, sampled)
        x = e[..., sampled]
        quad = np.einsum("...a,ab,...b->...", x, prec[np.ix_(sampled, sampled)], x)
        logdet = 2 * np.log(chol[sampled, sampled]).sum()
        return -0.5 * quad - 0.5 * logdet - 0.5 * sampled.size * math.log(2 * math.pi)

    def estep(self, eta, rng, omega, sampled, mcmc, block, scale2, scale3):
        """Metropolis pass; eta and rng move in place.

        Returns uy, pred and the acceptance rates of kernels 2 and 3 by parameter.
        """
        C = eta.shape[0]
        _, chol, prec = self.prior(np.asarray(omega, float), sampled)
        args = self.prob.kernel_args(C)
        chunks = args[-1]
        uy = np.empty((C, self.N))
        pred = self.prob.empty(C)
        acc2 = np.zeros((chunks, self.P))
        acc3 = np.zeros((chunks, self.P))
        try3 = np.zeros((chunks, self.P))
        call(_estep, self.prob._entries,
             (*args, *self._obs_args(), self.mean(), chol, prec,
              self.tkind, self.tlo, self.thi, self.p0, self.iov_of, max(self.n_iov, 1),
              np.ascontiguousarray(sampled, np.int64),
              int(mcmc[0]), int(mcmc[1]), int(mcmc[2]), int(mcmc[3]) if len(mcmc) > 3 else 0,
              int(block),
              np.ascontiguousarray(scale2, float), np.ascontiguousarray(scale3, float), eta, uy,
              rng, pred, acc2, acc3, try3))
        n2 = max(1, mcmc[1] * C * self.N)
        t3 = try3.sum(0)
        return uy, pred, acc2.sum(0) / n2, np.where(t3 > 0, acc3.sum(0) / np.maximum(t3, 1),
                                                    np.nan)

    def residual_sd(self, f, out) -> np.ndarray:
        """Residual sd g(f) for predictions f of outputs out (same shape); NaN on a hazard
        output, which has no residual error."""
        k, a, b = self.ek[out], self.ea[out], self.eb[out]
        g = np.select([k == 0, k == 1, k == 2], [a + 0 * f, b * np.abs(f), a + b * np.abs(f)],
                      np.sqrt(a * a + b * b * f * f))
        g = np.maximum(g, _G_MIN)
        return np.where(self.hcol[out] >= 0, np.nan, g) if self.model.hazards else g

    def check_initial(self):
        phi = self.mean()[None]
        pred, status = self.predict(phi)
        values = dict(zip(self.names, np.round(self.typical(), 6)))
        if self.on_hazard.any():
            self._check_hazards(pred[0], status[0], values)
        bad = int((status != 0).sum())
        if bad or not np.isfinite(self.nll(pred)).all():
            raise FitError(
                "the model cannot be evaluated at the initial values "
                f"({bad} of {self.N} subjects fail): check the equations and the initial "
                f"values {values}")

    def _check_hazards(self, pred, status, values):
        """At the initial values, on the subjects solved: a hazard negative at a record, zero or
        not finite at an event at a known time, or whose integral falls between two records,
        which only a negative hazard in between can do."""
        d = self.data
        rec = np.flatnonzero(self.on_hazard & (status[d.obs_subject] == 0))
        o, g = d.obs_out[rec], d.obs_grid[rec]
        h, H = pred[g, self.hcol[o]], pred[g, o]
        later = self.obs_prev[rec] >= 0
        gp = np.where(later, self.obs_prev[rec] + d.grid_ptr[d.obs_subject[rec]], g)
        Hp = pred[gp, o]
        rtol = self.prob.rtol
        # 10 times the solver's tolerance on the integral, whose floor is 1e-3 * rtol events
        falls = Hp - H > 10 * (rtol * np.maximum(np.abs(H), np.abs(Hp)) + 1e-3 * rtol)
        exact = later & (self.obs_y[rec] > 0) & (d.obs_cens[rec] == 0)
        for bad, why in (
                (h < 0, lambda k: f"is negative ({h[k]:.6g}) at time {d.obs_time[rec[k]]:g}"),
                (exact & ~((h > 0) & np.isfinite(h)),
                 lambda k: f"is {h[k]:.6g} at the event at time {d.obs_time[rec[k]]:g}, which "
                           "needs a positive finite hazard"),
                (falls, lambda k: f"has an integral falling from {Hp[k]:.6g} to {H[k]:.6g} "
                                  f"between times {d.grid_time[gp[k]]:g} and "
                                  f"{d.obs_time[rec[k]]:g}: it is negative in between")):
            if bad.any():
                k = int(np.flatnonzero(bad)[0])
                raise FitError(f"at the initial values, the hazard {self.model.outputs[o[k]]} of "
                               f"subject {d.ids[d.obs_subject[rec[k]]]} {why(k)}; check its "
                               f"expression and the initial values {values}")
