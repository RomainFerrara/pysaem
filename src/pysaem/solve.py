"""Compiled kernels: event timeline, closed form of linear systems, explicit and
implicit integrators, batch prediction.

A linear system with constant coefficients is advanced by the exponential of its
matrix. The explicit integrator is Dormand-Prince 5(4) with dense output and the
stiffness test of Hairer; the implicit one is Radau IIA order 5, ported from SciPy
(scipy.integrate._ivp.radau, BSD-3-Clause, copyright SciPy Developers).
"""

from __future__ import annotations

import contextlib
import copy
import math
import sys

import numba
import numpy as np
from numba import cfunc, njit, prange, types

from .data import DataError, Dataset, shown
from .model import HISTORY, Model, history_breakpoints, reads_inputs_history

OK, MAX_STEPS, NOT_FINITE, STIFF = 0, 1, 2, 3
_EVENT = 4  # an integrator stopped just past a switch on the states: restarted with a fresh step
_MAX_EVENTS = 100  # switches located in one segment before the subject goes on without locating
_REG, _BREAK, _OBS, _OFF, _BOLUS, _ON, _RESET, _SS, _OCC = 0, 1, 2, 3, 4, 5, 6, 7, 8
_METHODS = {"auto": 0, "explicit": 1, "implicit": 2}
_EXPLICIT_STEPS = 500  # explicit steps one segment may take before the solve goes to the implicit solver
_NS = 18  # state-sized work rows: 7 stages, ynew ysti ydiff bspl rc4 rc5 yint tol y u ymax
# bits of a model's structure; _INPUT_HISTORY: an input reads the signals of HISTORY at its record
_AUTONOMOUS, _CLOSED_FORM, _INPUT_HISTORY = 1, 2, 4
_HAZARDS = 8  # structure // _HAZARDS: how many last states integrate a hazard (Model.lowered)
_NUMERICAL, _IMPLICIT, _EXACT = 0, 1, 2  # how a solve was carried out
# Pade (6, 6) coefficients of the exponential
_PADE = np.array([1.0, 1 / 2, 5 / 44, 1 / 66, 1 / 792, 1 / 15840, 1 / 665280])

_S6 = 6**0.5
_C = np.array([(4 - _S6) / 10, (4 + _S6) / 10, 1.0])
_E = np.array([-13 - 7 * _S6, -13 + 7 * _S6, -1.0]) / 3
_MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
_MU_COMPLEX = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3)) - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6))
_T = np.array([
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1.0, 1.0, 0.0]])
_TI = np.array([
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
_P = np.array([
    [13 / 3 + 7 * _S6 / 3, -23 / 3 - 22 * _S6 / 3, 10 / 3 + 5 * _S6],
    [13 / 3 - 7 * _S6 / 3, -23 / 3 + 22 * _S6 / 3, 10 / 3 - 5 * _S6],
    [1 / 3, -8 / 3, 10 / 3]])
_NEWTON_MAXITER = 6
_EPS = np.finfo(float).eps


@njit(cache=True, error_model="numpy")
def _lu(A, piv):
    """In-place LU with partial pivoting; False when singular."""
    n = A.shape[0]
    for k in range(n):
        m = k
        big = abs(A[k, k])
        for i in range(k + 1, n):
            v = abs(A[i, k])
            if v > big:
                big = v
                m = i
        piv[k] = m
        if not big > 0.0:
            return False
        if m != k:
            for j in range(n):
                tmp = A[k, j]
                A[k, j] = A[m, j]
                A[m, j] = tmp
        for i in range(k + 1, n):
            A[i, k] /= A[k, k]
            for j in range(k + 1, n):
                A[i, j] -= A[i, k] * A[k, j]
    return True


@njit(cache=True, error_model="numpy")
def _lu_solve(A, piv, b):
    # _lu swaps whole rows, multipliers included: all the swaps go before the forward pass
    n = A.shape[0]
    for k in range(n):
        m = piv[k]
        if m != k:
            tmp = b[k]
            b[k] = b[m]
            b[m] = tmp
    for k in range(n):
        for i in range(k + 1, n):
            b[i] -= A[i, k] * b[k]
    for k in range(n - 1, -1, -1):
        for j in range(k + 1, n):
            b[k] -= A[k, j] * b[j]
        b[k] /= A[k, k]


@njit(cache=True, error_model="numpy")
def _lu_solve_scaled(A, piv, b, d):
    # A was factored as D^-1 M D, D = diag(d): solves M x = b
    for k in range(b.size):
        b[k] /= d[k]
    _lu_solve(A, piv, b)
    for k in range(b.size):
        b[k] *= d[k]


@njit(cache=True, error_model="numpy")
def _matmul(A, B, C):
    m = A.shape[0]
    for i in range(m):
        for j in range(m):
            v = 0.0
            for k in range(m):
                v += A[i, k] * B[k, j]
            C[i, j] = v


@njit(cache=True, error_model="numpy")
def _squarings(M, h):
    """Squarings _expm needs for exp(h M): halvings that bring its 1-norm to 0.5. Rounding
    grows by about 2**squarings eps (1e-11 for rates two decades apart over a week)."""
    norm = 0.0
    for j in range(M.shape[1]):
        v = 0.0
        for i in range(M.shape[0]):
            v += abs(M[i, j])
        norm = max(norm, v)
    norm *= abs(h)
    if not np.isfinite(norm):
        return -1
    squarings = 0
    while norm > 0.5:
        norm *= 0.5
        squarings += 1
    return squarings


@njit(cache=True, error_model="numpy")
def _expm(M, h, L, piv, col):
    """exp(h M) into L[0]; False when it is not finite.

    Pade (6, 6) approximant after scaling to a 1-norm of 0.5 at most, where its
    error is below machine precision, then repeated squaring.
    """
    m = M.shape[0]
    E, X, X2, X4, X6, T, U = L[0], L[1], L[2], L[3], L[4], L[5], L[6]
    squarings = _squarings(M, h)
    if squarings < 0:
        return False
    scale = h / 2.0**squarings
    for i in range(m):
        for j in range(m):
            X[i, j] = M[i, j] * scale
    _matmul(X, X, X2)
    _matmul(X2, X2, X4)
    _matmul(X4, X2, X6)
    for i in range(m):
        for j in range(m):
            T[i, j] = _PADE[3] * X2[i, j] + _PADE[5] * X4[i, j]
        T[i, i] += _PADE[1]
    _matmul(X, T, U)
    for i in range(m):
        for j in range(m):
            T[i, j] = _PADE[2] * X2[i, j] + _PADE[4] * X4[i, j] + _PADE[6] * X6[i, j]
        T[i, i] += _PADE[0]
    # exp(X) = (V - U)^-1 (V + U), V the even part held in T
    for i in range(m):
        for j in range(m):
            X2[i, j] = T[i, j] + U[i, j]
            X4[i, j] = T[i, j] - U[i, j]
    if not _lu(X4, piv):
        return False
    for j in range(m):
        for i in range(m):
            col[i] = X2[i, j]
        _lu_solve(X4, piv, col)
        for i in range(m):
            E[i, j] = col[i]
    for _ in range(squarings):
        _matmul(E, E, X)
        for i in range(m):
            for j in range(m):
                E[i, j] = X[i, j]
    return True


@njit(cache=True, error_model="numpy")
def _linear_system(rhs, jac, t, p, r, u, W, M):
    """Matrix of dy/dt = A y + c + u, augmented with its constant column when M
    has one more row than there are states."""
    n = W.shape[1]
    zero, none, c = W[0], W[1], W[3]
    for q in range(n):
        zero[q] = 0.0
        none[q] = 0.0
    rhs(t, zero, p, r, none, c)
    jac(t, zero, p, r, M)
    if M.shape[0] > n:
        for q in range(n):
            M[q, n] = c[q] + u[q]
            M[n, q] = 0.0
        M[n, n] = 0.0


@njit(cache=True, error_model="numpy")
def _transition(E, y, work):
    """y moved by the transition matrix E; False when the result is not finite."""
    n = y.size
    ok = True
    for i in range(n):
        v = E[i, n] if E.shape[0] > n else 0.0
        for j in range(n):
            v += E[i, j] * y[j]
        work[i] = v
        ok = ok and np.isfinite(v)
    for i in range(n):
        y[i] = work[i]
    return ok


@njit(cache=True, error_model="numpy")
def _sort_timeline(at, ak, ai, av, m):
    # stable bottom-up merge sort by time, O(m log m): equal times keep their fill order
    src = np.arange(m)
    dst = np.empty(m, np.int64)
    width = 1
    while width < m:
        for lo in range(0, m, 2 * width):
            mid, hi = min(lo + width, m), min(lo + 2 * width, m)
            a, b = lo, mid
            for c in range(lo, hi):
                if a < mid and (b >= hi or at[src[a]] <= at[src[b]]):
                    dst[c] = src[a]
                    a += 1
                else:
                    dst[c] = src[b]
                    b += 1
        src, dst = dst, src
        width *= 2
    t, k, i, v = at[:m].copy(), ak[:m].copy(), ai[:m].copy(), av[:m].copy()
    for c in range(m):
        q = src[c]
        at[c], ak[c], ai[c], av[c] = t[q], k[q], i[q], v[q]


@njit(cache=True, error_model="numpy")
def _event_rate(ea, er, edur, q):
    # an input with a duration gives each amount at the rate amount / duration; an amount of 0
    # is no infusion, whatever its rate
    if not ea[q] > 0.0:
        return 0.0
    return ea[q] / edur[q] if edur[q] > 0.0 else er[q]


@njit(cache=True, error_model="numpy")
def _ss_shift(lag, ii):
    # lag = n ii + rem, 0 <= rem < ii: n inputs of a steady-state series start at or after its
    # record
    rem = lag % ii
    return np.floor((lag - rem) / ii + 0.5), rem


@njit(cache=True, error_model="numpy")
def _on_time(et, ed, ess, ne, code):
    # start of an infusion entry: code q is the record's own input, q + ne the one of the
    # steady-state series running at the record, q + (j + 2) ne the j-th series input after it
    q, c = code % ne, code // ne
    if c == 0:
        return et[q] + ed[q]
    if c == 1:
        return et[q]
    return et[q] + _ss_shift(ed[q], ess[q])[1] + (c - 2) * ess[q]


@njit(cache=True, error_model="numpy")
def _interior(t, tend):
    eps = 1e-13 * max(1.0, abs(t), abs(tend))
    lo, hi = t + eps, tend - eps
    if hi < lo:
        lo = hi = 0.5 * (t + tend)
    return lo, hi


@njit(cache=True, error_model="numpy")
def _jac_fd(rhs, t, y, p, r, J, scale):
    # forward differences: the step starts at 1.5e-8 of the state (at least atol/rtol) and grows
    # x100, up to 1e-4 of the size of the states, while a row loses its change in rounding; each
    # entry is taken at the smallest step that shows the change of its row; nan where y or f is not finite
    m = y.size
    f0 = np.empty(m)
    f1 = np.empty(m)
    seen = np.empty(m, np.bool_)
    u0 = np.zeros(m)
    rhs(t, y, p, r, u0, f0)
    # the size of the states, or of their derivatives over a unit of time when all are at zero
    big = 0.0
    for k in range(m):
        if np.isfinite(y[k]):
            big = max(big, abs(y[k]))
    if big == 0.0:
        for k in range(m):
            if np.isfinite(f0[k]):
                big = max(big, abs(f0[k]))
    for j in range(m):
        keep = y[j]
        # an infinite scale marks a state that no equation reads (a hazard integral during
        # steady-state cycles): its column is zero
        for i in range(m):
            J[i, j] = np.nan if np.isfinite(scale[j]) else 0.0
            seen[i] = False
        if not np.isfinite(keep) or not np.isfinite(scale[j]):
            continue
        base = max(abs(keep), scale[j], 1e-12)
        h = 1.5e-8 * base
        top = 1e-4 * max(base, big)
        for _ in range(40):
            y[j] = keep + h
            step = y[j] - keep
            rhs(t, y, p, r, u0, f1)
            finite = True
            for i in range(m):
                if not np.isfinite(f1[i]):
                    finite = False
            if not finite:
                break
            lost = False
            for i in range(m):
                if seen[i]:
                    continue
                d = f1[i] - f0[i]
                J[i, j] = d / step
                if abs(d) > 1e-9 * max(abs(f0[i]), abs(f1[i])):
                    seen[i] = True
                elif f0[i] != 0.0:
                    lost = True
            if not lost or not h < top:
                break
            h = min(100.0 * h, top)
        y[j] = keep



@njit(cache=True, error_model="numpy")
def _grow_tol(y, ymax, tol, dyn):
    # the largest value of a state is followed step by step, not only at events: a state that
    # rose to 60 and fell to 1e-16 between two events kept the tolerance of 1e-16
    for j in range(y.size):
        a = abs(y[j])
        if a > ymax[j]:
            ymax[j] = a
            tol[j] = max(tol[j], dyn * a)


@njit(cache=True, error_model="numpy")
def _observe(out_fn, t, y, p, r, o, tol):
    """Outputs at y; when one is not finite, again with every state within its absolute tolerance
    below zero read as zero: at the accuracy of the solve it is zero (a state decayed to 1e-30 and
    returned at -1e-17 made C**0.8 nan and failed the subject)."""
    out_fn(t, y, p, r, o)
    finite = True
    for q in range(o.size):
        finite = finite and np.isfinite(o[q])
    if finite:
        return
    near = False
    for j in range(y.size):
        near = near or (y[j] < 0.0 and y[j] >= -tol[j])
    if not near:
        return
    z = y.copy()
    for j in range(z.size):
        if z[j] < 0.0 and z[j] >= -tol[j]:
            z[j] = 0.0
    out_fn(t, z, p, r, o)


@njit(cache=True, error_model="numpy")
def _jacobian(rhs, jac, has_jac, t, y, p, r, J, scale):
    """J from the generated derivatives, by differences where one is not finite (the derivative
    of C**g at C = 0 for g < 1, which a power of a state at zero meets)."""
    if has_jac:
        jac(t, y, p, r, J)
        finite = True
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                finite = finite and np.isfinite(J[i, j])
        if finite:
            return
    _jac_fd(rhs, t, y, p, r, J, scale)

@njit(cache=True, error_model="numpy")
def _hinit(rhs, t, y, f0, p, r, u, rtol, atol, order, span, y1, f1):
    n = y.size
    d0 = 0.0
    d1 = 0.0
    for j in range(n):
        sc = atol[j] + rtol * abs(y[j])
        d0 += (y[j] / sc) ** 2
        d1 += (f0[j] / sc) ** 2
    d0 = np.sqrt(d0 / n)
    d1 = np.sqrt(d1 / n)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    h0 = min(h0, span)
    for j in range(n):
        y1[j] = y[j] + h0 * f0[j]
    rhs(t + h0, y1, p, r, u, f1)
    d2 = 0.0
    for j in range(n):
        sc = atol[j] + rtol * abs(y[j])
        d2 += ((f1[j] - f0[j]) / sc) ** 2
    d2 = np.sqrt(d2 / n) / h0
    big = max(d1, d2)
    h1 = max(1e-6, h0 * 1e-3) if big <= 1e-15 else (0.01 / big) ** (1.0 / (order + 1))
    h = min(100 * h0, h1, span)
    return h if np.isfinite(h) and h > 0 else 1e-6


@njit(cache=True, error_model="numpy")
def _dopri_dense(y, W, hs):
    # coefficients of the dense output of a Dormand-Prince step, in rows 9 to 12 of W
    for j in range(y.size):
        W[9, j] = W[7, j] - y[j]
        W[10, j] = hs * W[0, j] - W[9, j]
        W[11, j] = W[9, j] - hs * W[6, j] - W[10, j]
        W[12, j] = hs * (-12715105075 / 11282082432 * W[0, j]
                         + 87487479700 / 32700410799 * W[2, j]
                         - 10690763975 / 1880347072 * W[3, j]
                         + 701980252875 / 199316789632 * W[4, j]
                         - 1453857185 / 822651844 * W[5, j]
                         + 69997945 / 29380423 * W[6, j])


@njit(cache=True, error_model="numpy")
def _dense(kind, y, D, th, out):
    # y at the fraction th of a step from y: Dormand-Prince (kind 0, D rows 9 to 12 of W) or
    # Radau (kind 1, D the collocation polynomial Q)
    if kind == 0:
        th1 = 1.0 - th
        for j in range(y.size):
            out[j] = y[j] + th * (D[9, j] + th1 * (D[10, j] + th * (D[11, j] + th1 * D[12, j])))
    else:
        for j in range(y.size):
            out[j] = y[j] + th * (D[j, 0] + th * (D[j, 1] + th * D[j, 2]))


@njit(cache=True, error_model="numpy")
def _locate(swf, kind, t, hs, y, D, p, r, G, n_sw, yint):
    """Fraction of the step [t, t + hs] just past the first root of the switching functions whose
    sign differs at its ends (G[0] at t, G[1] at the end), by the Illinois method on the dense
    output, to 1e-10 of the step; 1 when that is the end. G[2] and G[3] are work rows."""
    g0, gc, gb = G[0], G[2], G[3]
    for q in range(n_sw):
        gb[q] = G[1, q]
    tol = max(1e-10, 4 * _EPS * max(abs(t), abs(t + hs)) / hs)
    best = 1.0
    for q in range(n_sw):
        # the values at the current best: a switch with its old sign there crosses later
        if not g0[q] * gb[q] < 0.0:
            continue
        a, fa, b, fb = 0.0, g0[q], best, gb[q]
        side = 0
        for _ in range(200):
            if b - a <= tol:
                break
            c = (a * fb - b * fa) / (fb - fa)
            if not (a < c < b):
                c = 0.5 * (a + b)
            _dense(kind, y, D, c, yint)
            swf(t + c * hs, yint, p, r, gc)
            fc = gc[q]
            if fc == 0.0 or (fc > 0.0) == (g0[q] > 0.0):
                a, fa = c, fc
                if side == -1:
                    fb *= 0.5
                side = -1
            else:
                b, fb = c, fc
                for k in range(n_sw):
                    gb[k] = gc[k]
                if side == 1:
                    fa *= 0.5
                side = 1
        best = b
    return best


@njit(cache=True, error_model="numpy")
def _dopri(rhs, out_fn, autonomous, t, tend, y, p, r, u, h, rtol, atol, at, ai, i, i1,
           pred, W, o, max_steps, detect, swf, n_sw, G, ymax, dyn):
    n = y.size
    n_out = o.size
    k = W
    ynew, ysti, yint = W[7], W[8], W[13]
    # stage times stay strictly inside the segment: a switch at either end
    # must not be evaluated on the wrong side
    lo, hi = _interior(t, tend)
    rhs(lo, y, p, r, u, k[0])
    if autonomous:
        rest = True
        for j in range(n):
            if k[0, j] != 0.0:
                rest = False
                break
        if rest:
            # equilibrium of an autonomous segment: the state does not move
            while i < i1:
                _observe(out_fn, at[i], y, p, r, o, atol)
                for q in range(n_out):
                    pred[ai[i], q] = o[q]
                i += 1
            return tend, h, i, OK, 0
    if h <= 0.0:
        h = _hinit(rhs, lo, y, k[0], p, r, u, rtol, atol, 4, tend - t, ynew, k[1])
    if n_sw > 0:
        swf(t, y, p, r, G[0])
    # a step that crosses a switch is taken again up to just past it (stop, cut)
    stop = tend
    cut = False
    nstep = 0
    iasti = 0
    nonsti = 0
    while t < tend:
        if nstep >= max_steps:
            return t, h, i, MAX_STEPS, nstep
        nstep += 1
        last = t + 1.0000001 * h >= stop
        hs = stop - t if last else h
        for j in range(n):
            yint[j] = y[j] + hs * (k[0, j] / 5)
        rhs(min(max(t + hs / 5, lo), hi), yint, p, r, u, k[1])
        for j in range(n):
            yint[j] = y[j] + hs * (3 / 40 * k[0, j] + 9 / 40 * k[1, j])
        rhs(min(max(t + 3 * hs / 10, lo), hi), yint, p, r, u, k[2])
        for j in range(n):
            yint[j] = y[j] + hs * (44 / 45 * k[0, j] - 56 / 15 * k[1, j] + 32 / 9 * k[2, j])
        rhs(min(max(t + 4 * hs / 5, lo), hi), yint, p, r, u, k[3])
        for j in range(n):
            yint[j] = y[j] + hs * (19372 / 6561 * k[0, j] - 25360 / 2187 * k[1, j]
                                   + 64448 / 6561 * k[2, j] - 212 / 729 * k[3, j])
        rhs(min(max(t + 8 * hs / 9, lo), hi), yint, p, r, u, k[4])
        for j in range(n):
            ysti[j] = y[j] + hs * (9017 / 3168 * k[0, j] - 355 / 33 * k[1, j]
                                   + 46732 / 5247 * k[2, j] + 49 / 176 * k[3, j]
                                   - 5103 / 18656 * k[4, j])
        tnew = stop if last else t + hs
        rhs(min(tnew, hi), ysti, p, r, u, k[5])
        for j in range(n):
            ynew[j] = y[j] + hs * (35 / 384 * k[0, j] + 500 / 1113 * k[2, j]
                                   + 125 / 192 * k[3, j] - 2187 / 6784 * k[4, j]
                                   + 11 / 84 * k[5, j])
        rhs(min(tnew, hi), ynew, p, r, u, k[6])
        err = 0.0
        for j in range(n):
            e = hs * (71 / 57600 * k[0, j] - 71 / 16695 * k[2, j] + 71 / 1920 * k[3, j]
                      - 17253 / 339200 * k[4, j] + 22 / 525 * k[5, j] - 1 / 40 * k[6, j])
            sc = atol[j] + rtol * max(abs(y[j]), abs(ynew[j]))
            err += (e / sc) ** 2
        err = np.sqrt(err / n)
        if not np.isfinite(err):
            h = hs * 0.1
            if h < 1e-14 * max(1.0, abs(t)):
                return t, h, i, NOT_FINITE, nstep
            continue
        if err > 1.0:
            h = hs * max(0.2, 0.9 * err ** -0.2)
            continue
        dense = False
        if n_sw > 0:
            swf(tnew, ynew, p, r, G[1])
            cross = False
            for q in range(n_sw):
                cross = cross or G[0, q] * G[1, q] < 0.0
            if cut and last and not cross:
                # the step taken again ends short of the switch: it is located from the next one
                cut = False
                stop = tend
            elif cross and not (cut and last):
                _dopri_dense(y, W, hs)
                dense = True
                th = _locate(swf, 0, t, hs, y, W, p, r, G, n_sw, yint)
                cut = True
                stop = tnew
                if th < 1.0 and t + th * hs > t:
                    stop = t + th * hs
                    h = stop - t
                    continue
        if i < i1 and at[i] <= tnew:
            if not dense:
                _dopri_dense(y, W, hs)
            while i < i1 and at[i] <= tnew:
                _dense(0, y, W, (at[i] - t) / hs, yint)
                _observe(out_fn, at[i], yint, p, r, o, atol)
                for q in range(n_out):
                    pred[ai[i], q] = o[q]
                i += 1
        stiff = False
        if detect:
            num = 0.0
            den = 0.0
            for j in range(n):
                num += (k[6, j] - k[5, j]) ** 2
                den += (ynew[j] - ysti[j]) ** 2
            if den > 0.0 and hs * np.sqrt(num / den) > 3.25:
                nonsti = 0
                iasti += 1
                stiff = iasti == 15
            else:
                nonsti += 1
                if nonsti == 6:
                    iasti = 0
        for j in range(n):
            y[j] = ynew[j]
            k[0, j] = k[6, j]
        _grow_tol(y, ymax, atol, dyn)
        for q in range(n_sw):
            G[0, q] = G[1, q]
        t = tnew
        grow = 10.0 if err < 1e-10 else min(10.0, 0.9 * err ** -0.2)
        h = max(h, hs * grow) if last else hs * grow
        if cut and t >= stop:
            return t, h, i, _EVENT, nstep
        if stiff and t < tend:
            return t, h, i, STIFF, nstep
    return t, h, i, OK, nstep


@njit(cache=True, error_model="numpy")
def _radau(rhs, jac, has_jac, out_fn, t, tend, y, p, r, u, h_abs, rtol, atol, at, ai, i, i1,
           pred, o, max_steps, swf, n_sw, G, ymax, dyn):
    n = y.size
    n_out = o.size
    f = np.empty(n)
    f_new = np.empty(n)
    J = np.empty((n, n))
    lo, hi = _interior(t, tend)
    rhs(lo, y, p, r, u, f)
    _jacobian(rhs, jac, has_jac, lo, y, p, r, J, atol / rtol)
    current_jac = True
    LUr = np.empty((n, n))
    LUc = np.empty((n, n), np.complex128)
    pr = np.empty(n, np.int64)
    pc = np.empty(n, np.int64)
    d = np.ones(n)
    have_lu = False
    Z = np.zeros((3, n))
    Z0 = np.zeros((3, n))
    Wm = np.empty((3, n))
    dW = np.empty((3, n))
    dW_old = np.empty((3, n))
    F = np.empty((3, n))
    Q = np.zeros((n, 3))
    y_old = np.empty(n)
    y_new = np.empty(n)
    ytmp = np.empty(n)
    ze = np.empty(n)
    error = np.empty(n)
    scale = np.empty(n)
    fr = np.empty(n)
    fc = np.empty(n, np.complex128)
    have_sol = False
    t_old = t
    h_old = 0.0
    h_abs_old = -1.0
    error_norm_old = -1.0
    newton_tol = max(10 * _EPS / rtol, min(0.03, rtol ** 0.5))
    ti_c = _TI[1] + 1j * _TI[2]
    if h_abs <= 0.0:
        h_abs = _hinit(rhs, lo, y, f, p, r, u, rtol, atol, 3, tend - t, ytmp, f_new)
    if n_sw > 0:
        swf(t, y, p, r, G[0])
    stop = tend
    cut = False
    nstep = 0
    n_iter = 0
    rate = -1.0
    error_norm = 0.0
    safety = 0.9
    while t < tend:
        min_step = 2.3e-15 * max(abs(t), 1e-12)
        use_old = True
        if h_abs < min_step:
            h_abs = min_step
            use_old = False
        rejected = False
        accepted = False
        t_new = t
        h = h_abs
        while not accepted:
            nstep += 1
            if nstep > max_steps:
                return t, h_abs, i, MAX_STEPS, nstep
            if h_abs < min_step:
                return t, h_abs, i, NOT_FINITE, nstep
            t_new = t + h_abs
            if t + 1.0000001 * h_abs >= stop:
                t_new = stop
            h = t_new - t
            h_abs = h
            if have_sol:
                for s in range(3):
                    x = (t + h * _C[s] - t_old) / h_old
                    for j in range(n):
                        Z0[s, j] = (y_old[j] + x * (Q[j, 0] + x * (Q[j, 1] + x * Q[j, 2]))
                                    - y[j])
            else:
                Z0[:, :] = 0.0
            for j in range(n):
                scale[j] = atol[j] + abs(y[j]) * rtol
            converged = False
            while not converged:
                if not have_lu:
                    # the Newton matrix in units of each state's tolerance: unscaled, partial
                    # pivoting chose for a state with a tolerance of 1e-30 the row of a state at
                    # 3e4 coupled to it by 1e16, whose rounding (2e-26) then stalled Newton
                    for a in range(n):
                        d[a] = atol[a] + rtol * abs(y[a])
                        if not (d[a] > 0.0 and np.isfinite(d[a])):
                            d[a] = 1.0
                    for a in range(n):
                        for b in range(n):
                            LUr[a, b] = -J[a, b] * (d[b] / d[a])
                            LUc[a, b] = -J[a, b] * (d[b] / d[a])
                        LUr[a, a] += _MU_REAL / h
                        LUc[a, a] += _MU_COMPLEX / h
                    have_lu = _lu(LUr, pr) and _lu(LUc, pc)
                if have_lu:
                    # simplified Newton on the collocation system
                    for s in range(3):
                        for j in range(n):
                            Z[s, j] = Z0[s, j]
                            Wm[s, j] = (_TI[s, 0] * Z0[0, j] + _TI[s, 1] * Z0[1, j]
                                        + _TI[s, 2] * Z0[2, j])
                    rate = -1.0
                    n_iter = 0
                    for it in range(_NEWTON_MAXITER):
                        n_iter = it + 1
                        finite = True
                        for s in range(3):
                            for j in range(n):
                                ytmp[j] = y[j] + Z[s, j]
                            rhs(min(max(t + h * _C[s], lo), hi), ytmp, p, r, u, F[s])
                            for j in range(n):
                                if not np.isfinite(F[s, j]):
                                    finite = False
                        if not finite:
                            break
                        for j in range(n):
                            fr[j] = (F[0, j] * _TI[0, 0] + F[1, j] * _TI[0, 1]
                                     + F[2, j] * _TI[0, 2]) - _MU_REAL / h * Wm[0, j]
                            fc[j] = (F[0, j] * ti_c[0] + F[1, j] * ti_c[1]
                                     + F[2, j] * ti_c[2]) - _MU_COMPLEX / h * (
                                         Wm[1, j] + 1j * Wm[2, j])
                        _lu_solve_scaled(LUr, pr, fr, d)
                        _lu_solve_scaled(LUc, pc, fc, d)
                        # the scale follows the iterate, as the error scale does; the rate
                        # compares the last two corrections on that same scale, since a scale
                        # at atol (states at zero) would make the first one look huge
                        dW_norm = 0.0
                        dW_norm_old = 0.0
                        for j in range(n):
                            scale[j] = atol[j] + rtol * max(abs(y[j]), abs(y[j] + Z[2, j]))
                            dW[0, j] = fr[j]
                            dW[1, j] = fc[j].real
                            dW[2, j] = fc[j].imag
                            for s in range(3):
                                dW_norm += (dW[s, j] / scale[j]) ** 2
                                if it > 0:
                                    dW_norm_old += (dW_old[s, j] / scale[j]) ** 2
                        dW_norm = np.sqrt(dW_norm / (3 * n))
                        if it > 0:
                            dW_norm_old = np.sqrt(dW_norm_old / (3 * n))
                            rate = dW_norm / dW_norm_old if dW_norm_old > 0.0 else np.inf
                        if rate >= 0.0 and (rate >= 1.0 or rate ** (_NEWTON_MAXITER - it)
                                            / (1 - rate) * dW_norm > newton_tol):
                            break
                        for s in range(3):
                            for j in range(n):
                                Wm[s, j] += dW[s, j]
                        for s in range(3):
                            for j in range(n):
                                Z[s, j] = (_T[s, 0] * Wm[0, j] + _T[s, 1] * Wm[1, j]
                                           + _T[s, 2] * Wm[2, j])
                        if dW_norm == 0.0 or (rate >= 0.0
                                              and rate / (1 - rate) * dW_norm < newton_tol):
                            converged = True
                            break
                        for s in range(3):
                            for j in range(n):
                                dW_old[s, j] = dW[s, j]
                if not converged:
                    if current_jac:
                        break
                    _jacobian(rhs, jac, has_jac, max(t, lo), y, p, r, J, atol / rtol)
                    current_jac = True
                    have_lu = False
            if not converged:
                h_abs *= 0.5
                have_lu = False
                continue
            for j in range(n):
                y_new[j] = y[j] + Z[2, j]
                ze[j] = (Z[0, j] * _E[0] + Z[1, j] * _E[1] + Z[2, j] * _E[2]) / h
                error[j] = f[j] + ze[j]
            _lu_solve_scaled(LUr, pr, error, d)
            error_norm = 0.0
            for j in range(n):
                scale[j] = atol[j] + max(abs(y[j]), abs(y_new[j])) * rtol
                error_norm += (error[j] / scale[j]) ** 2
            error_norm = np.sqrt(error_norm / n)
            safety = 0.9 * (2 * _NEWTON_MAXITER + 1) / (2 * _NEWTON_MAXITER + n_iter)
            if rejected and error_norm > 1.0:
                for j in range(n):
                    ytmp[j] = y[j] + error[j]
                rhs(max(t, lo), ytmp, p, r, u, error)
                for j in range(n):
                    error[j] += ze[j]
                _lu_solve_scaled(LUr, pr, error, d)
                error_norm = 0.0
                for j in range(n):
                    error_norm += (error[j] / scale[j]) ** 2
                error_norm = np.sqrt(error_norm / n)
            if not np.isfinite(error_norm):
                h_abs *= 0.2
                have_lu = False
                rejected = True
            elif error_norm > 1.0:
                factor = _predict_factor(h_abs, h_abs_old, error_norm, error_norm_old,
                                         use_old)
                h_abs *= max(0.2, safety * factor)
                have_lu = False
                rejected = True
            else:
                accepted = True
        # the collocation polynomial of the step from t: dense output and next prediction
        for j in range(n):
            y_old[j] = y[j]
            for m in range(3):
                Q[j, m] = Z[0, j] * _P[0, m] + Z[1, j] * _P[1, m] + Z[2, j] * _P[2, m]
        t_old = t
        h_old = h
        have_sol = True
        if n_sw > 0:
            swf(t_new, y_new, p, r, G[1])
            cross = False
            for q in range(n_sw):
                cross = cross or G[0, q] * G[1, q] < 0.0
            if cut and t_new >= stop and not cross:
                # the step taken again ends short of the switch: it is located from the next one
                cut = False
                stop = tend
            elif cross and not (cut and t_new >= stop):
                th = _locate(swf, 1, t, h, y, Q, p, r, G, n_sw, ytmp)
                cut = True
                stop = t_new
                if th < 1.0 and th * h > min_step:
                    # the step is taken again up to just past the switch
                    stop = t + th * h
                    h_abs = stop - t
                    have_lu = False
                    continue
        recompute_jac = n_iter > 2 and rate > 1e-3
        factor = min(10.0, safety * _predict_factor(h_abs, h_abs_old, error_norm,
                                                    error_norm_old, use_old))
        if not recompute_jac and factor < 1.2:
            factor = 1.0
        else:
            have_lu = False
        rhs(min(t_new, hi), y_new, p, r, u, f_new)
        if recompute_jac:
            _jacobian(rhs, jac, has_jac, min(t_new, hi), y_new, p, r, J, atol / rtol)
            current_jac = True
        else:
            current_jac = False
        h_abs_old = h_abs
        error_norm_old = error_norm
        while i < i1 and at[i] <= t_new:
            x = (at[i] - t_old) / h_old
            for j in range(n):
                ytmp[j] = y_old[j] + x * (Q[j, 0] + x * (Q[j, 1] + x * Q[j, 2]))
            _observe(out_fn, at[i], ytmp, p, r, o, atol)
            for q in range(n_out):
                pred[ai[i], q] = o[q]
            i += 1
        for j in range(n):
            y[j] = y_new[j]
            f[j] = f_new[j]
        _grow_tol(y, ymax, atol, dyn)
        for q in range(n_sw):
            G[0, q] = G[1, q]
        t = t_new
        h_abs = h_abs * factor
        if cut and t >= stop:
            return t, h_abs, i, _EVENT, nstep
    return t, h_abs, i, OK, nstep


_SS_CYCLES = 1000  # dosing cycles a numerical steady state may take


@njit(cache=True, error_model="numpy")
def _ss_exact(rhs, jac, t, p, r, y, target, bolus, rate, dur, ii, lag, W, nc):
    """Steady state of a linear system dosed every ii, solved exactly: the pre-input state y*
    is the fixed point of one cycle, (I - T) y* = c, then y is moved to the record, a phase after
    the last input of the series before it (the later ones are in the timeline). False when the
    cycle has no fixed point or the result is not finite. Only the first nc states take part:
    the hazard integrals after them keep their value."""
    n = y.size
    N = n + 1
    u = np.zeros(n)
    M = np.empty((N, N))
    L = np.empty((7, N, N))
    piv = np.empty(N, np.int64)
    col = np.empty(N)
    on = np.eye(N)
    off = np.eye(N)
    if rate > 0.0:
        u[target] = rate
        _linear_system(rhs, jac, t, p, r, u, W, M)
        if not _expm(M, dur, L, piv, col):
            return False
        on[:, :] = L[0]
        u[target] = 0.0
    _linear_system(rhs, jac, t, p, r, u, W, M)
    if not _expm(M, ii - dur, L, piv, col):
        return False
    off[:, :] = L[0]
    B = np.eye(N)
    B[target, n] = bolus
    tmp = np.empty((N, N))
    T = np.empty((N, N))
    _matmul(on, B, tmp)
    _matmul(off, tmp, T)
    # no equation reads a hazard integral: its column of I - T is zero, the core block is solved
    A = np.empty((nc, nc))
    c = np.empty(nc)
    for i in range(nc):
        for j in range(nc):
            A[i, j] = (1.0 if i == j else 0.0) - T[i, j]
        c[i] = T[i, n]
    if not _lu(A, piv[:nc]):
        return False
    _lu_solve(A, piv[:nc], c)
    z = np.empty(N)
    for i in range(n):
        z[i] = c[i] if i < nc else y[i]
    z[n] = 1.0
    phase = ii - _ss_shift(lag, ii)[1]
    if phase < ii:
        # state at the record time: phase after an input of the cycle
        z[target] += bolus
        first = min(dur, phase)
        for seg in range(2):
            span = first if seg == 0 else phase - first
            if span <= 0.0:
                continue
            u[target] = rate if seg == 0 else 0.0
            _linear_system(rhs, jac, t, p, r, u, W, M)
            if not _expm(M, span, L, piv, col):
                return False
            if not _transition(L[0], z[:n], col):
                return False
    for i in range(nc):
        y[i] = z[i]
        if not np.isfinite(y[i]):
            return False
    return True


@njit(cache=True, error_model="numpy")
def _advance(rhs, jac, has_jac, out_fn, autonomous, t, tend, y, p, r, u, rtol, tol, W, o,
             max_steps, swf, n_sw, G):
    """y moved from t to tend with no observation, explicit then implicit if stiff."""
    at = np.empty(1)
    ai = np.zeros(1, np.int64)
    pred = np.empty((1, o.size))
    h = -1.0
    implicit = False
    events = 0
    while True:
        nsw = n_sw if events < _MAX_EVENTS else 0
        if not implicit:
            t, h, _, status, ns = _dopri(rhs, out_fn, autonomous, t, tend, y, p, r, u, h, rtol,
                                         tol, at, ai, 0, 0, pred, W, o, max_steps, True, swf, nsw,
                                         G, W[17], 0.0)
            implicit = status == STIFF or status == MAX_STEPS
        if implicit:
            t, h, _, status, ns = _radau(rhs, jac, has_jac, out_fn, t, tend, y, p, r, u, h, rtol,
                                         tol, at, ai, 0, 0, pred, o, max_steps, swf, nsw, G,
                                         W[17], 0.0)
        if status != _EVENT:
            return status
        if t >= tend:
            return OK
        events += 1
        h = -1.0


@njit(cache=True, error_model="numpy")
def _ss_numeric(rhs, jac, has_jac, out_fn, autonomous, p, r, y, target, bolus, rate, dur, ii,
                lag, rtol, tol, W, o, max_steps, nc, swf, n_sw, G):
    """Steady state by repeated dosing cycles until the pre-input state no longer moves.
    Only the first nc states take part: the hazard integrals after them keep their value."""
    n = y.size
    u = np.zeros(n)
    prev = np.empty(n)
    # a hazard integral grows at every cycle: it is left out of the error and of the
    # convergence test, and gets its value back on every exit
    keep = y[nc:].copy()
    if nc < n:
        tol = tol.copy()
        for i in range(nc, n):
            tol[i] = np.inf
    settled = False
    last = 0.0
    for cycle in range(_SS_CYCLES):
        for i in range(n):
            prev[i] = y[i]
        y[target] += bolus
        for seg in range(2):
            span = dur if seg == 0 else ii - dur
            if span <= 0.0:
                continue
            u[target] = rate if seg == 0 else 0.0
            status = _advance(rhs, jac, has_jac, out_fn, autonomous, 0.0, span, y, p, r, u, rtol,
                              tol, W, o, max_steps, swf, n_sw, G)
            if status != OK:
                y[nc:] = keep
                return status
        # the distance left to the fixed point is the last change times rho / (1 - rho),
        # rho the contraction of a cycle estimated from two successive changes
        change = 0.0
        for i in range(nc):
            change = max(change, abs(y[i] - prev[i]) / (rtol * abs(y[i]) + tol[i]))
        rho = min(change / last, 0.999) if last > 0.0 else 0.0
        last = change
        if cycle > 0 and change * max(1.0, rho / (1.0 - rho)) <= 1.0:
            settled = True
            break
    if not settled:
        y[nc:] = keep
        return MAX_STEPS
    phase = ii - _ss_shift(lag, ii)[1]
    if phase < ii:
        y[target] += bolus
        first = min(dur, phase)
        for seg in range(2):
            span = first if seg == 0 else phase - first
            if span <= 0.0:
                continue
            u[target] = rate if seg == 0 else 0.0
            status = _advance(rhs, jac, has_jac, out_fn, autonomous, 0.0, span, y, p, r, u, rtol,
                              tol, W, o, max_steps, swf, n_sw, G)
            if status != OK:
                y[nc:] = keep
                return status
    y[nc:] = keep
    return OK


@njit(cache=True, error_model="numpy")
def _abs_tol(q, nc, ymax, floor, atol, dyn, rtol):
    # a hazard integral (q >= nc) counts events, floor 1e-3 * rtol: a hazard infinite at the origin
    # fails without it. 1e-30 when nothing sets a scale, as 1e-300 collapses the step. Only a state
    # without a value of its own borrows the floor, or a state at 18 beside one at 1e36 is unchecked
    if q < nc:
        return max(atol, dyn * (ymax[q] if ymax[q] > 0.0 else floor), 1e-30)
    return max(atol, dyn * ymax[q], 1e-3 * rtol)


@njit(cache=True, error_model="numpy")
def _solve_one(rhs, jac, has_jac, out_fn, init, brk, inp, structure, targets, prows, occ_t,
               t_start, gt, et, ei, ea, er, ef, ess, rt, rv, rtol, atol, dyn, max_steps, method,
               pred, W, r, o, d, s, b, at, ak, ai, av, ed, es, edur, swf, n_sw, G, n_hbrk):
    """Predictions of one subject on its time grid; returns (status, how it was solved, switches
    located, 1 if locating was given up).

    prows holds the parameters of each occasion, which starts at occ_t (the first at -inf).
    A linear system with constant coefficients is solved exactly unless a method is
    imposed. Otherwise the absolute tolerance of a state is atol or dyn times the
    largest value the state has reached, whichever is larger: accuracy then does not
    depend on units. The last nh states integrate the hazards (Model.lowered); the columns
    of the hazards themselves follow those of the outputs. The n_sw switching functions swf
    (Model.symbolic.switches) are located by the integrators, until one segment meets
    _MAX_EVENTS of them (chattering, sliding): the subject then goes on without locating.
    The last n_hbrk breakpoints read the signals of HISTORY: they are evaluated on each row of
    the signals and kept within that row and its occasion.
    """
    events = 0
    unlocated = 0
    locate = n_sw > 0
    n = W.shape[1]
    n_out = o.size
    nh = structure // _HAZARDS
    nc = n - nh
    # b holds the breakpoints, then their periods
    n_brk = b.size // 2
    t_last = gt[gt.size - 1] if gt.size else t_start
    n_inp = s.size
    tol = W[14]
    y = W[15]
    u = W[16]
    ymax = W[17]
    for k in range(prows.shape[0]):
        for q in range(prows.shape[1]):
            if not np.isfinite(prows[k, q]):
                return NOT_FINITE, 0, events, unlocated
    p = prows[0]
    for q in range(r.size):
        r[q] = rv[0, q]
    init(p, r, y)
    # delay, scale and duration of each event, with the parameters of its occasion and, when
    # they read them, the signals at its record
    k_inp = -1
    per_record = structure & _INPUT_HISTORY != 0
    row = 0
    for q in range(et.size):
        k = 0
        while k + 1 < occ_t.size and occ_t[k + 1] <= et[q]:
            k += 1
        if per_record:
            while row + 1 < rt.size and rt[row + 1] <= et[q]:
                row += 1
            for c in range(r.size):
                r[c] = rv[row, c]
        if k != k_inp or per_record:
            inp(prows[k], r, d, s)
            for c in range(d.size):
                if not np.isfinite(d[c]):
                    return NOT_FINITE, 0, events, unlocated
            k_inp = k
        ed[q], es[q], edur[q] = d[ei[q]], s[ei[q]], d[n_inp + ei[q]]
    if per_record:
        for q in range(r.size):
            r[q] = rv[0, q]
    for q in range(n):
        ymax[q] = abs(y[q])
    for q in range(et.size):
        # an input sets the scale of its target before the state has moved
        ymax[targets[ei[q]]] = max(ymax[targets[ei[q]]], abs(ea[q] * es[q]))
    # a state still at zero borrows a scale from the largest one, six decades down
    floor = 0.0
    for q in range(nc):
        floor = max(floor, 1e-6 * ymax[q])
    m = 0
    for q in range(1, rt.size):
        at[m], ak[m], ai[m], av[m] = rt[q], _REG, q, 0.0
        m += 1
    # an occasion switches the parameters before anything else at its start
    for k in range(1, occ_t.size):
        at[m], ak[m], ai[m], av[m] = occ_t[k], _OCC, k, 0.0
        m += 1
    # r holds the first row: the covariates, constant
    n_cbrk = n_brk - n_hbrk
    span = max(t_last - t_start, 0.0)
    for k in range(occ_t.size):
        brk(prows[k], r, b)
        for q in range(n_cbrk):
            if b[n_brk + q] == 0.0:
                at[m], ak[m], ai[m], av[m] = b[q], _BREAK, 0, 0.0
                m += 1
                continue
            # a periodic switch: every b + j P from the start of the solve to the last record;
            # b reduced modulo P (exact), at most the ceil(span / P) + 1 slots Problem reserves
            if not np.isfinite(b[q]):
                return NOT_FINITE, 0, events, unlocated
            P = b[n_brk + q]
            b0 = np.fmod(b[q], P)
            left = math.ceil(span / P) + 1
            j = math.ceil((t_start - b0) / P)
            while b0 + j * P <= t_last:
                if left == 0:
                    return NOT_FINITE, 0, events, unlocated
                at[m], ak[m], ai[m], av[m] = b0 + j * P, _BREAK, 0, 0.0
                m += 1
                j += 1
                left -= 1
    # slots Problem reserves for each breakpoint that reads HISTORY; its period is a number
    left = np.empty(n_hbrk, np.int64)
    for q in range(n_hbrk):
        P = b[n_brk + n_cbrk + q]
        left[q] = rt.size + occ_t.size + (0 if P == 0.0 else math.ceil(span / P) + 1)
    for k in range(occ_t.size if n_hbrk > 0 else 0):
        k1 = occ_t[k + 1] if k + 1 < occ_t.size else np.inf
        for row in range(rt.size):
            lo = max(occ_t[k], rt[row] if row > 0 else -np.inf)
            hi = min(k1, rt[row + 1] if row + 1 < rt.size else np.inf)
            if lo >= hi:
                continue
            for c in range(r.size):
                r[c] = rv[row, c]
            brk(prows[k], r, b)
            for q in range(n_cbrk, n_brk):
                P = b[n_brk + q]
                # not finite on a row (before any amount, say): no switch there
                if not np.isfinite(b[q]):
                    continue
                if P == 0.0:
                    if lo <= b[q] < hi:
                        if left[q - n_cbrk] == 0:
                            return NOT_FINITE, 0, events, unlocated
                        at[m], ak[m], ai[m], av[m] = b[q], _BREAK, 0, 0.0
                        m += 1
                        left[q - n_cbrk] -= 1
                    continue
                b0 = np.fmod(b[q], P)
                j = math.ceil((max(lo, t_start) - b0) / P)
                while b0 + j * P < hi and b0 + j * P <= t_last:
                    if left[q - n_cbrk] == 0:
                        return NOT_FINITE, 0, events, unlocated
                    at[m], ak[m], ai[m], av[m] = b0 + j * P, _BREAK, 0, 0.0
                    m += 1
                    j += 1
                    left[q - n_cbrk] -= 1
    for q in range(r.size if n_hbrk > 0 else 0):
        r[q] = rv[0, q]
    for q in range(gt.size):
        at[m], ak[m], ai[m], av[m] = gt[q], _OBS, q, 0.0
        m += 1
    # at one time the fill order holds: observations, resets, steady states, inputs, infusions
    # ending (an infusion too short to outlast rounding starts before it ends); an infusion entry
    # carries its code (_on_time) so that a reset can cancel those under way
    for q in range(et.size):
        if ef[q] != 0:
            at[m], ak[m], ai[m], av[m] = et[q], _RESET, q, 0.0
            m += 1
    for q in range(et.size):
        if ess[q] > 0.0:
            at[m], ak[m], ai[m], av[m] = et[q], _SS, q, 0.0
            m += 1
    first_input = m
    for q in range(et.size):
        at[m] = et[q] + ed[q]
        rq = _event_rate(ea, er, edur, q)
        if rq > 0.0:
            ak[m], ai[m], av[m] = _ON, q, rq * es[q]
        else:
            ak[m], ai[m], av[m] = _BOLUS, targets[ei[q]], ea[q] * es[q]
        m += 1
    ne = et.size
    for q in range(ne):
        if ess[q] == 0.0:
            continue
        # the series of a steady state goes on every II shifted by the lag: its inputs at or
        # after the record up to the last one (lag >= II), and the rest of an infusion under way
        ii = ess[q]
        nk, rem = _ss_shift(ed[q], ii)
        rq = _event_rate(ea, er, edur, q)
        j = 0
        while j < nk and et[q] + rem + j * ii <= t_last:
            if m >= at.size:
                return NOT_FINITE, 0, events, unlocated
            at[m] = _on_time(et, ed, ess, ne, q + (j + 2) * ne)
            if rq > 0.0:
                ak[m], ai[m], av[m] = _ON, q + (j + 2) * ne, rq * es[q]
            else:
                ak[m], ai[m], av[m] = _BOLUS, targets[ei[q]], ea[q] * es[q]
            m += 1
            j += 1
        if rq > 0.0 and ea[q] / rq > ii - rem:
            if m >= at.size:
                return NOT_FINITE, 0, events, unlocated
            at[m], ak[m], ai[m], av[m] = et[q], _ON, q + ne, rq * es[q]
            m += 1
    last_input = m
    for a in range(first_input, last_input):
        if ak[a] != _ON:
            continue
        if m >= at.size:
            return NOT_FINITE, 0, events, unlocated
        q, c = ai[a] % ne, ai[a] // ne
        rq = _event_rate(ea, er, edur, q)
        # the infusion under way at a steady-state record started a phase before it
        start = et[q] + _ss_shift(ed[q], ess[q])[1] - ess[q] if c == 1 else at[a]
        at[m], ak[m], ai[m], av[m] = start + ea[q] / rq, _OFF, ai[a], av[a]
        m += 1
    for q in range(m):
        if not np.isfinite(at[q]) or not np.isfinite(av[q]):
            return NOT_FINITE, 0, events, unlocated
    _sort_timeline(at, ak, ai, av, m)

    for q in range(n):
        u[q] = 0.0
        tol[q] = _abs_tol(q, nc, ymax, floor, atol, dyn, rtol)
    active = 0
    last_reset = -np.inf
    t = t_start
    h = -1.0
    implicit = method == 2
    used_implicit = _NUMERICAL
    exact = method == 0 and structure & _CLOSED_FORM != 0 and n > 0
    wide = False
    size = 0
    if exact:
        used_implicit = _EXACT
        # a constant column is needed for input rates, regressors and constant terms
        wide = rt.size > 1
        for q in range(et.size):
            wide = wide or _event_rate(ea, er, edur, q) > 0.0
        zero, none, c = W[0], W[1], W[3]
        for q in range(n):
            zero[q] = 0.0
            none[q] = 0.0
        # a constant term in any occasion
        for k in range(prows.shape[0]):
            rhs(t, zero, prows[k], r, none, c)
            for q in range(n):
                wide = wide or c[q] != 0.0
        size = n + 1 if wide else n
    M = np.empty((size, size))
    L = np.empty((7, size, size))
    piv = np.empty(size, np.int64)
    col = np.empty(size)
    stale = True
    held = np.nan  # step whose transition matrix L[0] holds
    # the closed form is kept while its rounding stays a hundredth of rtol
    most = int(math.log2(0.01 * rtol / 2.220446049250313e-16))
    steps_left = max_steps
    # a stiff system whose fast modes sit at equilibrium escapes the stiffness test and crawls
    # at the stability limit: a step count per segment catches it, whatever the number of events
    explicit_left = _EXPLICIT_STEPS if method == 0 else max_steps
    # nothing after the last observation changes a prediction: the solve ends there
    end = 0
    for q in range(m):
        if ak[q] == _OBS:
            end = q + 1
    i = 0
    while i < end:
        ta = at[i]
        if ta > t:
            if n == 0:
                t = ta
                continue
            j = i
            while j < end and ak[j] == _OBS:
                j += 1
            target = at[j] if j < end else at[end - 1]
            i1 = i
            while i1 < j and at[i1] < target:
                i1 += 1
            if exact:
                if stale:
                    _linear_system(rhs, jac, t, p, r, u, W, M)
                    stale = False
                    held = np.nan
                while True:
                    stop = at[i] if i < i1 else target
                    step = stop - t
                    moving = False
                    for q in range(n):
                        moving = moving or y[q] != 0.0 or (wide and M[q, n] != 0.0)
                    if step > 0.0 and moving:
                        if step != held and _squarings(M, step) > most:
                            # rates too far apart for that step: the integrators go on from t
                            exact = False
                            used_implicit = _NUMERICAL
                            break
                        if step != held:
                            if not _expm(M, step, L, piv, col):
                                return NOT_FINITE, used_implicit, events, unlocated
                            held = step
                        if not _transition(L[0], y, col):
                            return NOT_FINITE, used_implicit, events, unlocated
                    t = stop
                    if i == i1:
                        break
                    _observe(out_fn, stop, y, p, r, o, tol)
                    for q in range(n_out):
                        pred[ai[i], q] = o[q]
                    i += 1
                if exact:
                    continue
            status = _EVENT
            met = 0
            while status == _EVENT and t < target:
                nsw = n_sw if locate else 0
                status = OK
                if not implicit:
                    t, h, i, status, ns = _dopri(rhs, out_fn, structure & _AUTONOMOUS != 0, t,
                                                 target, y, p, r, u, h, rtol, tol, at, ai, i, i1,
                                                 pred, W, o, min(steps_left, explicit_left),
                                                 method == 0, swf, nsw, G, ymax, dyn)
                    steps_left -= ns
                    explicit_left -= ns
                    if status == STIFF or (status == MAX_STEPS and method == 0):
                        implicit = True
                        steps_left = max_steps
                        status = OK
                if implicit and t < target and status == OK:
                    used_implicit = _IMPLICIT
                    t, h, i, status, ns = _radau(rhs, jac, has_jac, out_fn, t, target, y, p, r, u,
                                                 h, rtol, tol, at, ai, i, i1, pred, o, steps_left,
                                                 swf, nsw, G, ymax, dyn)
                    steps_left -= ns
                if status == _EVENT:
                    # just past a switch: a fresh step on the new branch
                    events += 1
                    met += 1
                    h = -1.0
                    if met >= _MAX_EVENTS:
                        locate = False
                        unlocated = 1
            if status == _EVENT:
                status = OK
            if status != OK:
                return status, used_implicit, events, unlocated
            for q in range(n):
                ymax[q] = max(ymax[q], abs(y[q]))
                tol[q] = _abs_tol(q, nc, ymax, floor, atol, dyn, rtol)
            explicit_left = _EXPLICIT_STEPS if method == 0 else max_steps
            continue
        kind = ak[i]
        if kind == _OBS:
            _observe(out_fn, ta, y, p, r, o, tol)
            for q in range(n_out):
                pred[ai[i], q] = o[q]
        elif kind == _REG:
            for q in range(r.size):
                r[q] = rv[ai[i], q]
            if exact:
                stale = True
        elif kind == _BOLUS:
            y[ai[i]] += av[i]
            ymax[ai[i]] = max(ymax[ai[i]], abs(y[ai[i]]))
            tol[ai[i]] = _abs_tol(ai[i], nc, ymax, floor, atol, dyn, rtol)
        elif kind == _ON:
            u[targets[ei[ai[i] % et.size]]] += av[i]
            active += 1
        elif kind == _OFF:
            # an infusion started before the last reset was cancelled by it
            if _on_time(et, ed, ess, et.size, ai[i]) >= last_reset:
                active -= 1
                u[targets[ei[ai[i] % et.size]]] -= av[i]
                if active == 0:
                    for q in range(n):
                        u[q] = 0.0
        elif kind == _OCC:
            p = prows[ai[i]]
        elif kind == _RESET:
            # a reset concerns the system, not the risk already run: hazard integrals go on
            if nh > 0:
                keep = y[nc:].copy()
                init(p, r, y)
                y[nc:] = keep
            else:
                init(p, r, y)
            active = 0
            last_reset = ta
            for q in range(n):
                u[q] = 0.0
                ymax[q] = max(ymax[q], abs(y[q]))
                tol[q] = _abs_tol(q, nc, ymax, floor, atol, dyn, rtol)
        elif kind == _SS:
            # the reset entry of the same record came first; the input itself follows
            q = ai[i]
            rq = _event_rate(ea, er, edur, q)
            dur = ea[q] / rq if rq > 0.0 else 0.0
            if dur > ess[q]:
                return NOT_FINITE, used_implicit, events, unlocated
            bolus = 0.0 if rq > 0.0 else ea[q] * es[q]
            tgt = targets[ei[q]]
            if exact:
                if not _ss_exact(rhs, jac, ta, p, r, y, tgt, bolus, rq * es[q], dur, ess[q],
                                 ed[q], W, nc):
                    return NOT_FINITE, used_implicit, events, unlocated
            else:
                status = _ss_numeric(rhs, jac, has_jac, out_fn, structure & _AUTONOMOUS != 0, p, r,
                                     y, tgt, bolus, rq * es[q], dur, ess[q], ed[q], rtol,
                                     tol, W, o, max_steps, nc, swf, n_sw if locate else 0, G)
                if status != OK:
                    return status, used_implicit, events, unlocated
            for q in range(n):
                ymax[q] = max(ymax[q], abs(y[q]))
                tol[q] = _abs_tol(q, nc, ymax, floor, atol, dyn, rtol)
        if exact and (kind == _ON or kind == _OFF or kind == _RESET or kind == _SS
                      or kind == _OCC):
            stale = True
        if kind != _OBS:
            # the state or the right-hand side jumps: the next step is chosen afresh, as at the
            # start (a step carried over left records 1e-4 off at rtol 1e-6)
            h = -1.0
        i += 1
    # the hazards themselves may be infinite at the origin: the likelihood checks them where
    # it reads them
    for a in range(pred.shape[0]):
        for q in range(n_out - nh):
            if not np.isfinite(pred[a, q]):
                return NOT_FINITE, used_implicit, events, unlocated
    return OK, used_implicit, events, unlocated


@njit(cache=True, parallel=True, error_model="numpy")
def _predict(rhs, jac, has_jac, out_fn, init, brk, inp, structure, targets, n, n_out, n_brk,
             n_hbrk, swf, n_sw, t_start, grid_ptr, grid_time, ev_ptr, ev_time, ev_input, ev_amt,
             ev_rate, ev_reset, ev_ss, occ_ptr, occ_time, reg_ptr, reg_time, reg_value, rtol, atol,
             dyn, max_steps, method, mmax, n_chunks, pred, status, implicit, located, PSI):
    # PSI: one row of parameters per occasion of each subject. located: the switches found, and
    # whether the search gave up, by chain and subject
    C = PSI.shape[0]
    N = grid_ptr.size - 1
    total = C * N
    n_reg = reg_value.shape[1]
    n_inp = targets.size
    # one set of work buffers per chunk, so no allocation inside a solve
    for chunk in prange(n_chunks):
        W = np.empty((_NS, n))
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
        for q in range(chunk * total // n_chunks, (chunk + 1) * total // n_chunks):
            c = q // N
            i = q % N
            g0, g1 = grid_ptr[i], grid_ptr[i + 1]
            e0, e1 = ev_ptr[i], ev_ptr[i + 1]
            r0, r1 = reg_ptr[i], reg_ptr[i + 1]
            k0, k1 = occ_ptr[i], occ_ptr[i + 1]
            st, imp, nev, gave_up = _solve_one(
                rhs, jac, has_jac, out_fn, init, brk, inp, structure, targets, PSI[c, k0:k1],
                occ_time[k0:k1], t_start[i], grid_time[g0:g1], ev_time[e0:e1], ev_input[e0:e1], ev_amt[e0:e1],
                ev_rate[e0:e1], ev_reset[e0:e1], ev_ss[e0:e1], reg_time[r0:r1], reg_value[r0:r1],
                rtol, atol, dyn[i], max_steps, method, pred[c, g0:g1], W, r, o, d, s, b, at, ak, ai,
                av, ed, es, edur, swf, n_sw, G, n_hbrk)
            status[c, i] = st
            implicit[c, i] = imp
            located[c, i, 0] = nev
            located[c, i, 1] = gave_up
            if st != OK:
                for a in range(g0, g1):
                    for k in range(n_out):
                        pred[c, a, k] = np.nan


@njit(cache=True, error_model="numpy")
def _predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, use_old):
    if not use_old or error_norm_old <= 0.0 or h_abs_old <= 0.0 or error_norm == 0.0:
        multiplier = 1.0
    else:
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25
    if error_norm == 0.0:
        return 10.0
    return min(1.0, multiplier) * error_norm ** -0.25


_F, _V, _M = types.float64, types.float64[::1], types.float64[:, ::1]
_SIGNATURES = {
    "rhs": types.void(_F, _V, _V, _V, _V, _V),
    "jac": types.void(_F, _V, _V, _V, _M),
    "out": types.void(_F, _V, _V, _V, _V),
    "init": types.void(_V, _V, _V),
    "brk": types.void(_V, _V, _V),
    "inp": types.void(_V, _V, _V, _V),
    "swf": types.void(_F, _V, _V, _V, _V),
}
_COMPILED: dict = {}
_RECURSION = 10_000  # Python depth allowed while Numba compiles a model; the C stack holds it


@contextlib.contextmanager
def deep_recursion():
    """Python's recursion limit raised to _RECURSION, then restored: Numba's analysis of the
    control flow recurses once per block, and a kernel has a few blocks per where()."""
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, _RECURSION))
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def compile_model(model: Model, n_hazards: int = 0) -> tuple:
    """Leading arguments of the kernels for a model, compiled once per process; the last
    n_hazards states of the model integrate a hazard (Model.lowered).

    Model functions are C callbacks, so the shared kernels are typed on function
    pointers, compile once for every model and load from the on-disk cache.
    """
    key = (model.hash, n_hazards)
    if key not in _COMPILED:
        ns = {"np": np, "math": math, **model.kernel_globals(jit=True)}
        src = model.source()
        has_jac = "jac" in src
        src.setdefault("jac", "def jac(t, y, p, r, J):\n    pass\n")
        src.setdefault("swf", "def swf(t, y, p, r, g):\n    pass\n")
        for code in src.values():
            exec(code, ns)  # noqa: S102 - generated source, or wrappers of code checked by reader.py
        with deep_recursion():
            f = {name: cfunc(sig, error_model="numpy")(ns[name])
                 for name, sig in _SIGNATURES.items()}
        sym = model.symbolic
        structure = _AUTONOMOUS * sym.autonomous + _CLOSED_FORM * (
            sym.linear and sym.autonomous and has_jac) + _INPUT_HISTORY * (
            sym.history and reads_inputs_history(model)) + _HAZARDS * n_hazards
        _COMPILED[key] = (f["rhs"], f["jac"], has_jac, f["out"], f["init"], f["brk"], f["inp"],
                          int(structure), np.array(sym.targets, np.int64), model.n_states,
                          model.n_outputs, 2 * len(sym.breakpoints), history_breakpoints(model),
                          f["swf"], len(sym.switches))
    return _COMPILED[key]


def call(kernel, entries: dict, args: tuple):
    """Call a compiled kernel through its entry point for these argument types.

    The dispatcher re-types every argument at each call, which costs more than a
    small batch of solves when six of them are function pointers. entries caches
    the entry point per kernel; argument types are those of the first call.
    """
    entry = entries.get(kernel)
    if entry is None:
        out = kernel(*args)  # compiles, or loads from the on-disk cache
        try:
            entries[kernel] = kernel.get_overload(tuple(numba.typeof(a) for a in args))
        except KeyError:
            entries[kernel] = kernel
        return out
    return entry(*args)


def _signals(names, columns, t, values):
    """Columns of the signal vector of one subject: a regressor as read, or for an interpolated
    one its value at the record (_at_) and the slope to the next record (_slope_, 0 at the last)."""
    out = np.empty((len(t), len(names)))
    for k, r in enumerate(names):
        base = r.split("_", 2)[2] if r.startswith(("_at_", "_slope_")) else r
        x = values[:, columns.index(base)]
        if r.startswith("_slope_"):
            dt = np.diff(t)
            s = np.zeros(len(t))
            s[:-1] = np.where(dt > 0, np.diff(x) / np.where(dt > 0, dt, 1.0), 0.0)
            x = s
        out[:, k] = x
    return out


def _in_force(t, v, x, at, slope):
    """Signal rows in force at the times x, from rows v starting at t: the last row at or before
    each time, the first before any; an interpolated value (columns at) moved along its slope."""
    k = np.searchsorted(t, x, "right") - 1
    before = np.flatnonzero(k < 0)
    k = np.maximum(k, 0)
    out = v[k].copy()
    out[:, at] += out[:, slope] * np.maximum(x - t[k], 0.0)[:, None]
    # nothing moves before the first record
    out[np.ix_(before, slope)] = 0.0
    return out


def _delivered_events(model: Model, data: Dataset) -> tuple:
    """Event arrays (ptr, time, input, amount, rate, reset, ss) on the inputs the kernels see:
    a record of an input with several targets is repeated once per target, the reset kept on
    the first copy only."""
    d = data
    groups = [[j for j, x in enumerate(model.delivered) if x[0] == k] for k in range(model.n_inputs)]
    arrays = (d.ev_time, d.ev_input, d.ev_amt, d.ev_rate, d.ev_reset, d.ev_ss)
    if all(len(g) == 1 for g in groups) or not d.n_events:
        return (d.ev_ptr, *arrays)
    if any(len(groups[i]) > 1 and ss > 0 for i, ss in zip(d.ev_input, d.ev_ss)):
        raise ValueError("a steady state of an input with several targets is not supported")
    rows, ptr = [], [0]
    for i in range(d.n_subjects):
        for e in range(d.ev_ptr[i], d.ev_ptr[i + 1]):
            for c, j in enumerate(groups[d.ev_input[e]]):
                rows.append((d.ev_time[e], j, d.ev_amt[e], d.ev_rate[e],
                             d.ev_reset[e] if c == 0 else 0, d.ev_ss[e]))
        ptr.append(len(rows))
    cols = list(zip(*rows))
    return (np.asarray(ptr, np.int64), *(np.asarray(c, a.dtype) for c, a in zip(cols, arrays)))


class Problem:
    """A model bound to a dataset: predictions for batches of individual parameters."""

    def __init__(self, model: Model, data: Dataset, *, rtol=1e-6, atol=0.0, dynamic_range=1e-12,
                 method="auto", max_steps=20_000):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}")
        if data.n_outputs > model.n_outputs:
            raise ValueError(f"the data have {data.n_outputs} observation types "
                             f"{data.output_labels}, the model {model.n_outputs} output(s)")
        # a model without input sees the amounts only through the signals of HISTORY
        signals_only = data.n_events and not model.n_inputs and model.symbolic.history
        if data.n_events and not model.n_inputs and not signals_only:
            raise ValueError(f"the data contain events but the model declares no input and "
                             f"reads none of {list(HISTORY)}")
        if data.n_events and model.n_inputs and data.ev_input.max() >= model.n_inputs:
            raise ValueError("an event refers to an input the model does not declare")
        if (data.ev_ss > 0).any() and model.symbolic.history:
            raise ValueError(f"a steady state needs a model that does not read {list(HISTORY)}: "
                             "the amounts before the steady-state record are not in the data")
        if (data.ev_ss > 0).any() and not model.symbolic.autonomous:
            raise ValueError("a steady state needs a model whose equations do not depend on time")
        if (data.ev_ss > 0).any() and model.interpolated:
            raise ValueError(f"a steady state needs a model without interpolated regressor "
                             f"{model.interpolated}: their signal before the record is not "
                             "known to be periodic")
        missing = [r for r in model.regressors if r not in data.regressor_names]
        if missing:
            raise ValueError(f"regressor(s) {missing} not found in the data")
        if model.t0 is not None and data.n_events:
            # given at t0, an amount recorded before it would change the data
            amount = (data.ev_amt != 0) | (data.ev_ss > 0)
            early = np.flatnonzero((data.ev_time < model.t0) & amount)
            if early.size:
                e = early[0]
                who = data.ids[np.searchsorted(data.ev_ptr, e, "right") - 1]
                raise DataError(f"subject {who}: an amount recorded at {data.ev_time[e]:g}, before "
                                f"the start of the solve t0 = {model.t0:g}")
        self.model, self.data = model, data
        self._events = _delivered_events(model, data)
        if signals_only:
            self._events = (np.zeros(data.n_subjects + 1, np.int64),
                            *(a[:0] for a in self._events[1:]))
        self.rtol, self.atol, self.max_steps = float(rtol), float(atol), int(max_steps)
        # one value for all subjects, or one per subject (Engine sets it from the data)
        self.dyn = np.ascontiguousarray(np.broadcast_to(np.asarray(dynamic_range, float),
                                                        (data.n_subjects,)))
        self.dynamic_range = float(self.dyn.max()) if data.n_subjects else float(dynamic_range)
        self.method = _METHODS[method]
        self._lowered = model.lowered()
        self._kernel = compile_model(self._lowered, len(model.hazards))
        # columns of a prediction: the outputs, then the hazards themselves
        self.n_columns = self._lowered.n_outputs
        self._t_start = (np.full(data.n_subjects, model.t0) if model.t0 is not None
                         else data.t_first.astype(float))
        self._reg = self._regressors()
        n = data.n_subjects
        n_occ = np.diff(data.occ_ptr)
        sym = self._lowered.symbolic
        # a periodic breakpoint is met once per period between the start and the last record
        span = np.zeros(n)
        for i in range(n):
            g0, g1 = data.grid_ptr[i], data.grid_ptr[i + 1]
            if g1 > g0:
                span[i] = max(data.grid_time[g1 - 1] - self._t_start[i], 0.0)
        n_h = self._kernel[12]
        n_c = len(sym.periods) - n_h
        per_occ = 1 + sum(1 if q == 0 else np.ceil(span / q) + 1 for q in sym.periods[:n_c])
        # one that reads HISTORY is met at most once per row of the signals and occasion
        n_rows = np.diff(self._reg[0])
        history = sum(n_rows + n_occ + (0 if q == 0 else np.ceil(span / q) + 1)
                      for q in sym.periods[n_c:])
        # a steady state adds the inputs of its series from the record to the last one, and the
        # end of an infusion under way at the record: an ON and an OFF each
        ev_ptr, ev_t, ev_ss = self._events[0], self._events[1], self._events[6]
        series = np.zeros(n)
        for i in range(n):
            g0, g1 = data.grid_ptr[i], data.grid_ptr[i + 1]
            t_last = data.grid_time[g1 - 1] if g1 > g0 else self._t_start[i]
            e = slice(ev_ptr[i], ev_ptr[i + 1])
            ss = ev_ss[e] > 0
            if ss.any():
                ahead = np.maximum(t_last - ev_t[e][ss], 0.0) / ev_ss[e][ss]
                series[i] = (2 * (np.floor(ahead) + 2) + 2).sum()
        counts = (np.diff(data.grid_ptr) + 4 * np.diff(ev_ptr) + n_rows
                  + per_occ * n_occ + history + series)
        self._mmax = int(counts.max()) + 1 if n else 1
        self.last_stats = {"implicit": 0, "exact": 0, "failed": 0, "events": 0, "unlocated": 0}
        # every subject solved since the problem was built, by how it was solved; events counts
        # the switches on the states located, unlocated the solves that gave up locating them
        self.totals = {"solves": 0, "exact": 0, "implicit": 0, "failed": 0, "events": 0,
                       "unlocated": 0}
        self._entries = {}

    def with_model(self, model: Model, n_hazards: int) -> "Problem":
        """The same data and timeline solved with another model of the same inputs, already
        lowered (its sensitivities, sensitivity.py); solves count in the same totals."""
        other = copy.copy(self)
        other.model = other._lowered = model
        other._kernel = compile_model(model, n_hazards)
        other.n_columns = model.n_outputs
        other._entries = {}
        return other

    def _regressors(self):
        """Rows of the signal vector read by the model: regressors in model order, one row per
        change of value from the start of the solve (the first row holds there) and, for an
        interpolated regressor, one at each reset, followed by the subject's covariates and,
        when the model reads them, the signals of HISTORY. A model that reads a signal has at
        least one row per subject."""
        d, model = self.data, self._lowered
        missing = [c for c in model.covariates if c not in d.covariates]
        if missing:
            raise ValueError(f"covariate(s) {missing} not found in the data")
        cov = np.column_stack([d.covariates[c] for c in model.covariates]) \
            if model.covariates else np.zeros((d.n_subjects, 0))
        if not model.regressors and not model.covariates:
            rows = np.zeros(d.n_subjects + 1, np.int64), np.zeros(0), np.zeros((0, 0))
        elif not model.regressors:
            rows = np.arange(d.n_subjects + 1, dtype=np.int64), d.t_first.astype(float), cov
        else:
            absent = [d.ids[i] for i in range(d.n_subjects) if d.reg_ptr[i + 1] == d.reg_ptr[i]]
            if absent:
                raise DataError(f"subject(s) {shown(absent[:5])} have no record of the "
                                f"regressor(s) {list(self.model.regressors)}")
            interp = set(self.model.interpolated)
            cols = [d.regressor_names.index(r) for r in self.model.regressors]
            at = [k for k, r in enumerate(model.regressors) if r.startswith("_at_")]
            slope = [k for k, r in enumerate(model.regressors) if r.startswith("_slope_")]
            reset = (d.ev_reset != 0) | (d.ev_ss > 0)
            ptr, times, values = [0], [], []
            for i in range(d.n_subjects):
                a, b = d.reg_ptr[i], d.reg_ptr[i + 1]
                given = d.reg_given[a:b]
                # a row without a record of any regressor read is no record
                use = given[:, cols].any(axis=1)
                t, x = d.reg_time[a:b][use], d.reg_value[a:b][use].copy()
                given = given[use]
                for c, name in enumerate(d.regressor_names):
                    if name in interp and not given[:, c].all():
                        # a blank cell of an interpolated regressor lies on the line
                        x[~given[:, c], c] = np.interp(t[~given[:, c]], t[given[:, c]],
                                                       x[given[:, c], c])
                v = _signals(model.regressors, d.regressor_names, t, x)
                t0 = self._t_start[i]
                grid = [[t0], t[t > t0]]
                if interp:
                    e = d.ev_time[d.ev_ptr[i]:d.ev_ptr[i + 1]][reset[d.ev_ptr[i]:d.ev_ptr[i + 1]]]
                    grid.append(e[e > t0])
                tg = np.unique(np.concatenate(grid))
                v = _in_force(t, v, tg, at, slope)
                keep = np.ones(len(tg), bool)
                keep[1:] = (v[1:] != v[:-1]).any(axis=1)
                times.append(tg[keep])
                values.append(np.column_stack([v[keep], np.repeat(cov[i][None], keep.sum(), 0)]))
                ptr.append(ptr[-1] + int(keep.sum()))
            rows = (np.array(ptr, np.int64), np.concatenate(times),
                    np.ascontiguousarray(np.concatenate(values)))
        return self._with_history(*rows) if model.symbolic.history else rows

    def _with_history(self, ptr, times, values):
        """The signal rows with the columns of HISTORY after them, one more row at each amount
        record: EVID 1 and 4, additional and steady-state records included; a reset without
        amount (EVID 3) is not one. The first row holds before any record."""
        d = self.data
        width = len(self._lowered.regressors) + len(self._lowered.covariates)
        out_ptr, out_t, out_v = [0], [], []
        for i in range(d.n_subjects):
            t0 = self._t_start[i]
            a, b = ptr[i], ptr[i + 1]
            tb, vb = (times[a:b], values[a:b]) if b > a else (np.array([t0]), np.zeros((1, width)))
            e0, e1 = d.ev_ptr[i], d.ev_ptr[i + 1]
            counted = ~((d.ev_reset[e0:e1] != 0) & (d.ev_amt[e0:e1] == 0))
            te, ae = d.ev_time[e0:e1][counted], d.ev_amt[e0:e1][counted]
            x = np.unique(np.concatenate([tb[1:], te]))
            x = np.concatenate([[min(tb[0], x[0]) if x.size else tb[0]], x])
            base = vb[np.maximum(np.searchsorted(tb, x, "right") - 1, 0)]
            k = np.searchsorted(te, x, "right")
            k[0] = 0
            seen = k > 0
            last = np.maximum(k - 1, 0)
            hist = np.column_stack([np.where(seen, te[last] if te.size else t0, t0),
                                    np.where(seen, ae[last] if te.size else 0.0, 0.0),
                                    np.where(seen, te[0] if te.size else t0, t0), k])
            v = np.column_stack([base, hist])
            keep = np.ones(len(x), bool)
            keep[1:] = (v[1:] != v[:-1]).any(axis=1)
            out_t.append(x[keep])
            out_v.append(v[keep])
            out_ptr.append(out_ptr[-1] + int(keep.sum()))
        return (np.array(out_ptr, np.int64), np.concatenate(out_t),
                np.ascontiguousarray(np.concatenate(out_v)))

    def predict(self, psi):
        """Outputs on each subject's time grid for parameters psi (chains, occasion rows, n):
        one row per occasion of each subject (one per subject without occasions).

        Returns pred (chains, grid points, n_columns), NaN where the solve failed, and status
        (chains, subjects), 0 on success. The column of a hazard output holds its integral from
        the start of the solve; the hazards themselves follow the outputs.
        """
        psi = np.ascontiguousarray(psi, float)
        if psi.ndim == 2:
            psi = psi[None]
        d = self.data
        rows = len(d.occ_time)
        if psi.shape[1:] != (rows, self.model.n_parameters):
            raise ValueError(f"psi has shape {psi.shape}, expected (chains, "
                             f"{rows}, {self.model.n_parameters})")
        C = psi.shape[0]
        pred = self.empty(C)
        status = np.empty((C, d.n_subjects), np.int64)
        implicit = np.empty((C, d.n_subjects), np.int64)
        located = np.empty((C, d.n_subjects, 2), np.int64)
        call(_predict, self._entries, (*self.kernel_args(C), pred, status, implicit, located, psi))
        # a subject whose rows are not all finite is not solved: it is left out of the counts
        asked = np.logical_and.reduceat(np.isfinite(psi).all(-1), d.occ_ptr[:-1], axis=1)
        self.last_stats = {"implicit": int((implicit == _IMPLICIT).sum()),
                           "exact": int((implicit == _EXACT).sum()),
                           "failed": int(((status != 0) & asked).sum()),
                           "events": int(located[..., 0].sum()),
                           "unlocated": int(located[..., 1].sum())}
        self.totals["solves"] += int(asked.sum())
        for k, v in self.last_stats.items():
            self.totals[k] += v
        return pred, status

    def empty(self, n_chains: int) -> np.ndarray:
        """Predictions to be filled by the kernels: (chains, grid points, n_columns)."""
        # the kernels write their n_out columns without bound checks
        if self._kernel[10] != self.n_columns:
            raise RuntimeError(f"the kernels write {self._kernel[10]} columns, "
                               f"{self.n_columns} are allocated")
        return np.empty((n_chains, len(self.data.grid_time), self.n_columns))

    def kernel_args(self, n_chains: int) -> tuple:
        """Model and data arguments shared by every compiled kernel."""
        d = self.data
        rp, rt, rv = self._reg
        chunks = max(1, min(4 * numba.get_num_threads(), n_chains * d.n_subjects))
        return (*self._kernel, self._t_start, d.grid_ptr, d.grid_time, *self._events,
                d.occ_ptr, d.occ_time, rp, rt,
                rv, self.rtol, self.atol,
                self.dyn, self.max_steps, self.method, self._mmax, chunks)

    def gather(self, pred) -> np.ndarray:
        """Predictions aligned with the observation records, shape (chains, n_obs)."""
        return pred[:, self.data.obs_grid, self.data.obs_out]
