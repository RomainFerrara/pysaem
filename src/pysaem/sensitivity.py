"""Exact derivatives of the predictions in the parameters by the forward sensitivity equations
(Hairer, Norsett and Wanner 1993, section I.14; Hindmarsh et al. 2005).

For a direction v, S = dy/dv solves dS/dt = J_y S + J_p v, S(t0) = (dy0/dp) v, and the
derivative of an output g is g_y S + g_p v. SymPy writes these equations from the symbolic form
of the model; the system and its sensitivities are solved together by the same kernels, so a
linear system stays linear and keeps its closed form, whose exponential then holds the
derivative of the exponential (Van Loan 1978).
"""

from __future__ import annotations

import hashlib

import numpy as np
import sympy as sp
from sympy.core.relational import Relational

from .model import T, Model, Symbolic

# functions whose value jumps: a parameter inside one with t moves the jump
_STEPS = (sp.floor, sp.ceiling, sp.Mod)


def _sym(prefix: str, i: int) -> sp.Symbol:
    return sp.Symbol(f"{prefix}{i}", real=True)


def exact_parameters(model: Model) -> list:
    """Per parameter of the model (lowered), whether its sensitivities are exact.

    Not exact: a parameter of an input (delay, scale, duration) or of a switching time, whose
    derivative needs the jump of the solution at the moved event; every parameter of a model
    whose equations switch on the states, or written as code compiled as is.
    """
    n = model.n_parameters
    if not getattr(model, "exact_derivatives", True):
        return [False] * n
    sym = model.symbolic
    if sym.jac is None:
        return [False] * n
    p = [_sym("_p", i) for i in range(n)]
    y = {_sym("_y", i) for i in range(model.n_states)}
    moved = set()
    for e in (*sym.delays, *sym.scales, *sym.durations, *sym.breakpoints):
        moved |= sp.sympify(e).free_symbols
    for e in (*sym.rhs, *sym.out):
        e = sp.sympify(e)
        for a in (*e.atoms(Relational), *(x for f in _STEPS for x in e.atoms(f))):
            fs = a.free_symbols
            if fs & y:
                return [False] * n
            if T in fs:
                moved |= fs
    return [s not in moved for s in p]


def _reach(sym: Symbolic, n: int, n_par: int) -> list:
    """Per state, the parameters it depends on through its equation, its initial value and the
    states its equation reads."""
    y = [_sym("_y", i) for i in range(n)]
    p = {_sym("_p", i) for i in range(n_par)}
    direct = [(sp.sympify(sym.rhs[i]).free_symbols | sp.sympify(sym.init[i]).free_symbols)
              for i in range(n)]
    reach = [d & p for d in direct]
    reads = [[k for k in range(n) if y[k] in direct[i]] for i in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            new = reach[i].union(*(reach[k] for k in reads[i]))
            if new != reach[i]:
                reach[i], changed = new, True
    return reach


class SensitivityModel(Model):
    """A lowered model with its forward sensitivities in the directions given: direction c
    differentiates in parameter directions[c], times a weight that follows the parameters
    (parameter n_parameters + c), the derivative of that parameter in what is differentiated.

    States: those before the hazard integrals, their sensitivities that are not zero, the
    hazard integrals, all their sensitivities. Columns: the outputs, their sensitivities
    (direction by direction), the hazards, their sensitivities, so that the solver still finds
    the hazard integrals and the hazards last (n_hazards of them).
    """

    exact_derivatives = False  # derivatives of derivatives are not needed

    def __init__(self, base: Model, n_hazards: int, directions):
        self.base, self.directions = base, tuple(int(a) for a in directions)
        sym = base.symbolic
        n, n_par, q = base.n_states, base.n_parameters, len(self.directions)
        nc = n - n_hazards
        y = [_sym("_y", i) for i in range(n)]
        p = [_sym("_p", i) for i in range(n_par)]
        w = [_sym("_p", n_par + c) for c in range(q)]
        reach = _reach(sym, n, n_par)
        # the slot of each state and sensitivity in the new state vector; a sensitivity a parameter
        # does not reach is zero and has none
        slot = {}
        for i in range(nc):
            slot[i] = i
        for c, a in enumerate(self.directions):
            for i in range(nc):
                if p[a] in reach[i]:
                    slot[(c, i)] = len(slot)
        for i in range(nc, n):
            slot[i] = len(slot)
        for c in range(q):
            for i in range(nc, n):
                slot[(c, i)] = len(slot)
        # the sensitivities are written with stand-ins, renamed with the states in one pass
        Y = {key: y[key] if isinstance(key, int) else sp.Dummy() for key in slot}
        rename = {Y[key]: _sym("_y", k) for key, k in slot.items()}

        def S(c, k):
            return Y.get((c, k), sp.S.Zero)

        def total(e, c, a):
            """Derivative of e in direction c, through the states and directly."""
            e = sp.sympify(e).rewrite(sp.Piecewise)
            d = sp.diff(e, p[a]) * w[c]
            for k in range(n):
                if e.has(y[k]) and S(c, k) != 0:
                    d += sp.diff(e, y[k]) * S(c, k)
            return d

        J = sym.jac
        order = sorted(slot, key=slot.get)
        rhs, init = [], []
        for key in order:
            if isinstance(key, int):
                rhs.append(sym.rhs[key])
                init.append(sym.init[key])
                continue
            c, i = key
            a = self.directions[c]
            d = sp.diff(sp.sympify(sym.rhs[i]).rewrite(sp.Piecewise), p[a]) * w[c]
            for k in range(n):
                if J[i, k] != 0 and S(c, k) != 0:
                    d += J[i, k] * S(c, k)
            rhs.append(d)
            init.append(sp.diff(sp.sympify(sym.init[i]), p[a]) * w[c])
        n_col = len(sym.out)
        n_o = n_col - n_hazards
        outs = list(sym.out[:n_o])
        outs += [total(sym.out[o], c, a) for c, a in enumerate(self.directions) for o in range(n_o)]
        outs += list(sym.out[n_o:])
        outs += [total(sym.out[o], c, a) for c, a in enumerate(self.directions)
                 for o in range(n_o, n_col)]
        rhs = [sp.sympify(e).xreplace(rename) for e in rhs]
        init = [sp.sympify(e).xreplace(rename) for e in init]
        outs = [sp.sympify(e).xreplace(rename) for e in outs]
        states = [rename[Y[key]] for key in order]
        jac = None
        if sym.linear:
            jac = sp.Matrix(rhs).jacobian(states) if states else sp.Matrix(0, 0, [])

        self.n_base_states, self.n_hazards_base, self.n_base_outputs = n, n_hazards, n_o
        self.name = base.name
        self.states = {s.name: "0" for s in states}
        self.parameters = list(base.parameters) + [f"_w{c}" for c in range(q)]
        self.outputs = [f"_o{k}" for k in range(len(outs))]
        self.inputs = base.inputs
        self.aux, self.regressors, self.covariates = {}, list(base.regressors), list(base.covariates)
        self.interpolated, self.t0, self.hazards = [], base.t0, {}
        self.odes = {}
        self.__dict__["symbolic"] = Symbolic(
            rhs, outs, init, list(sym.targets), list(sym.delays), list(sym.scales),
            list(sym.durations), list(sym.breakpoints), jac, sym.linear, sym.autonomous,
            list(sym.periods), [sp.sympify(g).xreplace(rename) for g in sym.switches],
            history=sym.history)

    @property
    def hash(self) -> str:
        key = f"{self.base.hash}/{self.n_hazards_base}/{self.directions}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def lowered(self) -> "Model":
        return self

    def split(self, pred):
        """Predictions of the model (chains, grid, outputs then hazards) and their derivatives
        (chains, grid, columns, directions) from the columns of this one."""
        q, n_o, n_h = len(self.directions), self.n_base_outputs, self.n_hazards_base
        C, G = pred.shape[:2]
        base = pred[..., :n_o]
        d = pred[..., n_o:n_o * (1 + q)].reshape(C, G, q, n_o)
        at = n_o * (1 + q)
        if n_h:
            base = pred[..., list(range(n_o)) + list(range(at, at + n_h))]
            dh = pred[..., at + n_h:].reshape(C, G, q, n_h)
            d = np.concatenate([d, dh], axis=-1)
        return base, d.transpose(0, 1, 3, 2)

