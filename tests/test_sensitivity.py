"""Exact derivatives of the predictions (sensitivity equations) against central differences, and
fits with exact derivatives against fits with finite differences."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import exact
import pysaem
from pysaem import Dataset, ErrorModel, Model, Population, Settings, saem
from pysaem.engine import Engine
from pysaem.post import modes
from pysaem.sensitivity import SensitivityModel, exact_parameters
from test_domains import CFDNA_TRUTH, cfdna_model
from test_hazards import DECAY, JOINT, WEIBULL, rows

# the structural models of the simulation study, named by content: spec, typical values,
# distributions other than lognormal
FAMILIES = {
    "oral one compartment": (
        {"parameters": ["ka", "V", "Cl"], "states": {"Ad": 0, "Ac": 0}, "aux": {"Cc": "Ac/V"},
         "odes": {"Ad": "-ka*Ad", "Ac": "ka*Ad - Cl/V*Ac"}, "outputs": ["Cc"],
         "inputs": {"oral": {"target": "Ad"}}, "t0": 0},
        {"ka": 1.0, "V": 50.0, "Cl": 5.0}, {}),
    "two compartments": (
        {"parameters": ["Cl", "V1", "Q", "V2"], "states": {"A1": 0, "A2": 0}, "aux": {"Cc": "A1/V1"},
         "odes": {"A1": "-Cl/V1*A1 - Q/V1*A1 + Q/V2*A2", "A2": "Q/V1*A1 - Q/V2*A2"},
         "outputs": ["Cc"], "inputs": {"iv": {"target": "A1"}}, "t0": 0},
        {"Cl": 10.0, "V1": 10.0, "Q": 5.0, "V2": 50.0}, {}),
    "saturable elimination": (
        {"parameters": ["V", "Vm", "Km"], "states": {"A": 0}, "aux": {"Cc": "A/V"},
         "odes": {"A": "-Vm*Cc/(Km + Cc)"}, "outputs": ["Cc"], "inputs": {"iv": {"target": "A"}},
         "t0": 0},
        {"V": 20.0, "Vm": 10.0, "Km": 2.0}, {}),
    "oral two compartments with a lag": (
        {"parameters": ["Tlag", "ka", "Cl", "V1", "Q", "V2"], "states": {"Ad": 0, "A1": 0, "A2": 0},
         "aux": {"Cc": "A1/V1"},
         "odes": {"Ad": "-ka*Ad", "A1": "ka*Ad - Cl/V1*A1 - Q/V1*A1 + Q/V2*A2",
                  "A2": "Q/V1*A1 - Q/V2*A2"},
         "outputs": ["Cc"], "inputs": {"oral": {"target": "Ad", "delay": "Tlag"}}, "t0": 0},
        {"Tlag": 0.5, "ka": 1.0, "Cl": 5.0, "V1": 20.0, "Q": 5.0, "V2": 50.0}, {}),
    "growth with a decaying kill rate": (
        {"parameters": ["TS0", "kg", "kd", "lambda"], "states": {"TS": "TS0"},
         "odes": {"TS": "kg*TS - kd*exp(-lambda*t)*TS"}, "outputs": ["TS"], "t0": 0},
        {"TS0": 50.0, "kg": 0.01, "kd": 0.03, "lambda": 0.02}, {}),
    "transit chain": (
        {"parameters": ["ktr", "V", "Cl"], "states": {"T1": 0, "T2": 0, "T3": 0, "Ac": 0},
         "aux": {"Cc": "Ac/V"},
         "odes": {"T1": "-ktr*T1", "T2": "ktr*T1 - ktr*T2", "T3": "ktr*T2 - ktr*T3",
                  "Ac": "ktr*T3 - Cl/V*Ac"},
         "outputs": ["Cc"], "inputs": {"oral": {"target": "T1"}}, "t0": 0},
        {"ktr": 2.0, "V": 20.0, "Cl": 2.0}, {}),
    "three compartments": (
        {"parameters": ["Cl", "V1", "Q2", "V2", "Q3", "V3"], "states": {"A1": 0, "A2": 0, "A3": 0},
         "aux": {"Cc": "A1/V1"},
         "odes": {"A1": "-Cl/V1*A1 - Q2/V1*A1 + Q2/V2*A2 - Q3/V1*A1 + Q3/V3*A3",
                  "A2": "Q2/V1*A1 - Q2/V2*A2", "A3": "Q3/V1*A1 - Q3/V3*A3"},
         "outputs": ["Cc"], "inputs": {"iv": {"target": "A1"}}, "t0": 0},
        {"Cl": 5.0, "V1": 10.0, "Q2": 10.0, "V2": 20.0, "Q3": 2.0, "V3": 100.0}, {}),
    "oral input and indirect response": (
        {"parameters": ["ka", "V", "Cl", "R0", "kout", "IC50", "Imax"],
         "states": {"Ad": 0, "Ac": 0, "R": "R0"}, "aux": {"Cc": "Ac/V"},
         "odes": {"Ad": "-ka*Ad", "Ac": "ka*Ad - Cl/V*Ac",
                  "R": "kout*R0*(1 - Imax*Cc/(IC50 + Cc)) - kout*R"},
         "outputs": ["Cc", "R"], "inputs": {"oral": {"target": "Ad"}}, "t0": 0},
        {"ka": 1.0, "V": 20.0, "Cl": 2.0, "R0": 100.0, "kout": 0.1, "IC50": 2.0, "Imax": 0.8},
        {"Imax": "logitnormal"}),
    "parent and metabolite": (
        {"parameters": ["ka", "Clp", "Vp", "Clm", "Vm"], "states": {"Ad": 0, "P": 0, "M": 0},
         "aux": {"Cp": "P/Vp", "Cm": "M/Vm"},
         "odes": {"Ad": "-ka*Ad", "P": "ka*Ad - Clp/Vp*P", "M": "Clp/Vp*P - Clm/Vm*M"},
         "outputs": ["Cp", "Cm"], "inputs": {"oral": {"target": "Ad"}}, "t0": 0},
        {"ka": 1.0, "Clp": 4.0, "Vp": 20.0, "Clm": 2.0, "Vm": 10.0}, {}),
    "oral three compartments": (
        {"parameters": ["ka", "Cl", "V1", "Q2", "V2", "Q3", "V3"],
         "states": {"Ad": 0, "A1": 0, "A2": 0, "A3": 0}, "aux": {"Cc": "A1/V1"},
         "odes": {"Ad": "-ka*Ad",
                  "A1": "ka*Ad - Cl/V1*A1 - Q2/V1*A1 + Q2/V2*A2 - Q3/V1*A1 + Q3/V3*A3",
                  "A2": "Q2/V1*A1 - Q2/V2*A2", "A3": "Q3/V1*A1 - Q3/V3*A3"},
         "outputs": ["Cc"], "inputs": {"oral": {"target": "Ad"}}, "t0": 0},
        {"ka": 1.5, "Cl": 5.0, "V1": 20.0, "Q2": 8.0, "V2": 40.0, "Q3": 1.0, "V3": 150.0}, {}),
    "oral saturable": (
        {"parameters": ["ka", "V", "Vm", "Km"], "states": {"Ad": 0, "A": 0}, "aux": {"Cc": "A/V"},
         "odes": {"Ad": "-ka*Ad", "A": "ka*Ad - Vm*Cc/(Km + Cc)"}, "outputs": ["Cc"],
         "inputs": {"oral": {"target": "Ad"}}, "t0": 0},
        {"ka": 1.0, "V": 20.0, "Vm": 8.0, "Km": 1.0}, {}),
    "two routes and a bioavailability": (
        {"parameters": ["F", "ka", "V", "Cl"], "states": {"Ad": 0, "Ac": 0}, "aux": {"Cc": "Ac/V"},
         "odes": {"Ad": "-ka*Ad", "Ac": "ka*Ad - Cl/V*Ac"}, "outputs": ["Cc"],
         "inputs": {"iv": {"target": "Ac"}, "oral": {"target": "Ad", "scale": "F"}}, "t0": 0},
        {"F": 0.6, "ka": 1.0, "V": 30.0, "Cl": 3.0}, {"F": "logitnormal"}),
    "target-mediated, quasi-steady state": (
        {"parameters": ["Cl", "V", "R0", "kint", "kdeg", "KSS"], "states": {"Atot": 0, "Rtot": "R0"},
         "aux": {"Ctot": "Atot/V",
                 "C": "0.5*((Ctot - Rtot - KSS) + sqrt((Ctot - Rtot - KSS)**2 + 4*KSS*Ctot))"},
         "odes": {"Atot": "-Cl*C - kint*V*(Ctot - C)",
                  "Rtot": "kdeg*R0 - kdeg*Rtot - (kint - kdeg)*(Ctot - C)"},
         "outputs": ["C"], "inputs": {"iv": {"target": "Atot"}}, "t0": 0},
        {"Cl": 0.2, "V": 3.0, "R0": 5.0, "kint": 0.05, "kdeg": 0.2, "KSS": 1.0}, {}),
    "zero-order input with a lag": (
        {"parameters": ["Tlag", "Tk0", "Cl", "V1", "Q", "V2"], "states": {"A1": 0, "A2": 0},
         "aux": {"Cc": "A1/V1"},
         "odes": {"A1": "-Cl/V1*A1 - Q/V1*A1 + Q/V2*A2", "A2": "Q/V1*A1 - Q/V2*A2"},
         "outputs": ["Cc"],
         "inputs": {"oral": {"target": "A1", "delay": "Tlag", "duration": "Tk0"}}, "t0": 0},
        {"Tlag": 0.5, "Tk0": 2.0, "Cl": 3.0, "V1": 15.0, "Q": 4.0, "V2": 40.0}, {}),
    "linear and saturable elimination": (
        {"parameters": ["V", "Cl", "Vm", "Km"], "states": {"A": 0}, "aux": {"Cc": "A/V"},
         "odes": {"A": "-Cl/V*A - Vm*Cc/(Km + Cc)"}, "outputs": ["Cc"],
         "inputs": {"iv": {"target": "A"}}, "t0": 0},
        {"V": 10.0, "Cl": 0.5, "Vm": 2.0, "Km": 2.0}, {}),
    "viral dynamics": (
        {"parameters": ["V0", "delta", "c", "eps"], "states": {"J": "c*V0", "V": "V0"},
         "aux": {"y": "log10(V)"},
         "odes": {"J": "(1 - eps)*c*delta*V - delta*J", "V": "J - c*V"}, "outputs": ["y"], "t0": 0},
        {"V0": 1e5, "delta": 0.5, "c": 5.0, "eps": 0.95}, {"eps": "logitnormal"}),
}
TIMES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 24.0, 36.0)


def records(model, n=3):
    """Every input at time 0, a second dose at 12, an infusion for subject 2; every output at
    TIMES."""
    out = []
    for i in range(1, n + 1):
        for m, _ in enumerate(model.inputs, 1):
            for t in (0.0, 12.0):
                out.append({"ID": i, "TIME": t, "DV": np.nan, "AMT": 100.0 * i, "ADM": m,
                            "RATE": 60.0 if i == 2 and m == 1 else np.nan, "YTYPE": np.nan})
        for o, _ in enumerate(model.outputs, 1):
            out += [{"ID": i, "TIME": t, "DV": 1.0, "AMT": np.nan, "ADM": np.nan, "RATE": np.nan,
                     "YTYPE": o} for t in TIMES]
    return pd.DataFrame(out).sort_values(["ID", "TIME"], kind="stable")


def dataset(model, df, **kw):
    kw.setdefault("output", "YTYPE")
    kw.setdefault("output_labels", list(range(1, model.n_outputs + 1)))
    if model.n_inputs:
        kw.setdefault("input", "ADM")
        kw.setdefault("input_labels", list(range(1, model.n_inputs + 1)))
    return Dataset.from_records(df, **kw)


def agreement(eng, idx=None, seed=0, h=1e-4):
    """Per column of idx: the largest error of the exact derivatives against central differences,
    relative to the largest derivative of that output (chain by chain). An output whose
    derivative stays below the noise of the differences, 10 rtol / h of its largest value, counts
    as having none, and the exact derivative must stay below that too. A derivative that is not a
    number is allowed on a hazard (Weibull in its shape at t = 0)."""
    rng = np.random.default_rng(seed)
    mean = eng.mean()
    spread = 0.3 * np.where(eng.tkind == 0, np.maximum(np.abs(mean), 1e-3), 1.0)
    phi = mean[None] + spread * rng.standard_normal((2, eng.N, eng.P))
    idx = list(range(eng.P)) if idx is None else idx
    pred, d = eng.derivatives(phi, idx)
    noise = 10 * eng.prob.rtol / h * np.abs(np.where(np.isfinite(pred), pred, 0.0)).max(1)
    errors = []
    for k, j in enumerate(idx):
        up, dn = phi.copy(), phi.copy()
        up[..., j] += h
        dn[..., j] -= h
        ref = (eng.predict(up)[0] - eng.predict(dn)[0]) / (2 * h)
        assert np.isfinite(ref).all()
        lost = ~np.isfinite(d[..., k])
        assert not lost[..., :eng.model.n_outputs].any()
        ref, dk = np.where(lost, 0.0, ref), np.where(lost, 0.0, d[..., k])
        size = np.abs(ref).max(1)
        flat = size <= noise
        assert (np.abs(dk).max(1)[flat] <= noise[flat]).all()
        err = np.abs(dk - ref).max(1) / np.where(flat, np.inf, size)
        errors.append(float(err.max()))
    return np.array(errors)


def engine(model, df, values, distributions=None, rtol=1e-10, method="auto", **kw):
    pop = Population.build(model, init=values, distribution=distributions or {}, omega=0.3,
                           error="proportional", **kw.pop("population", {}))
    return Engine(model, dataset(model, df, **kw), pop, rtol=rtol, method=method)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_exact_derivatives_of_each_study_family_agree_with_central_differences(name):
    spec, values, dist = FAMILIES[name]
    model = Model.from_dict(spec)
    eng = engine(model, records(model), values, dist)
    errors = agreement(eng)
    exact_ = np.array(exact_parameters(eng.prob._lowered))
    # exact columns: 1e-5 here, 2e-7 at worst in development; inputs keep forward differences
    assert (errors[exact_] < 1e-5).all(), dict(zip(model.parameters, errors))
    assert (errors[~exact_] < 2e-2).all(), dict(zip(model.parameters, errors))


def test_parameters_of_inputs_and_switching_times_keep_finite_differences():
    for name, moved in (("oral two compartments with a lag", ["Tlag"]),
                        ("two routes and a bioavailability", ["F"]),
                        ("zero-order input with a lag", ["Tlag", "Tk0"]),
                        ("oral one compartment", [])):
        model = Model.from_dict(FAMILIES[name][0])
        flags = exact_parameters(model.lowered())
        assert [p for p, f in zip(model.parameters, flags) if not f] == moved, name
    switch = Model(states={"A": "A0"}, parameters=["A0", "k", "t1"],
                   odes={"A": "where(t > t1, -k*A, 0)"}, outputs=["A"])
    assert exact_parameters(switch) == [True, True, False]
    on_states = Model(states={"A": "A0"}, parameters=["A0", "k"],
                      odes={"A": "where(A > 1, -k*A, -k)"}, outputs=["A"])
    assert exact_parameters(on_states) == [False, False]


def test_occasions_covariates_and_a_time_switch_agree():
    spec, values, _ = FAMILIES["oral one compartment"]
    model = Model.from_dict(spec | {"covariates": ["WT"],
                                    "odes": {"Ad": "-ka*Ad",
                                             "Ac": "ka*Ad - Cl*(WT/70)**0.75/V*Ac"
                                                   " - where(t > 6, 0.1, 0)*Ac"}})
    df = records(model)
    df["WT"] = df["ID"].map({1: 50.0, 2: 70.0, 3: 95.0})
    df["OCC"] = np.where(df["TIME"] < 12, 1, 2)
    eng = engine(model, df, values, covariates=["WT"],
                 population={"iov": {"ka": 0.2, "Cl": 0.2}})
    assert eng.P > eng.p0
    assert (agreement(eng) < 1e-5).all()


def test_steady_state_reset_and_interpolated_regressor_agree():
    model = Model(states={"A": "0", "E": "E0"}, parameters=["CL", "V", "E0", "kE"],
                  odes={"A": "-CL/V*A", "E": "kE*(E0 - E) + A/V"}, aux={"C": "A/V"},
                  outputs=["C", "E"], inputs={"dose": {"target": "A"}})
    out = []
    for i in (1, 2):
        out.append({"ID": i, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "SS": 1, "II": 12.0,
                    "EVID": 1, "RATE": 50.0 * (i - 1) or np.nan, "YTYPE": np.nan})
        out.append({"ID": i, "TIME": 30.0, "DV": np.nan, "AMT": 50.0, "SS": 0, "II": 0.0,
                    "EVID": 4, "RATE": np.nan, "YTYPE": np.nan})
        out += [{"ID": i, "TIME": t, "DV": 1.0, "AMT": np.nan, "SS": 0, "II": 0.0, "EVID": 0,
                 "RATE": np.nan, "YTYPE": o} for t in (1, 4, 11, 31, 40) for o in (1, 2)]
    df = pd.DataFrame(out).sort_values(["ID", "TIME"], kind="stable")
    values = {"CL": 2.0, "V": 10.0, "E0": 1.0, "kE": 0.3}
    for method in ("auto", "explicit"):
        # closed form, then the explicit integrator
        eng = engine(model, df, values, method=method)
        assert (agreement(eng) < 1e-5).all(), method
    signal = Model(states={"E": "E0"}, parameters=["E0", "kE"], odes={"E": "kE*(D - E)"},
                   outputs=["E"], regressors=["D"], interpolated=["D"])
    df = pd.DataFrame([{"ID": i, "TIME": t, "DV": 1.0, "D": 1.0 + i * t / 20, "YTYPE": 1}
                       for i in (1, 2) for t in (0.0, 2, 5, 9, 14)])
    eng = engine(signal, df, {"E0": 1.0, "kE": 0.3}, regressors=["D"])
    assert (agreement(eng) < 1e-5).all()


def test_domain_models_agree_including_a_stiff_one():
    model = cfdna_model()
    df = pd.DataFrame([{"ID": i, "TIME": t, "DV": 1.0, "YTYPE": o} for i in (1, 2)
                       for t in (-28.0, 0, 21, 63, 126) for o in (1, 2, 3)])
    eng = engine(model, df, CFDNA_TRUTH)
    assert (agreement(eng) < 1e-5).all()
    stiff = Model.from_dict({"equations": """
        R(0) = ksyn/kdeg
        Cf = Af/V
        d/dt(Af) = -kel*Af - V*(kon*Cf*R - koff*B)
        d/dt(R) = ksyn - kdeg*R - kon*Cf*R + koff*B
        d/dt(B) = kon*Cf*R - koff*B - kint*B
        Ctot = Cf + B
        """, "dose": "Af", "output": "Ctot",
        "parameters": ["V", "kel", "kon", "koff", "ksyn", "kdeg", "kint"]})
    df = pd.DataFrame([{"ID": 1, "TIME": 0.0, "DV": np.nan, "AMT": 10.0}]
                      + [{"ID": 1, "TIME": t, "DV": 1.0, "AMT": np.nan}
                         for t in (0.1, 1, 24, 168, 504)])
    eng = engine(stiff, df, dict(V=3, kel=0.01, kon=50, koff=0.05, ksyn=0.1, kdeg=0.05, kint=0.02),
                 output=None, output_labels=None)
    assert (agreement(eng, h=1e-5) < 1e-4).all()


@pytest.mark.parametrize("model, records_, values", [
    (JOINT, rows(1, "h", (0, 0), (6, 1)) + rows(1, "X", (0, 5.0), (3, 6.0))
     + rows(2, "h", (0, 0), (9, 0)) + rows(2, "X", (0, 4.0), (5, 6.5)),
     {"X0": 5.0, "g": 0.05, "h0": 0.02, "beta": 0.2}),
    (WEIBULL, rows(1, "S", (0, 0), (4, 1)) + rows(1, "R", (0, 0), (2, 1), (5, 2, 1), (8, 0))
     + rows(2, "S", (0, 0), (7, 0)) + rows(2, "R", (0, 0), (7, 1)),
     {"lam": 0.1, "kw": 1.5}),
    (DECAY, [{"ID": i, "TIME": 0.0, "DV": np.nan, "AMT": 100.0, "YTYPE": "C"} for i in (1, 2)]
     + rows(1, "h", (0, 0), (3, 1), (6, 2, 1)) + rows(1, "C", (1, 8.0), (4, 3.0))
     + rows(2, "h", (0, 0), (8, 0)) + rows(2, "C", (2, 5.0)),
     {"k": 0.3, "V": 10.0, "lam": 0.05, "beta": 0.1}),
])
def test_hazard_slopes_from_exact_derivatives_agree_with_central_differences(model, records_,
                                                                              values):
    df = pd.DataFrame(records_)
    pop = Population.build(model, init=values, distribution="normal", omega=0.3)
    data = Dataset.from_records(df, output="YTYPE", output_labels=model.outputs)
    fd, ex = (Engine(model, data, pop, rtol=1e-11, sensitivities=s) for s in ("fd", "exact"))
    assert (agreement(ex, h=1e-5) < 1e-5).all()
    phi = ex.mean()[None] + 0.2 * np.random.default_rng(1).standard_normal((2, ex.N, ex.P))
    idx = list(range(ex.P))
    _, f, D, _, raw = saem.sensitivities(ex, phi, idx)
    up, dn = (saem.sensitivities(fd, phi, idx, h=s)[4] for s in (1e-6, -1e-6))
    ref = (up + dn) / 2
    keep = np.isfinite(ref)
    assert np.allclose(raw[keep], ref[keep], rtol=1e-4, atol=1e-6 * np.abs(ref[keep]).max())


def test_log_scale_slopes_and_modes_agree_with_finite_differences():
    spec, values, dist = FAMILIES["oral saturable"]
    model = Model.from_dict(spec)
    df = records(model)
    pop = Population.build(model, init=values, omega=0.3, error="exponential")
    data = dataset(model, df)
    fd, ex = (Engine(model, data, pop, rtol=1e-11, sensitivities=s) for s in ("fd", "exact"))
    phi = ex.mean()[None] + 0.2 * np.random.default_rng(2).standard_normal((1, ex.N, ex.P))
    idx = list(range(ex.P))
    raw = saem.sensitivities(ex, phi, idx)[4]
    ref = (saem.sensitivities(fd, phi, idx, h=1e-6)[4] + saem.sensitivities(fd, phi, idx, h=-1e-6)[4]) / 2
    assert np.allclose(raw, ref, rtol=1e-5, atol=1e-8)
    start = np.zeros((ex.N, ex.P))
    assert np.allclose(modes(ex, start), modes(fd, start), atol=1e-4)


def test_the_sensitivity_model_keeps_the_hazards_last():
    aug = SensitivityModel(JOINT.lowered(), 1, (0, 3))
    # states X, dX/dX0, dX/dbeta is zero and left out, the hazard integral, its two derivatives
    assert aug.n_states == 1 + 1 + 1 + 2
    # outputs X, H; their derivatives; h; its derivatives
    assert aug.n_outputs == 2 + 2 * 2 + 1 + 2
    # dX/dX0 starts at the weight of its direction, the first parameter after the model's
    assert str(aug.symbolic.init[1]) == "_p4"
    assert aug.parameters[4:] == ["_w0", "_w1"]


def test_settings_choose_the_derivatives_and_refuse_others():
    assert Settings().sensitivities == "exact"
    model = Model.from_dict(FAMILIES["oral one compartment"][0])
    data = dataset(model, records(model))
    with pytest.raises(ValueError, match="sensitivities"):
        Engine(model, data, Population.build(model), sensitivities="symbolic")


ONE_CMT = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"},
                aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A"}})
QUICK = Settings(n_exploratory=200, n_smoothing=100, exploratory_autostop=False,
                 smoothing_autostop=False, seed=1, simulations=False, n_importance=2000,
                 n_information=500)


def _lag_case():
    """A lag (finite differences) next to exact parameters, two of them without variability."""
    spec, truth, _ = FAMILIES["oral two compartments with a lag"]
    model = Model.from_dict(spec)
    df = records(model, n=30)
    pop = Population.build(model, init=truth, omega=0.3, no_iiv=["Tlag", "Q"],
                           error=ErrorModel("proportional", b=0.15))
    eng = Engine(model, dataset(model, df), pop)
    rng = np.random.default_rng(3)
    phi = eng.mean() + 0.3 * rng.standard_normal((eng.N, eng.P)) * eng.iiv
    f = eng.prob.gather(eng.predict(phi[None])[0])[0]
    df.loc[df["DV"].notna(), "DV"] = f * (1 + 0.15 * rng.standard_normal(f.size))
    start = {k: v * (1.5 if i % 2 else 0.7) for i, (k, v) in enumerate(truth.items())}
    start["Tlag"] = 0.3
    return model, dataset(model, df), Population.build(
        model, init=start, omega=0.5, no_iiv=["Tlag", "Q"], error="proportional")


def _bolus_case():
    df, _ = exact.simulate()
    return ONE_CMT, Dataset.from_records(df), Population.build(
        ONE_CMT, init={"CL": 1.0, "V": 5.0}, omega=0.5, error="proportional")


@pytest.mark.parametrize("case", [_bolus_case, _lag_case], ids=["bolus", "oral with a lag"])
def test_fits_with_exact_and_finite_differences_agree_within_monte_carlo_error(case):
    model, data, pop = case()
    a = pysaem.fit(model, data, pop, replace(QUICK, sensitivities="fd"))
    b = pysaem.fit(model, data, pop, replace(QUICK, sensitivities="exact"))
    # over eight seeds of the case with a lag, the estimates moved by 0.02 to 0.19 standard
    # errors from seed to seed, by both methods alike (omega_V2: mean 0.2167 and 0.2174)
    for k, v in a.parameters.items():
        assert abs(b.parameters[k] - v) < 0.5 * a.se[k], (k, v, b.parameters[k], a.se[k])
    assert abs(a.ofv - b.ofv) < 0.5


def test_exact_derivatives_through_a_periodic_switch_and_a_split_input():
    # the sensitivity model keeps the periodic breakpoints and the inputs with several targets
    model = Model(states={"A": "0", "B": "0", "C": "0"}, parameters=["ka", "kb", "k", "f", "kin"],
                  odes={"A": "-ka*A", "B": "-kb*B", "C": "ka*A + kb*B - k*C + where(mod(t, 24) < 8, kin, 0)"},
                  outputs=["C"], inputs={"dose": {"target": ["A", "B"], "scale": ["f", "1 - f"]}})
    values = {"ka": 1.0, "kb": 0.2, "k": 0.1, "f": 0.6, "kin": 0.5}
    df = pd.DataFrame([{"ID": i, "TIME": 0.0, "DV": np.nan, "AMT": 100.0} for i in (1, 2)]
                      + [{"ID": i, "TIME": t, "DV": 1.0, "AMT": np.nan} for i in (1, 2) for t in (2, 7, 9, 30, 50)])
    pop = Population.build(model, init=values, distribution="normal", omega=0.3)
    data = Dataset.from_records(df)
    fd, ex = (Engine(model, data, pop, rtol=1e-11, sensitivities=s) for s in ("fd", "exact"))
    phi = ex.mean()[None] + 0.1 * np.random.default_rng(3).standard_normal((1, ex.N, ex.P))
    idx = list(range(ex.P))
    raw = saem.sensitivities(ex, phi, idx)[4]
    ref = (saem.sensitivities(fd, phi, idx, h=1e-6)[4] + saem.sensitivities(fd, phi, idx, h=-1e-6)[4]) / 2
    assert np.allclose(raw, ref, rtol=1e-4, atol=1e-7 * np.abs(ref).max())


@pytest.mark.parametrize("method", ["auto", "explicit"])
def test_exact_derivatives_through_a_switch_on_a_state_are_located(method):
    # zero order above A = 1, first order below; the kink is continuous, so the derivatives
    # are too and stay exact (exact_parameters), the switch located in the sensitivity solve
    from pysaem.solve import Problem
    m = Model(states={"A": "A0"}, parameters=["A0", "k"], odes={"A": "-k*min(A, 1)"}, outputs=["A"], t0=0.0)
    assert exact_parameters(m.lowered()) == [True, True]
    tt = np.array([0.5, 1.0, 1.5, 1.9, 2.1, 2.5, 3.0, 5.0, 8.0])
    data = Dataset.from_tables(pd.DataFrame({"id": 1, "time": tt, "value": 1.0}))
    A0, k = 3.0, 1.0
    ts = (A0 - 1) / k
    A = np.where(tt < ts, A0 - k * tt, np.exp(-k * (tt - ts)))
    exact_ = np.column_stack([np.where(tt < ts, 1.0, A), np.where(tt < ts, -tt, -tt * A)])
    prob = Problem(m, data, method=method)
    aug = SensitivityModel(prob._lowered, 0, (0, 1))
    sens = prob.with_model(aug, 0)
    pred, status = sens.predict(np.array([[A0, k, 1.0, 1.0]]))
    assert status.ravel().tolist() == [0] and sens.last_stats["events"] == 1
    d = aug.split(pred)[1][0, :, 0, :]
    # until 1 November 2026 the switch was crossed under error control: 2.8e-4 off
    assert np.abs(d - exact_).max() < 1e-5
    # central differences of the solution agree with the exact derivatives
    fine = Problem(m, data, rtol=1e-11)
    h = 1e-5
    cd = []
    for j in range(2):
        up, dn = np.array([[A0, k]]), np.array([[A0, k]])
        up[0, j] += h
        dn[0, j] -= h
        cd.append((fine.gather(fine.predict(up)[0]) - fine.gather(fine.predict(dn)[0]))[0] / (2 * h))
    np.testing.assert_allclose(np.column_stack(cd), exact_, atol=1e-6)
