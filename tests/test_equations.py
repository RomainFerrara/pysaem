"""Models written as equations on paper, from small to large, as a person or an agent writes them."""

import math

import numpy as np
import pandas as pd
import pytest

from pysaem import Dataset, Model, ModelError
from pysaem.solve import Problem


def solve(model, times, dose=100.0, psi=()):
    rows = [{"ID": 1, "TIME": 0.0, "DV": ".", "AMT": dose}] if dose else []
    rows += [{"ID": 1, "TIME": t, "DV": 1.0, "AMT": "."} for t in times]
    prob = Problem(model, Dataset.from_records(pd.DataFrame(rows)))
    pred, status = prob.predict(np.atleast_2d(np.asarray(psi, float)))
    assert (status == 0).all()
    return prob.gather(pred)[0]


def test_fifty_transit_compartments_match_the_gamma_density():
    lines = ["d/dt(A1) = -k*A1"] + [f"d/dt(A{i}) = k*A{i-1} - k*A{i}" for i in range(2, 51)]
    m = Model.from_dict({"equations": "\n".join(lines), "dose": "A1", "output": "A50",
                         "parameters": ["k"]})
    t = np.array([5.0, 20.0, 49.0, 80.0])
    k = 1.0
    exact = [100 * (k * x) ** 49 / math.factorial(49) * math.exp(-k * x) for x in t]
    assert np.allclose(solve(m, t, psi=[k]), exact, rtol=1e-4)


def test_if_block_is_the_same_model_as_where():
    text = """
    if t < 1                       # the gallbladder empties
      ke = kge
    elif t < 2
      ke = kge/2
    else
      ke = 0
    end
    d/dt(Ag) = -ka*Ag + ke*Ab
    d/dt(Ac) = ka*Ag - CL/V*Ac - kb*Ac
    d/dt(Ab) = kb*Ac - ke*Ab
    Cc = Ac/V
    """
    m = Model.from_dict({"equations": text, "dose": "Ag", "output": "Cc",
                         "parameters": ["ka", "CL", "V", "kb", "kge"]})
    assert m.aux["ke"] == "where(t < 1, kge, where(t < 2, kge/2, (0)))"
    ref = Model.from_dict({"equations": text.replace("""    if t < 1                       # the gallbladder empties
      ke = kge
    elif t < 2
      ke = kge/2
    else
      ke = 0
    end""", "    ke = where(t < 1, kge, where(t < 2, kge/2, 0))"), "dose": "Ag", "output": "Cc",
                          "parameters": ["ka", "CL", "V", "kb", "kge"]})
    t = [0.5, 1.5, 2.5, 6.0, 24.0]
    psi = [1.0, 1.5, 20.0, 0.3, 5.0]
    assert np.allclose(solve(m, t, psi=psi), solve(ref, t, psi=psi), rtol=1e-8)


def test_stiff_bispecific_binding_and_a_five_organ_pbpk_solve():
    tmdd = """
    R1(0) = ksyn1/kdeg1
    R2(0) = ksyn2/kdeg2
    Cf = Af/V
    d/dt(Af) = -kel*Af - V*(kon1*Cf*R1 - koff1*B1) - V*(kon2*Cf*R2 - koff2*B2)
    d/dt(R1) = ksyn1 - kdeg1*R1 - kon1*Cf*R1 + koff1*B1 - kon1*B2*R1 + koff1*T
    d/dt(R2) = ksyn2 - kdeg2*R2 - kon2*Cf*R2 + koff2*B2 - kon2*B1*R2 + koff2*T
    d/dt(B1) = kon1*Cf*R1 - koff1*B1 - kon2*B1*R2 + koff2*T - kint*B1
    d/dt(B2) = kon2*Cf*R2 - koff2*B2 - kon1*B2*R1 + koff1*T - kint*B2
    d/dt(T) = kon2*B1*R2 + kon1*B2*R1 - (koff1 + koff2)*T - kint*T
    Ctot = Cf + B1 + B2 + T
    """
    m = Model.from_dict({"equations": tmdd, "dose": "Af", "output": "Ctot",
                         "parameters": ["V", "kel", "kon1", "koff1", "kon2", "koff2", "ksyn1",
                                        "kdeg1", "ksyn2", "kdeg2", "kint"]})
    f = solve(m, [0.1, 1, 24, 168, 504], dose=10.0,
              psi=[3, 0.01, 0.5, 0.05, 0.3, 0.1, 0.1, 0.05, 0.2, 0.1, 0.02])
    assert np.isfinite(f).all() and f[0] > f[-1] > 0

    pbpk = """
    Qli = 90; Qki = 70; Qmu = 45
    Vbl = 5
    Vli = 1.8
    Vki = 0.3
    Vmu = 30
    Cbl = Abl/Vbl
    Cli = Ali/Vli
    Cki = Aki/Vki
    Cmu = Amu/Vmu
    d/dt(Abl) = Qli*Cli/Kpli + Qki*Cki/Kpki + Qmu*Cmu/Kpmu - (Qli + Qki + Qmu)*Cbl
    d/dt(Ali) = Qli*(Cbl - Cli/Kpli) - CLint*fu*Cli/Kpli
    d/dt(Aki) = Qki*(Cbl - Cki/Kpki)
    d/dt(Amu) = Qmu*(Cbl - Cmu/Kpmu)
    """
    m = Model.from_dict({"equations": pbpk, "dose": "Abl", "output": "Cbl",
                         "parameters": ["CLint", "Kpli", "Kpki", "Kpmu", "fu"]})
    f = solve(m, [0.1, 1, 4, 24], psi=[60, 3, 2, 1, 0.1])
    assert np.all(np.diff(f) < 0)


def test_habits_of_people_and_language_models_are_read():
    m = Model.from_dict({"equations": "ddt_A = -k*A;\ndA2/dt = k*A - k2^1*A2\ny = ln(A2 + 1)",
                         "dose": "A", "output": "y", "parameters": ["k", "k2"]})
    assert list(m.states) == ["A", "A2"]
    with pytest.raises(ModelError, match="else branch"):
        Model.from_dict({"equations": "if t < 1\n x = 1\nend\nd/dt(A) = -x*A", "output": "A",
                         "parameters": []})
    with pytest.raises(ModelError, match="not assigned"):
        Model.from_dict({"equations": "if t < 1\n x = 1\nelse\n y = 2\nend\nd/dt(A) = -x*A",
                         "output": "A", "parameters": []})
