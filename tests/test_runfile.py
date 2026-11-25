"""Run files: one TOML file describes a whole fit."""

from pathlib import Path

import pytest

from pysaem import cli, runfile
from pysaem.model import parse_equations

EXAMPLE = Path(__file__).parents[1] / "examples" / "theophylline.toml"


def test_example_run_file_builds_the_fit():
    run = runfile.load(EXAMPLE)
    assert run["model"].parameters == ["ka", "V", "CL"]
    assert run["data"].n_subjects == 12
    assert run["population"].parameters["CL"].covariates == {"WEIGHT": 0.0}
    assert run["population"].error["Cc"].kind == "combined1"
    assert run["settings"].seed == 1 and run["out"] == EXAMPLE.with_suffix(".json")


def test_run_file_fits_from_the_command_line(tmp_path):
    text = EXAMPLE.read_text().replace("seed = 1", "seed = 1\nn_exploratory = 60\nn_smoothing = 30\n"
                                       "fim = false\nn_importance = 500\nn_simulations = 50")
    (tmp_path / "run.toml").write_text(text)
    (tmp_path / "theophylline.csv").write_bytes((EXAMPLE.parent / "theophylline.csv").read_bytes())
    assert cli.main(["fit", str(tmp_path / "run.toml"), "--quiet"]) == 0
    assert (tmp_path / "run.json").exists()
    pytest.importorskip("matplotlib")
    assert cli.main(["plot", str(tmp_path / "run.json")]) == 0
    assert (tmp_path / "run.pdf").stat().st_size > 0


def test_unknown_keys_are_named(tmp_path):
    (tmp_path / "bad.toml").write_text(EXAMPLE.read_text().replace("omega = 0.5 }", "omga = 0.5 }", 1))
    with pytest.raises(ValueError, match="omga"):
        runfile.load(tmp_path / "bad.toml")


def test_equations_written_as_on_paper():
    m = parse_equations("""
        dA/dt = -k*A          # first form
        d/dt(B) = k*A - k2*B
        B(0) = B0
        y = B/V
    """)
    assert m == {"states": {"A": "0", "B": "B0"}, "odes": {"A": "-k*A", "B": "k*A - k2*B"},
                 "aux": {"y": "B/V"}}
    with pytest.raises(ValueError, match="line 2"):
        parse_equations("dA/dt = -A\nA' = 1")


def test_a_variable_defined_from_parameters_can_start_a_state():
    from pysaem import Model
    m = parse_equations("""
        R0 = kin/kout                  # baseline
        d/dt(R) = kin - kout*R
        R(0) = R0
    """)
    model = Model(**m, parameters=["kin", "kout"], outputs=["R"])
    assert str(model.symbolic.init[0]).count("/") == 1
    with pytest.raises(Exception, match="cannot be used here"):
        Model(**parse_equations("d/dt(R) = -R\nQ = 2*R\nR(0) = Q"), parameters=[], outputs=["R"])


PYTHON = '''
def rhs(t, y, p):
    return [-p['ka'] * y[0], p['ka'] * y[0] - p['CL'] / p['V'] * y[1]]

def observe(t, y, p):
    return y[1] / p['V']
'''


@pytest.mark.parametrize("given", [f'python = """{PYTHON}"""', 'python = "model.py"'])
def test_a_run_file_takes_a_model_written_in_python(tmp_path, given):
    head, rest = EXAMPLE.read_text().split("[model]")
    tail = rest.split("[population]")[1]
    model = (f'{given}\nstates = ["Ad", "Ac"]\ninputs = {{ dose = "Ad" }}\noutputs = ["Cc"]\n'
             'covariates = ["WEIGHT"]\n')
    (tmp_path / "run.toml").write_text(f"{head}[model]\n{model}\n[population]{tail}")
    (tmp_path / "model.py").write_text(PYTHON)
    (tmp_path / "theophylline.csv").write_bytes((EXAMPLE.parent / "theophylline.csv").read_bytes())
    run = runfile.load(tmp_path / "run.toml")
    assert run["model"].reading["form"] == "equations"
    assert run["model"].odes["Ad"] == "-Ad*ka" and run["model"].outputs == ["Cc"]


JOINT_CSV = """ID,TIME,DV,YTYPE,CENS
1,0,0,2,0
1,0,4.9,1,0
1,2,6.0,1,0
1,4,7.2,1,0
1,5.3,1,2,0
3,0,0,2,0
3,0,4.4,1,0
3,4,6.4,1,0
3,4,0,2,0
3,8,1,2,1
"""

JOINT_RUN = '''[data]
file = "joint.csv"
output = "YTYPE"

[model]
equations = """
X(0) = X0
d/dt(X) = g*X
h = h0*exp(beta*X)
"""
outputs = ["X", "h"]
hazards = { h = "terminal" }

[population]
X0 = { init = 5.0, omega = 0.3 }
g = { init = 0.1, omega = 0.3 }
h0 = { init = 0.01, iiv = false }
beta = { init = 0.1, iiv = false, distribution = "normal" }
'''


def test_a_run_file_gives_hazard_outputs_their_error_model(tmp_path):
    (tmp_path / "joint.csv").write_text(JOINT_CSV)
    for error, kind in (('{ X = "proportional" }', "proportional"), ('"constant"', "constant"),
                        ('{ kind = "combined2", a = 0.5, b = 0.1 }', "combined2"),
                        ('{ X = "constant", h = "hazard" }', "constant")):
        (tmp_path / "run.toml").write_text(f"{JOINT_RUN}error = {error}\n")
        run = runfile.load(tmp_path / "run.toml")
        assert run["model"].hazards == {"h": "terminal"} and run["data"].n_obs == 10
        assert run["population"].error["X"].kind == kind
        assert run["population"].error["h"].kind == "hazard"
    (tmp_path / "run.toml").write_text(f'{JOINT_RUN}error = {{ h = "proportional" }}\n')
    with pytest.raises(ValueError, match="output h is a hazard output"):
        runfile.load(tmp_path / "run.toml")
