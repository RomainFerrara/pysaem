import io
import re

import pytest

import exact
import pysaem
from pysaem import Dataset, FitError, Model, Population, Settings
from pysaem.display import Progress

ONE_CMT = Model(states={"A": "0"}, parameters=["CL", "V"], odes={"A": "-CL/V*A"},
                aux={"C": "A/V"}, outputs=["C"], inputs={"dose": {"target": "A"}})
FAST = Settings(n_exploratory=60, n_smoothing=30, exploratory_autostop=False,
                smoothing_autostop=False, n_importance=200, seed=3)


def run(mode, **kw):
    df, _ = exact.simulate(n=10)
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, error="proportional")
    buf = io.StringIO()
    pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, kw.pop("settings", FAST),
               progress=Progress(buf, mode, 100), **kw)
    return buf.getvalue()


def screen(raw):
    rows, r = [""], 0
    for tok in re.split(r"(\x1b\[2K|\x1b\[1A|\r|\n)", raw):
        if tok == "\n":
            r += 1
            rows += [""] * (r + 1 - len(rows))
        elif tok == "\x1b[1A":
            r -= 1
        elif tok == "\x1b[2K":
            rows[r] = ""
        elif tok and tok != "\r":
            rows[r] += tok
    return [x for x in rows if x]


def test_plain_output_has_no_control_codes_and_every_step():
    out = run("plain")
    assert "\x1b" not in out and "\r" not in out
    for word in ("10 subjects", "SAEM", "90 iterations", "log-likelihood", "information", "done"):
        assert word in out


def test_terminal_output_ends_with_one_line_per_step():
    rows = screen(run("terminal"))
    assert rows[0].startswith("fit  10 subjects") and rows[-1].lstrip().startswith("done")
    assert any("90 iterations, 60 exploratory, 30 smoothing" in x for x in rows)
    assert not any("/90" in x for x in rows)


def test_hidden_when_not_a_terminal(capsys):
    df, _ = exact.simulate(n=10)
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, error="proportional")
    pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, FAST)
    assert capsys.readouterr().err == ""


def test_callback_still_cancels_and_the_display_ends_its_line():
    buf = io.StringIO()
    df, _ = exact.simulate(n=10)
    pop = Population.build(ONE_CMT, init={"CL": 1.0, "V": 5.0}, error="proportional")
    with pytest.raises(FitError, match="cancel"):
        pysaem.fit(ONE_CMT, Dataset.from_records(df), pop, FAST, progress=Progress(buf, "terminal", 100),
                   on_progress=lambda e: e["iteration"] < 20)
    assert buf.getvalue().endswith("\n")
