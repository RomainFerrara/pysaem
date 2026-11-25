---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Comparison with other packages

Six open-source packages with a SAEM method fitted the same data from the same starting values. The
comparison follows the one of saemix against nlme (Comets, Lavenu and Lavielle, *Journal of
Statistical Software* 2017, section 5): the same model fitted by every package on a real dataset,
then a simulation study in two scenarios, then the computation time. The other ten scenarios of the
benchmark are on the [next page](benchmark_details.md).

| Package | Language | Version | Settings |
|---|---|---|---|
| saemix | R | 3.4 | 1000 + 500 iterations |
| nlmixr2, SAEM method | R | nlmixr2est 4.0.2 | 1000 + 500 iterations |
| FeRx | Rust | 0.4.0 | its defaults, 150 + 250 iterations |
| NoLimits.jl | Julia | 0.3.0 | its defaults, at most 300 iterations |
| pynlme | Python | 0.5.1 | its defaults, 150 + 150 + 100 iterations |
| PySAEM | Python | 0.1.0 | its defaults, at most 1000 + 500 iterations stopped automatically, 2 to 10 chains per subject |

Iterations are counted as those of the two phases of SAEM, exploratory then smoothing. saemix and
nlmixr2 ran more iterations than their defaults. Each fit was done once, and the
-2 log-likelihood (-2LL) of every fit was computed by one common method at its estimates, so that
the packages are compared on the same number.

```{code-cell} ipython3
:tags: [remove-input]
import pandas as pd

TOOLS = ["saemix", "nlmixr2", "FeRx", "NoLimits.jl", "pynlme", "PySAEM"]
PARAMS = ["ka", "V", "CL", "omega_ka", "omega_V", "omega_CL"]
sim = pd.read_csv("benchmark_data/simulation_parameters.csv")
designs = pd.read_csv("benchmark_data/designs.csv",
                      dtype={"subjects": "Int64", "datasets": "Int64", "failed": "Int64", "near_best": "Int64",
                      "samples": str, "tv_bias": str, "sd_bias": str, "lowest_coverage": str})
theo = pd.read_csv("benchmark_data/theophylline_estimates.csv")


def seconds(x):
    return x if pd.isna(x) else f"{x:.2f}" if x < 0.1 else f"{x:.1f}"


def percent(x, sign=True):
    if abs(x) > 1000:
        return ">1000"
    return f"{x:+.1f}" if sign else f"{x:.1f}"


def scenario(name, design):
    one = sim[sim.scenario == name].copy()
    one["cell"] = [f"{percent(b)} ({percent(r, False)})" for b, r in zip(one.bias, one.rrmse)]
    table = one.pivot(index="parameter", columns="tool", values="cell").loc[PARAMS, TOOLS]
    runs = designs[designs.design == design].set_index("tool").loc[TOOLS]
    table.loc["failed runs"] = runs.failed.astype(str)
    table.loc["near the best fit"] = runs.near_best.astype(str)
    return table.rename_axis(index=None, columns=None)
```

## The same model on theophylline

The model of the [getting-started vignette](theophylline.md), with a constant error `a`, the form
that all six packages can write. Every package started from the same values, read from a fit that
ignores the differences between people.

```{code-cell} ipython3
:tags: [remove-input]
table = theo.pivot(index="parameter", columns="tool", values="estimate").loc[PARAMS + ["a", "-2LL"], TOOLS]
shown = table.map(lambda x: f"{x:#.3g}")
shown.loc["-2LL"] = table.loc["-2LL"].map(lambda x: f"{x:.2f}")
shown.rename_axis(index=None, columns=None)
```

The typical values agree within 4 % and the -2LL within 0.15 points.

## Simulation study

The data were simulated from the theophylline model: an amount absorbed from the gut, then cleared
from the blood. The true typical values are `ka` = 1 per hour, `V` = 50 L and `CL` = 5 L/h, with a
spread of 30 % between subjects on each (`omega` = 0.3) and a measurement error of 15 % of the
prediction. Each subject took one dose of 100 mg, and could be sampled at 0.5, 1, 2, 4, 8, 12 and
24 hours. Two scenarios, 200 datasets each:

- **Scenario 1, rich sampling**: 50 subjects, all 7 samples each.
- **Scenario 2, sparse sampling**: 100 subjects, 3 of the 7 samples each.

Every package started each fit at twice or half the true typical values, with `omega` at 0.5 and
the error at 0.3. pynlme sets its own starting spreads and error.

Each cell gives the relative bias of the estimate, its mean of (estimate - truth) / truth over the
200 datasets, and in brackets the relative root mean square error (RRMSE), both in %. With 200
datasets, a relative bias is known to about RRMSE / 14. The last two lines count the failed fits
and the fits whose -2LL ended within 2 points of the lowest among the six packages on that dataset.

**Scenario 1, rich sampling**

```{code-cell} ipython3
:tags: [remove-input]
scenario("rich", "one compartment, rich")
```

**Scenario 2, sparse sampling**

```{code-cell} ipython3
:tags: [remove-input]
scenario("sparse", "one compartment, sparse")
```

From this start, pynlme stopped far from the best fit in both scenarios: its error parameter grew
until it took up most of the variation in the data, and its typical `ka` stayed far below 1.
Started at the true values, it came within 0.7 points of PySAEM (median over the one-compartment
scenarios of the benchmark).

## Computation time

Median time of one fit, in seconds, over datasets 1 to 10, one fit at a time on an idle Apple M4
Pro with 14 cores. Each package used its own number of threads: PySAEM 14, saemix 1, nlmixr2 2,
FeRx 8, NoLimits.jl 10, pynlme 1. Each time is that of a complete fit with its -2LL and standard
errors, except nlmixr2 and FeRx, which compute no likelihood after estimation. The time of nlmixr2
includes the compilation of its model; the first fit of NoLimits.jl, which compiles, is left out.

```{code-cell} ipython3
:tags: [remove-input]
time = designs[designs.design.isin(["one compartment, rich", "one compartment, sparse"])]
time = time.pivot(index="design", columns="tool", values="seconds").loc[:, TOOLS]
time.index = ["Scenario 1, rich sampling", "Scenario 2, sparse sampling"]
time.rename_axis(index=None, columns=None).map(seconds)
```
