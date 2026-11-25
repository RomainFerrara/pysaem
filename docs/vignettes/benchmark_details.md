---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Comparison: all scenarios

The same six packages, settings and starting rule as on the [previous page](benchmark.md), on the
twelve scenarios of the benchmark, 200 simulated datasets each. Every scenario is a model of how
an amount given to a subject, or a tumour under treatment, changes over time:

- **one compartment, rich and sparse**: scenarios 1 and 2, the theophylline model.
- **high variability**: the same model with a spread of 80 % between subjects, 20 or 100
  subjects, 8 samples each or 3 of the 8.
- **two compartments, lag, three doses**: the amount also moves to and from a second space,
  absorption starts after a delay, and each subject takes three doses, at one of three levels.
- **three compartments**: two such spaces, changed from its first version so that every parameter
  can be estimated.
- **intravenous and oral doses**: each subject gets one injection and one oral dose; the model
  estimates the fraction of the oral dose that reaches the blood.
- **zero-order input after a lag**: the amount enters at a constant rate for a while, after a delay.
- **linear and saturable elimination**: part of the clearance saturates at high concentrations.
- **tumour growth inhibition**: a tumour size that grows, and shrinks under a treatment whose
  effect fades.

`--`: the package cannot write this model.

```{code-cell} ipython3
:tags: [remove-input]
import pandas as pd

TOOLS = ["saemix", "nlmixr2", "FeRx", "NoLimits.jl", "pynlme", "PySAEM"]
designs = pd.read_csv("benchmark_data/designs.csv",
                      dtype={"subjects": "Int64", "datasets": "Int64", "failed": "Int64", "near_best": "Int64",
                      "samples": str, "tv_bias": str, "sd_bias": str, "lowest_coverage": str})
order = list(dict.fromkeys(designs.design))
size = designs.drop_duplicates("design").set_index("design")[["subjects", "samples"]].loc[order]


def by_design(values):
    table = designs.pivot(index="design", columns="tool", values=values).loc[order, TOOLS]
    return table.rename_axis(index=None, columns=None)


def seconds(x):
    return x if pd.isna(x) else f"{x:.2f}" if x < 0.1 else f"{x:.1f}"


def show(table):
    return table.map(lambda x: "--" if pd.isna(x) or x == "--" else x)
```

## Near the best fit

Number of the 200 datasets on which the package ended within 2 points of the lowest -2LL among
the six packages; a failed fit counts as farther. The two high-variability scenarios with 100
subjects have no likelihood method precise enough to compare the packages and are left out. No
fit failed in any scenario.

```{code-cell} ipython3
:tags: [remove-input]
near = by_design("near_best").dropna(how="all")
near = near.map(lambda x: x if pd.isna(x) else f"{int(x)}")
show(pd.concat([size.loc[near.index].astype(str), near], axis=1))
```

## Bias and coverage

Each cell gives three numbers: the largest relative bias (%) among the typical values, the same
among the spreads between subjects, and the lowest coverage (%) of the 97.5 % Wald intervals of
the typical values, built from each package's standard errors (PySAEM itself reports 95 %
intervals). A missing standard error counts as not covered. A good package has small biases and
coverages near 97.5.

```{code-cell} ipython3
:tags: [remove-input]
cells = designs.assign(cell=[
    "--" if tv == "--" else f"{tv} / {sd} / {cov}"
    for tv, sd, cov in zip(designs.tv_bias, designs.sd_bias, designs.lowest_coverage)])
table = cells.pivot(index="design", columns="tool", values="cell").loc[order, TOOLS]
table.rename_axis(index=None, columns=None)
```

## Computation time

Median time of one fit in seconds, measured as on the previous page. The scenario with linear and
saturable elimination was not timed.

```{code-cell} ipython3
:tags: [remove-input]
time = by_design("seconds").dropna(how="all")
show(time.map(seconds))
```
