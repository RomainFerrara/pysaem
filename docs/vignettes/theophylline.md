---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting started: theophylline

This vignette fits a population model to a small classic dataset, step by step: the data, the
model, the fit, then reading and checking the result.

## The data

Twelve people each took one oral amount of theophylline, a drug against asthma, and the
concentration in their blood was then measured about ten times over 24 hours (Boeckmann, Sheiner
and Beal). The table has one row per record: the amount taken (`AMT`) at time 0, then one row per
measurement (`DV`). Time is in hours, the amount in mg per kg of body weight, the concentration in
mg/L.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from pysaem.datasets import theophylline

table = theophylline()
table.head(4)
```

```{code-cell} ipython3
obs = table.dropna(subset=["DV"])
fig, ax = plt.subplots(figsize=(6, 3.5))
for _, one in obs.groupby("ID"):
    ax.plot(one.TIME, one.DV, marker=".", lw=1, alpha=0.7)
ax.set(xlabel="time (h)", ylabel="concentration (mg/L)")
plt.show()
```

Every curve has the same shape: the concentration rises while the drug is absorbed, then falls as
the body eliminates it. The curves differ in height and in speed. A population model describes
both: one typical curve, and how far each person is from it.

## The model

Two quantities change over time: the amount still in the gut, `Ad`, and the amount in the blood,
`Ac`. The gut passes its content to the blood at the rate `ka`; the blood is cleared at the rate
`CL / V`, where `V` is the volume the drug spreads in and `CL` the volume cleared per hour. What is
measured is the concentration `Ac / V`.

The model is written as two Python functions: the derivatives of the states, and what is observed.

```{code-cell} ipython3
from pysaem import Dataset, Model, Population, fit

def rhs(t, y, p):
    gut, blood = y
    return [-p["ka"] * gut, p["ka"] * gut - p["CL"] / p["V"] * blood]

def observe(t, y, p):
    return y[1] / p["V"]

model = Model.from_python(
    rhs=rhs, observe=observe, states=["Ad", "Ac"], outputs=["Cc"],
    parameters=["ka", "V", "CL"], inputs={"dose": "Ad"},
    values={"ka": 1.0, "V": 0.5, "CL": 0.04})
print(model.equations())
```

PySAEM reads the functions into equations and prints them, so the model can be checked before
fitting. `inputs={"dose": "Ad"}` says that the amounts of the table enter the gut. `values` are
typical values around which the equations are compared with the code; they are not used by the
fit.

## The population

Each person has their own `ka`, `V` and `CL`, spread around a typical value. The spread is
log-normal: a parameter stays positive, and its standard deviation on the log scale, `omega`,
reads roughly as a coefficient of variation (0.3 means about 30 % between people). The measurements
scatter around each person's curve with an error that has a constant part `a` and a part `b`
proportional to the prediction.

```{code-cell} ipython3
data = Dataset.from_records(table)
population = Population.build(
    model, init={"ka": 1.0, "V": 0.5, "CL": 0.04}, omega=0.5, error="combined1")
```

The initial values only need the right order of magnitude.

## The fit

```{code-cell} ipython3
result = fit(model, data, population)
print(result.summary())
```

The typical person absorbs the drug at `ka_pop` per hour, spreads it in `V_pop` L/kg and clears
`CL_pop` L/h/kg. People differ most in absorption (`omega_ka`) and least in volume (`omega_V`).
`se` and `rse %` give the precision of each estimate. The last lines give the -2 log-likelihood
and the information criteria used to compare models, and a message from the fit.

## Checking the fit

Each person's curve, at their most likely parameters, against their measurements:

```{code-cell} ipython3
curves, pred = result.curves, result.predictions
fig, axes = plt.subplots(3, 4, figsize=(10, 6.5), sharex=True, sharey=True)
for ax, (i, one) in zip(axes.flat, curves.groupby("id")):
    seen = pred[pred.id == i]
    ax.plot(one.time, one.ind_pred_mode, color="C0")
    ax.plot(seen.time, seen.value, ".", color="k")
    ax.set_title(f"subject {i}", fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("time (h)")
for ax in axes[:, 0]:
    ax.set_ylabel("mg/L")
plt.tight_layout()
plt.show()
```

The visual predictive check simulates the study many times under the fitted model and compares
the spread of the simulated data (bands) with the spread of the real data (lines): the 10th,
50th and 90th percentiles over time. When doses, covariates or schedules differ between subjects,
`result.pcvpc` gives the prediction-corrected check (Bergstrand et al. 2011): each observation and
each simulated value is scaled by the population prediction of its time bin over that of its own
record, so the percentiles show the variability of the subjects, not the spread of their doses.

```{code-cell} ipython3
v = result.vpc
fig, ax = plt.subplots(figsize=(6, 3.5))
for q, colour in (("p10", "C0"), ("p50", "C3"), ("p90", "C0")):
    ax.fill_between(v.t_mid, v[f"sim_{q}_low"], v[f"sim_{q}_high"], color=colour, alpha=0.2, lw=0)
    ax.plot(v.t_mid, v[f"obs_{q}"], color=colour)
ax.set(xlabel="time (h)", ylabel="concentration (mg/L)")
plt.show()
```

The observed percentiles stay inside the simulated bands: the model reproduces both the typical
curve and how much people differ.

## Comparing two models

The summary notes that people with a large volume also tend to clear more. A second model
estimates this correlation between `V` and `CL`; the two models are compared on BICc, lower being
better.

```{code-cell} ipython3
correlated = fit(model, data, Population.build(
    model, init={"ka": 1.0, "V": 0.5, "CL": 0.04}, omega=0.5, error="combined1",
    correlations=[["V", "CL"]]))
print(f"BICc without correlation {result.bicc:.1f}, with {correlated.bicc:.1f}")
```

The model with the correlation has the lower BICc: the data support it.

`result` also holds the individual parameters (`result.individual`), the predictions and residuals
(`result.predictions`), the iterations of the algorithm (`result.trace`) and
`result.to_json()` to save everything.
