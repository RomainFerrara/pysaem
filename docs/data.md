# Data

`Dataset.from_records` reads a table with one row per record, observations and events mixed, the
layout used by most population modelling tools. Column names are given as keyword arguments when
they differ from the defaults below. Empty cells, `.` and `NA` are missing values.

| Column | Default name | Meaning |
|---|---|---|
| subject | `ID` | identifier of the subject |
| time | `TIME` | time of the record |
| value | `DV` | observed value |
| amount | `AMT` | amount given at this time |
| rate, duration | `RATE`, `TINF` | an amount given over time, by its rate or its duration |
| event flag | `EVID` | 0 observation, 1 amount, 3 reset of the states, 4 reset then amount |
| missing flag | `MDV` | 1: the value of this row is not an observation |
| censoring | `CENS` | 1: the true value is below `DV`; -1: above it |
| limit | `LIMIT` | other bound of a censored value (minus or plus infinity by default) |
| additional amounts | `ADDL`, `II` | the amount is repeated `ADDL` times, every `II` |
| steady state | `SS` | 1: the subject was at steady state of this amount, given every `II` |
| occasion | `OCC` | occasion, for inter-occasion variability |

Without an event flag, a row with a non-zero amount is an event.

## Several outputs or inputs

With several observed outputs, a column gives the type of each observation:
`Dataset.from_records(table, output="TYPE")`. The same holds for amounts that enter different
states: `input="ROUTE"`.

The labels of these columns are matched to the outputs and inputs of the model when the fit
starts:

- A label that names an output or an input goes to it.
- Other labels keep their order: that of `output_labels` (or `input_labels`) when given, else
  numerical order, the smallest number going to the first output.
- Text labels that name nothing and have no given order are refused when the model has more than
  one output or input.

## Covariates and regressors

`covariates=[...]` names columns that are constant within a subject, such as weight or sex. A text
covariate with two labels becomes a code, 0 for the first label in sorted order and 1 for the
other. One with more labels becomes one indicator column per label after the first, named
`NAME_label`, and its name stands for all of them: `covariates={"CL": ["GENOTYPE"]}` estimates one
coefficient per label, the shift of that label from the first. The covariate screen and the
searches treat such a covariate as one. A covariate enters a parameter on its Gaussian scale: for a log-normal clearance,
$\log CL_i = \log CL_{pop} + \beta\, WT_i + \eta_i$. `CL_pop` is therefore the clearance at
`WT = 0`. To read it at a reference weight, centre the covariate first, for instance
`table["WT"] = table["WEIGHT"] - 70`.

`regressors=[...]` names columns that vary over time and are read by the model as functions of
time. A blank cell carries the previous value forward.

## Initial time

Each subject's states take their initial values at its first record, an observation or an
amount. A model given `t0` starts every subject at that time instead, and a warning is given when
a first record comes after time 0 while some initial values are not zero.
