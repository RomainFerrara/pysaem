# Run files and the command line

A run file describes the data, the model, the population and the settings of one fit:

```toml
[data]
file = "theophylline.csv"

[model]
equations = """
d/dt(Ad) = -ka*Ad              # gut
d/dt(Ac) = ka*Ad - CL/V*Ac     # blood
Cc = Ac/V
"""
dose = "Ad"
output = "Cc"

[population]
ka = { init = 1.0, omega = 0.5 }
V = { init = 0.5, omega = 0.5 }
CL = { init = 0.04, omega = 0.5 }
error = "combined1"

[settings]
seed = 1
```

```bash
pysaem fit run.toml
pysaem plot run.json
```

`pysaem fit` prints the estimates, standard errors, -2 log-likelihood and information criteria,
and writes the full result next to the run file as JSON. `pysaem plot` writes the diagnostic plots
(goodness of fit, visual predictive check, individual fits, iterations) to one PDF. A model and a
data file can also be given directly: `pysaem fit MODEL.json DATA.csv`.

## Sections

- `data`: the file, and the names of the covariate, regressor and observation-type columns, with
  the columns described in [Data](data.md).
- `model`: the equations described in [Writing a model](models.md), or Python functions
  (`python = """..."""` or `python = "model.py"`, with `states = [...]`). Then the state that
  receives the amounts (`dose`), the observed variable (`output`, or `outputs` for several), the
  covariates and regressors used in the equations, and `inputs` for several inputs or an input
  with a delay, a fraction or a duration.
- `population`: one entry per parameter, with `init`, `distribution` (`lognormal` by default,
  `normal`, or `logitnormal` with `bounds`), `omega`, `iiv = false`, `fixed`, `covariates`, `iov`,
  `gamma` and `prior = [m, s]`, a prior on the typical value described in
  [Population model](population.md). Then `error`, one of `constant`, `proportional`, `combined1`, `combined2` and
  `exponential`, or one per output, a hazard output taking `hazard` by itself. Then
  `correlations`, as blocks of parameters.
- `settings`: the fields of `pysaem.Settings`, such as the iterations, the seed, the number of
  starts (`n_starts`), the tolerances and a time budget.
