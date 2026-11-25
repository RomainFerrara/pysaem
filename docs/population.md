# Population model

`Population.build` describes how the individual parameters are distributed and how the
observations scatter around the predictions. One setting applies to every parameter unless a
dict overrides it by name.

```python
population = Population.build(
    model, init={"ka": 1.0, "V": 0.5, "CL": 0.04}, omega=0.5, error="combined1",
    distribution={"F": "logitnormal"}, no_iiv=["F"], covariates={"CL": ["WEIGHT"]},
    correlations=[["V", "CL"]], iov={"ka": 0.3}, fixed=["F"],
    prior={"ka": (1.5, 0.3)})
```

## Distributions

Each parameter has a typical value (`init`, the starting value of its estimation) and a random
effect of standard deviation `omega` on a transformed scale:

- `lognormal` (the default): the parameter stays positive, and `omega` reads as a coefficient of
  variation when it is small;
- `normal`: the parameter can take any sign;
- `logitnormal` with `bounds=(lo, hi)`: the parameter stays between the bounds.

`no_iiv` lists the parameters without random effect, estimated as one value for everybody.
`fixed` lists those whose typical value stays at `init`. `Parameter.omega_fixed` holds an omega.

## Covariates, correlations and occasions

`covariates` maps a parameter to the covariates of the data that enter its typical value, on the
transformed scale, with a coefficient per covariate (see [Data](data.md) for how covariates are
read). `correlations` lists blocks of parameters whose random effects are correlated. `iov` maps
a parameter to the initial standard deviation of a random effect per occasion, when the data
have an occasion column.

## Priors

`prior` maps a parameter to `(m, s)`: a normal prior on its typical value, on the transformed
scale, centred on the transform of `m` with standard deviation `s`. For a lognormal parameter
this is a lognormal prior of median `m`, and `s` then reads as a coefficient of variation when it is
small. The fit maximises the likelihood times the prior: the estimate is the posterior mode and
the standard errors are posterior standard deviations, computed from the same information
matrix with the precision of the prior added. The log-likelihood and the information criteria
reported are those of the data alone, without the prior. The summary names the parameters
with a prior.

A prior is useful when the data say little about a parameter, for instance an absorption rate
with sparse early samples, and a published value is available. Its weight relative to the data
is that of `1 / s^2` against the information of the data. A prior meant to hold a parameter at a
value is better written as `fixed`.

## Error models

`error` is one of `constant` (sd `a`), `proportional` (sd `b` times the prediction), `combined1`
(`a + b` times the prediction), `combined2` (the root of `a^2 + b^2` times the squared
prediction) and `exponential` (a constant error on the log of the observations), or a dict with
one of them per output. A hazard output takes `hazard` by itself. `ErrorModel(kind, a=, b=,
fixed=)` gives the initial values and holds them.
