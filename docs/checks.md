# Checking a fit

Three checks refit the model several times. Each takes the model, the data, a population model
and settings, as `fit` does, and returns a table. The population model is usually the one at the
estimates, `Population.from_dict(model, result.population)`. The refits compute no standard
errors and no simulations. `on_fit(k, result)` reports each one as it ends.

## Convergence from several starts

```python
from pysaem.assess import convergence

runs = convergence(model, data, population, n=5, spread=0.5)
print(runs.summary())
```

Each run uses a different seed and, with `spread` above zero, starting values moved on their
Gaussian scale by `spread` times a standard normal draw. The first run starts as given.
`summary()` gives the lowest and highest value of each estimate over the runs and their range as
a percentage of the mean, and the range of the -2LL in points. Runs that end within a few points
of -2LL and a few percent of each other show that the fit depends neither on the seed nor on the
start. A run that ends higher has stopped at a local maximum, and a run that ends lower than
the fit it checks says the fit itself did.

## Likelihood profile

```python
from pysaem.assess import profile

prof = profile(model, data, population, parameter="ka_pop", se=result.se["ka_pop"])
print(prof.table)
print(prof.interval(0.95))
```

The estimate named, a typical value `X_pop` or a standard deviation `omega_X`, is held at each
value of a grid while the others are refitted, and the -2LL is recorded with its Monte Carlo
standard error. The grid spans three standard errors on each side of the value, on the log
scale for a lognormal parameter or an omega, and `values=` gives another grid. `interval(level)`
reads the values where the profile rises above its lowest point by the chi-square quantile of
the level (Venzon and Moolgavkar 1988): an interval that the standard errors from the
information matrix cannot give when the likelihood is far from quadratic, for instance near a
boundary. A side the grid does not reach is `nan`.

## Bootstrap

```python
from pysaem.assess import bootstrap

boot = bootstrap(model, data, population, n=200, strata="SEX")
print(boot.summary(0.95))
```

Each replicate draws the subjects with replacement, keeping their records, and is refitted from
the population model given (Efron 1979). The sd of an estimate over the replicates and its
percentiles estimate its uncertainty without the information matrix (Thai et al. 2014). With
`strata`, a covariate of the data, the subjects are drawn within each of its levels, so that
every replicate keeps the levels' sizes. `failed` counts the replicates whose fit raised an error
and `samples` holds the subjects each one drew. The replicates are fitted without likelihood
unless `loglik=True`.
