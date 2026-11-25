<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/pysaem_logo_dark.png">
    <img alt="PySAEM" src="assets/pysaem_logo_light.png" width="420">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/RomainFerrara/pysaem/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/RomainFerrara/pysaem/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://pysaem.readthedocs.io"><img alt="Documentation" src="https://readthedocs.org/projects/pysaem/badge/?version=latest"></a>
  <img alt="Python 3.10 to 3.13" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue">
  <a href="LICENSE"><img alt="Licence: MIT" src="https://img.shields.io/badge/licence-MIT-green"></a>
</p>

PySAEM fits nonlinear mixed-effects models of ordinary differential equations in Python. Such a
model describes repeated measurements in a group of individuals by a typical curve and individual
parameters drawn around it. PySAEM estimates the population parameters by maximum likelihood with
the stochastic approximation expectation-maximisation algorithm (SAEM).

The model is written as plain Python functions or as equations. PySAEM reads the functions into
symbolic equations, checks them against the code and compiles a solver from them with Numba. It is
meant for scientists who model repeated measurements in Python, in pharmacology, biology or
medicine, and for programs that write and fit models without supervision.

## Installation

PySAEM needs Python 3.10 or later. Until its first release on PyPI, install it from GitHub:

```bash
pip install "pysaem[plot] @ git+https://github.com/RomainFerrara/pysaem"
```

`[plot]` adds matplotlib for the diagnostic plots. The first fit compiles the solver, which takes
about half a minute. The compiled code is cached for the next fits.

## Example

Twelve people took theophylline by mouth, and its concentration in their blood was measured over
24 hours. The amount moves from the gut to the blood, then leaves the blood:

```python
from pysaem import Dataset, Model, Population, fit
from pysaem.datasets import theophylline

def rhs(t, y, p):
    gut, blood = y
    return [-p["ka"] * gut, p["ka"] * gut - p["CL"] / p["V"] * blood]

def observe(t, y, p):
    return y[1] / p["V"]

model = Model.from_python(rhs=rhs, observe=observe, states=["gut", "blood"],
                          outputs=["concentration"], parameters=["ka", "V", "CL"],
                          inputs={"dose": "gut"}, values={"ka": 1.0, "V": 0.5, "CL": 0.04})
population = Population.build(model, init={"ka": 1.0, "V": 0.5, "CL": 0.04}, omega=0.5,
                              error="combined1")
result = fit(model, Dataset.from_records(theophylline()), population)
print(result.summary())
```

The same fit can be described in a run file and started from the command line:
`pysaem fit examples/theophylline.toml`.

## Features

- Models: systems of ordinary differential equations, or models without them, written as Python
  functions (conditions, loops, helper functions, NumPy, tables) or as equations.
- Data: amounts given at once or over time, with a lag or a fraction absorbed, resets, steady
  state and repeated amounts, several outputs, observations censored at a limit, time-to-event
  and repeated-event outputs, and occasions.
- Population model: normal, log-normal and logit-normal parameters, correlated random effects,
  inter-occasion variability, and an error model for each output.
- Covariates: continuous and categorical covariates act linearly on the scale where a parameter is
  normal, the log scale for a log-normal parameter, or in any form written in the model. The
  covariate model is searched by the covariate step of SAMBA, or by adding and removing one effect
  at a time. A third search uses any regressor with `fit` and `predict`, a random forest for
  instance. Its cross-validated prediction enters the model as one covariate per parameter.
- Solver: matrix exponential for linear systems, Dormand-Prince switching to Radau IIA when the
  system becomes stiff, exact Jacobians and sensitivities, subjects solved in parallel.
- Results: estimates with standard errors and confidence intervals, -2 log-likelihood by
  importance sampling, AIC, BIC and BICc, individual parameters and shrinkage, residuals and
  normalised prediction distribution errors, visual predictive checks, and the diagnostic plots
  in one PDF.

PySAEM has no FOCE, Laplace or Bayesian method, and does not fit count, categorical or mixture
models, or delay or stochastic differential equations.

## Data protection

PySAEM makes no network call and writes nothing outside the files you ask for, except a cache of
compiled kernels in its own `__pycache__` folder (or `NUMBA_CACHE_DIR`) that holds code, not
data. Error messages name the subject and quote the cell at fault so that you can find it. A
program that forwards them to a third party should redact them first.

## Documentation

The [documentation](https://pysaem.readthedocs.io) holds a getting-started vignette on the
theophylline data, how to write a model and lay out the data, run files and the command line, a
comparison with other packages, and the API reference.

## Citation

If you use PySAEM in your work, please cite it. GitHub gives the reference from
[`CITATION.cff`](CITATION.cff) under "Cite this repository".

## Contributing

Questions, bug reports and pull requests are welcome, as described in
[`CONTRIBUTING.md`](CONTRIBUTING.md).
Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).

## Licence

PySAEM is distributed under the [MIT licence](LICENSE). Its Radau IIA integrator is ported from
SciPy, under the BSD 3-clause licence reproduced in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
