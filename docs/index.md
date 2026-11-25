# PySAEM

PySAEM fits nonlinear mixed-effects models in Python. Such a model describes a group of
individuals measured repeatedly: one typical curve for the group, and how each individual departs
from it. The curve is written as ordinary Python functions or as differential equations, and the
population parameters are estimated by the stochastic approximation expectation-maximisation
(SAEM) algorithm.

The model is read into equations, compiled with Numba and fitted on all the cores of the machine.
The [getting-started vignette](vignettes/theophylline.md) fits a first model step by step.

```{toctree}
:maxdepth: 1
:caption: Getting started

install
vignettes/theophylline
```

```{toctree}
:maxdepth: 1
:caption: User guide

models
data
population
command_line
checks
```

```{toctree}
:maxdepth: 1
:caption: Comparison

vignettes/benchmark
vignettes/benchmark_details
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
```
