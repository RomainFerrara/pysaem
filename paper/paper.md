---
title: 'PySAEM: nonlinear mixed-effects models of ordinary differential equations in Python'
tags:
  - Python
  - nonlinear mixed-effects models
  - SAEM
  - ordinary differential equations
  - population models
authors:
  - name: Romain Ferrara
    corresponding: true
    affiliation: "1, 2"
  - name: Sébastien Benzekry
    affiliation: 1
affiliations:
 - name: COMPutational pharmacology and clinical Oncology (COMPO), Inria Center at University Côte d'Azur, Cancer Research Center of Marseille, Marseille, France
   index: 1
   ror: 02kvxyf05
 - name: Department of Nuclear Medicine, Institut Paoli-Calmettes, Marseille, France
   index: 2
   ror: 04s3t1g37
date: 11 November 2026
bibliography: paper.bib
---

<!-- AUTHOR, before submission:
- ORCID of each author (add `orcid:` under each name).
- JOSS screens out repositories with less than six months of public history, releases, issues and
  pull requests: the repository must be public long enough before submission.
- Results: the benchmark ran at commit 6509f90, the test count is that of the current main.
  Rerun the benchmark at the release tag, or state the version it ran on.
- The benchmark scripts, protocols and per-fit results must be public at submission (Research
  impact statement, last paragraph).
- ferrara2026: cite only if the preprint is posted, with its DOI or arXiv number.
-->

# Summary

Repeated measurements on a group of individuals, such as drug concentrations in patients or tumour
sizes under treatment, are often described by a mixed-effects model: a typical curve for the group,
and individual parameters drawn around it. When the curve is the solution of ordinary differential
equations (ODEs), the likelihood of the model has no closed form and its estimation is a hard
numerical problem. `PySAEM` estimates these models in Python by maximum likelihood, with the
stochastic approximation expectation-maximisation algorithm (SAEM) [@delyon1999; @kuhn2005]. The
model is written as ordinary Python functions. `PySAEM` reads them into symbolic equations, checks
the equations against the code and compiles a solver from them. A fit returns the estimates with
their standard errors, the likelihood, information criteria and diagnostics.

# Statement of need

`PySAEM` is meant for scientists who model repeated measurements with ODEs in Python, in
pharmacology, oncology or other fields, and for programs that write and fit models without
supervision. SAEM maximises the likelihood of a model that is nonlinear in its random effects
without linearising it [@lavielle2014]. Its open-source implementations are mostly written in R,
Julia and Rust, each with its own model language, so a Python user has to rewrite the model.
`PySAEM` fits the ODE system as it is written in Python, with the doses given as records of the
data, and derives from the code a compiled solver and exact derivatives.

The package was first written as the estimator of APOD, an agent in which a language model writes
structural models and reads their fits [@ferrara2026]. Such a program fits hundreds of models
without supervision, so the estimator must accept an ODE system as it was written and converge
without tuning. These two requirements shaped the package, which is now independent of APOD and
contains nothing specific to pharmacokinetics.

# State of the field

`saemix` [@comets2017] takes an R function, inside which the user solves an ODE model. `nlmixr2`
[@fidler2019] and `FeRx` [@keizer2026] read their own model language and also offer FOCEI.
`NoLimits.jl` [@huth2026] uses Julia macros and also offers Laplace, FOCEI and MCMC methods.
`pynlme` [@lovfors2026], the closest package in Python, follows the syntax of the MATLAB functions
`nlmefit` and `nlmefitsa`. It solves ODEs given as a vectorised right-hand side with SciPy, or as
model files compiled by the SUND toolbox, and has one error model for all outputs and no censored
observations. Commercial tools also implement SAEM, and Bayesian tools such as Stan
[@carpenter2017] fit the same models by Markov chain Monte Carlo.

`PySAEM` takes plain Python code, so that a model written for simulation, or written by a program,
is fitted unchanged. Reading free code into equations requires its own model interface, solver and
compiled likelihood. Added to `pynlme`, it would have replaced most of that package's interface and
solver, so we wrote a separate package. Each output has its own error model, observations may be
censored, and the package supports inter-occasion variability and time-to-event outputs. It has no
FOCE, Laplace or Bayesian method, and does not fit count, categorical or mixture models, or delay or
stochastic differential equations.

# Software design

The theophylline data [@boeckmann1994], which ship with the package, are fitted as follows:

```python
from pysaem import Dataset, Model, Population, fit
from pysaem.datasets import theophylline

def rhs(t, y, p):
    gut, blood = y
    return [-p["ka"] * gut, p["ka"] * gut - p["CL"] / p["V"] * blood]

def observe(t, y, p):
    return y[1] / p["V"]

model = Model.from_python(
    rhs=rhs, observe=observe, states=["gut", "blood"], outputs=["concentration"],
    parameters=["ka", "V", "CL"], inputs={"dose": "gut"},
    values={"ka": 1.0, "V": 0.5, "CL": 0.04})
population = Population.build(model, init={"ka": 1.0, "V": 0.5, "CL": 0.04},
                              omega=0.5, error="combined1")
result = fit(model, Dataset.from_records(theophylline()), population)
print(result.summary())
```

A model language gives the equations directly and must be learnt. Python functions need no new
language, and a solver sees them only through the numbers they return. `PySAEM` runs the functions with SymPy symbols
[@meurer2017] in place of numbers, once per branch of each condition, and writes the result as
equations. It checks these against the code at sampled points and on both sides of each condition,
and prints them with `model.equations()`. From the equations it derives whether the system is
linear, the times where the solver must stop, and the exact Jacobian and sensitivities
(\autoref{fig:code}). Valid code that cannot be written as equations is compiled as written, with a
warning, and loses these properties.

![(a) A model written as Python code, with three transit states built by a loop, a helper function
and a saturation that depends on a state. (b) The equations read by `PySAEM`, typeset here.
`model.equations()` prints them as text. (c) What the package derives from them.\label{fig:code}](figure1.png)

The solver is compiled with Numba [@lam2015], which keeps the package installable with `pip` alone
at the cost of about half a minute of compilation at the first fit. The compiled code is cached on
disk. Linear systems with constant coefficients are solved by the matrix exponential [@moler2003].
Other systems use the method of Dormand and Prince [@dormand1980] and switch to Radau IIA
[@hairer1996] when they become stiff [@hairer1993]. The Radau code is ported from SciPy
[@virtanen2020]. Subjects are solved in parallel threads.

SAEM follows @kuhn2005, with several Markov chains per subject. The estimate reported is the
average of the iterates of the second half of the smoothing phase [@polyak1992]. Their Metropolis-Hastings kernels draw from the population distribution
and move by random walks. To help fits run without tuning, `PySAEM` adds three steps. A kernel
redraws a group of random effects from the population distribution given the others. A
mode-jumping move [@tjelmeland2001] lets a chain reach the mode found from the population mean when
that mode is more probable by at least 2 log units, for instance when two rate constants are
swapped. A Fisher-scoring step updates the fixed effects with the standardised random effects held,
an interweaving adapted from @yu2011. The variances decrease by at most 5 % per iteration during the
exploratory phase [@lavielle2014]. The stopping rules of both phases are our own. These additions
are not covered by the convergence results of SAEM [@delyon1999], so the `converged` flag describes
the trajectory of the iterations and does not prove convergence.

A covariate acts on a parameter linearly on the scale where the parameter is normal, or in any form
written in the model code. Three searches select the covariate model. The covariate step of SAMBA
[@prague2022] regresses draws of the individual parameters on the covariates and refits the model
chosen. A stepwise search adds and removes one effect at a time, in the order of its association
with the random effects [@ayral2021], and decides each step by the BICc or a likelihood ratio test.
A learned search replaces the linear regression of SAMBA by any regressor with `fit` and `predict`,
a random forest for instance. The prediction for each subject comes from a regressor trained on the
other subjects. It enters the model as one covariate, which is removed when the likelihood does not
support it.

After SAEM, the likelihood is estimated by importance sampling, with its Monte Carlo error
[@lavielle2014]. Standard errors come from the observed information, the Hessian of this
estimate computed by central differences. The result also holds BICc [@delattre2014], shrinkage
[@savic2009], normalised prediction distribution errors [@comets2008] and visual predictive checks.
`PySAEM` requires Python 3.10 or later, NumPy [@harris2020], SciPy, pandas [@mckinney2010], SymPy and
Numba, and is distributed under the MIT licence.

# Research impact statement

The 471 tests of the package compare it with exact solutions, SciPy integrators at a tighter
tolerance, closed-form results and adaptive Gauss-Hermite quadrature [@pinheiro1995]. We
translated to Python 1905 published models from `nlmixr2lib` [@nlmixr2lib], textbooks and reviews,
covering 1615 distinct structures. For 1761 of them, every prediction agreed within $10^{-4}$
relative with SciPy's Radau method at a relative tolerance of $10^{-10}$, 83 failed and 61 had no
valid reference. These models were used during development. Of 300 further models, never used
before, 288 agreed. In the runs of a development version of APOD, the solver handled 268 of the 275
models written by the agent within the same tolerance.

`saemix`, `nlmixr2`, `FeRx`, `NoLimits.jl`, `pynlme` and `PySAEM` fitted the same 200 simulated
datasets of each of 12 designs (11 for `NoLimits.jl`), once per dataset, from twice or half the true
values. Each tool ran its defaults, `saemix` and `nlmixr2` with more iterations. The -2
log-likelihood of every fit was computed at its estimates by one method per design: adaptive
Gauss-Hermite quadrature, or the importance sampling of `PySAEM` when quadrature was not feasible.
A separate evaluator, sharing no code with `PySAEM`, agreed within its tolerance on 169 of 200
values it recomputed. On the 10 designs with a stable method, `PySAEM` ended within 2 points of the
lowest -2 log-likelihood on 1977 of 2000 datasets (98.8 %). The other tools did so on 78.2 %
(`nlmixr2`), 74.6 % of 1800 (`NoLimits.jl`), 72.6 % (`FeRx`), 65.8 % (`saemix`) and 33.5 %
(`pynlme`). Started at the true
values, `pynlme` came within 0.7 points of `PySAEM` on the one-compartment designs, where it had
ended farthest.

The relative biases of the typical values of `PySAEM` were within 14 %, except on the model with
linear and saturable elimination. There it overestimated Vmax and Km by 66 % and 70 % and
underestimated the linear clearance by 17 %. The other tools, `NoLimits.jl` excepted, had biases
of 0.3 % to 35.7 % on Vmax and Km. On this model the data mainly inform the ratio Vmax/Km, whose
median absolute error was 12.0 % with `PySAEM` and 13.6 % to 17.8 % with the other tools. The
coverage of the 97.5 % Wald intervals of `PySAEM` fell to 72 % on the design with a lag and 74 % on
the saturable one.

The median fit of `PySAEM` took 1.5 to 9.6 s on 14 threads. On their own thread counts (`saemix` and
`pynlme` one, `nlmixr2` two, `FeRx` eight, `NoLimits.jl` ten), `PySAEM` was faster than `saemix` and
`nlmixr2`, and slower than `pynlme`, `FeRx` and, on most designs, `NoLimits.jl`. It used more CPU
time than `saemix` on every design. On a dataset of 637 subjects with saturable elimination,
`PySAEM` took 26 min and `FeRx` 17 min, ending 4.8 points higher.

The designs and the defaults of `PySAEM` were set during its development, so this comparison
measures what `PySAEM` was tuned for, on new datasets. In an earlier run of the study, on the
original designs and against `saemix` and `nlmixr2` only, `PySAEM` passed 270 of 385
non-inferiority criteria set before the runs, failed 54, and 61 were inconclusive. The comparison
vignette of the documentation gives these results, and the scripts, protocols and per-fit results
are deposited with the package.

# AI usage disclosure

Claude Code (Anthropic), including the Claude Opus 5.5 model, was used to write code, tests and
documentation and to draft parts of this paper. The authors reviewed, edited and validated all of
its outputs and take full responsibility for the content of the package and of this paper.

# Acknowledgements

This work received no specific funding.

# References
