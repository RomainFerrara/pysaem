# Changelog

All notable changes to PySAEM are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/).

## Unreleased

## 0.3.0 - 2026-11-25

First public version.

### Added

- Models written as Python functions, read into symbolic equations and checked against the code,
  or written as equations. Code that cannot be read is compiled as written.
- Solver compiled with Numba: matrix exponential for linear systems, Dormand-Prince switching to
  Radau IIA when stiff, exact Jacobians and sensitivities.
- SAEM with several chains per subject, mode jumping, interweaving, stopping rules and several
  starts.
- Event records with amounts given at once or over time, lags, fractions, resets, steady state
  and repeated amounts, several outputs, censored observations, time-to-event outputs,
  occasions, covariates and regressors. A text covariate with more than two labels becomes one
  indicator column per label after the first.
- A prior on a typical value, `Parameter(prior=(m, s))` or `Population.build(prior={...})`, normal
  on the transformed scale. The estimate is then the posterior mode.
- Log-likelihood by importance sampling, standard errors, information criteria, individual
  parameters, residuals and NPDE, individual forecasts.
- Visual predictive checks, `result.vpc`, and prediction-corrected ones (Bergstrand et al. 2011),
  `result.pcvpc`. Censored records are read at the limit in the data and in the replicates. The
  correction by error model and the guard against near-zero predictions follow a design by
  Martin Soucail.
- Diagnostic plots in one PDF, framed on the observations.
- Covariate screening, and a covariate search by the covariate step of SAMBA, by adding and
  removing one effect at a time, or by any regressor with `fit` and `predict`
  (`covariates.search(method="learned", learner=...)`).
- `pysaem.assess`: `convergence` refits from several seeds and starts, `profile` holds one
  estimate on a grid, `bootstrap` refits replicates of the subjects. `Dataset.take` builds such
  replicates.
- Run files and the `pysaem` command line.
- The theophylline data, as `pysaem.datasets.theophylline()`.
