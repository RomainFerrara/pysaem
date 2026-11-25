# Writing a model

A model is written either as Python functions or as equations. Both give the same object, a
`Model`, whose equations can be printed with `model.equations()` before any fit.

## As Python functions

```python
def rhs(t, y, p):                      # derivatives of the states, in their order
    s, i, r = y
    beta = p["beta"] * (p["cut"] if t >= p["t_on"] else 1.0)   # fewer contacts from t_on
    return [-beta * s * i, beta * s * i - p["gamma"] * i, p["gamma"] * i]

def observe(t, y, p):                  # what is measured
    return y[1]

def init(p):                           # optional: values at the start (zero otherwise)
    return [1.0 - p["i0"], p["i0"], 0.0]
```

`y` holds the states in the order given, `p` the parameters, covariates and regressors by name
(`p["k"]` or `p.k`). The functions may use intermediate variables, `if`/`elif`/`else`, loops,
helper functions, NumPy, `math` and listed SciPy functions, tables (`np.interp`, `interp1d`,
`CubicSpline`) and comparisons used as numbers (`k * (t > t_on)`). A model without differential
equations has no states and computes its prediction in `observe`, which may return a list or a dict
when there are several outputs.

```python
from pysaem import Model

model = Model.from_python(rhs=rhs, observe=observe, init=init, states=["s", "i", "r"],
                          outputs=["infected"], parameters=["beta", "gamma", "cut", "t_on", "i0"],
                          values={"beta": 0.5, "gamma": 0.2, "cut": 0.4, "t_on": 30, "i0": 1e-4})
print(model.equations())
```

The functions can also be given as source text, as a path to a `.py` file or as a module, in
place of `rhs=`, `observe=` and `init=`. They are read from their source code, so they must be
defined in a file, a module or a notebook cell. In a plain interactive interpreter, which does not
keep the source, give the code as text.

`values` are typical values of the parameters. They are used to check the reading, described
below, and are not the starting values of the fit, which are given to `Population.build`.

### What happens when a model is read

PySAEM runs the functions once with symbols in place of numbers, once per branch of each
condition, and writes the result as SymPy equations. The equations are then compared with the code
around `values`, at the numbers the code contains and on both sides of each condition, and
`model.reading` says how many points and conditions were compared. From the equations, PySAEM
finds whether the system is linear (it is then solved exactly), the times where the solver must
stop, and the exact derivatives.

Valid code that cannot be written as equations, such as a test for `nan`, a `try`/`except`, a
loop whose number of turns depends on `t`, or an expression that grows at each step, is compiled
as written, with a warning. That code is slower and has no closed form and no exact derivatives, and
`model.reading` gives the reason. The code runs while it is read, with NumPy, `math` and SciPy as
its only modules and a time limit. This guards against mistakes and is not a sandbox, so a model
specification holding code is read only on request: `Model.from_dict(d, python=True)`.

## As equations

Equations are written one per line, as on paper, and are parsed, never run:

```python
from pysaem import Model
from pysaem.model import parse_equations

text = """
d/dt(Ad) = -ka*Ad              # gut
d/dt(Ac) = ka*Ad - CL/V*Ac     # blood
Cc = Ac/V
"""
model = Model(**parse_equations(text), parameters=["ka", "V", "CL"], outputs=["Cc"],
              inputs={"dose": "Ad"})
```

`d/dt(A) = ...` (or `dA/dt = ...`) defines a state, `A(0) = ...` its initial value (zero
otherwise), `name = ...` a variable, `#` starts a comment and `;` separates statements on one
line. A variable defined from parameters can start a state, for instance `R0 = kin/kout` then
`R(0) = R0`. A variable may depend on a condition:

```
if t < 24
  E = Emax
elif t < 48
  E = Emax/2
else
  E = 0
end
```

Expressions use the usual operators and functions (`exp`, `log`, `sqrt`, ...) and conditions
(`where(condition, a, b)` or `a if condition else b`).

## Inputs, outputs and initial time

- `inputs` maps each input of the data to the state it enters, for instance `{"dose": "Ad"}`, or
  to `{"target", "delay", "scale", "duration"}` for a lag, a fraction of the amount that enters
  or a zero-order input. The amounts enter outside `rhs`.
- `outputs` names what `observe` returns. Each output has its own error model.
- An output declared in `hazards`, for instance `hazards={"h": "terminal"}` (or `"single"`,
  `"repeated"`), is the hazard of an event. It is integrated with the equations, and its records
  give the times or counts of events.
- Each subject's states take their initial values at its first record. `t0=` sets another time,
  the same for every subject. A warning is given when the first record comes after time 0 and some
  initial values are not zero.

## Scope

PySAEM fits continuous outputs, possibly censored, and time-to-event outputs. It does not fit
count, categorical or mixture models, delay or stochastic differential equations, hazards that
depend on past events, or models with hazards compiled as written. It has no FOCE, Laplace or
Bayesian method.
