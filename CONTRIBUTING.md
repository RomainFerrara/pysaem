# Contributing to PySAEM

Thank you for your interest. Questions, bug reports, examples from your field and code are all
welcome. Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).

## Asking a question or reporting a problem

Open an [issue](https://github.com/RomainFerrara/pysaem/issues) and choose the template that fits.
For a bug, the most useful report holds the smallest model and dataset that show it (simulated
data are fine), the full error message and the versions of PySAEM, NumPy, SciPy, SymPy and Numba.
Please do not post data that cannot be shared.

## Proposing a change

1. Fork the repository and create a branch from `main`.
2. Install the package in editable mode with the test and documentation extras:

   ```bash
   git clone https://github.com/<your-account>/pysaem.git
   cd pysaem
   pip install -e ".[test,docs]"
   ```

3. Make the change, with a test that fails before it and passes after. Tests that fit a model
   for more than a few seconds are marked `@pytest.mark.slow`.
4. Run the tests and, if the documentation changed, build it:

   ```bash
   pytest -m "not slow"
   pytest -m slow
   sphinx-build -b html docs docs/_build/html
   ```

5. Add a line to `CHANGELOG.md` under "Unreleased" and open a pull request against `main`.
   The tests run on every pull request.

A new estimation method or change of default should cite the publication it comes from, and say
how it was checked: against an exact result, another solver or another estimation tool.

## Code style

The code follows PEP 8 with lines of at most 100 characters. Comments state the rule and the
number they rely on, and longer explanations belong in the documentation.
