# Install

PySAEM needs Python 3.10 or later and runs on Linux and macOS. Until its first release on PyPI,
install it from GitHub:

```bash
pip install "pysaem[plot] @ git+https://github.com/RomainFerrara/pysaem"
```

`[plot]` adds matplotlib, used by the vignettes and the diagnostic plots. The dependencies, NumPy,
SciPy, pandas, SymPy and Numba, are installed with it. The first fit compiles the solver, which
takes about half a minute. The compiled code is cached for the next fits.

To work on the package itself, clone the repository and install it in editable mode with the
test and documentation extras:

```bash
git clone https://github.com/RomainFerrara/pysaem.git
cd pysaem
pip install -e ".[plot,test,docs]"
pytest -m "not slow"
```
