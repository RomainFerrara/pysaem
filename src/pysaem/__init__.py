"""pysaem: population estimation for ODE models by SAEM."""

import warnings

from numba.core.errors import NumbaExperimentalFeatureWarning

# first-class functions are used on purpose (numba reports it from its own modules, so it is
# matched by its text); numpy 2 with the Accelerate library of macOS raises floating-point
# warnings on finite matrix products, ignored within pysaem only
warnings.filterwarnings("ignore", message="First-class function type feature is experimental",
                        category=NumbaExperimentalFeatureWarning)
warnings.filterwarnings("ignore", message=".* encountered in matmul", category=RuntimeWarning,
                        module=r"pysaem\.")


def _drop_stale_kernel_cache():
    # Numba checks only the file that defines a cached kernel, so a kernel calling a function of
    # another module keeps the old code; drop every pysaem kernel when any compiled module changes
    import hashlib
    import os
    from pathlib import Path

    import numba

    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    here = Path(__file__).parent
    digest = hashlib.sha256(numba.__version__.encode())
    for name in ("engine.py", "native.py", "post.py", "reader.py", "solve.py"):
        digest.update((here / name).read_bytes())
    stamp = here / "__pycache__" / "numba_sources.sha256"
    try:
        if stamp.exists() and stamp.read_text() == digest.hexdigest():
            return
        for f in list(stamp.parent.glob("*.nbi")) + list(stamp.parent.glob("*.nbc")):
            f.unlink(missing_ok=True)
        stamp.parent.mkdir(exist_ok=True)
        stamp.write_text(digest.hexdigest())
    except OSError:
        pass


_drop_stale_kernel_cache()

from .api import fit  # noqa: E402
from .data import DataError, Dataset  # noqa: E402
from .engine import FitError  # noqa: E402
from .model import Model, ModelError  # noqa: E402
from .population import ErrorModel, Parameter, Population  # noqa: E402
from .reader import Untranslatable, from_python  # noqa: E402
from .result import FitResult  # noqa: E402
from .saem import Settings  # noqa: E402
from ._version import __version__  # noqa: E402

__all__ = ["DataError", "Dataset", "ErrorModel", "FitError", "FitResult", "Model",
           "ModelError", "Parameter", "Population", "Settings", "Untranslatable", "fit",
           "from_python"]
