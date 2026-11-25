from pathlib import Path

import pysaem


def test_stale_kernel_cache_is_dropped():
    cache = Path(pysaem.__file__).parent / "__pycache__"
    stamp = cache / "numba_sources.sha256"
    fresh = stamp.read_text()
    fake = cache / "engine._stale_probe-1.py310.nbi"
    fake.write_bytes(b"")
    stamp.write_text("old")
    pysaem._drop_stale_kernel_cache()
    assert not fake.exists()
    assert stamp.read_text() == fresh


def test_current_kernel_cache_is_kept():
    cache = Path(pysaem.__file__).parent / "__pycache__"
    fake = cache / "engine._kept_probe-1.py310.nbi"
    fake.write_bytes(b"")
    pysaem._drop_stale_kernel_cache()
    assert fake.exists()
    fake.unlink()
