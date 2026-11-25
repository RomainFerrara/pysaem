"""Progress of a fit on stderr: rewritten in place in a terminal or a notebook, a line per tenth elsewhere."""

from __future__ import annotations

import shutil
import sys
import time

BAR = 24
COL = 62


def _mode(stream) -> str:
    if "ipykernel" in sys.modules:
        return "notebook"
    try:
        return "terminal" if stream.isatty() else "plain"
    except (AttributeError, ValueError):
        return "plain"


class Progress:
    """Header, SAEM bar with the current estimates, one line per post-estimation step, total."""

    def __init__(self, stream=None, mode: str | None = None, width: int | None = None):
        self.out = stream or sys.stderr
        self.mode = mode or _mode(self.out)
        self.width = width or (shutil.get_terminal_size((100, 20)).columns if self.mode == "terminal" else 100)
        enc = (getattr(self.out, "encoding", None) or "").lower().replace("-", "")
        utf = enc.startswith("utf")
        self.full, self.empty, self.dot, self.more = ("━", "─", " · ", "…") if utf else ("#", "-", " | ", "...")
        self.t0 = self.last = time.perf_counter()
        self.lines, self.shown, self.k, self.smooth_from = 0, -1, 0, None
        self.phase, self.label, self.setup = None, "SAEM", None

    @classmethod
    def create(cls, progress):
        """None: shown in a terminal or a notebook; True: always; False: never; or a Progress."""
        if isinstance(progress, Progress):
            return progress
        if progress is False or (progress is None and _mode(sys.stderr) == "plain"):
            return None
        return cls()

    def header(self, n_subjects: int, n_obs: int):
        self._freeze(f"fit  {n_subjects} subjects{self.dot}{n_obs} observations")
        if self.setup is not None:
            self._freeze(f"  {'model compiled and checked':<{COL - 2}}{self.setup:>8.1f} s")

    def start(self, label: str):
        self.label, self.smooth_from, self.k = label, None, 0

    def iteration(self, ev: dict):
        if ev["phase"] == "burn-in":
            return
        k, n, now = ev["iteration"], ev["n_iterations"], time.perf_counter()
        if ev["phase"] == "smoothing" and self.smooth_from is None:
            self.smooth_from = k
        self.k, self.phase = k, ev["phase"]
        if self.mode == "plain":
            if n and k * 10 // n != self.shown:
                self.shown = k * 10 // n
                self._put(self._bar(k, n, now) + "\n")
        elif now - self.last >= 0.1 or k == n:
            self.last = now
            self._live([self._bar(k, n, now), self._estimates(ev.get("parameters") or {})])

    def begin(self, step: str):
        if self.mode != "plain":
            self._live([f"  {step:<28}{self.more}"])

    def end(self, step: str, seconds: float):
        if step == "setup":
            self.setup = seconds
            return
        if step == "saem":
            e = self.smooth_from - 1 if self.smooth_from else self.k
            line = f"  {self.label:<12}{self.k} iterations, {e} exploratory, {self.k - e} smoothing"
        else:
            line = f"  {step:<28}"
        self._freeze(f"{line:<{COL}}{seconds:>8.1f} s")

    def close(self, result):
        self._freeze(f"  {'done':<{COL - 2}}{time.perf_counter() - self.t0:>8.1f} s   -2LL {result.ofv:.2f}"
                     + ("" if result.converged else "   not converged"))

    def abort(self):
        if self.lines:
            self._put("\n")
            self.lines = 0

    def _bar(self, k, n, now):
        f = min(1.0, k / n) if n else 0.0
        bar = self.full * round(BAR * f) + self.empty * (BAR - round(BAR * f))
        return f"  {self.label}  {self.phase:<13}{bar} {k:>5}/{n:<5}{now - self.t0:>7.1f} s"

    def _estimates(self, values: dict):
        text = "  " + self.dot.join(f"{k} {v:.4g}" for k, v in values.items())
        cut = self.width - 1 - len(self.more)
        return text if len(text) <= self.width - 1 else text[:cut] + self.more

    def _live(self, lines):
        if self.mode == "notebook":
            self._put("\r" + lines[0].ljust(self.width - 1)[: self.width - 1])
            self.lines = 1
            return
        self._put(self._clear() + "\n".join(x[: self.width - 1] for x in lines))
        self.lines = len(lines)

    def _freeze(self, line):
        if self.mode == "notebook":
            self._put("\r" + line.ljust(self.width - 1) + "\n")
        else:
            self._put(self._clear() + line + "\n")
        self.lines = 0

    def _clear(self):
        if self.mode != "terminal" or not self.lines:
            return ""
        return "\r\x1b[2K" + "\x1b[1A\r\x1b[2K" * (self.lines - 1)

    def _put(self, text):
        self.out.write(text)
        self.out.flush()
