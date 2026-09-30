"""Stage timing and memory measurement (stdlib; psutil used when installed)."""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

try:  # optional
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _win_memory() -> tuple[float, float] | None:
    """(working set MB, peak working set MB) via K32GetProcessMemoryInfo with correct 64-bit types."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return pmc.WorkingSetSize / 2**20, pmc.PeakWorkingSetSize / 2**20


def rss_mb() -> float:
    """Current resident memory of this process in MB."""
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    w = _win_memory()
    if w:
        return w[0]
    try:
        with open("/proc/self/statm") as fh:  # linux: resident pages
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except Exception:
        return float("nan")


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB."""
    w = _win_memory()
    if w:
        return w[1]
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 1024
    except Exception:
        return rss_mb()


class StageTimer:
    """Collects wall time and memory per named stage."""

    def __init__(self) -> None:
        self.stages: dict[str, dict[str, float]] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        m0 = rss_mb()
        try:
            yield
        finally:
            m1 = rss_mb()
            self.stages[name] = {"seconds": round(time.perf_counter() - t0, 3), "rss_mb": round(m1, 1),
                                 "rss_delta_mb": round(m1 - m0, 1)}

    def total(self) -> float:
        return round(sum(s["seconds"] for s in self.stages.values()), 3)

    def as_dict(self) -> dict:
        return {"stages": self.stages, "total_seconds": self.total(), "peak_rss_mb": round(peak_rss_mb(), 1)}
