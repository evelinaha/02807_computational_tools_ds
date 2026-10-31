"""Small helpers: timing and memory logging of the main pipeline steps."""

from __future__ import annotations

import resource
import sys
import time
import tracemalloc
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

# Every `timed` block appends one row here; `runtime_table()` turns it into a DataFrame for the report.
RUNTIME_LOG: list[dict] = []


def process_peak_rss_gb() -> float:
    """Peak resident memory of this process so far, in GB (ru_maxrss is bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6


@contextmanager
def timed(step: str, verbose: bool = True) -> Iterator[None]:
    """Time a block and record the Python-heap peak allocated inside it (numpy/scipy arrays included).

    DuckDB and igraph allocate outside the Python heap, so for those steps the process-wide
    peak RSS (also recorded) is the more meaningful number.
    """
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        secs = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        if not was_tracing:
            tracemalloc.stop()
        row = {
            "step": step,
            "seconds": round(secs, 2),
            "heap_peak_gb": round(peak / 1e9, 3),
            "process_peak_rss_gb": round(process_peak_rss_gb(), 2),
        }
        RUNTIME_LOG.append(row)
        if verbose:
            print(f"[{step}] {secs:.1f}s, heap peak {row['heap_peak_gb']:.2f} GB, "
                  f"process peak RSS {row['process_peak_rss_gb']:.2f} GB")


def runtime_table() -> pd.DataFrame:
    """All timed steps so far as a DataFrame."""
    return pd.DataFrame(RUNTIME_LOG)
