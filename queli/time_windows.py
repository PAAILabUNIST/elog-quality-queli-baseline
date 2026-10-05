"""Timestamp-granularity check and inference of the simultaneity windows."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import DetectorConfig
from .log_index import LogIndex


def timestamp_granularity(index: LogIndex) -> Tuple[bool, Optional[float]]:
    """Return (coarse, smallest positive within-case gap in seconds)."""
    gaps: list[float] = []
    for c in index.cases:
        t = c.times[~np.isnan(c.times)]
        if len(t) < 2:
            continue
        gaps.extend(np.diff(np.sort(t)).tolist())

    if not gaps:
        return False, None

    arr = np.asarray(gaps, dtype=float)
    positive = arr[arr > 0]
    min_pos = float(positive.min()) if len(positive) else None

    modulus = None
    for m in (86400.0, 3600.0, 60.0, 1.0):
        if len(positive) and np.all(np.isclose(positive % m, 0.0)):
            modulus = m
            break

    return bool(modulus is not None and modulus >= 3600.0), min_pos


def choose_windows(index: LogIndex, config: DetectorConfig) -> Tuple[float, float]:
    """Return (form-based window, collateral window) in seconds."""
    coarse, min_pos = timestamp_granularity(index)

    if not min_pos or coarse:
        form_window = 0.0
    else:
        form_window = min(min_pos, config.max_inferred_form_window_seconds)

    collateral_window = 1.0 if form_window == 0 else form_window * 10.0
    collateral_window = min(collateral_window, config.max_inferred_collateral_window_seconds)
    return form_window, max(collateral_window, form_window)
