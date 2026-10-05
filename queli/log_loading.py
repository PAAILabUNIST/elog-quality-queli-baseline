"""Reading a labelled event-log CSV into a canonical DataFrame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_NAMES = {
    "Case": "case_id",
    "Activity": "activity",
    "Timestamp": "timestamp",
    "Resource": "resource",
    "label": "label",
}


def load_log(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=COLUMN_NAMES)

    missing = [c for c in ("case_id", "activity", "timestamp") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")

    df["event_id"] = np.arange(len(df), dtype=np.int64)
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    df["case_id"] = df["case_id"].astype(str)
    df["activity"] = df["activity"].astype(str)
    df["_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df.sort_values(["case_id", "_timestamp", "_row_id"],
                          kind="mergesort", na_position="last").reset_index(drop=True)
