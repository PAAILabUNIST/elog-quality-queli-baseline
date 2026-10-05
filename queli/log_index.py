"""Per-case index of the log, built once and shared by every detector."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np
import pandas as pd


@dataclass
class CaseView:
    case_id: str
    index: np.ndarray
    acts: List[str]
    times: np.ndarray


@dataclass
class LogIndex:
    cases: List[CaseView]
    activity_counts: Counter = field(default_factory=Counter)
    cases_by_activity: Dict[str, Set[str]] = field(default_factory=dict)


def build_log_index(df: pd.DataFrame) -> LogIndex:
    cases: List[CaseView] = []
    counts: Counter = Counter()
    cases_by_activity: Dict[str, Set[str]] = defaultdict(set)

    ts = df["_timestamp"].to_numpy(dtype="datetime64[ns]")
    ts_sec = ts.astype("float64") / 1e9

    for case_id, case in df.groupby("case_id", sort=False):
        idx = case.index.to_numpy()
        acts = case["activity"].astype(str).tolist()
        cases.append(CaseView(str(case_id), idx, acts, ts_sec[idx]))
        counts.update(acts)
        for a in set(acts):
            cases_by_activity[a].add(str(case_id))

    return LogIndex(cases, counts, dict(cases_by_activity))
