"""Ground-truth parsing and per-pattern precision / recall / F1."""

from __future__ import annotations

import re
from typing import Any, Dict, Set

import pandas as pd

from .config import PATTERNS

_PATTERN_BY_TOKEN = {re.sub(r"[^A-Z0-9]", "", p): p for p in PATTERNS}


def _pattern_name(token: str) -> str | None:
    return _PATTERN_BY_TOKEN.get(re.sub(r"[^A-Z0-9]", "", token.upper()))


def parse_label(label: Any) -> Set[str]:
    """Patterns named in a label such as 'FORMBASED(2024-08-08 19:30:00)' or 'SYNONYMOUS|HOMONYMOUS'."""
    if pd.isna(label) or not str(label).strip():
        return set()
    text = str(label).strip()

    matches = re.findall(r"([A-Za-z0-9_\- ]+)\s*\(([^)]*)\)", text)
    tokens = [p for p, _ in matches] if matches else re.split(r"[|;,]+", text)
    return {name for name in (_pattern_name(t.strip()) for t in tokens if t.strip()) if name}


def truth_by_pattern(df: pd.DataFrame) -> Dict[str, Set[Any]]:
    result: Dict[str, Set[Any]] = {p: set() for p in PATTERNS}
    labels = df["label"]
    nonempty = labels.notna() & (labels.astype(str).str.strip() != "")
    sub = df.loc[nonempty, ["event_id", "label"]]
    for label_value, group in sub.groupby(sub["label"].astype(str), sort=False):
        for p in parse_label(label_value):
            result[p] |= set(group["event_id"])
    return result


def score(truth: Set[Any], flagged: Set[Any]) -> Dict[str, Any]:
    tp = len(flagged & truth)
    fp = len(flagged - truth)
    fn = len(truth - flagged)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else float("nan")
    return {"n_truth": len(truth), "n_flagged": len(flagged), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}
