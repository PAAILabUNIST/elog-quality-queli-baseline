"""Set, count-vector and label-text similarity measures."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Optional, Set

STOPWORDS = {"the", "a", "an", "of", "to", "for", "and", "or",
             "in", "on", "at", "by", "with", "from"}


def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine_counter(a: Counter, b: Counter,
                   na: Optional[float] = None, nb: Optional[float] = None) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b, na, nb = b, a, nb, na
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    if not dot:
        return 0.0
    na = na if na is not None else math.sqrt(sum(v * v for v in a.values()))
    nb = nb if nb is not None else math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def label_tokens(s: str) -> Set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(s).lower())
    return {t for t in tokens if t not in STOPWORDS}
