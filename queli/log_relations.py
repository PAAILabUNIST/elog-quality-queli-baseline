"""Relations mined from the log: simultaneous groups, conflict/parallel pairs, activity contexts."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .log_index import LogIndex
from .similarity import cosine_counter

CONTEXT_DIMENSIONS = ("preceding", "succeeding", "simultaneous")


def find_simultaneous_events(index: LogIndex, delta_seconds: float,
                             min_group_size: int = 2) -> List[List[int]]:
    """Maximal consecutive within-case groups whose timestamps span at most delta_seconds."""
    groups: List[List[int]] = []
    for c in index.cases:
        keep = ~np.isnan(c.times)
        idx = c.index[keep]
        times = c.times[keep]
        if len(idx) < min_group_size:
            continue

        start = 0
        for end in range(1, len(idx)):
            if times[end] - times[start] > delta_seconds:
                if end - start >= min_group_size:
                    groups.append(idx[start:end].tolist())
                start = end
        if len(idx) - start >= min_group_size:
            groups.append(idx[start:].tolist())
    return groups


def merge_overlapping_label_sets(label_sets: Iterable[Set[str]]) -> List[Set[str]]:
    """Union of all label sets that share at least one label."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for s in label_sets:
        for lbl in s:
            parent.setdefault(lbl, lbl)
        it = iter(s)
        first = next(it, None)
        if first is None:
            continue
        for lbl in it:
            union(first, lbl)

    clusters: Dict[str, Set[str]] = defaultdict(set)
    for lbl in parent:
        clusters[find(lbl)].add(lbl)
    return list(clusters.values())


def find_conflict_pairs(index: LogIndex, min_support: int = 1,
                        max_pairs: int = 200_000) -> List[Tuple[str, str]]:
    """Activity pairs that never occur together in the same case."""
    eligible = sorted(a for a, cs in index.cases_by_activity.items()
                      if len(cs) >= min_support)
    pairs: List[Tuple[str, str]] = []
    for a, b in combinations(eligible, 2):
        if not (index.cases_by_activity[a] & index.cases_by_activity[b]):
            pairs.append((a, b))
            if len(pairs) >= max_pairs:
                break
    return pairs


def find_parallel_event_pairs(index: LogIndex) -> List[Tuple[str, str]]:
    """Activity pairs seen in both orders (A before B in one case, B before A in another)."""
    order_counts: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for c in index.cases:
        first: Dict[str, int] = {}
        last: Dict[str, int] = {}
        for i, a in enumerate(c.acts):
            first.setdefault(a, i)
            last[a] = i
        for a, b in combinations(list(first), 2):
            if first[a] < last[b]:
                order_counts[(a, b)].add(c.case_id)
            if first[b] < last[a]:
                order_counts[(b, a)].add(c.case_id)

    pairs, seen = [], set()
    for (a, b), ab in order_counts.items():
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        if ab and order_counts.get((b, a)):
            pairs.append(key)
    return pairs


def activity_contexts(index: LogIndex, radius: int = 1) -> Dict[str, Dict[str, Counter]]:
    """Preceding, succeeding and same-timestamp neighbour counts for every activity."""
    ctx: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: {dim: Counter() for dim in CONTEXT_DIMENSIONS})

    for c in index.cases:
        acts, times, n = c.acts, c.times, len(c.acts)

        buckets: Dict[float, List[int]] = defaultdict(list)
        for i, t in enumerate(times):
            if not np.isnan(t):
                buckets[t].append(i)

        for i, a in enumerate(acts):
            e = ctx[a]
            for j in range(max(0, i - radius), i):
                e["preceding"][acts[j]] += 1
            for j in range(i + 1, min(n, i + radius + 1)):
                e["succeeding"][acts[j]] += 1
            t = times[i]
            if not np.isnan(t):
                for j in buckets[t]:
                    if j != i:
                        e["simultaneous"][acts[j]] += 1

    return dict(ctx)


def context_norms(ctx: Dict[str, Counter]) -> Dict[str, float]:
    return {k: math.sqrt(sum(v * v for v in c.values())) for k, c in ctx.items()}


def context_similarity(ca: Dict[str, Counter], cb: Dict[str, Counter],
                       na: Optional[Dict[str, float]] = None,
                       nb: Optional[Dict[str, float]] = None) -> float:
    """Mean cosine similarity over the context dimensions where either side has data."""
    total = 0.0
    n = 0
    for dim in CONTEXT_DIMENSIONS:
        if not ca[dim] and not cb[dim]:
            continue
        total += cosine_counter(ca[dim], cb[dim],
                                na[dim] if na else None, nb[dim] if nb else None)
        n += 1
    return total / n if n else 0.0
