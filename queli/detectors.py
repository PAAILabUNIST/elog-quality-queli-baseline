"""Blind detectors for the four imperfection patterns."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from .config import PATTERNS, DetectorConfig
from .log_index import LogIndex, build_log_index
from .log_relations import (
    activity_contexts,
    context_norms,
    context_similarity,
    find_conflict_pairs,
    find_parallel_event_pairs,
    find_simultaneous_events,
    merge_overlapping_label_sets,
)
from .similarity import jaccard, label_tokens
from .time_windows import choose_windows


def detect_simultaneous_pattern(df: pd.DataFrame, index: LogIndex, delta_seconds: float,
                                config: DetectorConfig) -> Set[Any]:
    """Algorithm 1 (form-based capture / collateral events) for a given window."""
    groups = find_simultaneous_events(index, delta_seconds, config.min_group_size)
    if not groups:
        return set()

    act = df["activity"].astype(str).to_numpy()
    label_sets = [set(act[g]) for g in groups]
    merged = merge_overlapping_label_sets(label_sets)

    cluster_of: Dict[str, int] = {}
    for cid, cluster in enumerate(merged):
        for lbl in cluster:
            cluster_of[lbl] = cid

    support: Dict[int, List[List[int]]] = defaultdict(list)
    for g, labels in zip(groups, label_sets):
        cids = {cluster_of[l] for l in labels}
        if len(cids) == 1:
            support[cids.pop()].append(g)

    flagged_rows: Set[int] = set()
    for cid in range(len(merged)):
        support_groups = support.get(cid, [])
        if len(support_groups) >= config.min_group_occurrences:
            flagged_rows.update(r for g in support_groups for r in g)

    return set(df.loc[sorted(flagged_rows), "event_id"])


def detect_synonymous(df: pd.DataFrame, index: LogIndex, config: DetectorConfig) -> Set[Any]:
    conflict_pairs = find_conflict_pairs(index, config.synonym_min_support,
                                         config.max_conflict_pairs)
    if not conflict_pairs:
        return set()

    needed = {a for pair in conflict_pairs for a in pair
              if index.activity_counts[a] >= config.synonym_min_support}
    if not needed:
        return set()

    contexts = {a: c for a, c in activity_contexts(index, config.context_radius).items()
                if a in needed}
    norms = {a: context_norms(c) for a, c in contexts.items()}
    tokens = {a: label_tokens(a) for a in contexts}

    flagged_activities: Set[str] = set()
    for a, b in conflict_pairs:
        if a not in contexts or b not in contexts:
            continue
        ctx_sim = context_similarity(contexts[a], contexts[b], norms[a], norms[b])
        lex_sim = jaccard(tokens[a], tokens[b])
        score = 0.85 * ctx_sim + 0.15 * lex_sim
        if (ctx_sim >= config.synonym_context_similarity or
                (score >= config.synonym_context_similarity and
                 lex_sim >= config.synonym_lexical_similarity)):
            flagged_activities.update((a, b))

    mask = df["activity"].astype(str).isin(flagged_activities)
    return set(df.loc[mask, "event_id"])


def detect_homonymous(df: pd.DataFrame, index: LogIndex, config: DetectorConfig) -> Set[Any]:
    in_parallel = {a for pair in find_parallel_event_pairs(index) for a in pair}
    if not in_parallel:
        return set()

    occ: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    r = config.context_radius
    for c in index.cases:
        n = len(c.acts)
        positions: Dict[str, List[int]] = defaultdict(list)
        for i, a in enumerate(c.acts):
            if a in in_parallel:
                positions[a].append(i)

        for a, pos_list in positions.items():
            if len(pos_list) < config.homonym_min_occurrences:
                continue
            for pos in pos_list:
                occ[a].append({
                    "row_index": int(c.index[pos]),
                    "prev": set(c.acts[max(0, pos - r):pos]),
                    "next": set(c.acts[pos + 1:min(n, pos + r + 1)]),
                })

    rng = random.Random(config.random_seed)
    flagged_rows: Set[int] = set()

    for activity in sorted(occ):
        records = occ[activity]
        signatures = Counter(
            (tuple(sorted(x["prev"])), tuple(sorted(x["next"]))) for x in records)
        if len(signatures) < 2:
            continue

        sample = records
        if len(records) > config.homonym_pair_sample:
            sample = rng.sample(records, config.homonym_pair_sample)

        pair_dissim = [
            1.0 - 0.5 * (jaccard(x["prev"], y["prev"]) + jaccard(x["next"], y["next"]))
            for x, y in combinations(sample, 2)
        ]
        diversity = float(np.mean(pair_dissim)) if pair_dissim else 0.0
        if diversity >= config.homonym_context_diversity:
            flagged_rows.update(x["row_index"] for x in records)

    return set(df.loc[sorted(flagged_rows), "event_id"])


def detect_all(df: pd.DataFrame, config: DetectorConfig | None = None) -> Dict[str, Set[Any]]:
    """Run every detector on the log with its label column removed; return flagged event ids per pattern."""
    config = config or DetectorConfig()
    work = df.drop(columns=["label"], errors="ignore")
    work = work.sort_values(["case_id", "_timestamp", "_row_id"],
                            kind="mergesort", na_position="last").reset_index(drop=True)

    index = build_log_index(work)
    form_window, collateral_window = choose_windows(index, config)

    detections = {
        "FORM_BASED": detect_simultaneous_pattern(work, index, form_window, config),
        "COLLATERAL": detect_simultaneous_pattern(work, index, collateral_window, config),
        "SYNONYMOUS": detect_synonymous(work, index, config),
        "HOMONYMOUS": detect_homonymous(work, index, config),
    }
    return {p: detections[p] for p in PATTERNS}
