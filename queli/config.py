"""Pattern names and detector parameters."""

from __future__ import annotations

from dataclasses import dataclass

PATTERNS = ["FORM_BASED", "COLLATERAL", "SYNONYMOUS", "HOMONYMOUS"]


@dataclass
class DetectorConfig:
    max_inferred_form_window_seconds: float = 60.0
    max_inferred_collateral_window_seconds: float = 3600.0

    min_group_occurrences: int = 2
    min_group_size: int = 2

    context_radius: int = 1

    synonym_context_similarity: float = 0.60
    synonym_lexical_similarity: float = 0.20
    synonym_min_support: int = 2

    homonym_context_diversity: float = 0.50
    homonym_min_occurrences: int = 2
    homonym_pair_sample: int = 60

    max_conflict_pairs: int = 200_000
    random_seed: int = 0
