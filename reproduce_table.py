#!/usr/bin/env python3
"""Reproduce the per-pattern QUELI detection table (tab:queli-per-pattern).

Each pattern is evaluated on its own labelled log, data/<dataset>/<pattern>.csv.
Detection is blind: the label column is removed before any detector runs and is
only used afterwards to compute precision, recall and F1.

Usage:
    python reproduce_table.py [--data data] [--out results]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from queli.detectors import detect_all
from queli.log_loading import load_log
from queli.scoring import score, truth_by_pattern

DATASETS = {"bpic11": "BPIC11", "bpic15": "BPIC15", "credit": "Credit", "pub": "Pub"}
PATTERN_FILES = {
    "FORM_BASED": ("form_based", "Form-based"),
    "COLLATERAL": ("collateral", "Collateral"),
    "SYNONYMOUS": ("synonymous", "Synonymous"),
    "HOMONYMOUS": ("homonymous", "Homonymous"),
}


def fmt(x: float) -> str:
    return "--" if math.isnan(x) else f"{x:.3f}"


def run(data_root: Path) -> pd.DataFrame:
    rows = []
    for folder, dataset in DATASETS.items():
        for pattern, (stem, pattern_name) in PATTERN_FILES.items():
            path = data_root / folder / f"{stem}.csv"
            print(f"{dataset:<7} {pattern_name:<11} {path}")
            df = load_log(path)
            flagged = detect_all(df)[pattern]
            truth = truth_by_pattern(df)[pattern]
            rows.append({"dataset": dataset, "pattern": pattern_name,
                         "file": path.as_posix(), **score(truth, flagged)})
    return pd.DataFrame(rows)


def write_latex(results: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{QUELI detection performance by dataset and imperfection pattern.}",
        r"  \label{tab:queli-per-pattern}",
        r"  \small",
        r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Dataset & Pattern & Prec. & Rec. & $F_1$ \\",
        r"    \midrule",
    ]
    for i, (dataset, group) in enumerate(results.groupby("dataset", sort=False)):
        if i:
            lines.append(r"    \addlinespace")
        for r in group.itertuples():
            lines.append(f"    {dataset} & {r.pattern} & {fmt(r.precision)} & "
                         f"{fmt(r.recall)} & {fmt(r.f1)} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_markdown(results: pd.DataFrame, path: Path) -> None:
    lines = [
        "# QUELI detection performance by dataset and imperfection pattern",
        "",
        "| Dataset | Pattern | Precision | Recall | F1 | Truth | Flagged | TP | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results.itertuples():
        lines.append(f"| {r.dataset} | {r.pattern} | {fmt(r.precision)} | {fmt(r.recall)} | "
                     f"{fmt(r.f1)} | {r.n_truth:,} | {r.n_flagged:,} | {r.tp:,} | "
                     f"{r.fp:,} | {r.fn:,} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    results = run(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(args.out / "per_pattern_metrics.csv", index=False)
    write_latex(results, args.out / "per_pattern_table.tex")
    write_markdown(results, args.out / "per_pattern_table.md")
    print(f"Wrote results to {args.out.resolve()}")


if __name__ == "__main__":
    main()
