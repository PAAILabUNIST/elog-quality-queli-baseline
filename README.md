# QUELI per-pattern detection results

This repository contains the code and data that produce the table
**"QUELI detection performance by dataset and imperfection pattern"**
(`tab:queli-per-pattern`).

The detectors follow the QUELI approach of Andrews, Suriadi, Ouyang & Poppe (2018),
*Towards Event Log Querying for Data Quality: Let's Start with Detecting Log Imperfections*
(OTM 2018, LNCS 11229). Four imperfection patterns are covered: form-based event capture,
collateral events, synonymous labels and homonymous labels.

## Results

| Dataset | Pattern    | Prec. | Rec.  | F1    |
|---------|------------|------:|------:|------:|
| BPIC11  | Form-based | 0.008 | 0.999 | 0.015 |
| BPIC11  | Collateral | 0.000 | 0.002 | 0.000 |
| BPIC11  | Synonymous | 0.059 | 0.086 | 0.070 |
| BPIC11  | Homonymous | 0.005 | 1.000 | 0.011 |
| BPIC15  | Form-based | 0.034 | 1.000 | 0.066 |
| BPIC15  | Collateral | 0.000 | 0.004 | 0.000 |
| BPIC15  | Synonymous | 0.699 | 0.544 | 0.612 |
| BPIC15  | Homonymous | 0.115 | 1.000 | 0.206 |
| Credit  | Form-based | 1.000 | 1.000 | 1.000 |
| Credit  | Collateral | 0.500 | 0.001 | 0.001 |
| Credit  | Synonymous | 0.165 | 0.194 | 0.178 |
| Credit  | Homonymous | 0.212 | 1.000 | 0.350 |
| Pub     | Form-based | 1.000 | 1.000 | 1.000 |
| Pub     | Collateral | 0.333 | 0.001 | 0.001 |
| Pub     | Synonymous | 0.113 | 0.545 | 0.188 |
| Pub     | Homonymous | 1.000 | 0.583 | 0.736 |

The underlying counts (true positives, false positives, false negatives) are in
[results/per_pattern_metrics.csv](results/per_pattern_metrics.csv).

## Reproducing the table

Requires Python 3.9 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce_table.py
```

The script runs all 16 logs and writes:

| File | Contents |
|---|---|
| `results/per_pattern_table.tex` | The LaTeX table, ready to paste into the paper |
| `results/per_pattern_table.md` | The same table with TP / FP / FN counts |
| `results/per_pattern_metrics.csv` | Raw counts and unrounded scores for each row |

Use `--data` and `--out` to change the input and output folders.

## Evaluation protocol

- **One log per pattern.** Each row of the table is scored on a log in which only that
  pattern was injected: `data/<dataset>/<pattern>.csv`.
- **Blind detection.** The `label` column is dropped before any detector runs. It is
  read afterwards only to score the output.
- **Event-level scoring.** For the pattern being evaluated, an event is a true positive
  if the detector flags it and its label names that pattern. Precision, recall and F1
  are computed over event IDs. A value is shown as `--` when it is undefined
  (for example, when nothing was flagged).
- **Form-based detection is always run.** It is not skipped for logs with coarse,
  date-level timestamps such as BPIC11. This means the detector never uses the file name
  to decide whether to look for form-based capture.

## Data

The `data/` folder holds four event logs, each with four injected-imperfection
variants. All variants were generated with random seed 52 and an injection ratio of 0.3.

```
data/
  bpic11/   BPI Challenge 2011
  bpic15/   BPI Challenge 2015
  credit/   Credit log
  pub/      Pub log
    form_based.csv  collateral.csv  synonymous.csv  homonymous.csv
```

Every file has the columns `Case, Activity, Timestamp, Resource, label`. The `label` column
is empty for clean events. For injected events, it names the pattern, for example
`FORMBASED(2024-08-08 19:30:00)`.

## Code layout

| Module | Purpose |
|---|---|
| `reproduce_table.py` | Entry point: runs every log, scores the results and writes the tables |
| `queli/detectors.py` | The four pattern detectors and `detect_all()` |
| `queli/log_relations.py` | Simultaneous-event groups, conflict and parallel activity pairs, activity contexts |
| `queli/time_windows.py` | Timestamp granularity check and the inferred simultaneity windows |
| `queli/log_index.py` | Per-case index of the log, shared by all detectors |
| `queli/similarity.py` | Jaccard, cosine and label-token similarity |
| `queli/log_loading.py` | Reads a CSV log into canonical columns |
| `queli/scoring.py` | Parses ground-truth labels and computes precision, recall and F1 |
| `queli/config.py` | Pattern names and detector parameters |

### How each pattern is detected

- **Form-based / collateral** (QUELI Algorithm 1): finds groups of events within a case
  whose timestamps fall inside a small window, merges groups that share activity labels,
  and flags label clusters that recur in at least two groups. The form-based window is
  inferred from the smallest within-case time gap, capped at 60 s, and set to 0 s for
  coarse timestamps. The collateral window is 10 times larger, capped at 1 h.
- **Synonymous**: compares activity pairs that never co-occur in a case. A pair is
  flagged when its preceding, succeeding and simultaneous contexts are similar (cosine
  ≥ 0.60), or when a combined context and label-text score passes the same threshold.
- **Homonymous**: takes activities that appear in both orders relative to another
  activity and repeat within a case, then flags them when their surrounding contexts
  are diverse (mean pairwise dissimilarity ≥ 0.50).

All parameters are listed in `queli/config.py`.
