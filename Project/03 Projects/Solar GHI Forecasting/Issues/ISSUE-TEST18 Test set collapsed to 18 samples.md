---
tags: [issue, data]
type: issue
status: open
severity: critical
code: TEST18
---

# TEST18 — Test set collapsed to 18 samples

> [!danger] Status: **OPEN** · severity **critical**

The dataset was extended to 2026-08-16 but the satellite archive stops at **2026-06-07**, so
1,931 rows have no image. The split is positional 70/15/15 over *all* rows, and
`valid_indices` filters to rows that have an image — so the image-less tail landed almost
entirely in test.

| Split | rows | with image |
| --- | ---: | ---: |
| train | 6,708 | 6,708 (100%) |
| val | 1,438 | 930 (64.7%) |
| **test** | **1,438** | **18 (1.3%)** |

The notebook printed it: `Test: 18 | Test batches: 1`.

The 18 rows span **2 unique days** (2026-05-15, 2026-06-07), mean GHI 318.5 vs 492.8 for the
full test set — unusually dark hours.

## Why it invalidates the comparison

The two result CSVs use different test sets. One RandomForest, scored both ways:

| Test set | n | RF avg MAE |
| --- | ---: | ---: |
| Full (what [[Baseline results]] reports) | 1,438 | **61.6** |
| The 18 imaged rows (what the fusion CSV reports) | 18 | **121.8** |

So the implied "tabular beats deep" gap of ~87 W/m² is mostly a test-set artefact.
Like-for-like it is ~27.

**Root cause:** `himawari_data.py` — the historical satellite collector — was deleted, so the
archive can't be backfilled. See [[Code map]].

## Fix

One line in notebook cell 19, *before* the split:

```python
df_clean = df_clean[df_clean["image_path"].notna()].reset_index(drop=True)
```

Gives train 5,359 · val 1,148 · **test 1,149 (~1,123 deep samples over 116 days, 7 months,
two monsoon seasons)** — better than the original single-season test set, so this also closes
[[ISSUE-E4 Single-season test set]].

Then add a guard:

```python
assert len(test_dataset) > 500, f"Test set collapsed to {len(test_dataset)} samples"
```

Also: restore `himawari_data.py` (`git show 0dc4dc7:himawari_data.py > himawari_data.py`) and
backfill 2026-02-03 → 2026-08-16. And score the tabular baselines on the **same**
`valid_indices` rows.
