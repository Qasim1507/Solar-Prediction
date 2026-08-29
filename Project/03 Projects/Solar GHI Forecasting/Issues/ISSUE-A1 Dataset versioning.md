---
tags: [issue, data]
type: issue
status: partial
severity: high
code: A1
---

# A1 — Dataset versioning

> [!bug] Status: **PARTIAL** · severity **high**

There were two different `combined_dataset.csv` files with different row counts, and the one
the notebook reads (`data/`) is **gitignored**. Re-running the notebook trained on different
data with a different split, so no reported number reproduced.

Both copies are now identical (9,590 rows) — but `data/` is still gitignored, so the file that
matters is untracked and can silently diverge again. See [[Dataset]].

## Fix

Freeze a versioned copy, commit it (add `!data/combined_dataset*.csv` to `.gitignore`), point
the notebook and `predict.py` / `verify.py` at one shared constant, record the dataset SHA-256
in `train_stats.json`, and assert row count + GHI mean at the top of the notebook.
