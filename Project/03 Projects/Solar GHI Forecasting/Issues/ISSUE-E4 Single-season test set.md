---
tags: [issue, evaluation]
type: issue
status: open
severity: medium
code: E4
---

# E4 — Single-season test set

> [!bug] Status: **OPEN** · severity **medium**

The earlier valid test set ran 11 Oct 2025 → 2 Feb 2026 — **NE monsoon only**. Not listed in
the limitations. See [[Limitations]].

## Fix

Closing [[ISSUE-TEST18 Test set collapsed to 18 samples]] fixes this too — the imaged-rows split spans 116 days across 7 months and two seasons. Otherwise: blocked rolling-origin CV, 3–4 folds.
