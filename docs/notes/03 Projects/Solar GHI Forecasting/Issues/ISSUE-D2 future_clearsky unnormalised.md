---
tags: [issue, code]
type: issue
status: open
severity: low
code: D2
---

# D2 — future_clearsky unnormalised

> [!bug] Status: **OPEN** · severity **low**

Raw 0–1000 W/m² values concatenated onto a 256-dim z-scored vector before `enrich`, in both
training and inference. Consistent, so not a skew bug — but three inputs dominate that layer's
scale, and it contradicts the "per-column z-scoring" claim. See [[Prediction head]].

## Fix

Normalise with clear-sky mean/std, persist to `train_stats.json`, retrain as v3.
