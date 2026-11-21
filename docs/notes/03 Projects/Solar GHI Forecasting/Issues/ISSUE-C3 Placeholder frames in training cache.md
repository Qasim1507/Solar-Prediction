---
tags: [issue, data]
type: issue
status: open
severity: low
code: C3
---

# C3 — Placeholder frames in training cache

> [!bug] Status: **OPEN** · severity **low**

23 of 7,631 cached training tensors carry the same "No Image" fingerprint as
[[ISSUE-B1 No Image placeholder]] (`mean = 0.00217`, `frac_black = 0.9887`). 0.3% of frames,
so training impact is negligible.

## Fix

Filter them in `save_images_as_npy` and state that you did — it's the same unguarded failure that killed the deployment.
