---
tags: [issue, code]
type: issue
status: fixed
severity: medium
code: D1
---

# D1 — load_model could not load concat checkpoint

> [!success] Status: **FIXED** · severity **medium**

`load_model` always constructed the model with `ablation=None`, giving
`enrich[0] = Linear(259, 256)`. The concat checkpoint has `Linear(515, 256)`, so strict
`load_state_dict` raised.

## Fix

`.config.json` sidecars now written for all six checkpoints, with shape-inference fallback.
