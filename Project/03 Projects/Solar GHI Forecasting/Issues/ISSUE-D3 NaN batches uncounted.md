---
tags: [issue, training]
type: issue
status: open
severity: medium
code: D3
---

# D3 — NaN batches uncounted

> [!bug] Status: **OPEN** · severity **medium**

`if torch.isnan(loss) or torch.isinf(loss): continue` under fp16 autocast, with `log σ` and
`1/σ²` in the loss — a known-fragile combination. Batches are silently dropped and never
counted. See [[Training setup]].

## Fix

Add a per-epoch counter. **If it exceeds ~1% of batches that is itself a result**, and the model should be retrained in bf16 or fp32.
