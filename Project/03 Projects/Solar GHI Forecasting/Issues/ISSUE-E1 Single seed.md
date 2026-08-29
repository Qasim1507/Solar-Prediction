---
tags: [issue, evaluation]
type: issue
status: open
severity: critical
code: E1
---

# E1 — Single seed

> [!danger] Status: **OPEN** · severity **critical**

Every conclusion rests on **one run per variant**. The headline claim is a ~12–13 W/m² gap
between LSTM-only and the fusion model — and **the ablation ordering inverted between two
runs**, which is direct evidence that run-to-run variance is comparable to the effect being
claimed. See [[Ablation results]].

## Fix

3 seeds × 4 variants ≈ **1 hour of GPU**. Report mean ± sd.

**Highest value-per-minute item in the entire project.** It is the difference between a
defensible conclusion and a coin flip.

Fallback if unrun: *"One seed. I would not defend the ordering within the deep variants on
that basis; the claim I defend is the direction."* → [[Three hard questions]]
