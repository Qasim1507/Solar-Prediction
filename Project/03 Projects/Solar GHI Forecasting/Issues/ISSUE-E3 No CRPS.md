---
tags: [issue, evaluation]
type: issue
status: open
severity: medium
code: E3
---

# E3 — No CRPS

> [!bug] Status: **OPEN** · severity **medium**

CRPS is the proper scoring rule for probabilistic forecasts and the natural companion to
[[PICP and interval width]]. Not computed. Conceded in deck A8.

## Fix

`properscoring.crps_gaussian(y, mu, sigma)` — μ and σ are already saved. ~20 minutes.
