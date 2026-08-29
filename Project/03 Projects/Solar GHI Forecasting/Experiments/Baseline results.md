---
tags: [results]
type: experiment
n_test: 1437
status: valid
---

# Baseline results — the strongest thing you have

`results/baseline_ghi_comparison.csv` · **n = 1,437** · chronological hold-out

| Model | t+1h | t+2h | t+3h |
| --- | ---: | ---: | ---: |
| [[Persistence]] | 157.7 | 265.5 | 335.4 |
| [[Smart persistence]] | 73.5 | 117.2 | 146.6 |
| Linear Regression | 64.3 | 94.5 | 115.5 |
| **Random Forest** | **48.1** | **64.7** | **72.3** |
| LightGBM | 48.5 | 64.8 | 72.9 |

MAE in W/m².

## [[Forecast skill score]] vs smart persistence

**+35% / +45% / +51%** — and skill *grows* with horizon, because persistence decays faster
than learned dynamics. That is physically sensible and it is a **positive result**.

> [!tip] Lead with this
> It is currently buried in a baselines table. It deserves a slide of its own.

> [!caution] One caveat to state
> These are better than the earlier run (63.5 / 82.5 / 90.0) partly because the test period
> moved to a genuinely easier stretch — mean hour-to-hour |Δk_t| of 0.163 versus 0.179, about
> 9% less volatile.

Compare: [[Ablation results]]
