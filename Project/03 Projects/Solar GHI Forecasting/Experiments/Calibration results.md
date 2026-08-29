---
tags: [results]
type: experiment
status: success
---

# Calibration results — the clear success

The probabilistic head worked. This is the one unambiguously positive deep-model result.

| Nominal | LSTM-only coverage | width | Fusion coverage | width |
| --- | ---: | ---: | ---: | ---: |
| 50% | 57.7% | 163 | 55.5% | 175 |
| 80% | 84.7% | 310 | 82.6% | 333 |
| **90%** | **91.8%** | 398 | **90.2%** | 427 |
| 95% | 95.0% | 474 | 93.7% | 509 |

Widths in W/m², over 1,120 samples × 3 horizons.

## Standardised-residual test

z = (y − μ)/σ should be standard normal if calibrated.

- LSTM-only: mean **−0.003**, sd **0.931**
- Fusion: mean **−0.075**, sd **0.991**

Both means essentially zero (no systematic bias), both sds within 7% of one.

## Reading

Both over-cover at 50% and 80%, and land almost exactly on nominal at 90% and 95%. The
residuals are **more peaked than a Gaussian** — too many small errors for the fitted σ — so
narrow intervals over-cover.

> [!important] Report width, always
> Coverage alone is meaningless. See [[PICP and interval width]].

Missing: **CRPS** → [[ISSUE-E3 No CRPS]]
