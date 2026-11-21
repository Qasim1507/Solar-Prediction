---
tags: [concept, metric]
type: note
---

# PICP and interval width

**PICP** — Prediction Interval Coverage Probability. The fraction of true values that fell
inside the predicted 90% interval. You want ≈ 90%.

> [!danger] PICP alone is meaningless
> A forecast of "somewhere between 0 and 1400 W/m²" scores 100%. **Always report mean
> interval width alongside it.**

The old broken deployment posted 100% coverage with a band ~540 W/m² wide — about 74% of
the mean signal. That is not calibration, that is a shrug.

**Proper reporting** — coverage *and* width at several nominal levels, plus a
standardised-residual test: z = (y − μ)/σ should be standard normal.

See [[Calibration results]] for the numbers.

Still missing: **CRPS**, the proper scoring rule for probabilistic forecasts. See
[[ISSUE-E3 No CRPS]].
