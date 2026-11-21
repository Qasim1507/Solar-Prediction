---
tags: [deployment, method, key-finding]
type: note
---

# The k_t bias diagnostic

**The test that MAE cannot do.** Built into `verify.py --report`.

## The idea

Absolute error scales with how much irradiance there is, so a forecast issued at 08:00 and
one at noon are not comparable. Divide by [[Clear-sky GHI]] and the sun drops out:

$$\text{k}_t\text{ bias} = k_{t,\text{forecast}} - k_{t,\text{actual}}$$

**If the bias is large but its spread is small, the model is not confused — it is _offset_.**
A constant offset has a findable cause. Scattered bias means the model genuinely cannot read
the sky. Completely different problems; MAE cannot tell them apart.

Decision rule: `sd / |mean|` < 0.5 → strongly systematic · < 1.0 → mostly systematic ·
> 1.0 → scatter dominates.

## What it found

| Run | mean k_t bias | spread |
| --- | ---: | ---: |
| 19 Aug — broken satellite | **−0.220** | 0.011 |
| 21 Aug — satellite fixed | **−0.180** | 0.054 |

Same pathology, barely improved. Subtracting that single constant from the 21 Aug forecast
takes MAE from **99.0 → 12.0 W/m²** — i.e. **88% of all error is one offset**.

## What that points at

A near-constant multiplicative under-bias that survived the satellite fix points hard at
**[[ISSUE-B5 Lookback window night contamination]]**: ~14 of the 24 inference window rows are
night with GHI ≈ 0, dragging the model's read of current conditions down by a roughly fixed
amount every run.

> [!tip] Track `kt_bias`, not MAE
> MAE bounces around with the weather. The bias doesn't.

> [!warning] n = 1 day on each side. The *pattern* is the signal, not the magnitude.
