---
tags: [results]
type: experiment
---

# Error analysis

## By cloud regime (LSTM-only / Fusion, W/m²)

| k_t bin | n | MAE |
| --- | ---: | --- |
| < 0.3 (overcast) | 67 | 116.9 / 155.0 |
| 0.3 – 0.5 | 177 | 95.7 / 112.9 |
| 0.5 – 0.7 | 241 | 91.9 / 105.1 |
| 0.7 – 0.85 | 269 | 84.1 / 92.8 |
| > 0.85 (clear) | 366 | 69.1 / 78.3 |

Error rises monotonically with cloudiness — 69% increase for LSTM-only, 98% for fusion.
Physically expected: [[Clear-sky index]] is nearly deterministic under clear sky and
near-chaotic under broken convection.

## By hour

MAE peaks at 12:00–13:00 (109.8 / 127.8) where mean GHI is 683–727, and falls to 54.8 / 57.4
at 16:00 where mean GHI is 497. **Normalised by the hourly mean the relative error is far
flatter** — this is a scale effect, not a time-of-day weakness.

## Worst failures

The three largest errors in the whole test set fall on **2 Dec 2025**, a persistently
overcast morning (k_t ≈ 0.1) where the model reverted toward clear-sky values — forecasting
662 against an actual 62.

**The one regime where the image branch pays:** large ramps. At |ΔGHI| > 200 W/m², fusion
(85.0) nearly matches LSTM-only (82.4) and both crush persistence (281.3).
