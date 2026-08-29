---
tags: [issue, data]
type: issue
status: open
severity: high
code: C2
---

# C2 — Overnight gap

> [!bug] Status: **OPEN** · severity **high**

Because only 08:00–17:00 is kept and targets are built with a positional `df.ghi.shift(-h)`,
a "t+3h" target issued at 16:00 is **not 19:00 — it is 09:00 the next morning.**

Affects **10.0% / 20.1% / 30.1%** of t+1/2/3h targets.

## Measured impact

| | MAE as reported | MAE, same-day rows only |
| --- | ---: | ---: |
| t+1h | 63.5 | **68.9** |
| t+2h | 82.5 | **93.5** |
| t+3h | 90.0 | **104.2** |

Headline MAEs are **8–16% optimistic**. Affects every model identically, so the ranking holds.

## The part the deck misses

The same positional shift corrupts `ghi_lag1/2/3`, the **t−1 and t−2 image frames**, and the
**[[Optical flow]] pairs**. For 10–20% of samples the "previous hour's frame" is *yesterday
evening's*, and the motion vector between them is noise.

**That is a better explanation for the image branch underperforming than "not enough data"** —
a fifth of the gate's conditioning signal was garbage.

## Fix

Reindex onto a continuous hourly `DatetimeIndex`; build targets, lags, frame offsets and flow pairs by **timestamp difference**, not row position; drop samples whose window or horizon crosses the overnight gap. Expect MAEs to rise — say so.
