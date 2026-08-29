---
tags: [issue, deployment]
type: issue
status: open
severity: critical
code: B5
---

# B5 — Lookback window night contamination

> [!danger] Status: **OPEN** · severity **critical**

**Most likely cause of the remaining forecast bias.** See [[kt bias diagnostic]].

- **Training:** 24 consecutive rows of a dataframe filtered to 08:00–17:00 → 24 *daylight*
  steps ≈ 2.4 calendar days. **Zero night rows, ever.**
- **Inference:** `extend_with_recent()` appends every hour Open-Meteo returns (`past_days=3`,
  no daylight filter); `build_lookback_window` takes `.tail(24)` → 24 *clock* hours.

Confirmed by the system's own diagnostics: `lookback_ends: 2026-08-21 08:00` from an 08:00
issue means the window spans 2026-08-20 09:00 → 2026-08-21 08:00 —
**14 of 24 rows (58%) are night hours the model never saw in training.**

Part of [[Train-serve skew]].

## Fix

One line in `model.py: build_lookback_window`:

```python
past = df[(df["timestamp"] <= ref_naive) &
          (df["timestamp"].dt.hour.between(8, 17))].tail(WINDOW_SIZE)
```

Add a runtime assertion that all 24 rows fall in 08:00–17:00, and log the window start.

> [!important] Do this **before** accumulating more verification days, or the "after" data
> measures a system that is still half-broken.
