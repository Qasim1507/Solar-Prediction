---
tags: [issue, data]
type: issue
status: open
severity: high
code: C1
---

# C1 — Clear-sky time convention

> [!bug] Status: **OPEN** · severity **high**

Empirical **maximum** [[Clear-sky index]] per hour across the whole dataset:

| SGT | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max k_t | 0.69 | 0.78 | 0.88 | 0.94 | 0.99 | 1.02 | 1.07 | 1.12 | 1.21 | **1.43** |

A physically meaningful k_t should cap near 1.0–1.1 at *every* hour. This monotone
morning→evening drift is the signature of a **timestamp convention mismatch**: Open-Meteo
labels each hourly GHI with the **end** of its averaging window; `pvlib` returns the
**instantaneous** value.

**This was already diagnosed once** — `compute_clearsky_hour_mean()` exists in `model.py`
(commit `97bb429`) — but was only applied to the [[Physics clamp]]. The dataset's
`ghi_clearsky` column, and therefore `clearsky_ratio` (feature #1 **and** gate input #1) and
`future_clearsky`, still use the instantaneous value.

## Fix

In `historical_data.py` / `combined_dataset.py`:

```python
times = pd.date_range(t - pd.Timedelta(minutes=50), t, freq="10min")
ghi_clearsky = loc.get_clearsky(pd.DatetimeIndex(times))["ghi"].mean()
```

Rebuild the dataset as a **new file**, verify max k_t is now flat across hours, retrain as v3.
Report before/after for deep models *and* LightGBM.

**This is a genuine physics correction and deserves its own slide.**
