---
tags: [issue, deployment]
type: issue
status: open
severity: high
code: B4
---

# B4 — Gate cloud cover synthesised

> [!bug] Status: **OPEN** · severity **high**

The [[Physics gate]]'s second input is a **different variable** at train and serve time.

**Training** (notebook cell 19):
```python
gate_features = [cr_arr[t], cc_arr[t]/100.0, flow[0], flow[1]]
#                            ^^^^^^^^^^^^^^ real ERA5 cloud fraction
```

**Inference** (`model.py: compute_gate_features`):
```python
cc = float(np.clip((1 - cr) * 100, 0, 100))   # SYNTHESISED — this is just 1 − k_t
```

At serve time input #2 is a deterministic function of input #1, perfectly anti-correlated
with it. Deck slide 18 says "cc = ERA5 cloud cover, scaled to [0,1]" — true in training, false
in deployment.

Part of [[Train-serve skew]].

## Fix

`fetch_recent_df` already pulls `cloud_cover`. Use it:

```python
row = df[df["timestamp"] <= ref_naive].tail(1)
cc  = float(row["cloud_cover"].iloc[0]) / 100.0
```

Add a unit test asserting serve-time gate features equal training-time values for a known row.
