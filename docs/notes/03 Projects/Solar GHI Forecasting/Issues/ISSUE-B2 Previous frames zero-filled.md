---
tags: [issue, deployment]
type: issue
status: fixed
severity: high
code: B2
---

# B2 — Previous frames zero-filled

> [!success] Status: **FIXED** · severity **high**

`predict.py` called `run_model(...)` without `prev_paths`, defaulting to `(None, None)`.
`load_satellite_inputs` then zero-filled frames t−1 and t−2. Training used three real frames.

## Fix

`fetch_frame_series()` added; `himawari_prev1.png` / `prev2.png` fetched and passed. Verified present and real (264 KB / 194 KB).
