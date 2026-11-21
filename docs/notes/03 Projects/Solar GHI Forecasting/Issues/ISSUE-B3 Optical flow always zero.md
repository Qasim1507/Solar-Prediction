---
tags: [issue, deployment]
type: issue
status: fixed
severity: high
code: B3
---

# B3 — Optical flow always zero

> [!success] Status: **FIXED** · severity **high**

Consequence of [[ISSUE-B2 Previous frames zero-filled]]:
`_compute_optical_flow(zeros, frame_t)` returns ~0, so **two of the four [[Physics gate]]
inputs were pinned to zero** on every live run.

## Fix

Resolved by B2. Consider asserting flow ≠ 0 when both frames are valid.
