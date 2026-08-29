---
tags: [issue, deployment]
type: issue
status: fixed
severity: critical
code: B1
---

# B1 — No Image placeholder

> [!success] Status: **FIXED** · severity **critical**

NICT serves a black **"No Image" placeholder** with **HTTP 200**, so the
`if resp.status_code != 200` guard never fired.

Fingerprint: `mean pixel = 0.00217`, `fraction black = 0.9887`. A real daylight frame has mean
brightness 0.16–0.26. **23 of 7,631 cached training frames** carry the identical signature —
and in production it appeared to hit every run.

Effect: 14.4M of 15.5M parameters fed a constant black square on every deployed forecast.

## Fix

`validate_tile()` added to `current_data.py` with brightness/std checks and sun-angle
awareness; multi-source fallback (`nict,slider,gk2a,jaxa`); 10-minute backward retry.

**Verified:** `himawari_current.png` is now 292 KB, mean brightness 64.1, 0% black.
