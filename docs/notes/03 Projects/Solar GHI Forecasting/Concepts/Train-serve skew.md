---
tags: [concept, mlops]
type: note
---

# Train/serve skew

When the inputs a model receives **in production** differ in distribution from the inputs it
saw **in training** — even though the code "works" and throws no errors.

The most dangerous class of ML bug, because every test passes and the model still silently
produces garbage.

## Found in this project — five instances

| # | Divergence | Status |
| --- | --- | --- |
| [[ISSUE-B1 No Image placeholder]] | Satellite fetch returned a black placeholder with HTTP 200 | ✅ fixed |
| [[ISSUE-B2 Previous frames zero-filled]] | t−1, t−2 frames never fetched | ✅ fixed |
| [[ISSUE-B3 Optical flow always zero]] | Consequence of B2 | ✅ fixed |
| [[ISSUE-B4 Gate cloud cover synthesised]] | Real ERA5 value in training, `1 − k_t` at serve | ❌ open |
| [[ISSUE-B5 Lookback window night contamination]] | 24 daylight rows in training, 24 clock hours at serve | ❌ open |

**How they were caught:** the [[Verification loop]]. That is the whole argument for building
one. See [[kt bias diagnostic]] for the tool that measures the damage.
