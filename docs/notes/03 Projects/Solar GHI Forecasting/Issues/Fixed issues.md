---
tags: [moc, issue]
type: moc
---

# Fixed issues

Keep these — they are **evidence of engineering judgement**, and the B-series in particular is
the raw material for the best story in the defence. See [[Train-serve skew]].

| Issue | What it was |
| --- | --- |
| [[ISSUE-B1 No Image placeholder]] | Satellite fetch returned a black placeholder with HTTP 200 |
| [[ISSUE-B2 Previous frames zero-filled]] | t−1, t−2 frames never fetched |
| [[ISSUE-B3 Optical flow always zero]] | Two of four gate inputs pinned to zero |
| [[ISSUE-D1 load_model could not load concat checkpoint]] | Architecture mismatch on load |

Also done: notebook outputs are now saved; persistence baselines and skill scores added to both
result tables; diagnostics block added to `forecast_latest.json`; configurable architecture
presets and a no-lag ablation; the [[kt bias diagnostic]] built into `verify.py`.
