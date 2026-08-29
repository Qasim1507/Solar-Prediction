---
tags: [log]
type: moc
---

# Daily log

Running log for the thesis. Newest first. Full daily notes live in `02 Daily`.

## 2026-08-24
- Built this vault.

## 2026-08-21
- Re-audit of commit `56b7371`. **Found [[ISSUE-TEST18 Test set collapsed to 18 samples]]** —
  deep models scored on 18 samples, baselines on 1,437.
- Rewrote `verify.py` with the [[kt bias diagnostic]].
- First post-fix verification: k_t bias **−0.180** vs **−0.220** pre-fix. Same pathology.
  Removing that one constant takes MAE 99.0 → 12.0.

## 2026-08-20
- Fix round landed: [[ISSUE-B1 No Image placeholder]], [[ISSUE-B2 Previous frames zero-filled]],
  [[ISSUE-B3 Optical flow always zero]], [[ISSUE-D1 load_model could not load concat checkpoint]] closed.
- Full retrain; added `small` preset and no-lag ablation.

## 2026-08-19
- First audit. Live deployment found to have **negative** forecast/actual correlation.
