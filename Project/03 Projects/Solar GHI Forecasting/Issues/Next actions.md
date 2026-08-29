---
tags: [project, todo]
type: note
---

# Next actions

- [ ] **Fix the split** — one line, before the 70/15/15 → [[ISSUE-TEST18 Test set collapsed to 18 samples]]
- [ ] **Fix the lookback window** — daylight filter → [[ISSUE-B5 Lookback window night contamination]]
- [ ] **Fix the gate cloud cover** — use the real column → [[ISSUE-B4 Gate cloud cover synthesised]]
- [ ] Restore `himawari_data.py`; backfill images Feb → Aug 2026
- [ ] Fix `daily_run.sh` expiry date (`20260816`) and restart the cron
- [ ] **Re-run the experiment grid** on the corrected split; score baselines on the same rows
- [ ] **3 seeds × 4 variants** → [[ISSUE-E1 Single seed]]
- [ ] Add CRPS + Diebold–Mariano → [[ISSUE-E3 No CRPS]] · [[ISSUE-E2 No Diebold-Mariano test]]
- [ ] Rename the CNN-only ablation → [[ISSUE-D6 CNN-only ablation mislabeled]]
- [ ] Run 5+ days of post-fix verification → before/after table for [[kt bias diagnostic]]
- [ ] Rewrite deck slide 33 as a [[Train-serve skew]] finding
- [ ] Add a slide for [[ISSUE-C1 Clear-sky time convention]]
- [ ] Promote [[Forecast skill score]] to a headline table

## If there is time before the retrain

- [ ] **CNN frozen throughout** — direct test of the [[Overfitting]] hypothesis, ~25 min GPU
- [ ] ImageNet-init control — tests whether [[SwimSeg pretraining]] helped at all

## Longer term

- [ ] Predict **k_t** instead of GHI — removes the diurnal scale from the target
- [ ] Continuous hourly index for targets/lags/frames/flow → [[ISSUE-C2 Overnight gap]]
- [ ] 256-dim gate vector instead of a scalar → [[Gate collapse]]
- [ ] Infrared channels; more years of imagery
