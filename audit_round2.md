# Re-audit — commit `56b7371` (after the fix round)

Compared against my previous audit at `0dc4dc7`. Ten commits of work landed.

---

## Bottom line

**The live pipeline is genuinely fixed** — I verified the satellite fetch is now pulling
real imagery, not the placeholder. That was the worst defect and it's gone.

**But the retrain broke the evaluation.** Your deep models were scored on **18 test
samples spanning 2 days**, while your tabular baselines were scored on 1,437. Every
number in `results/fusion_ghi_comparison.csv` is meaningless, and the deck's central
"tabular beats deep" comparison is now an artefact of two different test sets.

Your notebook printed the warning and it wasn't caught:

```
cell  6 (baselines):  Train: 6701 | Val: 1436 | Test: 1437
cell 19 (deep):       Train: 6675 | Val:  914 | Test: 18
                      Train batches: 53 | Val: 8 | Test: 1
```

**Do not present the current `fusion_ghi_comparison.csv`.** Fix the split and re-run —
it's a one-line change and about an hour of GPU.

---

# Part 1 — What actually got fixed (verified, not just claimed)

| ID | Issue | Status | Evidence |
|---|---|---|---|
| **B1** | "No Image" placeholder | ✅ **Fixed** | `validate_tile()` added with brightness/std checks + sun-angle awareness; multi-source fallback (`nict,slider,gk2a,jaxa` via `SATELLITE_SOURCES`); 10-min backward retry. On-disk `himawari_current.png` is now **292 KB, mean brightness 64.1, 0% black** — real imagery. It was 4 KB / 98.9% black before. |
| **B2** | t−1, t−2 frames zero-filled | ✅ **Fixed** | `fetch_frame_series()`; `himawari_prev1.png` (264 KB, mean 44.3) and `prev2.png` (194 KB, mean 19.2) exist and are passed via `prev_paths=(...)` at `predict.py:220`. |
| **B3** | Optical flow pinned to (0,0) | ✅ **Fixed** | Follows from B2 — real previous frames now reach `_compute_optical_flow`. |
| **D1** | `load_model` couldn't load concat | ✅ **Fixed** | `.config.json` sidecars written for all 6 checkpoints, with shape-inference fallback. |
| **E5** | Skill vs smart persistence missing | ✅ **Fixed** | `skill_score()` + `Skill_vs_SP` column, and persistence baselines now computed on the *same* `valid_indices` as the deep models. Correct design. |
| **A2** | No saved notebook outputs | ✅ **Fixed** | 18/30 cells have outputs. This is what let me diagnose the split bug. |
| — | Diagnostics in output | ✅ **New, good** | `forecast_latest.json` now carries `image_ok`, `outside_training_hours`, `lookback_ends`, `capped`. Exactly the right instinct. |
| — | Configurable architecture | ✅ **New, good** | `small`/`large` presets and a `no-ghi_lag1` ablation. The no-lag ablation is a genuinely valuable addition — it isolates how much the model is just doing persistence. |

That's real progress, and the B1 fix in particular is the difference between a deployment
that can work and one that cannot.

---

# Part 2 — The new critical break: an 18-sample test set

## 2.1 What happened

The dataset was extended from 7,640 rows (→ 2026-02-02) to **9,590 rows (→ 2026-08-16)**.
The satellite archive was **not** extended with it:

- Last row **with** an image: `2026-06-07 16:00`
- First row **missing** an image: `2026-02-03 09:00`
- Rows with an image: 7,659 / 9,590. **1,931 rows have no image.**

The split is still positional 70/15/15 over all rows (notebook cell 19, unchanged), and
`GHIForecastDataset.valid_indices` filters to rows that have an image. So:

| Split | rows | with image | date range |
|---|---|---|---|
| train | 6,708 | 6,708 (100%) | 2024-01-01 → 2025-11-02 |
| val | 1,438 | 930 (**64.7%**) | 2025-11-02 → 2026-03-25 |
| **test** | **1,438** | **18 (1.3%)** | 2026-03-25 → 2026-08-16 |

The image-less tail landed almost entirely in the test split.

## 2.2 What the 18 samples actually are

- **2 unique days only** — 2026-05-15 and 2026-06-07
- Mean GHI **318.5 W/m²** against 492.8 for the full test set — they are unusually dark,
  cloudy hours
- One batch. `Test batches: 1`.

Every MAE, RMSE, R², PICP and skill score in `fusion_ghi_comparison.csv` comes from two
cloudy days. The 5–15 W/m² gaps separating your model variants are pure noise at n=18.

## 2.3 Why the "tabular beats deep" conclusion is now invalid

The two CSVs use different test sets. I trained one RandomForest and scored it on both:

| Test set used | n | RF MAE t+1/t+2/t+3 | avg |
|---|---|---|---|
| Full test set — **what `baseline_ghi_comparison.csv` reports** | 1,438 | 47.9 / 64.6 / 72.3 | **61.6** |
| The 18 imaged rows — **what `fusion_ghi_comparison.csv` reports** | 18 | 62.7 / 128.0 / 174.8 | **121.8** |

Same model, same training data. The test set alone moves MAE by a factor of two.

So the deck's implied gap — RF at ~62 vs Fusion at 148, about **87 W/m²** — is mostly a
test-set artefact. Scored like-for-like on the same 18 rows, the gap is about **27 W/m²**.
Still favours tabular, but it is a different claim with a different magnitude, and it
rests on 18 samples.

Note also that smart persistence scores 112.7 on the full test set and 146.5 on the 18
rows — which is why the fusion CSV shows almost nothing beating it. That is the test set
talking, not the models.

## 2.4 The fix

Drop image-less rows **before** splitting, not after. One line in cell 19:

```python
df_clean = df_clean[df_clean["image_path"].notna()].reset_index(drop=True)
```

What you get:

| Split | n | date range | deep samples |
|---|---|---|---|
| train | 5,359 | 2024-01-01 → 2025-06-20 | ~5,333 |
| val | 1,148 | 2025-06-20 → 2025-10-12 | ~1,122 |
| **test** | **1,149** | **2025-10-13 → 2026-06-07** | **~1,123** |

That test set is 1,123 samples over **116 days spanning 7 months and two monsoon
seasons** — materially better than the single-season test set you had at `0dc4dc7`, so
this also closes issue **E4**.

Then add a hard guard so this can never pass silently again:

```python
assert len(test_dataset) > 500, (
    f"Test set collapsed to {len(test_dataset)} samples — "
    f"check image coverage in the test date range")
```

And to compare fairly across the two CSVs, score the tabular baselines on the **same**
`test_dataset.valid_indices` rows, not on all 1,438.

## 2.5 Root cause: `himawari_data.py` was deleted

The historical satellite collector (190 lines, `fetch_hourly_range`) is gone from the
repo. `current_data.py` can fetch an arbitrary timestamp, but there is no bulk backfill
any more — which is exactly why the weather table ran ahead to August while the image
archive stalled in June.

**Restore it** (`git show 0dc4dc7:himawari_data.py > himawari_data.py`), or add a
`--backfill FROM TO` mode to `current_data.py`. Then backfill 2026-02-03 → 2026-08-16 so
the extra 1,931 rows become usable. With the new `validate_tile()` in place the backfill
will also skip dead tiles properly.

## 2.6 `verification_log.csv` is now empty (0 bytes)

The eight days of live verification data are gone from the working tree. They're
recoverable — `git show 0dc4dc7:verification_log.csv > verification_log.csv` — and you
should recover them, because that log is the *evidence* for the train/serve failure and
the "before" half of your before/after story. Right now you have no live data at all, so
you cannot demonstrate that the B1/B2/B3 fixes actually improved anything.

**Restart the cron immediately.** You need at least 5 days of post-fix forecasts, and
`daily_run.sh` still has `[ "$(date +%Y%m%d)" -le 20260816 ] || exit 0` — an expiry date
that has already passed, so it silently exits. Change it.

---

# Part 3 — Issues from the last audit that are still open

I re-verified each of these against the current code.

| ID | Issue | Status |
|---|---|---|
| **B4** | Gate's `cloud_cover` synthesised as `(1−k_t)` at inference, real ERA5 value in training | ❌ **Unchanged** — `model.py:compute_gate_features` still does `cc = clip((1 - cr) * 100, 0, 100)` |
| **B5** | Lookback window is 24 *clock* hours at inference, 24 *daylight* rows in training | ❌ **Unchanged** — no daylight filter in `build_lookback_window`. Your own diagnostics prove it: `lookback_ends: 2026-08-21 08:00` from an 08:00 issue means the window spans 2026-08-20 09:00 → 2026-08-21 08:00, i.e. **14 of 24 rows are night hours the model never saw in training (58%)** |
| **B6** | Live station weather fetched, printed, never used | ❌ **Unchanged** — `predict.py:190` loads it, prints it, and it never enters `tabular_seq` |
| **C1** | Clear-sky instantaneous vs preceding-hour-mean | ❌ **Unchanged** — max k_t by hour is still exactly 0.69 → 1.43 across 08:00–17:00 |
| **C2** | Overnight-gap targets/lags/frames/flow | ❌ **Unchanged** — still 10.0% / 20.0% / 30.0% cross-day at t+1/2/3 |
| **D2** | `future_clearsky` fed raw, unnormalised | ❌ **Unchanged** |
| **D6** | "CNN-only" ablation still runs the BiLSTM as the attention query | ❌ **Unchanged** — `ablation == "cnn"` uses `H_a = cross_attn(H_t, patches)` |
| **E1** | Single seed | ❌ **Not done** — no seed loop in the experiment grid |
| **E2** | Diebold–Mariano test | ❌ **Not done** — zero occurrences in the notebook |
| **E3** | CRPS | ❌ **Not done** — zero occurrences |
| **E6** | Interval width alongside PICP | ❌ **Not done** — no PINAW/width computed anywhere |
| **A1** | Dataset versioning | ⚠️ **Half done** — both copies are now identical (good), but `data/` is still gitignored, so the file the notebook actually reads is untracked. Freeze and commit a versioned copy. |

B5 is worth singling out. It is one line, it is the most likely remaining cause of
degraded live behaviour, and your own diagnostics field now proves it is happening on
every run.

---

# Part 4 — Do the results make sense, or are they pointless?

Split by table.

### `baseline_ghi_comparison.csv` — **meaningful, and better than before**

n = 1,437, honest chronological split, and persistence baselines are now included.

| Model | t+1h | t+2h | t+3h |
|---|---|---|---|
| Persistence | 157.7 | 265.5 | 335.4 |
| Smart persistence | 73.5 | 117.2 | 146.6 |
| Linear Regression | 64.3 | 94.5 | 115.5 |
| Random Forest | **48.1** | **64.7** | **72.3** |
| LightGBM | 48.5 | 64.8 | 72.9 |

Skill vs smart persistence: **+35% / +45% / +51%**. That is a strong, physically sensible
result — skill grows with horizon because persistence decays faster than learned
dynamics. **This is your best result and it should be a headline.**

One caveat: these numbers are better than the 63.5/82.5/90.0 you had at `0dc4dc7`, and
some of that is a genuinely easier test period, not a better model. The appended
post-Feb-2026 data is measurably smoother — mean hour-to-hour |Δk_t| of 0.163 versus
0.179 for the archive period, about 9% less volatile. Say so, or report on the Option-B
split where the periods are mixed.

### `fusion_ghi_comparison.csv` — **currently pointless**

n = 18, two days. Nothing in it can be defended. Specifically:

- The ordering (concat 133.1 < nolag 138.3 < CNN 142.6 < LSTM 147.3 < fusion 148.4) is
  noise at this sample size, and it **contradicts** your previous ordering, where
  LSTM-only was *best* at 84.7 and concat was second-worst. You cannot claim a stable
  ablation ranking when it inverts between runs.
- `Physics-Gated (small)` at MAE 262.9, R² **−1.08** is a failed run, not a result — it
  is worse than holding GHI constant. Either it diverged or the preset is misconfigured.
  Investigate before reporting it.
- PICP fell from ~90–92% to 70–76% at a nominal 90% level. Calibration was described in
  your deck as "the one clear success" — that claim is currently unsupported. It may well
  come back after the split fix; on 18 samples it means nothing either way.
- `Skill_vs_SP` ≈ 0 for everything says "no model beats smart persistence", which the
  notebook prints as a warning. On this test set that conclusion is unfounded.

### The live deployment — **fixed on the image side, unproven overall**

`forecast_latest.json` from 2026-08-21 08:00 reads 148.4 / 351.9 / 512.6 W/m² for
t+1/2/3 against clear-sky 356/586/768. That is a sensible rising morning profile that
tracks the sun, and it is a marked improvement on the old near-constant ~600/700/765
output. `image_ok: true` and the three real frames on disk confirm B1–B3 are working.

But there is **zero verification data** to prove it, because the log was wiped and the
cron expired. Until you have a week of post-fix verified forecasts you cannot claim the
deployment works — and with B4 and B5 still open, two of the four gate inputs and 58% of
the lookback window are still wrong.

---

# Part 5 — What to do, in order

**Today (≈2 hours, unblocks everything)**

1. Add the one-line image filter before the split (§2.4) plus the `assert`.
2. Restore `himawari_data.py` and `verification_log.csv` from `0dc4dc7`.
3. Fix the `daily_run.sh` expiry date and restart the cron. Every day you wait is a day
   of before/after evidence you don't get.
4. Fix **B5** (daylight filter in `build_lookback_window`) and **B4** (use the real
   `cloud_cover` column). Both are one-liners and both are live-path correctness.

**This week**

5. Re-run the full experiment grid on the corrected split (~1 h GPU). Score the tabular
   baselines on the *same* `valid_indices` so the two CSVs are comparable.
6. Add the seed loop — 3 seeds × the variants you care about. With the ranking having
   already inverted once between runs, this is no longer optional; it is the difference
   between a defensible conclusion and a coin flip.
7. Add CRPS, DM test, and interval width (E2/E3/E6). Together well under a day.
8. Backfill satellite images 2026-02-03 → 2026-08-16 so the extra 1,931 rows are usable.

**Before any retrain you intend to report**

9. Fix **C1** (clear-sky convention) and **C2** (continuous hourly index for targets,
   lags, frames and flow), then retrain. These change every number, so do them once,
   deliberately, and report v1 vs v3 side by side.
10. Fix or rename the **D6** "CNN-only" ablation.

**Housekeeping:** `_to_delete/` still holds my two scratch PNGs; `HANDOFF_PROMPT.md` got
committed into the repo root and probably belongs in `docs/` or out of the tree.

---

## The honest read

The last round fixed the hardest problem in the project — the deployment was
unrecoverable before and it is working now, and you built proper diagnostics while you
were in there. That's the right kind of progress.

What it also did was extend the dataset without extending the imagery, and the split
silently ate the consequence. That is a normal failure and it is cheap to fix, but you
must fix it before the results mean anything again. Right now you have one table worth
presenting (baselines) and one that would sink you if an examiner asked "how many test
samples?"

Ask that question of every table you put on a slide.
