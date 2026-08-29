#!/usr/bin/env python3
"""Adds the Issues section to the vault. Idempotent."""
import os

VAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Project")
P = "03 Projects/Solar GHI Forecasting"
I = f"{P}/Issues"
N = {}

def note(path, front, body):
    fm = "---\n" + "\n".join(f"{k}: {v}" for k, v in front.items()) + "\n---\n\n"
    N[path] = fm + body.strip() + "\n"

def issue(code, title, status, severity, body, area="", fix=""):
    cal = 'success' if status == 'fixed' else ('danger' if severity == 'critical' else 'bug')
    note(f"{I}/ISSUE-{code} {title}.md",
         {"tags": f"[issue, {area or 'general'}]", "type": "issue",
          "status": status, "severity": severity, "code": code},
         f"# {code} — {title}\n\n"
         f"> [!{cal}] Status: **{status.upper()}** · severity **{severity}**\n\n"
         + body.strip()
         + ("\n\n## Fix\n\n" + fix.strip() if fix else ""))

issue("TEST18", "Test set collapsed to 18 samples", "open", "critical",
"""
The dataset was extended to 2026-08-16 but the satellite archive stops at **2026-06-07**, so
1,931 rows have no image. The split is positional 70/15/15 over *all* rows, and
`valid_indices` filters to rows that have an image — so the image-less tail landed almost
entirely in test.

| Split | rows | with image |
| --- | ---: | ---: |
| train | 6,708 | 6,708 (100%) |
| val | 1,438 | 930 (64.7%) |
| **test** | **1,438** | **18 (1.3%)** |

The notebook printed it: `Test: 18 | Test batches: 1`.

The 18 rows span **2 unique days** (2026-05-15, 2026-06-07), mean GHI 318.5 vs 492.8 for the
full test set — unusually dark hours.

## Why it invalidates the comparison

The two result CSVs use different test sets. One RandomForest, scored both ways:

| Test set | n | RF avg MAE |
| --- | ---: | ---: |
| Full (what [[Baseline results]] reports) | 1,438 | **61.6** |
| The 18 imaged rows (what the fusion CSV reports) | 18 | **121.8** |

So the implied "tabular beats deep" gap of ~87 W/m² is mostly a test-set artefact.
Like-for-like it is ~27.

**Root cause:** `himawari_data.py` — the historical satellite collector — was deleted, so the
archive can't be backfilled. See [[Code map]].
""", area="data",
fix="""
One line in notebook cell 19, *before* the split:

```python
df_clean = df_clean[df_clean["image_path"].notna()].reset_index(drop=True)
```

Gives train 5,359 · val 1,148 · **test 1,149 (~1,123 deep samples over 116 days, 7 months,
two monsoon seasons)** — better than the original single-season test set, so this also closes
[[ISSUE-E4 Single-season test set]].

Then add a guard:

```python
assert len(test_dataset) > 500, f"Test set collapsed to {len(test_dataset)} samples"
```

Also: restore `himawari_data.py` (`git show 0dc4dc7:himawari_data.py > himawari_data.py`) and
backfill 2026-02-03 → 2026-08-16. And score the tabular baselines on the **same**
`valid_indices` rows.
""")

issue("B1", "No Image placeholder", "fixed", "critical",
"""
NICT serves a black **"No Image" placeholder** with **HTTP 200**, so the
`if resp.status_code != 200` guard never fired.

Fingerprint: `mean pixel = 0.00217`, `fraction black = 0.9887`. A real daylight frame has mean
brightness 0.16–0.26. **23 of 7,631 cached training frames** carry the identical signature —
and in production it appeared to hit every run.

Effect: 14.4M of 15.5M parameters fed a constant black square on every deployed forecast.
""", area="deployment",
fix="""
`validate_tile()` added to `current_data.py` with brightness/std checks and sun-angle
awareness; multi-source fallback (`nict,slider,gk2a,jaxa`); 10-minute backward retry.

**Verified:** `himawari_current.png` is now 292 KB, mean brightness 64.1, 0% black.
""")

issue("B2", "Previous frames zero-filled", "fixed", "high",
"""
`predict.py` called `run_model(...)` without `prev_paths`, defaulting to `(None, None)`.
`load_satellite_inputs` then zero-filled frames t−1 and t−2. Training used three real frames.
""", area="deployment",
fix="`fetch_frame_series()` added; `himawari_prev1.png` / `prev2.png` fetched and passed. Verified present and real (264 KB / 194 KB).")

issue("B3", "Optical flow always zero", "fixed", "high",
"""
Consequence of [[ISSUE-B2 Previous frames zero-filled]]:
`_compute_optical_flow(zeros, frame_t)` returns ~0, so **two of the four [[Physics gate]]
inputs were pinned to zero** on every live run.
""", area="deployment",
fix="Resolved by B2. Consider asserting flow ≠ 0 when both frames are valid.")

issue("B4", "Gate cloud cover synthesised", "open", "high",
"""
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
""", area="deployment",
fix="""
`fetch_recent_df` already pulls `cloud_cover`. Use it:

```python
row = df[df["timestamp"] <= ref_naive].tail(1)
cc  = float(row["cloud_cover"].iloc[0]) / 100.0
```

Add a unit test asserting serve-time gate features equal training-time values for a known row.
""")

issue("B5", "Lookback window night contamination", "open", "critical",
"""
**Most likely cause of the remaining forecast bias.** See [[kt bias diagnostic]].

- **Training:** 24 consecutive rows of a dataframe filtered to 08:00–17:00 → 24 *daylight*
  steps ≈ 2.4 calendar days. **Zero night rows, ever.**
- **Inference:** `extend_with_recent()` appends every hour Open-Meteo returns (`past_days=3`,
  no daylight filter); `build_lookback_window` takes `.tail(24)` → 24 *clock* hours.

Confirmed by the system's own diagnostics: `lookback_ends: 2026-08-21 08:00` from an 08:00
issue means the window spans 2026-08-20 09:00 → 2026-08-21 08:00 —
**14 of 24 rows (58%) are night hours the model never saw in training.**

Part of [[Train-serve skew]].
""", area="deployment",
fix="""
One line in `model.py: build_lookback_window`:

```python
past = df[(df["timestamp"] <= ref_naive) &
          (df["timestamp"].dt.hour.between(8, 17))].tail(WINDOW_SIZE)
```

Add a runtime assertion that all 24 rows fall in 08:00–17:00, and log the window start.

> [!important] Do this **before** accumulating more verification days, or the "after" data
> measures a system that is still half-broken.
""")

issue("B6", "Live weather fetched but unused", "open", "low",
"""
`predict.py` loads `weather_current.json` from data.gov.sg, prints it, and writes it into
`forecast_latest.json` — but it **never enters `tabular_seq`**. The window is built entirely
from the CSV plus the Open-Meteo top-up. Deck slide 32 step 2 implies otherwise.
""", area="deployment",
fix="Either wire the station readings into the final window row, or remove the fetch and correct the architecture diagram. Don't leave it as-is — it's a claim you can't support.")

issue("C1", "Clear-sky time convention", "open", "high",
"""
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
""", area="data",
fix="""
In `historical_data.py` / `combined_dataset.py`:

```python
times = pd.date_range(t - pd.Timedelta(minutes=50), t, freq="10min")
ghi_clearsky = loc.get_clearsky(pd.DatetimeIndex(times))["ghi"].mean()
```

Rebuild the dataset as a **new file**, verify max k_t is now flat across hours, retrain as v3.
Report before/after for deep models *and* LightGBM.

**This is a genuine physics correction and deserves its own slide.**
""")

issue("C2", "Overnight gap", "open", "high",
"""
Because only 08:00–17:00 is kept and targets are built with a positional `df.ghi.shift(-h)`,
a "t+3h" target issued at 16:00 is **not 19:00 — it is 09:00 the next morning.**

Affects **10.0% / 20.1% / 30.1%** of t+1/2/3h targets.

## Measured impact

| | MAE as reported | MAE, same-day rows only |
| --- | ---: | ---: |
| t+1h | 63.5 | **68.9** |
| t+2h | 82.5 | **93.5** |
| t+3h | 90.0 | **104.2** |

Headline MAEs are **8–16% optimistic**. Affects every model identically, so the ranking holds.

## The part the deck misses

The same positional shift corrupts `ghi_lag1/2/3`, the **t−1 and t−2 image frames**, and the
**[[Optical flow]] pairs**. For 10–20% of samples the "previous hour's frame" is *yesterday
evening's*, and the motion vector between them is noise.

**That is a better explanation for the image branch underperforming than "not enough data"** —
a fifth of the gate's conditioning signal was garbage.
""", area="data",
fix="Reindex onto a continuous hourly `DatetimeIndex`; build targets, lags, frame offsets and flow pairs by **timestamp difference**, not row position; drop samples whose window or horizon crosses the overnight gap. Expect MAEs to rise — say so.")

issue("C3", "Placeholder frames in training cache", "open", "low",
"""
23 of 7,631 cached training tensors carry the same "No Image" fingerprint as
[[ISSUE-B1 No Image placeholder]] (`mean = 0.00217`, `frac_black = 0.9887`). 0.3% of frames,
so training impact is negligible.
""", area="data",
fix="Filter them in `save_images_as_npy` and state that you did — it's the same unguarded failure that killed the deployment.")

issue("D1", "load_model could not load concat checkpoint", "fixed", "medium",
"""
`load_model` always constructed the model with `ablation=None`, giving
`enrich[0] = Linear(259, 256)`. The concat checkpoint has `Linear(515, 256)`, so strict
`load_state_dict` raised.
""", area="code",
fix="`.config.json` sidecars now written for all six checkpoints, with shape-inference fallback.")

issue("D2", "future_clearsky unnormalised", "open", "low",
"""
Raw 0–1000 W/m² values concatenated onto a 256-dim z-scored vector before `enrich`, in both
training and inference. Consistent, so not a skew bug — but three inputs dominate that layer's
scale, and it contradicts the "per-column z-scoring" claim. See [[Prediction head]].
""", area="code",
fix="Normalise with clear-sky mean/std, persist to `train_stats.json`, retrain as v3.")

issue("D3", "NaN batches uncounted", "open", "medium",
"""
`if torch.isnan(loss) or torch.isinf(loss): continue` under fp16 autocast, with `log σ` and
`1/σ²` in the loss — a known-fragile combination. Batches are silently dropped and never
counted. See [[Training setup]].
""", area="training",
fix="Add a per-epoch counter. **If it exceeds ~1% of batches that is itself a result**, and the model should be retrained in bf16 or fp32.")

issue("D5", "Reanalysis labelled measured", "open", "medium",
"""
`verify.py` logs `source = "measured"` for Open-Meteo `shortwave_radiation` — **the same
product used to build the training targets**. The verification is therefore model-vs-model,
and "verified against measured irradiance" (deck slides 1, 5, 32) is too strong.
See [[Verification loop]].
""", area="code",
fix="Relabel the column `openmeteo_analysis`; change deck wording to \"verified against Open-Meteo analysis (the same reanalysis product used as the training target — not independent ground truth)\".")

issue("D6", "CNN-only ablation mislabeled", "open", "medium",
"""
`ablation == "cnn"` uses `H_a`, and `H_a = cross_attn(H_t, patches)` where **the query is the
BiLSTM output**. So the temporal branch is still trained and still drives the attention
pooling. It is not "CNN-only" — it is "image features pooled by a temporal query".

Deck slide 22 says "α forced to 0; H_t discarded". **H_t is not discarded.** An examiner will
spot this. See [[Ablation study]] · [[Cross-attention]].
""", area="code",
fix="Rename the variant in code, CSVs and deck (\"cross-attention only\" / \"image branch only\") **or** add a genuine image-only variant that pools patches with a learned constant query. Renaming is the honest, cheap option.")

issue("E1", "Single seed", "open", "critical",
"""
Every conclusion rests on **one run per variant**. The headline claim is a ~12–13 W/m² gap
between LSTM-only and the fusion model — and **the ablation ordering inverted between two
runs**, which is direct evidence that run-to-run variance is comparable to the effect being
claimed. See [[Ablation results]].
""", area="evaluation",
fix="""
3 seeds × 4 variants ≈ **1 hour of GPU**. Report mean ± sd.

**Highest value-per-minute item in the entire project.** It is the difference between a
defensible conclusion and a coin flip.

Fallback if unrun: *"One seed. I would not defend the ordering within the deep variants on
that basis; the claim I defend is the direction."* → [[Three hard questions]]
""")

issue("E2", "No Diebold-Mariano test", "open", "medium",
"""
No statistical test of whether the LightGBM-vs-fusion gap is significant. A paired
**Diebold–Mariano** test on the forecast errors is the standard instrument. Already conceded
in deck appendix A8.
""", area="evaluation",
fix="~10 lines of scipy on the paired errors, per horizon. ~20 minutes.")

issue("E3", "No CRPS", "open", "medium",
"""
CRPS is the proper scoring rule for probabilistic forecasts and the natural companion to
[[PICP and interval width]]. Not computed. Conceded in deck A8.
""", area="evaluation",
fix="`properscoring.crps_gaussian(y, mu, sigma)` — μ and σ are already saved. ~20 minutes.")

issue("E4", "Single-season test set", "open", "medium",
"""
The earlier valid test set ran 11 Oct 2025 → 2 Feb 2026 — **NE monsoon only**. Not listed in
the limitations. See [[Limitations]].
""", area="evaluation",
fix="Closing [[ISSUE-TEST18 Test set collapsed to 18 samples]] fixes this too — the imaged-rows split spans 116 days across 7 months and two seasons. Otherwise: blocked rolling-origin CV, 3–4 folds.")

issue("A1", "Dataset versioning", "partial", "high",
"""
There were two different `combined_dataset.csv` files with different row counts, and the one
the notebook reads (`data/`) is **gitignored**. Re-running the notebook trained on different
data with a different split, so no reported number reproduced.

Both copies are now identical (9,590 rows) — but `data/` is still gitignored, so the file that
matters is untracked and can silently diverge again. See [[Dataset]].
""", area="data",
fix="""
Freeze a versioned copy, commit it (add `!data/combined_dataset*.csv` to `.gitignore`), point
the notebook and `predict.py` / `verify.py` at one shared constant, record the dataset SHA-256
in `train_stats.json`, and assert row count + GHI mean at the top of the notebook.
""")

note(f"{I}/Open issues.md", {"tags": "[moc, issue]", "type": "moc"}, """
# Open issues

Sorted by what to do first. Closed ones live in [[Fixed issues]].

## Blocking the defence

| Issue | Severity | Cost |
| --- | --- | --- |
| [[ISSUE-TEST18 Test set collapsed to 18 samples]] | 🔴 critical | one line + 1h GPU |
| [[ISSUE-E1 Single seed]] | 🔴 critical | ~1h GPU |

## Live-path correctness

| Issue | Severity | Cost |
| --- | --- | --- |
| [[ISSUE-B5 Lookback window night contamination]] | 🔴 critical | one line |
| [[ISSUE-B4 Gate cloud cover synthesised]] | 🟠 high | one line |
| [[ISSUE-B6 Live weather fetched but unused]] | 🟡 low | minutes |

## Data correctness — needs a rebuild + retrain

| Issue | Severity | Cost |
| --- | --- | --- |
| [[ISSUE-C1 Clear-sky time convention]] | 🟠 high | days |
| [[ISSUE-C2 Overnight gap]] | 🟠 high | days |
| [[ISSUE-A1 Dataset versioning]] | 🟠 high | hours |
| [[ISSUE-C3 Placeholder frames in training cache]] | 🟡 low | minutes |

## Evaluation gaps

| Issue | Severity | Cost |
| --- | --- | --- |
| [[ISSUE-E2 No Diebold-Mariano test]] | 🟡 medium | 20 min |
| [[ISSUE-E3 No CRPS]] | 🟡 medium | 20 min |
| [[ISSUE-E4 Single-season test set]] | 🟡 medium | free with TEST18 |

## Code hygiene

| Issue | Severity |
| --- | --- |
| [[ISSUE-D6 CNN-only ablation mislabeled]] | 🟡 medium |
| [[ISSUE-D5 Reanalysis labelled measured]] | 🟡 medium |
| [[ISSUE-D3 NaN batches uncounted]] | 🟡 medium |
| [[ISSUE-D2 future_clearsky unnormalised]] | 🟡 low |

→ [[Next actions]] · [[Limitations]]
""")

note(f"{I}/Fixed issues.md", {"tags": "[moc, issue]", "type": "moc"}, """
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
""")

note(f"{I}/Limitations.md", {"tags": "[defence]", "type": "note"}, """
# Limitations — volunteer these

Stating them first is what separates a defence from an interrogation.

1. **Data scale.** ~5,300 training samples against 14.4M image-branch parameters. Root cause of
   the negative result, and the training curves show it directly. → [[Overfitting]]
2. **Ground truth is reanalysis.** ERA5 shortwave radiation, not a pyranometer. Smoothed, so
   true point variability is understated and every error figure is against a smoothed field.
   → [[Data sources]]
3. **Single visible band.** An 8-bit RGB browse product converted to greyscale — effectively
   one channel. **Infrared bands, which carry cloud-top temperature and therefore optical
   depth, are never used.**
4. **Target-construction artefact.** 30% of t+3h targets land next morning. Absolute MAEs are
   8–16% optimistic; the ranking is unaffected. → [[ISSUE-C2 Overnight gap]]
5. **Single seed.** → [[ISSUE-E1 Single seed]]
6. **Single-season test set** in the valid run. → [[ISSUE-E4 Single-season test set]]
7. **No CRPS, no significance test.** → [[ISSUE-E3 No CRPS]] · [[ISSUE-E2 No Diebold-Mariano test]]
8. **SwimSeg transfer untested.** Ground-based upward-looking photos vs top-down satellite —
   related textures, but the ImageNet-init control was never run. → [[SwimSeg pretraining]]
""")

note(f"{I}/Next actions.md", {"tags": "[project, todo]", "type": "note"}, """
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
""")

written = skipped = 0
for rel, content in N.items():
    path = os.path.join(VAULT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and open(path, encoding="utf-8").read() == content:
        skipped += 1
        continue
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    written += 1

print(f"issue notes written : {written}")
print(f"unchanged           : {skipped}")
print(f"total in this batch : {len(N)}")
