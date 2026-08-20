# Task: fix the audited defects in this solar GHI forecasting project

An independent audit of this repository found **19 defects**. Several invalidate the live
deployment; one breaks reproducibility of every reported number. This prompt lists all of
them with evidence and file references. Work through them in the order given.

## Ground rules — read before touching anything

1. **Do not overwrite `best_model*.pt` or `train_stats.json`.** The defence deck's numbers
   are derived from those exact checkpoints. If you retrain, write to new filenames
   (`best_model_v3*.pt`) and keep both sets of results side by side.
2. **Verify each claim below before fixing it.** Every one includes the evidence I used —
   reproduce it first, then fix. If you find a claim is wrong, say so instead of patching
   around it.
3. **Do not "fix" the negative result.** The finding that the image branch hurts and that
   LightGBM beats every deep variant is real and is the thesis's main contribution. The
   goal is to make the numbers *mean what they say*, not to make them look better.
4. After each fix, state what changed numerically. Silent fixes are useless — I need to
   report the deltas in a defence.
5. Work in a branch. Commit each numbered group separately with the issue IDs in the
   message.

---

# GROUP A — Reproducibility (do this first; everything else depends on it)

### A1. There are two different `combined_dataset.csv` files and the tracked one is not the one used

- `./combined_dataset.csv` — 7,640 rows, 2024-01-01 → **2026-02-02**, `ghi` mean 475.88.
  **This is the one in git.** It matches the deck (7,640 rows, train GHI 475.05) and it
  matches `train_stats.json` (`ghi_mean: 475.0499`). This is the dataset the checkpoints
  were trained on.
- `./data/combined_dataset.csv` — 9,570 rows, 2024-01-01 → **2026-08-14**, `ghi` mean
  480.98, and it has an extra `pv_power_predicted` column. **This is the one the notebook
  actually reads** (`CSV_PATH = f"{ROOT}/data/combined_dataset.csv"`, cell 2) and the one
  `predict.py` / `verify.py` prefer. **And `data/` is gitignored** (`.gitignore:6`).

Consequence: re-running `solar_pv_main.ipynb` today trains on 25% more data with a
different 70/15/15 boundary — the test set becomes roughly Feb–Aug 2026 instead of
Oct 2025 – Feb 2026 — and produces different normalisation stats. **Not one reported
number would reproduce.** The deck's slide 1 claim ("all figures reproduced from the
repository") and appendix A3 (reproducibility commands) are currently false.

**Fix:**
- Freeze the 7,640-row file as `data/combined_dataset_v1_frozen.csv`, commit it (add a
  `!data/combined_dataset*.csv` negation to `.gitignore`), and point the notebook,
  `predict.py` and `verify.py` at it via a single shared constant.
- Keep the 9,570-row file as `data/combined_dataset_extended.csv` for future retraining.
- Add an assertion at the top of the notebook: `assert len(df) == 7640 and df.ghi.mean().round(2) == 475.88`,
  so a dataset swap fails loudly instead of silently.
- Record the dataset SHA-256 in `train_stats.json` alongside the normalisation stats.

### A2. The notebook has no saved cell outputs

There is no execution record for any training run. Re-execute the notebook end-to-end on
the frozen dataset and commit it **with outputs**, or export a run log.

---

# GROUP B — The live deployment is broken (5 defects, highest severity)

Evidence that it is broken, from `verification_log.csv`, restricted to the seven runs
issued at 10:00 SGT so clock time and clear-sky geometry are held constant:

| Horizon | forecast sd | actual sd | corr(forecast, actual) |
|---|---|---|---|
| t+1h | **24.1** | 39.2 | **−0.46** |
| t+2h | **15.3** | 53.7 | **−0.53** |
| t+3h | **7.7** | 85.5 | **−0.32** |

Over all 24 logged hours: −0.66 / −0.74 / −0.49. Day-to-day forecast variance is **7% of
actual variance**. Against the best-possible constant per horizon the model scores
**−55% / −64% / −5% skill**. The reported 79.4 W/m² live MAE reflects a low-variance
sample, not skill — a flat 729 W/m² would score 84.8. The "100% inside the 90% band" is
an artefact of a band ~540 W/m² wide, i.e. **74% of the mean signal**.

Here is why, in descending order of severity.

### B1. The live satellite image is NICT's "No Image" placeholder — CRITICAL

`datanow/satellite/himawari_current.png` is 4,358 bytes, 550×550, **98.9% pure black**,
with the words "No Image" rendered in grey in the centre. NICT serves this placeholder
with **HTTP 200**, so the guard at `current_data.py:129` (`if resp.status_code != 200`)
never fires.

Fingerprint (exact, reproducible): `mean pixel = 0.00217`, `fraction of pixels < 0.02 = 0.9887`.
A real daylight frame has mean brightness 0.16–0.26.

Scanning all 7,631 cached training tensors under `data/satellite_npy/`, **23 frames carry
that identical fingerprint** — so the placeholder is in the archive too, and it is
unguarded everywhere.

Net effect: 14.4M of the model's 15.5M parameters have been fed a constant black square on
every deployed forecast.

**Fix — validate the fetched image and retry backwards:**
```python
# current_data.py, in fetch_image_nict, after Image.open(...)
gray = img.convert("L")
a = np.asarray(gray, dtype=np.float32) / 255.0
if a.mean() < 0.02 or (a < 0.02).mean() > 0.95:
    print(f"  ✗ NICT returned a No-Image placeholder for {date_str}")
    return None

# current_data.py, in fetch_image — walk back up to 6 × 10-min slots
base = date_time or self.get_latest_timestamp()
for back in range(7):
    if (r := self.fetch_image_nict(base - timedelta(minutes=10 * back))):
        return r
return self.fetch_image_jaxa(base)
```
Then make `predict.py` **abort rather than forecast** if no valid frame is available.
A missing forecast is defensible; a confident wrong one is not.

Also add the same filter to the training-cache build (notebook §3, `save_images_as_npy`)
and drop the 23 contaminated frames.

### B2. Frames t−1 and t−2 are always zero tensors in production

`predict.py:120` calls `run_model(...)` without `prev_paths`, so it defaults to
`(None, None)`. `run_model` → `load_satellite_inputs(sat_path, None, None)` →
`model.py:540-545` zero-fills both previous frames. Training used three real frames
(`multi_frame = cat([frame_t, frame_t−1, frame_t−2])`, notebook cell 19).

**Fix:** fetch t−1h and t−2h via `SatelliteCollector().fetch_image(dt)` (it already
accepts a timestamp), cache them under `datanow/satellite/`, and pass the paths:
```python
prev1 = collector.fetch_image(now_sgt - timedelta(hours=1))
prev2 = collector.fetch_image(now_sgt - timedelta(hours=2))
mu, sigma = run_model(..., prev_paths=(prev1, prev2))
```

### B3. Optical flow is always exactly (0, 0) in production

Consequence of B2: `_compute_optical_flow(zeros, frame_t)` (`model.py:498`) returns ~0, so
two of the physics gate's four inputs are pinned to zero on every live run. Fixing B2 fixes
this — add an assertion that flow is non-zero when both frames are valid.

### B4. The gate's cloud-cover input is a different variable at train and serve time

- **Training** (notebook cell 19): `gate_features = [cr, cloud_cover/100, flow_vx, flow_vy]`
  where `cloud_cover` is the real ERA5 total cloud fraction.
- **Inference** (`model.py:484` `compute_gate_features`): `cc = clip((1 - cr) * 100, 0, 100)`
  — i.e. **synthesised as `1 − k_t`**, a deterministic function of gate input #1 and
  perfectly anti-correlated with it.

Deck slide 18 states "cc = ERA5 cloud cover, scaled to [0,1]". True in training, false in
deployment.

**Fix:** `fetch_recent_df` already pulls `cloud_cover` from Open-Meteo. Use it:
```python
row = df[df["timestamp"] <= ref_naive].tail(1)
cc  = float(row["cloud_cover"].iloc[0]) / 100.0
```
Add a unit test that builds gate features from a known dataframe row and asserts they equal
the training-time values for that row.

### B5. The lookback window has the wrong time base in production

- **Training:** 24 consecutive rows of a dataframe filtered to 08:00–17:00 SGT → 24
  *daylight* steps ≈ 2.4 calendar days, **containing no night hours at all**.
- **Inference:** `extend_with_recent()` (`model.py:433`) appends every hour Open-Meteo
  returns (`past_days=3`, no daylight filter). `build_lookback_window` then takes
  `.tail(24)`, which at a 10:00 issue time is 24 *clock* hours — roughly **13 night rows
  with GHI ≈ 0 and k_t ≈ 0**.

Over half the input window is a regime that does not exist anywhere in training. This is
probably the single largest contributor to the flat, low-biased live forecasts.

**Fix — one line in `build_lookback_window` (`model.py:466`):**
```python
past = df[(df["timestamp"] <= ref_naive) &
          (df["timestamp"].dt.hour.between(8, 17))].tail(WINDOW_SIZE)
```
Add a runtime assertion that all 24 rows fall in 08:00–17:00, and log the window's start
timestamp so a regression is visible in the cron log.

### B6. Live station weather is fetched, printed, and never used

`predict.py:88` loads `weather_current.json` from data.gov.sg, prints it, and writes it into
`forecast_latest.json` — but it never enters `tabular_seq`. The window is built entirely
from the CSV plus the Open-Meteo top-up. Deck slide 32 step 2 implies otherwise.

**Fix:** either wire the station readings into the final row of the lookback window, or
remove the fetch and correct the architecture diagram. Do not leave it as-is.

### B7. Re-run the deployment after B1–B5 and compare

Run `predict.py` + `verify.py` daily for at least 5 days with the fixes in, keeping the old
log. Produce a before/after table with per-horizon MAE, forecast sd, and
corr(forecast, actual). **This before/after is the most valuable new artefact you can
produce for the defence** — it turns a broken deployment into a demonstrated debugging
result.

---

# GROUP C — Data defects that also affect the offline results

### C1. Clear-sky GHI is misaligned with GHI by half an hour of solar geometry

Empirical `k_t = ghi / ghi_clearsky`, **maximum** per hour over the whole dataset:

| SGT hour | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|
| max k_t | 0.69 | 0.78 | 0.88 | 0.94 | 0.99 | 1.02 | 1.07 | 1.12 | 1.21 | **1.43** |

A physically meaningful clear-sky index should cap near 1.0–1.1 at *every* hour. This
monotone morning→evening drift is the signature of a **timestamp convention mismatch**:
Open-Meteo labels each hourly GHI with the **end** of its averaging window (the 14:00 value
is the 13:00→14:00 mean), while `pvlib.get_clearsky(T)` returns the **instantaneous** value
at T. Rising sun → instantaneous > hourly mean → k_t suppressed; setting sun → reverse.

This was already diagnosed once — `compute_clearsky_hour_mean()` exists in `model.py:370`
(commit `97bb429`) — **but it was only applied to the physics clamp.** The dataset's
`ghi_clearsky` column still uses the instantaneous value, and therefore so do
`clearsky_ratio` (tabular feature #1 **and** gate input #1) and `future_clearsky`.

**Fix:** in `historical_data.py` / `combined_dataset.py`, replace the instantaneous
clear-sky computation with the preceding-hour mean already written:
```python
times = pd.date_range(t - pd.Timedelta(minutes=50), t, freq="10min")
ghi_clearsky = loc.get_clearsky(pd.DatetimeIndex(times))["ghi"].mean()
```
Rebuild the dataset (as a **new** file — see A1), re-verify that max k_t is now roughly flat
across hours, and retrain into `best_model_v3*.pt`. Report the before/after MAE for both the
deep models and LightGBM. This is a genuine physics correction and deserves its own slide.

### C2. The overnight-gap artefact extends beyond the targets

Already documented on deck slide 31 for targets — confirmed independently: **10.0% / 20.1% /
30.1%** of t+1h / t+2h / t+3h targets land on the next morning, because the dataset keeps
only 08:00–17:00 and targets are built with a positional `df.ghi.shift(-h)`.

Quantified impact (RandomForest, same split):

| | MAE as reported | MAE, same-day rows only |
|---|---|---|
| t+1h | 63.5 | **68.9** |
| t+2h | 82.5 | **93.5** |
| t+3h | 90.0 | **104.2** |

So every headline MAE is **8–16% optimistic**. The ranking is unaffected.

**What the deck misses:** the same positional shift corrupts `ghi_lag1/2/3`, the t−1 and t−2
image frames (`_get_frame(t-1)`, cell 19), and the optical-flow pairs (cell 17, which pairs
`all_paths[i]` with `all_paths[i-1]`). For 10–20% of samples the "previous hour's frame" is
**yesterday evening's**, and the flow vector between them is meaningless noise.

This is a materially better explanation for the image branch underperforming than "not
enough data" — a fifth of the motion signal the gate was conditioned on was garbage.

**Fix:** reindex onto a continuous hourly `DatetimeIndex`, build targets, lags, frame
offsets and flow pairs by **timestamp difference** rather than row position, and drop
samples whose window or horizon crosses the overnight gap. Report the new sample counts and
re-run everything. Expect MAEs to rise; say so explicitly.

### C3. 23 placeholder frames in the training cache

Same "No Image" fingerprint as B1 (`mean = 0.00217`, `frac_black = 0.9887`). 0.3% of frames,
so training impact is negligible — but filter them and state that you did, because it is the
same unguarded failure that killed the deployment.

---

# GROUP D — Code correctness

### D1. `load_model()` cannot load `best_model_concat.pt`

`model.py:183` always constructs `PhysicsGatedFusionModelV2()` with `ablation=None`, giving
`enrich[0] = Linear(259, 256)`. The concat checkpoint has `Linear(515, 256)` (because
`in_dim = self.D * 2` when `ablation == "concat"`, `model.py:261`). Strict
`load_state_dict` will raise.

Deck appendix A4 says "load_model auto-detects, so nothing breaks at runtime" — true for the
v1/v2 distinction, **not** true for the concat ablation.

**Fix:** infer the ablation from `state["enrich.0.weight"].shape[1]` (259 → gated/lstm/cnn,
515 → concat) and pass it to the constructor. Add a test that round-trips all four
checkpoints through `load_model`.

### D2. `future_clearsky` is fed unnormalised

Raw values in 0–1000 W/m² are concatenated onto a 256-d z-scored vector before
`enrich` (`model.py:311`), in both training and inference. Consistent, so not a train/serve
bug — but it means three inputs dominate that layer's scale, and it contradicts deck slide
12's "per-column z-scoring". Normalise with the clear-sky mean/std, persist them to
`train_stats.json`, retrain as v3.

### D3. Skipped NaN batches are never counted

`train_model` (notebook cell 23) does `if torch.isnan(loss) or torch.isinf(loss): continue`
under fp16 autocast, with `log σ` and `1/σ²` in the loss — a known-fragile combination. Add a
counter and print it per epoch. **If it exceeds ~1% of batches, that is itself a result** and
the model should be retrained in bf16 or fp32.

### D4. Stale docstrings claim v1 is deployed

`model.py:9-11` says `PhysicsGatedFusionModel` (v1) is "DEPLOYED" and "matches the trained
checkpoints on disk"; line 201 says v2 is "experimental, no trained checkpoint yet".
**All four checkpoints contain `global_enc.*` keys — they are v2.** `predict.py:7` repeats
the error. Already noted in deck A4; just fix the comments so code and deck agree.

### D5. `verify.py` labels reanalysis as "measured"

`fetch_actual_ghi` pulls Open-Meteo `shortwave_radiation`, and the log records
`source = "measured"`. That is **the same product used to build the training targets** — it
is model analysis, not an independent pyranometer. The verification is therefore
model-vs-model, and the deck's "verified against measured irradiance" (slides 1, 5, 32) is
too strong.

**Fix:** relabel the column `openmeteo_analysis`, and change the deck wording to "verified
against Open-Meteo analysis (the same reanalysis product used as the training target — not
independent ground truth)". Note this is a *known* limitation on slide 35 item 2; the fix is
to use consistent language everywhere.

### D6. The "CNN-only" ablation still runs the BiLSTM

`ablation == "cnn"` uses `H_a`, and `H_a = cross_attn(H_t, patches)` where the **query is the
BiLSTM output** (`model.py:296-307`). So the temporal branch is still trained and still
drives the attention pooling. It is not "CNN-only" — it is "image features pooled by a
temporal query".

Deck slide 22 (E4) says "α forced to 0; H_t discarded" — H_t is *not* discarded.

**Fix:** either rename the variant to something accurate ("cross-attention only" / "image
branch only") in code, CSVs and deck, **or** add a genuine image-only variant that pools the
patches with a learned constant query. Renaming is the honest, cheap option; an examiner
will spot this.

---

# GROUP E — Evaluation gaps

### E1. Every conclusion rests on a single seed
Conclusion #1 (the 12.9 W/m² "image hurts" gap) comes from one run per variant. You already
decline to defend a 3 W/m² ordering — 12.9 W/m² with no variance estimate is the same
category of claim. **Run 3 seeds × 4 variants** (~1 h GPU total at 12.7 min/run) and report
mean ± sd. Highest value-per-minute item in this entire list.

### E2. No Diebold–Mariano test
Deck A8 already concedes this. Run a paired DM test on LightGBM vs Fusion forecast errors
over the 1,120 test samples, per horizon. ~10 lines of scipy, ~20 minutes.

### E3. No CRPS
Deck A8 concedes this too. You already have μ and σ — `properscoring.crps_gaussian(y, mu, sigma)`,
or implement the closed form. ~20 minutes. Report it next to PICP for all four variants.

### E4. Test set is a single season
Test runs 11 Oct 2025 → 2 Feb 2026 — NE monsoon only. This is **not** in the limitations
list. Add it, and if time allows run blocked rolling-origin CV (3–4 folds) so the ranking
isn't a one-season accident.

### E5. Skill score against smart persistence is missing from the headline
Recomputed independently and confirmed: persistence 243.7 W/m², smart persistence 117.2 W/m²
(deck says 117.7 — close enough, the difference is k_t clipping).

But restricting **both** train and test to genuine same-day pairs, tabular models show
strong, horizon-increasing skill:

| Horizon | RF MAE | Smart persistence | **Skill score** |
|---|---|---|---|
| t+1h | 69.4 | 81.6 | **+15.0%** |
| t+2h | 93.0 | 123.4 | **+24.6%** |
| t+3h | 104.5 | 155.4 | **+32.8%** |

Skill score vs smart persistence is the **standard currency in solar forecasting** and this
is a genuinely positive, physically sensible result (skill rises with horizon because
persistence decays faster than learned dynamics). It is currently buried in a baselines
table. **Compute it for every model and every horizon and promote it to a headline table.**

### E6. Live intervals reported without width
Deck slide 27 correctly reports interval width alongside coverage for the offline model.
Slide 33 reports "100% inside the 90% band" for the live model with no width. Apply the same
standard: the live band is ~504–575 W/m², about 74% of the mean signal.

---

# GROUP F — Deck corrections required by the above

- **Slide 33 (operational) — rewrite entirely.** Current framing ("live MAE 79.4, 100%
  coverage, an easy sample") understates the problem. Report the negative correlations, the
  7%-of-variance figure, the negative skill against a constant, and the interval width. Then
  add a **new slide "Five train/serve divergences found in the live path"** covering B1–B5.
  This converts the weakest section into a strength — it demonstrates debugging a deployed
  system, which is harder than training one.
- **Slide 32 step 2** — remove or fix the implication that data.gov.sg station weather feeds
  the model (B6).
- **Slide 18** — the "cc = ERA5 cloud cover" claim is true only in training (B4).
- **Slide 22 (E4)** — "H_t discarded" is false for the CNN-only ablation (D6).
- **Appendix A4** — the `load_model` "nothing breaks at runtime" claim is false for the
  concat checkpoint (D1).
- **Appendix A3 / slide 1** — the reproducibility claim is currently false because the
  notebook reads a gitignored, since-extended dataset (A1).
- **Slide 35 (limitations)** — add single-season test set (E4), and the overnight-gap
  corruption of lags/frames/flow (C2).
- **Add a new slide** for the clear-sky convention finding (C1) with the max-k_t-by-hour
  table. It is a real physics bug that was diagnosed and corrected — currently invisible.
- **Add the skill-score table** (E5) as a headline result.
- **Soften** "removing the image branch improves the model by 12.9 W/m²" to "in a single
  run" until E1 is done.

---

# Suggested execution order

**Phase 1 — do not defend anything until these are done (half a day)**
A1, A2 (reproducibility) → D1, D4, D6 (correctness/labelling) → E2, E3 (DM test, CRPS) →
E4, E5, E6 (evaluation framing) → F (deck corrections).

**Phase 2 — the highest-value new results (1–2 days)**
E1 (3 seeds × 4 variants) → B1–B6 (live path) → B7 (5 days of corrected live forecasts,
before/after table).

**Phase 3 — requires a rebuild and retrain (several days)**
C1 (clear-sky convention) → C2 (continuous hourly index for targets/lags/frames/flow) →
C3, D2, D3 → full retrain as v3 → re-run all baselines and ablations on the corrected
dataset → report v1 vs v3 side by side.

---

# Deliverables I want back

1. A branch with each group committed separately, issue IDs in the commit messages.
2. `AUDIT_RESPONSE.md` — one row per issue: ID, confirmed/refuted, what changed, numeric
   before/after where applicable.
3. A regression test file covering: image-placeholder rejection, gate-feature train/serve
   equality, lookback-window daylight-only invariant, and `load_model` round-tripping all
   four checkpoints.
4. Updated `results/*.csv` with CRPS, DM p-values, per-horizon skill scores vs smart
   persistence, and seed mean ± sd.
5. A before/after live-deployment table once B7 has run.

**Start by reproducing the evidence for A1, B1 and C1 and telling me whether you agree,
before changing any code.**
