# Solar GHI Forecasting — Independent Audit

**Scope:** full repo walk — `combined_dataset.py`, `historical_data.py`, `himawari_data.py`,
`current_data.py`, `model.py`, `predict.py`, `verify.py`, `solar_pv_main.ipynb`,
`results/*.csv`, `verification_log.csv`, the cached image tensors, the four checkpoints,
and the defence deck.

---

## Bottom line

**Your offline results are sound and your deck is unusually honest.** The negative
finding is real, correctly diagnosed, and defensible. It is not pointless work — it is
the strongest part of the thesis.

**Your live deployment is broken, and slide 33 currently claims the opposite.** The
deployed forecaster has been running on a blank satellite image, two zero-filled video
frames, and a zeroed motion vector for all eight logged days. Its day-to-day forecasts
are *negatively* correlated with reality at every horizon. This is the one section that
will not survive an examiner who looks at the log.

I found nine issues that are not in your deck. Five of them are in the live path. One
of them (the clear-sky time convention) contaminates your offline features too.

---

# Part 1 — The live system is broken

Five defects, all in the `predict.py` → `model.py` path. None appear in the deck.

## 1.1 The live satellite image is a "No Image" placeholder — the critical one

`datanow/satellite/himawari_current.png` is 4,358 bytes, 550×550, 98.9% pure black,
with the words **"No Image"** rendered in grey in the middle. That is NICT's
missing-frame placeholder, and **NICT serves it with HTTP 200**, so this guard never
fires:

```python
# current_data.py:129
if resp.status_code != 200:
    return None          # never triggers — the placeholder is a 200
```

I fingerprinted it. The placeholder has an exact, reproducible signature:
`mean pixel = 0.00217`, `fraction black = 0.9887`. Scanning all 7,631 cached training
frames, **23 of them carry that identical fingerprint** — so this failure mode is
present in your archive too, just rarely (0.3%). In production it appears to have hit
every single run.

For comparison, a real daylight frame has mean brightness 0.16–0.26.

**Consequence:** the entire image branch — 14.4M of the model's 15.5M parameters —
has been fed a constant black square on every deployed forecast. Whatever the offline
model learned from imagery, the live model never sees it.

**Fix** (`current_data.py`, `fetch_image_nict`):

```python
img = Image.open(BytesIO(resp.content))
gray = img.convert("L")
a = np.asarray(gray, dtype=np.float32) / 255.0
if a.mean() < 0.02 or (a < 0.02).mean() > 0.95:
    print(f"  ✗ NICT returned a No-Image placeholder for {date_str}")
    return None                      # let the caller step back 10 min and retry
```

and in `fetch_image`, walk backwards up to ~6 slots:

```python
for back in range(0, 7):
    t = (date_time or self.get_latest_timestamp()) - timedelta(minutes=10*back)
    if (r := self.fetch_image_nict(t)):
        return r
print("  ✗ no valid NICT frame in the last hour — aborting forecast")
return None
```

And make `predict.py` **refuse to forecast** on a blank image rather than silently
proceeding. A missing forecast is defensible; a confident wrong one is not.

## 1.2 Two of the three input frames are always zeros in production

`predict.py:120` calls:

```python
mu, sigma = run_model(model, tabular_seq, SATELLITE_IMG, future_clearsky,
                      df, now_sgt, device)          # prev_paths defaults to (None, None)
```

`run_model` passes that straight into `load_satellite_inputs(sat_path, None, None)`,
which zero-fills frames *t−1* and *t−2*. The model was trained on
`multi_frame = [frame_t, frame_t−1, frame_t−2]` with three real frames. In deployment
it gets `[placeholder, zeros, zeros]`.

**Fix:** `current_data.py` already knows how to fetch an arbitrary timestamp. Fetch
*t−1h* and *t−2h* too, cache them in `datanow/satellite/`, and pass the paths:

```python
prev1 = SatelliteCollector().fetch_image(now_sgt - timedelta(hours=1))
prev2 = SatelliteCollector().fetch_image(now_sgt - timedelta(hours=2))
mu, sigma = run_model(..., prev_paths=(prev1, prev2))
```

## 1.3 Optical flow is always exactly (0, 0) in production

Because 1.2 zeroes `frame_t1`, `_compute_optical_flow(zeros, frame_t)` returns ~0.
Two of the physics gate's four inputs are pinned to zero on every live run. Fixing 1.2
fixes this automatically.

## 1.4 The gate's cloud-cover input is a *different variable* at train and serve time

Training (`solar_pv_main.ipynb` §5):

```python
gate_features = [cr_arr[t], cc_arr[t]/100.0, flow[0], flow[1]]
#                            ^^^^^^^^^^^^^^ real ERA5 cloud fraction
```

Inference (`model.py:484 compute_gate_features`):

```python
cr = clip(ghi_last / ghi_cs, 0, 1.5)
cc = clip((1 - cr) * 100, 0, 100)     # SYNTHESISED — this is just 1 − k_t
return torch.tensor([[cr, cc / 100.0]])
```

At serve time input #2 is a deterministic function of input #1 (`1 − k_t`), perfectly
anti-correlated with it. At train time it was an independent ERA5 measurement. Slide 18
says "cc = ERA5 cloud cover, scaled to [0,1]" — true in training, false in deployment.

**Fix:** `fetch_recent_df` already pulls `cloud_cover` from Open-Meteo. Use the real
column:

```python
row = df[df["timestamp"] <= ref_naive].tail(1)
cc  = float(row["cloud_cover"].iloc[0]) / 100.0    # real value, same as training
```

## 1.5 The lookback window has the wrong time base in production

This one is subtle and probably the largest single contributor to the flat forecasts.

- **Training:** 24 consecutive rows of `df_clean`, which is filtered to 08:00–17:00 SGT.
  So the window is 24 *daylight* steps ≈ 2.4 calendar days. **No night hours, ever.**
  Slide 13 says this correctly.
- **Inference:** `extend_with_recent()` appends *every* hour Open-Meteo returns —
  `past_days=3`, no daylight filter. `build_lookback_window` then takes `.tail(24)`,
  which at an 10:00 issue time is 24 *clock* hours: **roughly 13 night rows with
  GHI ≈ 0 and k_t ≈ 0.**

The BiLSTM has never seen an input window like that. Over half of it is a regime that
does not exist anywhere in the training distribution.

**Fix** — one line in `build_lookback_window`:

```python
past = df[(df["timestamp"] <= ref_naive) &
          (df["timestamp"].dt.hour.between(8, 17))].tail(WINDOW_SIZE)
```

## 1.6 (minor) Live station weather is fetched, printed, and never used

`predict.py:88` loads `weather_current.json` from data.gov.sg, prints it, and writes it
into `forecast_latest.json`. It never enters `tabular_seq` — the window is built
entirely from the CSV plus the Open-Meteo top-up. Slide 32 step 2 implies otherwise.
Either wire it in or drop it from the diagram; as written it is a claim you can't
support.

---

## What all of that does to the forecasts

From `verification_log.csv`, restricted to the seven runs issued at 10:00 SGT so that
clock time and clear-sky geometry are held constant:

| Horizon | forecast mean ± sd | actual mean ± sd | corr(forecast, actual) |
|---|---|---|---|
| t+1h | 602.3 ± **24.1** | 613.6 ± 39.2 | **−0.46** |
| t+2h | 699.1 ± **15.3** | 752.7 ± 53.7 | **−0.53** |
| t+3h | 762.7 ± **7.7** | 776.9 ± 85.5 | **−0.32** |

Over all 24 logged hours the correlations are −0.66 / −0.74 / −0.49.

Three things follow, and you should state all three yourself before an examiner does:

1. **The model emits a near-constant.** At t+3h the forecast varies by ±7.7 W/m² across
   eight days while reality varies by ±85.5. Day-to-day forecast variance is **7% of
   actual variance**.
2. **What little variation exists points the wrong way.** Negative correlation at every
   horizon. The live system has *worse than zero* discriminative skill.
3. **The 79.4 W/m² live MAE is an artefact of a low-variance sample, not skill.** A
   constant 729 W/m² scores 84.8. The best-possible constant per horizon scores
   30.4 / 43.0 / 65.9 — the model scores 47.0 / 70.5 / 68.9, i.e. **−55% / −64% / −5%
   skill against a constant.**

And the 100% interval coverage is not a success: the 90% band is ~504–575 W/m² wide,
about **74% of the mean signal**. A band that wide captures everything by construction.
Coverage without width is not a calibration result — you make exactly this point on
slide 27 for the offline model, so apply the same standard here.

**Slide 33 needs to be rewritten.** The current framing ("live MAE 79.4, 100% inside the
band, an easy sample") understates it. The honest version is: *the deployment
demonstrates the engineering, and its verification loop caught a real train/serve
divergence — the live forecasts carry no skill, and here are the five specific defects
that explain why.* That is a **better** slide. It shows you can debug a deployed system,
which is a harder skill than training one.

---

# Part 2 — Issues that also affect the offline results

## 2.1 Clear-sky GHI is misaligned with GHI by half an hour of solar geometry

This is a real data bug and it is *not* in the deck.

Empirical clear-sky ratio `k_t = ghi / ghi_clearsky`, maximum per hour across the whole
dataset:

| SGT hour | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|
| max k_t | 0.69 | 0.78 | 0.88 | 0.94 | 0.99 | 1.02 | 1.07 | 1.12 | 1.21 | **1.43** |

A physically meaningful clear-sky index should top out near 1.0–1.1 at every hour. This
monotone morning→evening drift is the exact signature of a **timestamp convention
mismatch**: Open-Meteo labels each hourly GHI with the *end* of its averaging window
(the 14:00 value is the 13:00→14:00 mean), while `pvlib.get_clearsky(T)` returns the
*instantaneous* value at T. Morning: the sun is rising, instantaneous > hourly mean, so
k_t is suppressed. Afternoon: the reverse.

You already diagnosed this — `compute_clearsky_hour_mean()` in `model.py`, commit
`97bb429` — but **you only applied it to the physics clamp.** The dataset's
`ghi_clearsky` column, and therefore `clearsky_ratio` (feature #1 and gate input #1) and
`future_clearsky`, still use the instantaneous value.

**Consequence:** your most important feature carries a deterministic time-of-day bias,
and the model has to spend capacity undoing it via `sin_hour`/`cos_hour`.

**Fix:** in `historical_data.py` / `combined_dataset.py`, replace the instantaneous
clear-sky computation with the preceding-hour mean you already wrote:

```python
times = pd.date_range(t - pd.Timedelta(minutes=50), t, freq="10min")
ghi_clearsky = loc.get_clearsky(pd.DatetimeIndex(times))["ghi"].mean()
```

Rebuild the CSV and retrain. This is a genuine "I found and corrected a physical
inconsistency" contribution — worth its own slide.

## 2.2 The overnight-gap target bug — your numbers check out

Slide 31 is correct and I independently reproduced it: 10.0% / 20.1% / 30.1% of
t+1h/t+2h/t+3h targets land on the next morning. Confirming your conclusion that it
flatters every model:

| | RF MAE as reported | RF MAE, same-day rows only |
|---|---|---|
| t+1h | 63.5 | **68.9** |
| t+2h | 82.5 | **93.5** |
| t+3h | 90.0 | **104.2** |

So headline MAEs are **8–16% optimistic**. Your framing ("affects every model
identically, ranking unchanged") is right, and I'd add the table above so the examiner
sees the magnitude.

Worth noting the artefact goes further than targets: `ghi_lag1–3`, the t−1/t−2 image
frames, and the optical-flow pairs are all built by positional shift too. So for 10–20%
of samples the "previous hour's frame" is *yesterday evening's*, and the flow vector
between them is meaningless noise. That is an additional reason the image branch
underperformed, and it's a more interesting explanation than "not enough data."

## 2.3 23 placeholder frames in the training cache

Same "No Image" fingerprint as §1.1. Only 0.3% of frames, so the training impact is
negligible — but you should filter them and say you did, because it is the same bug
that killed the deployment.

---

# Part 3 — Do the results actually mean anything?

**Yes, mostly.** Here is my honest split.

### Solid — keep and lead with these

- **The negative result.** LSTM-only 84.7 → Fusion 97.0 under an identical setup, with
  monotone ordering by how much image information reaches the head. That is a clean,
  controlled experiment.
- **The gate-collapse diagnosis.** α ∈ [0.425, 0.477] with correct sign and no
  magnitude, plus the flat-loss explanation. This is the best analysis in the deck.
- **The baselines.** I recomputed persistence and smart persistence independently and
  got 243.7 and 117.2 W/m² against your 243.7 and 117.7 — you're right.
- **Calibration.** Coverage at four nominal levels *with interval widths*, plus the
  standardised-residual test. Properly done.
- **The audit slides (31, 44).** Finding and reporting your own artefacts is exactly
  what a defence should look like.

### The result you're underselling

Restricting both training and test to genuine same-day pairs, tabular models beat smart
persistence by a wide margin:

| Horizon | RF MAE | Smart persistence | **Forecast skill score** |
|---|---|---|---|
| t+1h | 69.4 | 81.6 | **+15.0%** |
| t+2h | 93.0 | 123.4 | **+24.6%** |
| t+3h | 104.5 | 155.4 | **+32.8%** |

That is a real, publishable, physically sensible result — skill rising with horizon
because persistence decays faster than the learned dynamics. **Skill score against
smart persistence is the standard currency in solar forecasting**, and right now it's
buried in a baselines table. Make it a headline.

### Weak — fix or caveat before you're asked

| Issue | Why it matters | Cost to fix |
|---|---|---|
| **Single seed** | Conclusion #1 (the 12.9 W/m² image-hurts gap) rests on n=1 run per variant. You decline to defend a 3 W/m² ordering (A8) — but 12.9 W/m² with no variance estimate is the same category of claim. | 3 seeds × 4 variants ≈ **1 h GPU**. Highest value-per-minute item you have. |
| **Single-season test set** | Test is 11 Oct 2025 → 2 Feb 2026 — NE monsoon only. Not in your limitations list. | State it (5 min), or blocked rolling-origin CV (1 day). |
| **No Diebold–Mariano test** | You flag this yourself in A8. A paired DM test on 1,120 forecast errors is ~10 lines of scipy. | **20 min.** Just run it. |
| **No CRPS** | You flag it in A8. `properscoring.crps_gaussian(y, mu, sigma)` — you already have μ and σ. | **20 min.** |
| **`future_clearsky` fed unnormalised** | Raw 0–1000 W/m² concatenated onto a 256-d standardised vector into `enrich`. Not fatal, but it means three inputs dominate the layer's scale and it's inconsistent with your "per-column z-scoring" claim on slide 12. | 1 line + retrain. |
| **Skipped NaN batches uncounted** | fp16 autocast + `log σ` + `1/σ²` is a known-fragile combination; `continue` hides how often it fired. Add a counter — if it's >1% of batches, that's a result in itself. | 2 lines. |

### One factual error in the deck

`load_model()` **cannot load `best_model_concat.pt`.** It always constructs
`PhysicsGatedFusionModelV2()` with `ablation=None`, giving `enrich[0] = Linear(259, 256)`;
the concat checkpoint has `Linear(515, 256)`. Strict `load_state_dict` will raise. Your
A4 audit says "load_model auto-detects, so nothing breaks at runtime" — true for v1/v2,
not true for the concat ablation. Add an `ablation=` argument to `load_model`, or fix
the sentence.

---

# Part 4 — What to do, in priority order

### Before the defence (a few hours, all high-value)

1. **Rewrite slide 33.** Report the negative correlations, the 7%-of-variance figure,
   and the skill-vs-constant numbers. Frame the deployment as *the thing that exposed
   the train/serve gap*. Add a new slide: "Five train/serve divergences found in the
   live path" (§1.1–1.5). This turns your weakest section into a strong one.
2. **Run the Diebold–Mariano test** on LightGBM vs Fusion paired errors, and **CRPS**
   for the probabilistic head. Both are <1 h total and both are questions you have
   already predicted you'll be asked.
3. **Add the clear-sky convention finding (§2.1)** with the max-k_t-by-hour table. It's
   a genuine physics bug you diagnosed; currently it's invisible.
4. **Add the skill-score table** against smart persistence as a headline result.
5. **Add "single-season test set"** to slide 35.
6. **Fix the `load_model` / concat claim** in A4.

### If you have a day or two

7. **Three seeds per variant** (~1 h GPU). Report mean ± sd. This is what makes
   conclusion #1 defensible rather than suggestive.
8. **Patch the five live-path bugs** (§1.1–1.5) and re-run the forecaster for a few
   days. Even 3–4 days of *correctly-fed* forecasts alongside the broken ones is a
   compelling before/after slide.
9. **Rebuild the dataset with hour-mean clear-sky** (§2.1) and retrain. Expect a
   measurable improvement in everything, deep and tabular alike.

### If you have weeks (this is your future-work section, and it's already right)

10. **Predict k_t, not GHI.** Your own future-work item #2, and it's the correct one —
    it removes the diurnal scale that currently makes midday errors dominate the loss.
11. **Build targets on a continuous hourly index** and drop overnight-crossing samples.
12. **Freeze the CNN throughout** — you already identified this as the untested control,
    and the training curves point straight at it. 25 minutes of GPU for a direct test of
    your central hypothesis. Honestly, do this one in the "before the defence" block.

---

# Part 5 — On whether the project is worth anything

It is, and I'd push back on any framing where a negative result is a failure.

What you have is: a complete multi-source data pipeline, a controlled four-way ablation
under identical conditions, a correctly-diagnosed mechanism failure, a well-calibrated
probabilistic head, honest baselines you've verified independently, and a deployed
service **whose verification loop caught a real bug** — which is precisely what
verification loops are for.

The thing to fix is not the result. It's that the deck currently presents the deployment
as a modest success when it is a diagnostic. Say what actually happened, show the five
defects, and the deployment stops being the weak slide and becomes evidence that you can
find bugs in your own system that the metrics were hiding. Very few master's projects
contain a train/serve skew analysis at all.

Two claims to soften, and one to strengthen:

- **Soften:** "live MAE 79.4 on a different season" — it's a low-variance sample, and the
  forecast is anti-correlated with truth.
- **Soften:** "removing the image branch improves the model by 12.9 W/m²" — until you
  have seeds, say "by 12.9 W/m² in a single run; a seed study is the obvious control."
- **Strengthen:** the skill scores against smart persistence (+15/+25/+33%). That is a
  positive result and it's currently buried.
