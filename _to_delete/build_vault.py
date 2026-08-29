#!/usr/bin/env python3
"""Builds the Obsidian second-brain vault. Idempotent: safe to re-run."""
import os, sys, json, datetime

VAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Project")
N = {}   # path -> content

def note(path, front, body):
    fm = "---\n" + "\n".join(f"{k}: {v}" for k, v in front.items()) + "\n---\n\n"
    N[path] = fm + body.strip() + "\n"

TODAY = datetime.date.today().isoformat()

# ══════════════════════════════════════════════ HOME
note("00 Home.md",
 {"tags":"[moc]", "type":"moc"}, f"""
# Second Brain

> [!tip] How this vault works
> Notes are small and linked. Follow `[[wikilinks]]`, use the **graph view** (Ctrl/Cmd+G)
> to see clusters, and search with Ctrl/Cmd+Shift+F. Everything about the thesis lives
> under **03 Projects**; everything else has a home in the folders below.

## Active

- [[Solar GHI Forecasting]] — M.Tech thesis. Defence pending.
  - [[Defence prep]] ← **start here before the viva**
  - [[Open issues]] ← what still needs fixing
  - [[Daily log]]

## Structure

| Folder | For |
| --- | --- |
| `01 Inbox` | Anything uncategorised. Empty it weekly. |
| `02 Daily` | One note per working day. Log what you did and what broke. |
| `03 Projects` | Things with a deadline and an end state. |
| `04 Areas` | Ongoing responsibilities with no end date. |
| `05 Resources` | Reference material — papers, links, snippets. |
| `06 Archive` | Finished or abandoned. Never delete, just move here. |
| `99 Templates` | Note templates. Enable the core **Templates** plugin to use them. |

## Entry points

- **Concepts** — [[GHI]] · [[Clear-sky index]] · [[Smart persistence]] · [[Forecast skill score]] · [[Gaussian NLL]] · [[Train-serve skew]]
- **The model** — [[Model architecture]] · [[Physics gate]] · [[Temporal branch]] · [[Image branch]]
- **The results** — [[Baseline results]] · [[Ablation results]] · [[Gate collapse]] · [[Calibration results]]
- **The system** — [[Live pipeline]] · [[Verification loop]] · [[kt bias diagnostic]]

## Setup (once)

Settings → Core plugins → enable **Templates** (folder: `99 Templates`), **Daily notes**
(folder: `02 Daily`, template: `99 Templates/Daily note`), **Graph view**, **Backlinks**.
Optional: install **Dataview** — the frontmatter here is already structured for it.

*Vault generated {TODAY}.*
""")

note("01 Inbox/README.md", {"tags":"[meta]"}, """
# Inbox

Drop anything here when you don't want to decide where it goes. Process weekly:
move to a project, an area, a resource — or delete it.
""")

# ══════════════════════════════════════════════ PROJECT MOC
P = "03 Projects/Solar GHI Forecasting"

note(f"{P}/Solar GHI Forecasting.md",
 {"tags":"[project, thesis]", "type":"project", "status":"active",
  "deadline":"defence pending"}, """
# Solar GHI Forecasting

**Physics-Gated Cross-Modal Fusion for Short-Horizon Solar Irradiance Forecasting**
M.Tech Computer Engineering · Singapore · 1.3521°N 103.8198°E

## In one paragraph

PV output is almost exactly proportional to irradiance, and irradiance without cloud is
pure astronomy — so the entire forecasting problem is cloud. This project forecasts
[[GHI]] at t+1h, t+2h and t+3h by fusing a weather time series ([[Temporal branch]]) with
geostationary satellite imagery of the cloud field ([[Image branch]]), blended by a small
[[Physics gate]] conditioned on physical state. It outputs a mean *and* an uncertainty per
horizon, because reserve sizing needs an interval.

## The finding

**Negative, and defensible.** Adding the satellite branch made forecasts *worse*, the gate
collapsed to a near-constant, and gradient-boosted trees on 11 tabular features beat every
deep variant. See [[Ablation results]] and [[Gate collapse]] for the mechanism.

## Map

- **Background** → [[Problem statement]] · [[Why 1-3 hours]]
- **Data** → [[Data sources]] · [[Dataset]] · [[Feature list]] · [[Pipeline]]
- **Model** → [[Model architecture]] · [[Temporal branch]] · [[Image branch]] · [[Physics gate]] · [[Prediction head]] · [[Training setup]]
- **Results** → [[Baseline results]] · [[Ablation results]] · [[Calibration results]] · [[Gate collapse]]
- **Deployment** → [[Live pipeline]] · [[Verification loop]] · [[kt bias diagnostic]]
- **Quality** → [[Open issues]] · [[Fixed issues]] · [[Limitations]]
- **Defence** → [[Defence prep]] · [[60-second answer]] · [[Q and A bank]] · [[Three hard questions]]
- **Repo** → [[Code map]]

## Status

> [!danger] Blocking before the defence
> [[ISSUE-TEST18 Test set collapsed to 18 samples]] — the deep-model results table is
> currently scored on 18 samples over 2 days. One-line fix. Do this first.
""")

# ══════════════════════════════════════════════ BACKGROUND
note(f"{P}/Background/Problem statement.md",
 {"tags":"[concept, background]", "type":"note"}, """
# Problem statement

Formally: **given observations up to time _t_, estimate the conditional distribution of
[[GHI]] at t+1h, t+2h and t+3h.**

Distribution, not point estimate — a grid operator sizing reserve capacity needs to know
the uncertainty. That is why the model emits μ and σ per horizon and trains on
[[Gaussian NLL]].

## The chain of reasoning

1. **PV output ≈ irradiance.** In this dataset the simulated PV series correlates with GHI
   at r = 0.99994. So forecasting solar power reduces to forecasting irradiance.
2. **Irradiance without cloud is solved.** [[Clear-sky GHI]] depends only on solar geometry
   and is computable decades ahead.
3. **Therefore the whole problem is cloud.** This single sentence justifies the entire
   architecture — the satellite branch exists because that is what shows the cloud field.
4. **Singapore is the hard case.** Equatorial maritime tropics: deep convective cloud forms
   and dissipates on 10–60 minute scales, so [[Clear-sky index]] swings between 0.1 and 1.0
   within one afternoon.

See also [[Why 1-3 hours]].
""")

note(f"{P}/Background/Why 1-3 hours.md",
 {"tags":"[concept, background]", "type":"note"}, """
# Why 1–3 hours

The horizon is chosen by economics *and* by a genuine gap between technologies.

**Economics.** Reserve capacity must be committed in advance. Under-forecasting wastes
spinning reserve; over-forecasting risks a shortfall. The cost is asymmetric and it scales
with forecast error.

**The technology gap.**

| Horizon | What wins | Why it stops working |
| --- | --- | --- |
| < 30 min | Ground sky cameras | Field of view only covers ~30 min of cloud travel |
| **1–3 h** | **← this project** | Neither of the others reaches |
| > 6 h | Numerical weather prediction | Needs a full model cycle to refresh |

A geostationary satellite sees cloud *upwind* of the site, which is exactly the information
neither a sky camera nor a stale NWP run can supply at this range.
""")

# ══════════════════════════════════════════════ CONCEPTS
C = f"{P}/Concepts"

note(f"{C}/GHI.md", {"tags":"[concept, metric]", "type":"note", "unit":"W/m²"}, """
# GHI — Global Horizontal Irradiance

Total solar power arriving on one square metre of **flat ground**, counting both the direct
beam from the sun's disc and the diffuse light scattered by the sky. The standard input for
flat-panel PV.

- In this [[Dataset]]: range ~3–997 W/m², mean ≈ 481 W/m².
- Source is Open-Meteo `shortwave_radiation` — **ERA5 reanalysis, not a pyranometer**. See
  [[Limitations]].
- Related: [[Clear-sky GHI]] · [[Clear-sky index]]

> DNI (direct normal) and DHI (diffuse horizontal) are the other two components. GHI ≈
> DNI·cos(zenith) + DHI. Only GHI is forecast here.
""")

note(f"{C}/Clear-sky GHI.md", {"tags":"[concept]", "type":"note"}, """
# Clear-sky GHI

What [[GHI]] *would* be at a given time and place with no cloud at all. Computed from solar
geometry by `pvlib`'s **Ineichen** model, using solar zenith angle, air mass and a
climatological turbidity coefficient.

**Deterministic and known arbitrarily far ahead.** This is why feeding the model *future*
clear-sky values is not leakage — it depends only on the ephemeris and the site latitude.

> [!warning] Timestamp convention
> Open-Meteo labels each hourly GHI with the **end** of its averaging window (the 14:00
> value is the 13:00→14:00 mean). `pvlib.get_clearsky(T)` returns the **instantaneous**
> value at T. Mixing the two produces a fake morning→evening drift in [[Clear-sky index]].
> See [[ISSUE-C1 Clear-sky time convention]].

Related: [[Physics clamp]]
""")

note(f"{C}/Clear-sky index.md",
 {"tags":"[concept, metric]", "type":"note", "symbol":"k_t"}, """
# Clear-sky index (k_t)

$$k_t = \\frac{GHI}{GHI_{clearsky}}$$

The fraction of the maximum possible sunlight that actually got through. **1.0 = cloudless,
0.1 = heavy overcast.**

The single most important variable in solar forecasting, because it **strips out the daily
sun cycle and leaves only the cloud signal**. Everything interesting is in k_t.

Used in this project as:
- tabular feature #1 (`clearsky_ratio`) — see [[Feature list]]
- gate input #1 — see [[Physics gate]]
- the basis of [[Smart persistence]]
- the basis of the [[kt bias diagnostic]]

> [!tip] Why it matters for evaluation
> Absolute error scales with how much irradiance there is, so a forecast issued at 08:00
> and one issued at noon are **not comparable on MAE**. Convert to k_t first.
""")

note(f"{C}/Persistence.md", {"tags":"[concept, baseline]", "type":"note"}, """
# Persistence

*"It will be the same in three hours as it is now."*

$$\\hat{GHI}(t+h) = GHI(t)$$

The null hypothesis. Free, needs no training. Terrible near sunrise and sunset because it
ignores the sun moving.

**On the test set:** 157.7 / 265.5 / 335.4 W/m² MAE at t+1/2/3h.

Any forecast that cannot beat this is worthless. The one that actually matters, though, is
[[Smart persistence]].
""")

note(f"{C}/Smart persistence.md", {"tags":"[concept, baseline]", "type":"note"}, """
# Smart persistence

**The benchmark that matters.** Hold the [[Clear-sky index]] constant instead of the
irradiance, then multiply by the known future [[Clear-sky GHI]]:

$$\\hat{GHI}(t+h) = k_t(t) \\times GHI_{clearsky}(t+h)$$

In plain words: *"the clouds will stay as they are, but the sun will keep moving."*

It removes the diurnal trend, so it only fails when the cloud field actually changes. Free,
no training, and strong.

**On the test set:** 73.5 / 117.2 / 146.6 W/m² MAE.

> [!important]
> If an examiner asks for "the baseline", this is the one they mean. Report
> [[Forecast skill score]] against it, not raw MAE.

Implemented in `model.py: smart_persistence_forecast()`, and computed per-row by the
[[Verification loop]].
""")

note(f"{C}/Forecast skill score.md", {"tags":"[concept, metric]", "type":"note"}, """
# Forecast skill score

$$SS = 1 - \\frac{MAE_{model}}{MAE_{reference}}$$

with [[Smart persistence]] as the reference. **The standard currency in solar forecasting.**

"+35% skill" means your errors are 35% smaller than the free benchmark's. Report this
rather than raw MAE when comparing to literature — raw MAE depends entirely on the site's
climate, so it is not portable.

**Current values** (Random Forest, see [[Baseline results]]):

| Horizon | Skill |
| --- | --- |
| t+1h | +35% |
| t+2h | +45% |
| t+3h | +51% |

Skill *grows* with horizon because persistence decays faster than learned dynamics do.
That is a physically sensible signature and worth pointing out.
""")

note(f"{C}/MAE RMSE R2.md", {"tags":"[concept, metric]", "type":"note"}, """
# MAE, RMSE and R²

- **MAE** — mean absolute error, in W/m². Directly interpretable. The headline metric here.
- **RMSE** — squares errors first, so it punishes big misses harder. Reported alongside.
- **R²** — fraction of variance explained.

> [!danger] The R² trap
> GHI has an enormous deterministic **daily cycle**. Knowing only the time of day already
> gives a high R². A useless forecast can score 0.8. This is exactly why
> [[Smart persistence]] is the real benchmark — it already captures the sun's motion, so
> beating *it* means genuinely predicting cloud.

**If asked "your R² is 0.79, isn't that good?"** → "It measures how much of the *sun* I
explained, not how much of the *weather*."

See also [[Forecast skill score]] · [[kt bias diagnostic]]
""")

note(f"{C}/Gaussian NLL.md", {"tags":"[concept, training]", "type":"note"}, """
# Heteroscedastic Gaussian NLL

The training objective.

$$L = \\sum_{h \\in \\{1,2,3\\}} \\left[ \\log \\sigma_h + \\frac{(y_h - \\mu_h)^2}{2\\sigma_h^2} \\right]$$

Read it as a **tug of war**:
- the second term punishes being wrong;
- dividing by σ² lets the model *soften* that punishment by admitting uncertainty;
- the first term, log σ, **charges rent** for that admission.

The optimum is honest calibration — σ ends up tracking how wrong the model actually tends
to be in that situation. "Heteroscedastic" just means σ may differ per sample.

σ is produced as `softplus(output) + 1e-4` to keep it positive.

> [!warning] The cost — volunteer this
> Because checkpointing is on **validation NLL**, you are selecting for *calibration*, not
> point accuracy. LightGBM optimises squared error directly. So the MAE comparison against
> it is not perfectly like-for-like.

Related: [[Calibration results]] · [[PICP and interval width]] · [[Training setup]]
""")

note(f"{C}/PICP and interval width.md", {"tags":"[concept, metric]", "type":"note"}, """
# PICP and interval width

**PICP** — Prediction Interval Coverage Probability. The fraction of true values that fell
inside the predicted 90% interval. You want ≈ 90%.

> [!danger] PICP alone is meaningless
> A forecast of "somewhere between 0 and 1400 W/m²" scores 100%. **Always report mean
> interval width alongside it.**

The old broken deployment posted 100% coverage with a band ~540 W/m² wide — about 74% of
the mean signal. That is not calibration, that is a shrug.

**Proper reporting** — coverage *and* width at several nominal levels, plus a
standardised-residual test: z = (y − μ)/σ should be standard normal.

See [[Calibration results]] for the numbers.

Still missing: **CRPS**, the proper scoring rule for probabilistic forecasts. See
[[ISSUE-E3 No CRPS]].
""")

note(f"{C}/Ablation study.md", {"tags":"[concept, method]", "type":"note"}, """
# Ablation study

Retrain the same model with **one component removed or changed**, holding everything else
identical — data, loss, optimiser, schedule, seed — to isolate that component's
contribution.

The variants in this project:

| Variant | Fusion rule | Question it answers |
| --- | --- | --- |
| LSTM-only | `enrich([H_t ‖ c])` | Is the image branch contributing anything? |
| CNN-only | `enrich([H_a ‖ c])` | Is the temporal branch contributing? |
| Naive concat | `enrich([H_t ‖ H_a ‖ c])` | Does the gate beat plain concatenation? |
| **Physics-gated** | `enrich([α·H_t + (1−α)·H_a ‖ c])` | The proposal |
| No-lag | drops `ghi_lag1` | How much is just persistence? |
| Small | B0 backbone, half width | Is it overfitting? |

> [!warning] Naming problem
> "CNN-only" **still runs the BiLSTM** — `H_a` is cross-attention with `H_t` as the query.
> See [[ISSUE-D6 CNN-only ablation mislabeled]].

Results → [[Ablation results]]
""")

note(f"{C}/Cross-attention.md", {"tags":"[concept, architecture]", "type":"note"}, """
# Cross-attention

Attention answers: *"given what I'm looking for, which parts of this should I pay attention
to?"*

$$H_a = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d}}\\right)V$$

Here:
- **Query** = `H_t`, the temporal summary from the [[Temporal branch]] — 1 token
- **Keys / Values** = the 196 image patches from the [[Image branch]]
- 8 heads × 32 dims each, so different heads can specialise

Output: `H_a`, a 256-dim summary of the sky **as seen through the lens of the current
weather state**.

> [!question] "Why is the query length 1? Isn't that wasting the mechanism?"
> It reduces attention to a **learned pooling** over the 196 patches, weighted by where the
> weather sequence currently sits. Deliberate — you want one sky summary conditioned on the
> current state. A longer query (one token per horizon) would let each horizon attend
> differently, and is reasonable future work.
""")

note(f"{C}/BiLSTM.md", {"tags":"[concept, architecture]", "type":"note"}, """
# BiLSTM

A **recurrent** network that carries a memory forward step by step, deciding at each step
what to keep and what to forget (that is the "LSTM" gating). **Bidirectional** means it
reads the sequence forwards *and* backwards.

Configuration here: `nn.LSTM(11, 128, num_layers=2, bidirectional=True, dropout=0.1)`
→ 24 output vectors of 256 dims (128 × 2 directions).

> [!question] "Isn't reading backwards cheating?"
> No. **Every one of the 24 steps is already in the past** at forecast time. The backward
> pass just gives each step context from both sides of the window. No future information
> enters.

Followed by **attention pooling** rather than taking the last hidden state — see
[[Temporal branch]].
""")

note(f"{C}/EfficientNet.md", {"tags":"[concept, architecture]", "type":"note"}, """
# EfficientNet-B2

A **convolutional** network — it slides learned filters across an image, building up from
edges to textures to shapes. Used here as a *feature extractor*, not a classifier.

Input 224×224×3 → output **352 channels at 7×7** = 49 "patches", each describing one region
of sky.

> [!question] "Why B2 rather than ResNet or a ViT?"
> B2 is compound-scaled with a good accuracy-per-parameter ratio at 224×224 — ~7.2M params
> vs ~25M for ResNet-50. With ~5k training samples, **parameter efficiency was the binding
> constraint**. A ViT would be strictly worse: no convolutional locality prior, more
> data-hungry. In hindsight even B2 was too big — hence the `small` (B0) preset.

Pretrained on cloud segmentation first — see [[SwimSeg pretraining]].
""")

note(f"{C}/Optical flow.md", {"tags":"[concept]", "type":"note"}, """
# Optical flow (Farnebäck)

Estimates how pixels **moved** between two consecutive frames, producing a velocity vector
per pixel. Averaged over the centre region around Singapore, it gives two numbers:
`v_x` (east-west) and `v_y` (north-south) **cloud motion**.

Fed to the [[Physics gate]] as inputs 3 and 4 — an explicit cloud-advection prior.

Computed with `cv2.calcOpticalFlowFarneback` at 64×64 for speed; whole dataset in ~8s.

> [!bug] Two problems found
> 1. In production it was always exactly (0,0) because the previous frame was never fetched
>    — see [[ISSUE-B3 Optical flow always zero]] (**fixed**).
> 2. Flow pairs are built by **row position**, so 10% of them pair this morning's frame with
>    *yesterday evening's* — see [[ISSUE-C2 Overnight gap]] (**open**).
""")

note(f"{C}/Train-serve skew.md", {"tags":"[concept, mlops]", "type":"note"}, """
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
""")

note(f"{C}/Overfitting.md", {"tags":"[concept, training]", "type":"note"}, """
# Overfitting — the root cause here

The model memorises the training set instead of learning generalisable structure. Diagnosed
by training loss continuing to fall while validation loss rises.

## The evidence in this project

Training and validation NLL track each other until **epoch 22**, then diverge. Validation
bottoms around epoch 26 and climbs for 20 epochs until early stopping.

**The divergence begins immediately after the CNN unfreeze at epoch 20**, and epoch time
doubles from 10s to 20s at the same point — confirming the extra parameters went live.

## Why it was inevitable

| | |
| --- | --- |
| Training samples | ~5,300 |
| Image-branch parameters | **14,405,124** (92.9% of the model) |

Releasing 14.4M parameters onto 5.3k samples is textbook. This is the mechanism behind
[[Ablation results]] and [[Gate collapse]] — and you have a plot of it.

The untested control that would prove it: **keep the CNN frozen throughout**. ~25 min of
GPU. See [[Next actions]].
""")

note(f"{C}/SwimSeg pretraining.md", {"tags":"[concept, method]", "type":"note"}, """
# SwimSeg pretraining

Before the [[Image branch]] encoder ever sees a satellite tile, it is trained to **segment
cloud from sky** on 1,013 ground-based whole-sky photos from NTU's SwimSeg dataset.

**Rationale:** ImageNet features encode dogs, cars and furniture. Cloud-segmentation
features encode *what a cloud looks like*.

| | |
| --- | --- |
| Architecture | [[EfficientNet]]-B2 + FPN decoder, 128-ch laterals |
| Loss | 0.5·BCE + 0.5·Dice + 0.2·MSE(cloud ratio) |
| Split | by shooting date, 80/20 — never random |
| Output | `swimseg_encoder.pt`, 7.27M params |

> [!warning] The honest caveat
> SwimSeg is **ground-based, upward-looking**; Himawari is **top-down satellite**. The
> textures are related but not the same. Whether this actually helped was **never tested in
> isolation** — the ImageNet-init control is unrun. See [[Next actions]].
""")

note(f"{C}/Physics clamp.md", {"tags":"[concept, deployment]", "type":"note"}, """
# Physics clamp

A safety rail on the deployed forecast: cap it at **1.15 × the preceding-hour-mean
[[Clear-sky GHI]]**, since irradiance cannot meaningfully exceed the clear-sky value.

Uses the *hour-mean* clear-sky, not the instantaneous value, because Open-Meteo's hourly
labelling convention makes the instantaneous value badly wrong near sunrise and sunset.
(The same convention issue that [[ISSUE-C1 Clear-sky time convention]] is about — the clamp
got fixed, the features did not.)

Also forces true night forecasts to zero.

> [!question] "Isn't a clamp hiding model failures?"
> It is a safety rail on a *deployed* system, and a `capped` flag is logged whenever it
> fires, so it is auditable rather than silent. It is **not applied during offline
> evaluation**, so it doesn't flatter any reported metric. A clamp firing often would be a
> symptom — which is exactly why it's logged.
""")

# ══════════════════════════════════════════════ ARCHITECTURE
A = f"{P}/Architecture"

note(f"{A}/Model architecture.md",
 {"tags":"[architecture, moc]", "type":"note", "params":"15505804"}, """
# Model architecture

**Physics-Gated Cross-Modal Fusion** (`PhysicsGatedFusionModelV2` in `model.py`).

```
tabular_seq (24,11) ──► [[Temporal branch]] ──────────► H_t (256) ──┐
                                                                    ├─► [[Physics gate]] ─► fused (256)
multi_frame (9,224,224) ─┐                                          │
roi_image   (3,224,224) ─┴► [[Image branch]] ─► 196 patches ─► [[Cross-attention]] ─► H_a (256) ─┘
                                                                    │
gate_features (4) ──────────────────────────────────────────────────┘

fused ‖ future_clearsky (3) → 259 → [[Prediction head]] → μ(3), σ(3)
```

## Parameter budget — memorise this ratio

| | Count | Share |
| --- | ---: | ---: |
| Total | 15,505,804 | 100% |
| **Image encoders** | **14,405,124** | **92.9%** |
| Trainable while CNN frozen | 1,100,680 | 7.1% |
| **Physics gate** | **193** | 0.001% |

That 92.9% against ~5,300 training samples is the seed of the whole negative result —
see [[Overfitting]].

## Presets

| Preset | Backbone | LSTM | Heads | D | Dropout |
| --- | --- | --- | --- | --- | --- |
| `large` | efficientnet_b2 | 128 | 8 | 256 | 0.15 |
| `small` | efficientnet_b0 | 64 | 4 | 128 | 0.30 |
""")

note(f"{A}/Temporal branch.md", {"tags":"[architecture]", "type":"note"}, """
# Temporal branch

Reads the 24-step window of 11 features ([[Feature list]]) and produces one 256-dim vector.

1. **[[BiLSTM]]** — 2 layers, 128 hidden, bidirectional, dropout 0.1
   → (B, 24, 256), one vector per timestep
2. **Attention pooling** — a 256→128→1 scoring MLP produces one logit per timestep; softmax
   over the 24 steps gives weights summing to 1; output is the weighted sum
   → `H_t` (B, 256)

> [!tip] Why attention pooling, not the last hidden state?
> Taking only the last state forces one vector to carry 24 steps of history through a
> bottleneck. Attention lets the model **emphasise the last three hours during a ramp** and
> a longer stretch on a stable day.

**This is where the performance actually comes from.** In [[Ablation results]] the
LSTM-only variant is the *best* model.

Params: 539,648 (LSTM) + 33,025 (attention).
""")

note(f"{A}/Image branch.md", {"tags":"[architecture]", "type":"note"}, """
# Image branch

Two [[EfficientNet]]-B2 encoders, initialised from [[SwimSeg pretraining]]:

- **`global_enc`** — shared across the three time-stacked frames (t, t−1, t−2)
- **`roi_enc`** — the centre 112×112 crop of frame t, upscaled back to 224×224
  (roughly the inner 2°×2° around Singapore)

Each 224×224 image → 352 channels at 7×7 → flattened to **49 patches** of 352 dims.

$$3 \\text{ global frames} \\times 49 + 1 \\text{ RoI} \\times 49 = \\mathbf{196 \\text{ patches}}$$

These become the keys and values for [[Cross-attention]].

**Cost:** 4 EfficientNet forward passes per sample — this dominates everything else in the
model. Frozen for the first 20 epochs, then unfrozen at lr/10 (see [[Training setup]] and
[[Overfitting]]).

Params: 7,202,562 each = **14.4M**, i.e. 92.9% of the model.
""")

note(f"{A}/Physics gate.md",
 {"tags":"[architecture, contribution]", "type":"note", "params":"193"}, """
# Physics gate — the contribution

**193 parameters. The whole thesis.**

$$\\alpha = \\sigma\\big(W_2 \\cdot \\text{ReLU}(W_1 g + b_1) + b_2\\big), \\quad g = [k_t,\\ cc/100,\\ v_x,\\ v_y]$$

$$\\text{fused} = \\alpha \\cdot H_t + (1-\\alpha) \\cdot H_a$$

| Input | Meaning |
| --- | --- |
| `k_t` | [[Clear-sky index]] at issue time, clipped to [0, 1.5] |
| `cc` | ERA5 cloud cover, scaled to [0,1] |
| `v_x`, `v_y` | Mean Farnebäck [[Optical flow]] in the centre RoI |

## The hypothesis

We already know *from physics* when each modality should matter — on a clear day the
temporal trend is nearly sufficient; under broken cloud the spatial field dominates. So
supply that physical state directly rather than making the fusion layer infer the regime
from scratch. **Intended:** clear sky → α→1 (trust time series); broken cloud → α→0 (trust
satellite).

## What actually happened

It collapsed. → **[[Gate collapse]]**

> [!bug] Serve-time bug
> Input #2 is **synthesised as `1 − k_t`** at inference instead of using the real cloud
> cover. See [[ISSUE-B4 Gate cloud cover synthesised]].
""")

note(f"{A}/Prediction head.md", {"tags":"[architecture]", "type":"note"}, """
# Prediction head

```
fused (256) ‖ future_clearsky (3)  →  259
  → enrich: Linear(259, 256) + ReLU + Dropout(0.15)
  → Linear(256,128) + LayerNorm + GELU + Dropout(0.15)
  → Linear(128, 64) + LayerNorm + GELU
  → Linear(64, 6)
```

The 6 outputs split into **μ (3 horizons)** and **σ (3 horizons)**;
`σ = softplus(out[3:]) + 1e-4` keeps it strictly positive so `log σ` and `1/σ²` stay finite
in the [[Gaussian NLL]].

> [!note] Three marginal Gaussians, not a joint distribution
> The horizons are given equal weight and their errors are treated as independent. There is
> no covariance between t+1h and t+3h. Worth stating if asked.

> [!bug] `future_clearsky` is fed **raw** (0–1000 W/m²) alongside a standardised 256-dim
> vector. Consistent between train and serve, so not a skew bug — but poorly scaled, and it
> contradicts the "per-column z-scoring" claim. See [[ISSUE-D2 future_clearsky unnormalised]].
""")

note(f"{A}/Training setup.md", {"tags":"[training]", "type":"note"}, """
# Training setup

| Setting | Value | Reasoning |
| --- | --- | --- |
| Loss | [[Gaussian NLL]] | Reserve sizing needs an interval |
| Optimiser | AdamW | Decoupled weight decay |
| Learning rate | 3e-5 | Conventional CNN fine-tuning rate |
| Weight decay | **0.1** | Deliberately aggressive — overfitting anticipated |
| Scheduler | ReduceLROnPlateau ×0.5, patience 8 | |
| Batch size | 128 (CUDA) / 16 (CPU/MPS) | |
| Max epochs | 150, early stop patience 15–20 on val NLL | |
| Gradient clip | global norm 1.0 | |
| Mixed precision | fp16 autocast + GradScaler | |
| **CNN frozen** | epochs 1–20 | A random head backprop'ing into a pretrained backbone destroys it |
| **CNN unfrozen** | epoch 20, lr ÷ 10 | Let the head settle first |
| Seeds | torch 42, numpy 42 — **one run only** | ← [[ISSUE-E1 Single seed]] |
| Hardware | RTX 4090, ~12.7 min per model | |

## What the curves show

→ [[Overfitting]]. Divergence starts at epoch 22, right after the unfreeze.

> [!bug] Non-finite batches are skipped with a bare `continue` and **never counted**.
> fp16 + `log σ` + `1/σ²` is a fragile combination. See [[ISSUE-D3 NaN batches uncounted]].
""")

# ══════════════════════════════════════════════ DATA
D = f"{P}/Data"

note(f"{D}/Data sources.md", {"tags":"[data]", "type":"note"}, """
# Data sources

Four public sources, all free.

| Source | Provides | Access | Role |
| --- | --- | --- | --- |
| **Open-Meteo ERA5 archive** | temperature, RH, rain, wind, cloud cover, shortwave radiation (→ [[GHI]]), DNI, DHI — hourly | REST, no key | **Target** + 6 of 11 features |
| **Himawari-8/9 via NICT** | full-disk visible tile, level `4d`, tile `1_1`, 550px, 10-min cadence | public HTTP | The [[Image branch]] |
| **pvlib (Ineichen)** | [[Clear-sky GHI]] at any timestamp | local library | k_t feature, future clear-sky input, [[Physics clamp]] |
| **SwimSeg (NTU)** | 1,013 Singapore whole-sky images + cloud masks | research dataset | [[SwimSeg pretraining]] only |

> [!warning] The target is reanalysis, not a measurement
> Open-Meteo `shortwave_radiation` is ERA5-family model analysis — spatially and temporally
> smoothed. The [[Verification loop]] uses the **same product**, so live verification is a
> consistency check, not independent ground truth. See [[Limitations]].

A fifth source, **PVGIS**, was fetched for PV reference series then excluded — it
correlates 0.99994 with GHI because it is a simulation *of* GHI.
""")

note(f"{D}/Dataset.md", {"tags":"[data]", "type":"note"}, """
# Dataset

`data/combined_dataset.csv`

| | |
| --- | --- |
| Rows | **9,590** |
| Period | 2024-01-01 → 2026-08-16 |
| Hours kept | **08:00–17:00 SGT** (10 rows/day) |
| Rows with a satellite image | **7,659** |
| Last row with an image | 2026-06-07 |
| GHI mean ± sd (train) | 476.9 ± 258.4 W/m² |
| Split | 70 / 15 / 15, **strictly chronological** |

## Why daylight-only

Night GHI ≈ 0, trivially predictable. Keeping it would inflate every accuracy metric and let
the loss be dominated by samples where the answer is "zero".

> [!important] Consequence — say "steps", not "hours"
> A 24-**step** window spans about **2.4 calendar days**, not 24 hours. `sin_hour`/`cos_hour`
> tell the model where in the day each step sits.

## Two live problems

> [!danger] [[ISSUE-TEST18 Test set collapsed to 18 samples]]
> Weather runs to August 2026, images stop in June → the image-less rows all landed in the
> test split. **Deep models were scored on 18 samples over 2 days.**

> [!bug] [[ISSUE-C2 Overnight gap]]
> Targets are built by positional `shift(-h)`, so 10 / 20 / 30% of t+1/2/3h targets are
> actually *next morning*.
""")

note(f"{D}/Feature list.md", {"tags":"[data]", "type":"note"}, """
# The eleven tabular features

`TABULAR_COLS` in `model.py`. Each is z-scored using **training-split statistics only**,
persisted to `train_stats.json`.

| # | Feature | Why it's there |
| --- | --- | --- |
| 1 | `clearsky_ratio` | The [[Clear-sky index]] — cloud attenuation itself. Detrends the diurnal cycle. |
| 2 | `cloud_cover` | ERA5 total cloud fraction, %. Direct attenuation proxy. |
| 3 | `temperature_2m` | Correlates ~0.86 with GHI — largely a proxy for accumulated insolation. |
| 4 | `rain` | Non-zero rain implies deep cloud, strong attenuation. |
| 5 | `wind_speed_10m` | Surface proxy for advection speed of the cloud field. |
| 6 | `relative_humidity_2m` | High RH precedes convective cloud formation. |
| 7–8 | `sin_hour`, `cos_hour` | Cyclical encoding — 23:00 and 00:00 must be adjacent. |
| 9–10 | `sin_month`, `cos_month` | Season, without a Dec/Jan discontinuity. |
| 11 | `ghi_lag1` | Previous hour's GHI — the persistence signal, made explicit. |

The **tabular baselines get 14 features** — these plus `ghi_lag2`, `ghi_lag3` and raw
`ghi_clearsky`. So [[Baseline results]] are achieved on a handicap that *favours* the
baselines. Worth saying out loud.

The `no-lag` ablation drops #11 to measure how much of the model is just persistence.
""")

note(f"{D}/Pipeline.md", {"tags":"[data, pipeline]", "type":"note"}, """
# Data pipeline

Sequential — each stage consumes the last one's output. This is the order to walk someone
through the repo. See [[Code map]] for the files.

1. **Collect weather** — Open-Meteo ERA5 + pvlib clear-sky → `historical_data.py`
2. **Collect satellite tiles** — hourly Himawari tiles → `himawari_data.py` ⚠️ *deleted, see [[ISSUE-TEST18 Test set collapsed to 18 samples]]*
3. **Align** — floor to hour, filter 08:00–17:00 SGT, match nearest image within ±60 min → `combined_dataset.py`
4. **Cache images** — PNG → greyscale → 224×224 → 0–1 → 3 channels → ImageNet-normalised → `.npy` in RAM
5. **Precompute [[Optical flow]]** — Farnebäck between consecutive frames
6. **[[SwimSeg pretraining]]** — encoder learns cloud vs sky
7. **Build samples** — `GHIForecastDataset`, see below
8. **Train / ablate / evaluate** — notebook §7–8 → `results/*.csv`
9. **Deploy & verify** — [[Live pipeline]] · [[Verification loop]]

## One training sample

| Tensor | Shape | Contents |
| --- | --- | --- |
| `tabular_seq` | (24, 11) | 24 daylight steps of normalised [[Feature list]] |
| `multi_frame` | (9, 224, 224) | Frames at t, t−1, t−2 stacked channel-wise |
| `roi_image` | (3, 224, 224) | Centre 112² crop of frame t, upscaled |
| `future_clearsky` | (3,) | [[Clear-sky GHI]] at t+1/2/3 — pure astronomy |
| `gate_features` | (4,) | k_t, cc/100, v_x, v_y → [[Physics gate]] |
| `targets` | (3,) | Standardised GHI at t+1/2/3 |
""")

# ══════════════════════════════════════════════ RESULTS
R = f"{P}/Experiments"

note(f"{R}/Baseline results.md",
 {"tags":"[results]", "type":"experiment", "n_test":"1437", "status":"valid"}, """
# Baseline results — the strongest thing you have

`results/baseline_ghi_comparison.csv` · **n = 1,437** · chronological hold-out

| Model | t+1h | t+2h | t+3h |
| --- | ---: | ---: | ---: |
| [[Persistence]] | 157.7 | 265.5 | 335.4 |
| [[Smart persistence]] | 73.5 | 117.2 | 146.6 |
| Linear Regression | 64.3 | 94.5 | 115.5 |
| **Random Forest** | **48.1** | **64.7** | **72.3** |
| LightGBM | 48.5 | 64.8 | 72.9 |

MAE in W/m².

## [[Forecast skill score]] vs smart persistence

**+35% / +45% / +51%** — and skill *grows* with horizon, because persistence decays faster
than learned dynamics. That is physically sensible and it is a **positive result**.

> [!tip] Lead with this
> It is currently buried in a baselines table. It deserves a slide of its own.

> [!caution] One caveat to state
> These are better than the earlier run (63.5 / 82.5 / 90.0) partly because the test period
> moved to a genuinely easier stretch — mean hour-to-hour |Δk_t| of 0.163 versus 0.179, about
> 9% less volatile.

Compare: [[Ablation results]]
""")

note(f"{R}/Ablation results.md",
 {"tags":"[results]", "type":"experiment", "n_test":"1120", "status":"valid-earlier-run"}, """
# Ablation results

> [!danger] Which numbers to use
> The **current** `fusion_ghi_comparison.csv` is scored on **18 samples** and must not be
> presented — see [[ISSUE-TEST18 Test set collapsed to 18 samples]]. The table below is from
> the earlier **valid** run at n = 1,120.

| Variant | MAE | RMSE | R² | PICP@90 | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| **LSTM-only** | **84.7** | 116.6 | 0.791 | 91.9 | best — image removal *helps* |
| CNN-only | 91.2 | 125.0 | 0.760 | 89.4 | image alone is weaker |
| Naive concat | 93.4 | 129.7 | 0.742 | 89.2 | both together: worse than either |
| Physics-Gated ★ | 97.0 | 133.8 | 0.725 | 90.2 | gating worse than concat |

## What it proves

1. **Removing the image branch improves MAE by 12.3 W/m²** under an otherwise identical
   setup. The image branch is not merely unhelpful — it is actively harmful.
2. The variants order **monotonically** by how much image information reaches the output.
3. The gate is worse than plain concatenation → [[Gate collapse]].

**Mechanism:** [[Overfitting]] — 14.4M image params on ~5.3k samples.

> [!warning] Single seed
> This ordering **inverted** between two runs. Until [[ISSUE-E1 Single seed]] is closed,
> defend the *direction*, not the ordering.
""")

note(f"{R}/Gate collapse.md",
 {"tags":"[results, key-finding]", "type":"experiment"}, """
# Gate collapse — memorise this

Measured over 1,120 test samples:

| | |
| --- | --- |
| mean α | **0.4461** |
| standard deviation | 0.0096 |
| min / max | 0.4246 / 0.4773 |
| total range | **0.053** |
| design intent | 0 → 1 |

**The gate spans 5% of its available range. It is not a switch; it is a constant.**

α never leaves [0.425, 0.477] on any test sample, at any hour, under any cloud condition.

## But the sign is right

Mean α rises monotonically from **0.438** in the cloudiest decile to **0.454** in the
clearest. Clearer sky → slightly more weight on the temporal branch, exactly as designed.
**Correct direction, no usable magnitude.**

## Why — the one-sentence answer

> [!quote] Say this
> If `H_a` carries little predictive information, the loss is nearly **flat in α**, so
> gradient descent has no reason to move it anywhere. The gate didn't fail because it was
> badly designed — **there was nothing worth gating.**

## Why it's worse than concatenation

Gating is a **bottleneck**. Concatenation keeps both 256-dim vectors and lets the next layer
use whatever is useful. Gating collapses them into a single convex combination *first*,
discarding information, and only pays off if α carries real regime information. It doesn't.
So you pay the bottleneck cost for no conditioning benefit.

Related: [[Physics gate]] · [[Ablation results]] · [[Overfitting]]
""")

note(f"{R}/Calibration results.md",
 {"tags":"[results]", "type":"experiment", "status":"success"}, """
# Calibration results — the clear success

The probabilistic head worked. This is the one unambiguously positive deep-model result.

| Nominal | LSTM-only coverage | width | Fusion coverage | width |
| --- | ---: | ---: | ---: | ---: |
| 50% | 57.7% | 163 | 55.5% | 175 |
| 80% | 84.7% | 310 | 82.6% | 333 |
| **90%** | **91.8%** | 398 | **90.2%** | 427 |
| 95% | 95.0% | 474 | 93.7% | 509 |

Widths in W/m², over 1,120 samples × 3 horizons.

## Standardised-residual test

z = (y − μ)/σ should be standard normal if calibrated.

- LSTM-only: mean **−0.003**, sd **0.931**
- Fusion: mean **−0.075**, sd **0.991**

Both means essentially zero (no systematic bias), both sds within 7% of one.

## Reading

Both over-cover at 50% and 80%, and land almost exactly on nominal at 90% and 95%. The
residuals are **more peaked than a Gaussian** — too many small errors for the fitted σ — so
narrow intervals over-cover.

> [!important] Report width, always
> Coverage alone is meaningless. See [[PICP and interval width]].

Missing: **CRPS** → [[ISSUE-E3 No CRPS]]
""")

note(f"{R}/Error analysis.md", {"tags":"[results]", "type":"experiment"}, """
# Error analysis

## By cloud regime (LSTM-only / Fusion, W/m²)

| k_t bin | n | MAE |
| --- | ---: | --- |
| < 0.3 (overcast) | 67 | 116.9 / 155.0 |
| 0.3 – 0.5 | 177 | 95.7 / 112.9 |
| 0.5 – 0.7 | 241 | 91.9 / 105.1 |
| 0.7 – 0.85 | 269 | 84.1 / 92.8 |
| > 0.85 (clear) | 366 | 69.1 / 78.3 |

Error rises monotonically with cloudiness — 69% increase for LSTM-only, 98% for fusion.
Physically expected: [[Clear-sky index]] is nearly deterministic under clear sky and
near-chaotic under broken convection.

## By hour

MAE peaks at 12:00–13:00 (109.8 / 127.8) where mean GHI is 683–727, and falls to 54.8 / 57.4
at 16:00 where mean GHI is 497. **Normalised by the hourly mean the relative error is far
flatter** — this is a scale effect, not a time-of-day weakness.

## Worst failures

The three largest errors in the whole test set fall on **2 Dec 2025**, a persistently
overcast morning (k_t ≈ 0.1) where the model reverted toward clear-sky values — forecasting
662 against an actual 62.

**The one regime where the image branch pays:** large ramps. At |ΔGHI| > 200 W/m², fusion
(85.0) nearly matches LSTM-only (82.4) and both crush persistence (281.3).
""")

# ══════════════════════════════════════════════ DEPLOYMENT
DP = f"{P}/Deployment"

note(f"{DP}/Live pipeline.md", {"tags":"[deployment]", "type":"note"}, """
# Live pipeline

A cron job that forecasts, then grades itself.

1. **10:00 SGT** — `daily_run.sh predict`
2. **Fetch** — `current_data.py`: current satellite tile + t−1h + t−2h, each validated for
   real content; recent hourly weather to top up the archive lag
3. **Build & run** — `predict.py`: 24-step window → `load_model` (architecture auto-detected
   from the `.config.json` sidecar) → μ, σ
4. **[[Physics clamp]]** — cap at 1.15 × preceding-hour-mean clear-sky
5. **Write** `forecast_latest.json` with a `diagnostics` block (`image_ok`,
   `outside_training_hours`, `lookback_ends`, `capped`)
6. **Later** — [[Verification loop]]

## Health checks now in place

- Satellite tile rejected if mean brightness < 0.02 or > 95% black → retries backwards in
  10-min steps, then falls back to other sources (`SATELLITE_SOURCES=nict,slider,gk2a,jaxa`)
- Warns if the image is > 2h old or the lookback data is > 6h stale

> [!bug] Two divergences still open
> [[ISSUE-B4 Gate cloud cover synthesised]] and
> [[ISSUE-B5 Lookback window night contamination]] — see [[Train-serve skew]].

> [!warning] `daily_run.sh` has a hard expiry date of `20260816` and silently exits after
> it. Change it or the cron does nothing.
""")

note(f"{DP}/Verification loop.md", {"tags":"[deployment]", "type":"note"}, """
# Verification loop

`verify.py` — fetches what actually happened and grades the forecast.

## Why it matters more than the forecast

**The verification loop is what caught [[Train-serve skew]].** Across eight logged days the
old system's forecasts barely moved — at t+3h the forecast varied by ±7.7 W/m² while
reality varied by ±85.5 — and what variation existed was **negatively correlated** with the
truth (−0.66 / −0.74 / −0.49).

That is invisible in MAE. The live MAE of 79.4 looked *better* than the offline 97.

> [!quote] How to frame it in the defence
> Not "my deployment worked". Instead: *"I deployed it, the verification loop caught that
> the live forecasts had no skill, and I traced it to five concrete train/serve
> divergences."* Very few master's projects contain a train/serve skew analysis at all.

## What it logs now

Per row: forecast, actual, error, CI hit — **plus** `clearsky_wm2`, `kt_forecast`,
`kt_actual`, `kt_bias`, and a [[Smart persistence]] reference.

## The report

`python3 verify.py --report` → five sections. See [[kt bias diagnostic]].

> [!caution] "measured" is a misnomer
> Actuals come from Open-Meteo — **the same product used as the training target**. It is a
> consistency check, not independent ground truth. See
> [[ISSUE-D5 Reanalysis labelled measured]].
""")

note(f"{DP}/kt bias diagnostic.md",
 {"tags":"[deployment, method, key-finding]", "type":"note"}, """
# The k_t bias diagnostic

**The test that MAE cannot do.** Built into `verify.py --report`.

## The idea

Absolute error scales with how much irradiance there is, so a forecast issued at 08:00 and
one at noon are not comparable. Divide by [[Clear-sky GHI]] and the sun drops out:

$$\\text{k}_t\\text{ bias} = k_{t,\\text{forecast}} - k_{t,\\text{actual}}$$

**If the bias is large but its spread is small, the model is not confused — it is _offset_.**
A constant offset has a findable cause. Scattered bias means the model genuinely cannot read
the sky. Completely different problems; MAE cannot tell them apart.

Decision rule: `sd / |mean|` < 0.5 → strongly systematic · < 1.0 → mostly systematic ·
> 1.0 → scatter dominates.

## What it found

| Run | mean k_t bias | spread |
| --- | ---: | ---: |
| 19 Aug — broken satellite | **−0.220** | 0.011 |
| 21 Aug — satellite fixed | **−0.180** | 0.054 |

Same pathology, barely improved. Subtracting that single constant from the 21 Aug forecast
takes MAE from **99.0 → 12.0 W/m²** — i.e. **88% of all error is one offset**.

## What that points at

A near-constant multiplicative under-bias that survived the satellite fix points hard at
**[[ISSUE-B5 Lookback window night contamination]]**: ~14 of the 24 inference window rows are
night with GHI ≈ 0, dragging the model's read of current conditions down by a roughly fixed
amount every run.

> [!tip] Track `kt_bias`, not MAE
> MAE bounces around with the weather. The bias doesn't.

> [!warning] n = 1 day on each side. The *pattern* is the signal, not the magnitude.
""")


# ══════════════════════════════════════════════ DEFENCE
F = f"{P}/Defence"

note(f"{F}/Defence prep.md", {"tags":"[defence, moc]", "type":"moc"}, """
# Defence prep

> [!tip] Rehearse in this order
> [[60-second answer]] → [[Three hard questions]] → [[Q and A bank]] → skim
> [[Limitations]] → check [[Open issues]] is honest.

## The frame

Lead with the **negative result**. A negative result you can explain mechanistically is
stronger than a positive one you cannot. You have the mechanism: [[Overfitting]] beginning
at the exact epoch the CNN unfreezes, [[Gate collapse]] with a measured α range, and a
monotone [[Ablation results]] ordering.

## What is genuinely strong

1. **[[Baseline results]]** — +35/+45/+51% [[Forecast skill score]] vs [[Smart persistence]]. A real positive result.
2. **[[Gate collapse]]** — diagnosed, not just observed. The flat-loss explanation is the best analysis in the project.
3. **[[Calibration results]]** — coverage *with* widths *and* a residual test.
4. **[[Train-serve skew]]** — five divergences found by your own verification loop. Almost no master's project has this.
5. **[[kt bias diagnostic]]** — you built the tool that shows 88% of live error is one constant offset.

## What to soften

- ~~"live MAE 79.4"~~ → it was a low-variance sample, anti-correlated with truth
- ~~"removing the image branch improves by 12.9"~~ → "in a single run" until [[ISSUE-E1 Single seed]] closes

## The golden rule

> [!important]
> **Ask "how many test samples?" of every table you put on a slide.** Every claim needs a
> sample size; every comparison must be on the same rows.
""")

note(f"{F}/60-second answer.md", {"tags":"[defence, script]", "type":"note"}, """
# The 60-second answer

For *"so, tell me what you did."* Say roughly this:

> Solar panels produce power in proportion to the sunlight hitting them, and the only thing
> that meaningfully interrupts that sunlight is cloud. Grid operators have to decide hours in
> advance how much backup generation to keep running, so they need to know how much sun there
> will be one to three hours from now.

> I built a system that forecasts solar irradiance over Singapore at one, two and three hours
> ahead. It combines two very different kinds of information: a **time series** of weather at
> the site, and **satellite images** of the cloud field around Singapore, which show weather
> that hasn't arrived yet.

> The novel part is how I combine them. Most work just glues the two representations together
> and lets the network figure it out. I argued we already know from physics *when* each source
> should matter — on a clear day the local trend is enough; under broken cloud the spatial
> picture dominates. So I built a small gate that takes physical state — clear-sky index,
> cloud cover, cloud motion — and produces a single number α that decides how much to weight
> each branch.

> I tested it against three ablations of itself and against classical baselines. The headline
> result is **negative**: adding the satellite branch made forecasts worse, the gate collapsed
> to a near-constant, and gradient-boosted trees on eleven tabular features beat every deep
> variant. I think that's the most defensible finding in the project, and I can explain
> exactly why it happened.

Then stop. Let them ask.

Related: [[Problem statement]] · [[Ablation results]] · [[Gate collapse]]
""")

note(f"{F}/Three hard questions.md",
 {"tags":"[defence, critical]", "type":"note"}, """
# The three hard questions

These can actually go badly. Volunteer two of them before you're asked.

---

## 1. "How many samples is your test set?"

**Fix it before the defence** → [[ISSUE-TEST18 Test set collapsed to 18 samples]]. Then:
*"About 1,123 samples across 116 days and two monsoon seasons."*

If unfixed, say it **first, unprompted**:

> "The deep-model table is currently scored on 18 samples because my satellite archive stops
> before my weather data does. It's a split bug, I've identified it, here's the fix, and I'm
> not drawing conclusions from that table."

---

## 2. "How many random seeds did you run?"

Currently one — and **the ablation ordering already inverted between two runs**, which is
direct evidence that run-to-run variance is comparable to the claimed effect.

3 seeds × 4 variants ≈ 1 hour GPU. → [[ISSUE-E1 Single seed]]

Fallback if unrun:

> "One seed. I would not defend the ordering within the deep variants on that basis; the
> claim I'd defend is the direction — that image information consistently did not help."

---

## 3. "Your live system reports 79 W/m² MAE — better than offline. How?"

**Do not accept the compliment.**

> "That number is from a favourable low-variance sample, not better performance. Across those
> eight days the forecast varied by only about 8 W/m² at t+3h while reality varied by 85, and
> the forecast was actually *negatively* correlated with the truth at every horizon. A
> constant would have scored similarly. The verification loop is what surfaced that, and it
> led me to five concrete train/serve divergences — three of which I've fixed."

Delivered that way this becomes one of the **strongest** moments in the defence.

→ [[Train-serve skew]] · [[kt bias diagnostic]] · [[Verification loop]]
""")

note(f"{F}/Q and A bank.md", {"tags":"[defence]", "type":"note"}, """
# Q&A bank

Answer out loud *before* reading the answer.

## Concepts

**Why forecast irradiance rather than PV power?**
PV output is close to a deterministic function of irradiance — 0.99994 correlation here — so
irradiance is the physically fundamental quantity, and forecasting it makes the model
transferable to any panel configuration. It also avoids circularity: the PV column is a
simulation derived from GHI.

**Why 1–3 hours?** → [[Why 1-3 hours]]

**Isn't future clear-sky GHI leakage?**
No. It's a closed-form function of solar zenith angle, air mass and climatological turbidity —
it depends only on the ephemeris and site latitude. Nothing about future *weather* enters it.
→ [[Clear-sky GHI]]

## Data

**How do you know there's no leakage?**
Four controls: strict chronological cut, no shuffling; each split wrapped in its own Dataset
so a lookback window can't cross the boundary (5,343 rows → 5,312 samples); normalisation
statistics from the training split only, persisted to `train_stats.json`; and `pv_actual`
excluded everywhere because it correlates 0.99994 with GHI.

**Why only 08:00–17:00?** → [[Dataset]]

**Your ground truth is reanalysis — does that invalidate the results?**
It qualifies them. ERA5 is smoothed, so true point variability is understated and every error
is against a smoothed field — which likely makes the problem look *easier*. The relative
comparison between my models is unaffected since they all face the same target. For absolute
operational accuracy I'd need pyranometer data (SERIS). → [[Limitations]]

## Architecture

**Why EfficientNet-B2?** → [[EfficientNet]]
**Why pretrain on SwimSeg?** → [[SwimSeg pretraining]]
**Why a scalar gate?**
It's the most direct test of the hypothesis, and it's *interpretable* — I can plot α against
cloudiness and check whether it behaves as designed, which is exactly how I diagnosed the
collapse. A 256-dim gate is my first future-work item, but it would have **hidden** the
failure rather than exposing it. → [[Physics gate]]

**Why is the cross-attention query length 1?** → [[Cross-attention]]

## Training and evaluation

**Why Gaussian NLL rather than MSE?** → [[Gaussian NLL]]
**Your R² is 0.79, isn't that good?** → [[MAE RMSE R2]]
**Is the LightGBM gap significant?** → [[ISSUE-E2 No Diebold-Mariano test]]
**Doesn't a wide interval guarantee high PICP?** → [[PICP and interval width]]

## Results

**Your novel contribution performed worst — isn't the project a failure?**
The project asked whether satellite fusion pays off at this data scale and answered clearly:
at ~5,000 samples against 14.4M image-branch parameters, it does not. That's a real finding
with a mechanism. A negative result I can explain is more useful to the next person than a
positive result I can't. → [[Overfitting]]

**Why does the gate do worse than naive concatenation?** → [[Gate collapse]]

**What would the satellite branch need to actually help?**
More data — years, not months. A much smaller image branch. And **infrared channels** — the
current input is effectively one visible band, so it carries no cloud-top temperature and
therefore no optical depth. I'd also predict k_t rather than GHI.

## Deployment

**How would you deploy at scale?**
One satellite tile covers many sites, so compute the patch embeddings once per tile per
timestep and share them across every site inside it, letting each site's temporal query attend
to its own neighbourhood. Per-site marginal cost collapses to roughly the BiLSTM. Latency
isn't the constraint — 64 ms against an hourly cadence — the satellite's 20–30 min
publication delay is.

**Isn't the physics clamp hiding failures?** → [[Physics clamp]]
""")

# ══════════════════════════════════════════════ CODE MAP
note(f"{P}/Code map.md", {"tags":"[reference]", "type":"note"}, """
# Code map

| File | Does what | Notes |
| --- | --- | --- |
| `model.py` | **Single source of truth** for architecture + inference helpers | [[Model architecture]] |
| `solar_pv_main.ipynb` | Training, ablations, evaluation | §1 EDA · §2 baselines · §3 images · §3.5 SwimSeg · §4 RAM+flow · §5 dataset · §6 model · §7 train · §8 evaluate · §9 plots |
| `historical_data.py` | Open-Meteo ERA5 + pvlib clear-sky | [[Data sources]] |
| ~~`himawari_data.py`~~ | Historical satellite tiles | ⚠️ **deleted** — restore it |
| `combined_dataset.py` | Aligns everything into one table | [[Pipeline]] |
| `current_data.py` | Live weather + satellite, with `validate_tile()` | [[Live pipeline]] |
| `predict.py` | Live forecast → `forecast_latest.json` | [[Live pipeline]] |
| `verify.py` | Verification + [[kt bias diagnostic]] report | [[Verification loop]] |
| `daily_run.sh` | Cron entry point | ⚠️ expiry `20260816` |
| `fetch_solcast.py` | PVGIS PV reference (fetched, excluded as leaky) | |

## Artefacts

| Path | What |
| --- | --- |
| `data/combined_dataset.csv` | 9,590 rows → [[Dataset]] |
| `best_model*.pt` + `.config.json` | 6 checkpoints with architecture sidecars |
| `swimseg_encoder.pt` | [[SwimSeg pretraining]] output |
| `train_stats.json` | Normalisation statistics |
| `results/*.csv` | [[Baseline results]] · [[Ablation results]] |
| `results/plots/` | eda · training_curves · horizon_degradation · gate_analysis · forecast_vs_actual |
| `analysis/` | Per-model prediction dumps, α analysis, figure builder |
| `deck/` | `build.py` → the defence pptx |
| `verification_log.csv` | Post-fix live log |
| `verification_log_prefix.csv` | Pre-fix live log (the broken era) |

## Key functions

- `model.py: build_lookback_window` ← [[ISSUE-B5 Lookback window night contamination]]
- `model.py: compute_gate_features` ← [[ISSUE-B4 Gate cloud cover synthesised]]
- `model.py: compute_clearsky_hour_mean` ← the fix that [[ISSUE-C1 Clear-sky time convention]] needs applied upstream
- `model.py: smart_persistence_forecast` / `skill_score` → [[Forecast skill score]]
- notebook cell 19 `GHIForecastDataset` ← [[ISSUE-TEST18 Test set collapsed to 18 samples]]
""")

# ══════════════════════════════════════════════ DAILY
note(f"{P}/Daily log.md", {"tags":"[log]", "type":"moc"}, """
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
""")

note("02 Daily/2026-08-24.md", {"tags":"[daily]", "type":"daily", "date":"2026-08-24"}, """
# 2026-08-24

## Focus
- [ ] [[ISSUE-TEST18 Test set collapsed to 18 samples]] — the one-line split fix
- [ ] [[ISSUE-B5 Lookback window night contamination]] — daylight filter

## Log
- Set up this vault.

## Notes

## Tomorrow
""")

# ══════════════════════════════════════════════ AREAS / RESOURCES
note("04 Areas/README.md", {"tags":"[meta]"}, """
# Areas

Ongoing responsibilities with **no end date** — as opposed to `03 Projects`, which have a
deadline and a finish line.

Examples you might add: `Research skills`, `Python & PyTorch`, `Writing`, `Career`.

When a project finishes, its durable knowledge often graduates into an area; the project
itself goes to `06 Archive`.
""")

note("05 Resources/README.md", {"tags":"[meta]"}, """
# Resources

Reference material that isn't tied to one project — papers, links, code snippets, course
notes.

See [[References]].
""")

note("05 Resources/References.md", {"tags":"[reference]", "type":"note"}, """
# References

## Data

- **Open-Meteo Historical Weather API** — ERA5 reanalysis, hourly. `archive-api.open-meteo.com`
- **Open-Meteo Forecast API** — `past_days` analysis values; live top-up and verification
- **Himawari-8/9 real-time imagery, NICT** — `himawari8-dl.nict.go.jp`, D531106 level 4d, tile 1_1, 550px
- **JAXA P-Tree** — `ftp.ptree.jaxa.jp`, NetCDF `albedo_03` (fallback source)
- **PVGIS 5.3, EU JRC** — PV reference series (fetched, then excluded as leaky)
- **data.gov.sg** real-time environment API — station temperature, rainfall, humidity, wind

## Software

- **timm / EfficientNet** — Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019
- **PyTorch** — `nn.LSTM`, `nn.MultiheadAttention` (Vaswani et al., *Attention Is All You Need*, 2017)
- **pvlib** — Ineichen–Perez clear-sky model
- **OpenCV** — Farnebäck, *Two-Frame Motion Estimation Based on Polynomial Expansion*, 2003
- **LightGBM**, **scikit-learn**

## Datasets

- **SwimSeg** (NTU) — 1,013 Singapore whole-sky images with binary cloud/sky masks, Oct 2013 – Jul 2015

## Worth reading for the viva

- Diebold & Mariano (1995), *Comparing Predictive Accuracy* — the test in [[ISSUE-E2 No Diebold-Mariano test]]
- Gneiting & Raftery (2007), *Strictly Proper Scoring Rules* — CRPS, [[ISSUE-E3 No CRPS]]
- Yang et al., *Verification of deterministic solar forecasts* — where [[Forecast skill score]] conventions come from
""")

note("06 Archive/README.md", {"tags":"[meta]"}, """
# Archive

Finished or abandoned. **Never delete — just move here.** Links from other notes keep
working, and old context is often what you need six months later.
""")

# ══════════════════════════════════════════════ TEMPLATES
T = "99 Templates"

note(f"{T}/Daily note.md", {"tags":"[daily]", "type":"daily", "date":"{{date}}"}, """
# {{date}}

## Focus
- [ ]

## Log

## Notes

## Tomorrow
""")

note(f"{T}/Concept note.md", {"tags":"[concept]", "type":"note"}, """
# {{title}}

**One-sentence definition.**

## Why it matters here

## Detail

## Related
- [[ ]]
""")

note(f"{T}/Experiment note.md",
 {"tags":"[results]", "type":"experiment", "date":"{{date}}", "n_test":"", "status":"draft"}, """
# {{title}}

## Question

## Setup

| | |
| --- | --- |
| Test set / n | |
| Split | |
| Seeds | |
| Checkpoint | |

## Results

| Model | t+1h | t+2h | t+3h |
| --- | ---: | ---: | ---: |

## Reading

## Caveats
- Sample size:
- Confounds:

## Related
- [[ ]]
""")

note(f"{T}/Issue note.md",
 {"tags":"[issue]", "type":"issue", "status":"open", "severity":"medium", "code":""}, """
# {{title}}

> [!bug] Status: **OPEN** · severity **medium**

## Symptom

## Evidence

## Root cause

## Fix

## Related
- [[ ]]
""")

note(f"{T}/Meeting note.md",
 {"tags":"[meeting]", "type":"meeting", "date":"{{date}}", "with":""}, """
# {{title}}

**Date:** {{date}} · **With:**

## Agenda

## Notes

## Decisions

## Actions
- [ ]
""")

# ══════════════════════════════════════════════ WRITE
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

# Obsidian config — wikilinks, attachments, and a readable graph
cfgdir = os.path.join(VAULT, ".obsidian")
os.makedirs(cfgdir, exist_ok=True)
def merge_json(name, updates):
    p = os.path.join(cfgdir, name)
    try:
        cur = json.load(open(p, encoding="utf-8"))
        if not isinstance(cur, dict):
            cur = {}
    except Exception:
        cur = {}
    cur.update(updates)
    json.dump(cur, open(p, "w", encoding="utf-8"), indent=2)

merge_json("app.json", {
    "useMarkdownLinks": False,
    "newLinkFormat": "shortest",
    "attachmentFolderPath": "05 Resources/attachments",
    "alwaysUpdateLinks": True,
    "showUnsupportedFiles": False,
})
merge_json("appearance.json", {"accentColor": "#B8660A"})
merge_json("core-plugins.json", {
    "templates": True, "daily-notes": True, "graph": True, "backlink": True,
    "outgoing-link": True, "tag-pane": True, "outline": True,
    "page-preview": True, "search": True, "file-explorer": True,
    "global-search": True, "switcher": True, "command-palette": True,
    "bookmarks": True, "properties": True,
})
json.dump({"folder": "99 Templates"},
          open(os.path.join(cfgdir, "templates.json"), "w"), indent=2)
json.dump({"folder": "02 Daily", "template": "99 Templates/Daily note",
           "format": "YYYY-MM-DD"},
          open(os.path.join(cfgdir, "daily-notes.json"), "w"), indent=2)

# Remove Obsidian's stub welcome note now that a real Home exists
stub = os.path.join(VAULT, "Welcome.md")
if os.path.exists(stub) and "This is your new" in open(stub, encoding="utf-8").read():
    os.remove(stub)

print(f"notes written : {written}")
print(f"unchanged     : {skipped}")
print(f"total notes   : {len(N)}")
print(f"vault         : {VAULT}")
