---
tags: [defence]
type: note
---

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
