---
tags: [concept, baseline]
type: note
---

# Smart persistence

**The benchmark that matters.** Hold the [[Clear-sky index]] constant instead of the
irradiance, then multiply by the known future [[Clear-sky GHI]]:

$$\hat{GHI}(t+h) = k_t(t) \times GHI_{clearsky}(t+h)$$

In plain words: *"the clouds will stay as they are, but the sun will keep moving."*

It removes the diurnal trend, so it only fails when the cloud field actually changes. Free,
no training, and strong.

**On the test set:** 73.5 / 117.2 / 146.6 W/m² MAE.

> [!important]
> If an examiner asks for "the baseline", this is the one they mean. Report
> [[Forecast skill score]] against it, not raw MAE.

Implemented in `model.py: smart_persistence_forecast()`, and computed per-row by the
[[Verification loop]].
