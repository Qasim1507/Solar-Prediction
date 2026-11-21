---
tags: [concept, metric]
type: note
---

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
