---
tags: [concept, metric]
type: note
symbol: k_t
---

# Clear-sky index (k_t)

$$k_t = \frac{GHI}{GHI_{clearsky}}$$

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
