---
tags: [concept, background]
type: note
---

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
