---
tags: [concept, metric]
type: note
---

# Forecast skill score

$$SS = 1 - \frac{MAE_{model}}{MAE_{reference}}$$

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
