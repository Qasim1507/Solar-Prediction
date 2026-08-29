---
tags: [architecture, contribution]
type: note
params: 193
---

# Physics gate — the contribution

**193 parameters. The whole thesis.**

$$\alpha = \sigma\big(W_2 \cdot \text{ReLU}(W_1 g + b_1) + b_2\big), \quad g = [k_t,\ cc/100,\ v_x,\ v_y]$$

$$\text{fused} = \alpha \cdot H_t + (1-\alpha) \cdot H_a$$

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
