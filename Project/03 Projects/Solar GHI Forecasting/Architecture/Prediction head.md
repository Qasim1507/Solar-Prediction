---
tags: [architecture]
type: note
---

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
