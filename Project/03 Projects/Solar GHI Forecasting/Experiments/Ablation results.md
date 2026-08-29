---
tags: [results]
type: experiment
n_test: 1120
status: valid-earlier-run
---

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
