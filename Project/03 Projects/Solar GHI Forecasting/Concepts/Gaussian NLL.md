---
tags: [concept, training]
type: note
---

# Heteroscedastic Gaussian NLL

The training objective.

$$L = \sum_{h \in \{1,2,3\}} \left[ \log \sigma_h + \frac{(y_h - \mu_h)^2}{2\sigma_h^2} \right]$$

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
