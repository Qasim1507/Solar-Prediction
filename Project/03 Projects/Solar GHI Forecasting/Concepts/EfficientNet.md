---
tags: [concept, architecture]
type: note
---

# EfficientNet-B2

A **convolutional** network — it slides learned filters across an image, building up from
edges to textures to shapes. Used here as a *feature extractor*, not a classifier.

Input 224×224×3 → output **352 channels at 7×7** = 49 "patches", each describing one region
of sky.

> [!question] "Why B2 rather than ResNet or a ViT?"
> B2 is compound-scaled with a good accuracy-per-parameter ratio at 224×224 — ~7.2M params
> vs ~25M for ResNet-50. With ~5k training samples, **parameter efficiency was the binding
> constraint**. A ViT would be strictly worse: no convolutional locality prior, more
> data-hungry. In hindsight even B2 was too big — hence the `small` (B0) preset.

Pretrained on cloud segmentation first — see [[SwimSeg pretraining]].
