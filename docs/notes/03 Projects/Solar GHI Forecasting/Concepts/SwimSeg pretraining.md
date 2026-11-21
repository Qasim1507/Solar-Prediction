---
tags: [concept, method]
type: note
---

# SwimSeg pretraining

Before the [[Image branch]] encoder ever sees a satellite tile, it is trained to **segment
cloud from sky** on 1,013 ground-based whole-sky photos from NTU's SwimSeg dataset.

**Rationale:** ImageNet features encode dogs, cars and furniture. Cloud-segmentation
features encode *what a cloud looks like*.

| | |
| --- | --- |
| Architecture | [[EfficientNet]]-B2 + FPN decoder, 128-ch laterals |
| Loss | 0.5·BCE + 0.5·Dice + 0.2·MSE(cloud ratio) |
| Split | by shooting date, 80/20 — never random |
| Output | `swimseg_encoder.pt`, 7.27M params |

> [!warning] The honest caveat
> SwimSeg is **ground-based, upward-looking**; Himawari is **top-down satellite**. The
> textures are related but not the same. Whether this actually helped was **never tested in
> isolation** — the ImageNet-init control is unrun. See [[Next actions]].
