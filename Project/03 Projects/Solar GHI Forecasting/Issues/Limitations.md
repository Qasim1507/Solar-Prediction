---
tags: [defence]
type: note
---

# Limitations — volunteer these

Stating them first is what separates a defence from an interrogation.

1. **Data scale.** ~5,300 training samples against 14.4M image-branch parameters. Root cause of
   the negative result, and the training curves show it directly. → [[Overfitting]]
2. **Ground truth is reanalysis.** ERA5 shortwave radiation, not a pyranometer. Smoothed, so
   true point variability is understated and every error figure is against a smoothed field.
   → [[Data sources]]
3. **Single visible band.** An 8-bit RGB browse product converted to greyscale — effectively
   one channel. **Infrared bands, which carry cloud-top temperature and therefore optical
   depth, are never used.**
4. **Target-construction artefact.** 30% of t+3h targets land next morning. Absolute MAEs are
   8–16% optimistic; the ranking is unaffected. → [[ISSUE-C2 Overnight gap]]
5. **Single seed.** → [[ISSUE-E1 Single seed]]
6. **Single-season test set** in the valid run. → [[ISSUE-E4 Single-season test set]]
7. **No CRPS, no significance test.** → [[ISSUE-E3 No CRPS]] · [[ISSUE-E2 No Diebold-Mariano test]]
8. **SwimSeg transfer untested.** Ground-based upward-looking photos vs top-down satellite —
   related textures, but the ImageNet-init control was never run. → [[SwimSeg pretraining]]
