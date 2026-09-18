---
tags: [data, pipeline]
type: note
---

# Data pipeline

Sequential — each stage consumes the last one's output. This is the order to walk someone
through the repo. See [[Code map]] for the files.

1. **Collect weather** — Open-Meteo ERA5 + pvlib clear-sky → `historical_data.py`
2. **Collect satellite tiles** — hourly Himawari tiles → `himawari_data.py` ⚠️ *deleted, see [[ISSUE-TEST18 Test set collapsed to 18 samples]]*
3. **Align** — floor to hour, filter 08:00–17:00 SGT, match nearest image within ±60 min → `combined_dataset.py`
4. **Cache images** — PNG → greyscale → 224×224 → 0–1 → 3 channels → ImageNet-normalised → `.npy` in RAM
5. **Precompute [[Optical flow]]** — Farnebäck between consecutive frames
6. **[[SwimSeg pretraining]]** — encoder learns cloud vs sky
7. **Build samples** — `GHIForecastDataset`, see below
8. **Train / ablate / evaluate** — notebook §7–8 → `results/*.csv`
9. **Deploy & verify** — [[Live pipeline]] · [[Verification loop]]

## One training sample

| Tensor | Shape | Contents |
| --- | --- | --- |
| `tabular_seq` | (24, 11) | 24 daylight steps of normalised [[Feature list]] |
| `multi_frame` | (9, 224, 224) | Frames at t, t−1, t−2 stacked channel-wise |
| `roi_image` | (3, 224, 224) | Centre 112² crop of frame t, upscaled |
| `future_clearsky` | (3,) | [[Clear-sky GHI]] at t+1/2/3 — pure astronomy |
| `gate_features` | (4,) | k_t, cc/100, v_x, v_y → [[Physics gate]] |
| `targets` | (3,) | Standardised GHI at t+1/2/3 |
