---
tags: [reference]
type: note
---

# References

## Data

- **Open-Meteo Historical Weather API** — ERA5 reanalysis, hourly. `archive-api.open-meteo.com`
- **Open-Meteo Forecast API** — `past_days` analysis values; live top-up and verification
- **Himawari-8/9 real-time imagery, NICT** — `himawari8-dl.nict.go.jp`, D531106 level 4d, tile 1_1, 550px
- **JAXA P-Tree** — `ftp.ptree.jaxa.jp`, NetCDF `albedo_03` (fallback source)
- **PVGIS 5.3, EU JRC** — PV reference series (fetched, then excluded as leaky)
- **data.gov.sg** real-time environment API — station temperature, rainfall, humidity, wind

## Software

- **timm / EfficientNet** — Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019
- **PyTorch** — `nn.LSTM`, `nn.MultiheadAttention` (Vaswani et al., *Attention Is All You Need*, 2017)
- **pvlib** — Ineichen–Perez clear-sky model
- **OpenCV** — Farnebäck, *Two-Frame Motion Estimation Based on Polynomial Expansion*, 2003
- **LightGBM**, **scikit-learn**

## Datasets

- **SwimSeg** (NTU) — 1,013 Singapore whole-sky images with binary cloud/sky masks, Oct 2013 – Jul 2015

## Worth reading for the viva

- Diebold & Mariano (1995), *Comparing Predictive Accuracy* — the test in [[ISSUE-E2 No Diebold-Mariano test]]
- Gneiting & Raftery (2007), *Strictly Proper Scoring Rules* — CRPS, [[ISSUE-E3 No CRPS]]
- Yang et al., *Verification of deterministic solar forecasts* — where [[Forecast skill score]] conventions come from
