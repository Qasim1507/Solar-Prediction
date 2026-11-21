---
tags: [data]
type: note
---

# Data sources

Four public sources, all free.

| Source | Provides | Access | Role |
| --- | --- | --- | --- |
| **Open-Meteo ERA5 archive** | temperature, RH, rain, wind, cloud cover, shortwave radiation (→ [[GHI]]), DNI, DHI — hourly | REST, no key | **Target** + 6 of 11 features |
| **Himawari-8/9 via NICT** | full-disk visible tile, level `4d`, tile `1_1`, 550px, 10-min cadence | public HTTP | The [[Image branch]] |
| **pvlib (Ineichen)** | [[Clear-sky GHI]] at any timestamp | local library | k_t feature, future clear-sky input, [[Physics clamp]] |
| **SwimSeg (NTU)** | 1,013 Singapore whole-sky images + cloud masks | research dataset | [[SwimSeg pretraining]] only |

> [!warning] The target is reanalysis, not a measurement
> Open-Meteo `shortwave_radiation` is ERA5-family model analysis — spatially and temporally
> smoothed. The [[Verification loop]] uses the **same product**, so live verification is a
> consistency check, not independent ground truth. See [[Limitations]].

A fifth source, **PVGIS**, was fetched for PV reference series then excluded — it
correlates 0.99994 with GHI because it is a simulation *of* GHI.
