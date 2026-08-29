---
tags: [concept, metric]
type: note
unit: W/m²
---

# GHI — Global Horizontal Irradiance

Total solar power arriving on one square metre of **flat ground**, counting both the direct
beam from the sun's disc and the diffuse light scattered by the sky. The standard input for
flat-panel PV.

- In this [[Dataset]]: range ~3–997 W/m², mean ≈ 481 W/m².
- Source is Open-Meteo `shortwave_radiation` — **ERA5 reanalysis, not a pyranometer**. See
  [[Limitations]].
- Related: [[Clear-sky GHI]] · [[Clear-sky index]]

> DNI (direct normal) and DHI (diffuse horizontal) are the other two components. GHI ≈
> DNI·cos(zenith) + DHI. Only GHI is forecast here.
