---
tags: [concept]
type: note
---

# Clear-sky GHI

What [[GHI]] *would* be at a given time and place with no cloud at all. Computed from solar
geometry by `pvlib`'s **Ineichen** model, using solar zenith angle, air mass and a
climatological turbidity coefficient.

**Deterministic and known arbitrarily far ahead.** This is why feeding the model *future*
clear-sky values is not leakage — it depends only on the ephemeris and the site latitude.

> [!warning] Timestamp convention
> Open-Meteo labels each hourly GHI with the **end** of its averaging window (the 14:00
> value is the 13:00→14:00 mean). `pvlib.get_clearsky(T)` returns the **instantaneous**
> value at T. Mixing the two produces a fake morning→evening drift in [[Clear-sky index]].
> See [[ISSUE-C1 Clear-sky time convention]].

Related: [[Physics clamp]]
