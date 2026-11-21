---
tags: [concept, deployment]
type: note
---

# Physics clamp

A safety rail on the deployed forecast: cap it at **1.15 × the preceding-hour-mean
[[Clear-sky GHI]]**, since irradiance cannot meaningfully exceed the clear-sky value.

Uses the *hour-mean* clear-sky, not the instantaneous value, because Open-Meteo's hourly
labelling convention makes the instantaneous value badly wrong near sunrise and sunset.
(The same convention issue that [[ISSUE-C1 Clear-sky time convention]] is about — the clamp
got fixed, the features did not.)

Also forces true night forecasts to zero.

> [!question] "Isn't a clamp hiding model failures?"
> It is a safety rail on a *deployed* system, and a `capped` flag is logged whenever it
> fires, so it is auditable rather than silent. It is **not applied during offline
> evaluation**, so it doesn't flatter any reported metric. A clamp firing often would be a
> symptom — which is exactly why it's logged.
