---
tags: [defence, script]
type: note
---

# The 60-second answer

For *"so, tell me what you did."* Say roughly this:

> Solar panels produce power in proportion to the sunlight hitting them, and the only thing
> that meaningfully interrupts that sunlight is cloud. Grid operators have to decide hours in
> advance how much backup generation to keep running, so they need to know how much sun there
> will be one to three hours from now.

> I built a system that forecasts solar irradiance over Singapore at one, two and three hours
> ahead. It combines two very different kinds of information: a **time series** of weather at
> the site, and **satellite images** of the cloud field around Singapore, which show weather
> that hasn't arrived yet.

> The novel part is how I combine them. Most work just glues the two representations together
> and lets the network figure it out. I argued we already know from physics *when* each source
> should matter — on a clear day the local trend is enough; under broken cloud the spatial
> picture dominates. So I built a small gate that takes physical state — clear-sky index,
> cloud cover, cloud motion — and produces a single number α that decides how much to weight
> each branch.

> I tested it against three ablations of itself and against classical baselines. The headline
> result is **negative**: adding the satellite branch made forecasts worse, the gate collapsed
> to a near-constant, and gradient-boosted trees on eleven tabular features beat every deep
> variant. I think that's the most defensible finding in the project, and I can explain
> exactly why it happened.

Then stop. Let them ask.

Related: [[Problem statement]] · [[Ablation results]] · [[Gate collapse]]
