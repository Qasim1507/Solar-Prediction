---
tags: [concept]
type: note
---

# Optical flow (Farnebäck)

Estimates how pixels **moved** between two consecutive frames, producing a velocity vector
per pixel. Averaged over the centre region around Singapore, it gives two numbers:
`v_x` (east-west) and `v_y` (north-south) **cloud motion**.

Fed to the [[Physics gate]] as inputs 3 and 4 — an explicit cloud-advection prior.

Computed with `cv2.calcOpticalFlowFarneback` at 64×64 for speed; whole dataset in ~8s.

> [!bug] Two problems found
> 1. In production it was always exactly (0,0) because the previous frame was never fetched
>    — see [[ISSUE-B3 Optical flow always zero]] (**fixed**).
> 2. Flow pairs are built by **row position**, so 10% of them pair this morning's frame with
>    *yesterday evening's* — see [[ISSUE-C2 Overnight gap]] (**open**).
