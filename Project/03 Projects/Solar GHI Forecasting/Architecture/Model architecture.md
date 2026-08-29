---
tags: [architecture, moc]
type: note
params: 15505804
---

# Model architecture

**Physics-Gated Cross-Modal Fusion** (`PhysicsGatedFusionModelV2` in `model.py`).

```
tabular_seq (24,11) ──► [[Temporal branch]] ──────────► H_t (256) ──┐
                                                                    ├─► [[Physics gate]] ─► fused (256)
multi_frame (9,224,224) ─┐                                          │
roi_image   (3,224,224) ─┴► [[Image branch]] ─► 196 patches ─► [[Cross-attention]] ─► H_a (256) ─┘
                                                                    │
gate_features (4) ──────────────────────────────────────────────────┘

fused ‖ future_clearsky (3) → 259 → [[Prediction head]] → μ(3), σ(3)
```

## Parameter budget — memorise this ratio

| | Count | Share |
| --- | ---: | ---: |
| Total | 15,505,804 | 100% |
| **Image encoders** | **14,405,124** | **92.9%** |
| Trainable while CNN frozen | 1,100,680 | 7.1% |
| **Physics gate** | **193** | 0.001% |

That 92.9% against ~5,300 training samples is the seed of the whole negative result —
see [[Overfitting]].

## Presets

| Preset | Backbone | LSTM | Heads | D | Dropout |
| --- | --- | --- | --- | --- | --- |
| `large` | efficientnet_b2 | 128 | 8 | 256 | 0.15 |
| `small` | efficientnet_b0 | 64 | 4 | 128 | 0.30 |
