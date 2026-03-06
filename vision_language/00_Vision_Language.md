# Vision-Language Models

> Parent: [Multimodal Models](../00_Multimodal.md)

## Overview

Vision-Language Models (VLMs) combine visual perception with language understanding in a single model. They can answer questions about images, describe visual content, follow visual instructions, and — in their latest unified forms — also generate images. VLMs bridge the visual encoders and LLMs we've studied separately.

## Core Idea

```
            ┌──────────────┐    ┌───────────┐    ┌──────────────┐
 Image ──→  │Visual Encoder│──→ │ Connector │──→ │     LLM      │ ──→ Text output
            │(ViT/CLIP)    │    │(Projection)│   │(Decoder-only)│
            └──────────────┘    └───────────┘    └──────────────┘
                                                        ▲
 Text prompt ──────────────────────────────────────────┘
```

The key challenge: how to effectively combine visual information with the LLM's text processing.

## Topics

| # | Topic | File | Description |
|---|-------|------|-------------|
| 1 | Architecture | [01_Architecture.md](01_Architecture.md) | VLM design patterns — fusion strategies, projectors |
| 2 | Models | [02_Models.md](02_Models.md) | LLaVA, Qwen-VL, InternVL — understanding-focused |
| 3 | Unified Models | [03_Unified_Models.md](03_Unified_Models.md) | Janus, BAGEL — generation + understanding |
| 4 | Training | [04_Training.md](04_Training.md) | Pretraining, instruction tuning, data pipelines |

## VLM Landscape

```
Understanding-only:           Unified (understand + generate):
┌─────────────────────┐      ┌─────────────────────────────┐
│ LLaVA               │      │ Janus (dual encoder)        │
│ Qwen-VL             │      │ BAGEL (dual encoder)        │
│ InternVL             │      │ Show-o (single encoder)     │
│ PaliGemma            │      │ Chameleon (VQ tokens)       │
│ Phi-3-Vision         │      │ Transfusion                 │
└─────────────────────┘      └─────────────────────────────┘
```

## Related

- [Visual Encoders](../visual_encoder/00_Visual_Encoder.md) — the vision backbone of VLMs
- [Diffusion Models](../diffusion/00_Diffusion.md) — used by unified models for image generation
- [Applications](../applications/00_Applications.md) — downstream tasks powered by VLMs
- [AI_Infra: Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving VLMs at scale
