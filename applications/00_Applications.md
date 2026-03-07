# Multimodal Applications

> Parent: [Multimodal Models](../00_Multimodal.md)

## Overview

This section covers downstream applications of multimodal models — where the visual encoders, diffusion models, and VLMs come together to solve real-world problems. Applications span image generation, video, and robotics.

## Topics

| # | Topic | File | Description |
|---|-------|------|-------------|
| 1 | Image Generation | [01_Image_Generation.md](01_Image_Generation.md) | Text-to-image, editing, inpainting |
| 2 | Video | [02_Video.md](02_Video.md) | Video generation and understanding |
| 3 | Robotics | [03_Robotics.md](03_Robotics.md) | VLA models, DiT for robotics, embodied AI |
| 4 | Omni-Modality Serving | [04_Omni_Serving.md](04_Omni_Serving.md) | Multi-stage pipelines, disaggregated inference (vLLM-Omni case study) |

## Application Landscape

```
                          Multimodal Models
                                │
          ┌─────────────┬───────┼───────┬──────────────┐
          ▼             ▼       ▼       ▼              ▼
    Image/Creative   Video  Robotics/  Omni-Modality
    ┌──────────┐  ┌────────┐ Embodied  Serving
    │Text2Image│  │Text2Vid│ ┌──────┐  ┌──────────┐
    │Editing   │  │VideoQA │ │VLA   │  │Multi-Stage│
    │Inpainting│  │Editing │ │DiT   │  │Pipelines │
    │ControlNet│  │Stream  │ │Sim2R │  │Disaggreg.│
    └──────────┘  └────────┘ └──────┘  └──────────┘
```

## Related

- [Generation Patterns](../multimodal_generation/01_Generation_Patterns.md) — how AR and DiT compose into end-to-end generation systems
- [Diffusion Models](../diffusion/00_Diffusion.md) — generation backbone
- [Vision-Language Models](../vision_language/00_Vision_Language.md) — understanding backbone
- [AI_Infra: Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving these applications

## Code

Use vLLM-omni as example for learning

https://docs.vllm.ai/projects/vllm-omni/en/latest/#about
https://blog.vllm.ai/2025/11/30/vllm-omni.html
https://docs.google.com/presentation/d/1qv4qMW1rKAqDREMXiUDLIgqqHQe7TDPj/edit?slide=id.p57#slide=id.p57
