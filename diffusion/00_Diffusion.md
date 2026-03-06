# Diffusion Models

> Parent: [Multimodal Models](../00_Multimodal.md)

## Overview

Diffusion models generate data by learning to reverse a gradual noising process. Starting from pure noise, they iteratively denoise to produce high-quality images, video, or other outputs. Diffusion models have become the dominant approach for image generation, surpassing GANs in both quality and diversity.

## Core Idea

```
Forward Process (fixed):   x₀ ──→ x₁ ──→ x₂ ──→ ... ──→ x_T
                          clean   slightly   more          pure
                          image   noisy      noisy         noise

Reverse Process (learned): x_T ──→ x_{T-1} ──→ ... ──→ x₀
                           pure    slightly               clean
                           noise   cleaner                image

The model learns to predict and remove noise at each step.
```

## Topics

| # | Topic | File | Description |
|---|-------|------|-------------|
| 1 | Diffusion Basics | [01_Diffusion_Basics.md](01_Diffusion_Basics.md) | DDPM, forward/reverse process, noise schedules |
| 2 | Sampling | [02_Sampling.md](02_Sampling.md) | DDIM, DPM-Solver, classifier-free guidance |
| 3 | Latent Diffusion | [03_Latent_Diffusion.md](03_Latent_Diffusion.md) | Stable Diffusion architecture, conditioning |
| 4 | DiT | [04_DiT.md](04_DiT.md) | Diffusion Transformer — replacing U-Net with transformers |

## Evolution

```
Timeline:
2020  DDPM ────────── Foundation: simple denoising objective
2021  Guided Diff. ── Classifier guidance for conditional generation
2021  DDIM ────────── Deterministic sampling, fewer steps
2022  Stable Diff. ── Latent diffusion + text conditioning (U-Net)
2022  DiT ─────────── Replace U-Net with transformer
2023  SDXL ────────── Improved SD with larger U-Net
2023  SD 3 ────────── MMDiT (multimodal DiT)
2024  FLUX.1 ───────── Improved flow matching + DiT
2024  Sora ─────────── Video generation with DiT
```

## Diffusion vs Other Generative Models

| Aspect | Diffusion | GAN | VAE | Autoregressive |
|--------|-----------|-----|-----|----------------|
| Quality | Excellent | Excellent | Good | Excellent |
| Diversity | High | Mode collapse risk | High | High |
| Training | Stable | Unstable | Stable | Stable |
| Sampling speed | Slow (many steps) | Fast (1 step) | Fast (1 step) | Slow (sequential) |
| Controllability | High (guidance) | Limited | Limited | High (prompting) |

## Related

- [Visual Encoders](../visual_encoder/00_Visual_Encoder.md) — VAE provides latent space for diffusion
- [Applications: Image Generation](../applications/01_Image_Generation.md) — downstream use of diffusion models
- [AI_Infra: Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving diffusion models at scale
