# Multimodal Generation Architecture

> Parent: [Multimodal Models](../00_Multimodal.md)

## Overview

This section explains how components (visual encoders, diffusion models, VLMs) **compose into complete generation systems**. It bridges the gap between understanding individual components and understanding end-to-end generation pipelines.

## Topics

| # | Topic | File | Description |
|---|-------|------|-------------|
| 1 | Generation Patterns | [01_Generation_Patterns.md](01_Generation_Patterns.md) | AR+DiT composition patterns, data flows, component roles |

## Prerequisites

You should be familiar with:
- [Visual Encoders](../visual_encoder/00_Visual_Encoder.md) — CLIP/SigLIP, VAE
- [Diffusion Models](../diffusion/00_Diffusion.md) — DDPM, latent diffusion, DiT
- [Vision-Language Models](../vision_language/00_Vision_Language.md) — VLM architectures, unified models

## Related

- [Applications](../applications/00_Applications.md) — practical deployment of these patterns
- [Omni-Modality Serving](../applications/04_Omni_Serving.md) — how vLLM-Omni serves these architectures
