# Visual Encoders

> Parent: [Multimodal Models](../00_Multimodal.md)

## Overview

Visual encoders transform raw images into representations that neural networks can process. This section covers the evolution from CNNs to Vision Transformers, contrastive/self-supervised encoders (CLIP, SigLIP 2, DINOv2), and Variational Autoencoders (VAE) used in generative models.

A key recent insight: semantic encoders (like SigLIP 2) can outperform VAEs for *both* understanding and generation tasks, challenging the assumption that separate encoders are needed.

## Encoder Taxonomy

```
Visual Encoders
├── Pixel-based
│   ├── CNN (ResNet, ConvNeXt)         → local features, translation equivariance
│   └── ViT (patch embedding)          → global attention, scalable
├── Semantic (high-level meaning)
│   ├── Language-supervised: CLIP, SigLIP 2
│   ├── Self-supervised: DINOv2, WebSSL
│   └── Properties: good for understanding, works for generation with RAE decoder
└── Reconstruction (pixel-faithful)
    ├── VAE: SD-VAE, FLUX VAE
    ├── RAE: Representation Autoencoder
    └── Properties: preserves spatial detail, traditional choice for generation
```

## Topics

| # | Topic | File | Description |
|---|-------|------|-------------|
| 1 | CNN Basics | [01_CNN_Basics.md](01_CNN_Basics.md) | Convolutions, pooling, ResNet — prerequisite foundations |
| 2 | Vision Transformer | [02_ViT.md](02_ViT.md) | Patch embedding, positional encoding, ViT architecture |
| 3 | Semantic Encoders | [03_Semantic_Encoders.md](03_Semantic_Encoders.md) | CLIP, SigLIP 2, DINOv2, WebSSL — contrastive & self-supervised |
| 4 | VAE | [04_VAE.md](04_VAE.md) | Variational Autoencoders, SD-VAE, FLUX VAE, RAE |
| 5 | Hands-On Lab | [05_Hands_On_Lab.md](05_Hands_On_Lab.md) | Trace VLM execution in vLLM — instrument CLIP, LLaVA, projectors |

## Key Insight: Unified Visual Representation

Traditional multimodal models (Janus, BAGEL) use **dual encoders** — a VAE for generation and a semantic encoder (SigLIP 2) for understanding. Recent work on Representation Autoencoders (RAE) shows that a single semantic encoder can handle both tasks, simplifying architecture significantly.

## Related

- [Diffusion Models](../diffusion/00_Diffusion.md) — use visual encoders for latent space
- [Vision-Language Models](../vision_language/00_Vision_Language.md) — use visual encoders for image understanding
- [LLM KB: Transformer Architecture](../../LLM/transformer/) — ViT builds on transformer attention
