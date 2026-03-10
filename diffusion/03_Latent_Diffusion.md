# Latent Diffusion Models

> Parent: [Diffusion Models](00_Diffusion.md)

## Overview

Latent Diffusion Models (LDM) perform the diffusion process in the latent space of a pretrained autoencoder (VAE) rather than in pixel space. This dramatically reduces computational cost while maintaining generation quality. Stable Diffusion is the most well-known implementation of this idea.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Latent Diffusion Model                           │
│                                                                     │
│  ┌─────────┐    ┌──────────────────────────┐    ┌─────────┐        │
│  │   VAE   │    │    Denoising Network      │    │   VAE   │        │
│  │ Encoder │──→ │  (U-Net or DiT)           │ ──→│ Decoder │        │
│  │         │    │                            │    │         │        │
│  │ 512x512 │    │  Operates on 64x64x4      │    │ 64x64x4 │        │
│  │   → z   │    │  latent space              │    │→ 512x512│        │
│  └─────────┘    │                            │    └─────────┘        │
│                 │  ┌──────────────────────┐  │                      │
│                 │  │ Conditioning         │  │                      │
│                 │  │ (text, image, etc.)  │  │                      │
│                 │  └──────────────────────┘  │                      │
│                 └──────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Latent Space?

| Aspect | Pixel Diffusion | Latent Diffusion |
|--------|----------------|-----------------|
| Resolution | 512x512x3 = 786K | 64x64x4 = 16K |
| Compression | None | ~48x fewer values |
| Training cost | Very high | Feasible on consumer GPUs |
| Quality | Excellent | Excellent (VAE is near-lossless) |

The VAE is trained separately and frozen during diffusion training.

## Stable Diffusion Architecture (v1/v2)

### Components

```
Text Prompt: "a cat sitting on a mat"
       │
       ▼
┌──────────────┐
│ Text Encoder │    CLIP text encoder (frozen)
│  (CLIP)      │    → sequence of text embeddings
└──────┬───────┘
       │
       ▼ cross-attention
┌──────────────────────────────────────────┐
│              U-Net                        │
│                                          │
│  Down blocks         Mid block    Up blocks
│  ┌────┐             ┌────┐      ┌────┐  │
│  │ResB│→│AttnB│→    │ResB│→    │ResB│→  │
│  │+Dwn│  │+Cross│   │Attn│    │AttnB│  │
│  └────┘  └─────┘    └────┘    │+Up  │  │
│     ↓                           └────┘  │
│  ┌────┐                        ┌────┐  │
│  │ResB│→│AttnB│→    ───────→   │ResB│  │
│  │+Dwn│  │+Cross│  skip conn  │AttnB│  │
│  └────┘  └─────┘              │+Up  │  │
│                                └────┘  │
│  Timestep t → sinusoidal embedding     │
│  → injected via adaptive LayerNorm     │
└──────────────────────────────────────────┘
       │
       ▼
  Predicted noise ε_θ(z_t, t, c_text)
```

### Key U-Net Components

- **ResNet blocks**: convolutional processing with residual connections
- **Self-attention**: spatial attention within the latent features
- **Cross-attention**: conditions on text embeddings (how text guides generation)
- **Skip connections**: U-Net's encoder-decoder skip connections preserve detail
- **Timestep embedding**: tells the model the current noise level

### Cross-Attention Mechanism

```
Q = Linear(latent_features)     # What the image "asks about"
K = Linear(text_embeddings)     # What the text provides
V = Linear(text_embeddings)     # Text information to inject

Attention = softmax(QK^T / √d) V

→ Each spatial position in the latent attends to relevant text tokens
→ "cat" tokens activate cat-related spatial regions
```

## Stable Diffusion 3 / MMDiT

SD3 replaces the U-Net with a **Multimodal Diffusion Transformer (MMDiT)**:

```
┌────────────────────────────────────────────────┐
│                   MMDiT Block                   │
│                                                │
│  Image tokens:  [z₁, z₂, ..., z_n]            │
│  Text tokens:   [t₁, t₂, ..., t_m]            │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │ Joint Self-Attention                      │  │
│  │                                          │  │
│  │  Concatenate image + text tokens         │  │
│  │  → full bidirectional attention          │  │
│  │  → image and text tokens interact freely │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  Separate FFNs for image and text              │
│  Image FFN: process image tokens               │
│  Text FFN: process text tokens                 │
│                                                │
│  Timestep → adaLN-Zero (adaptive LayerNorm)    │
└────────────────────────────────────────────────┘
```

Key differences from SD 1/2:
- No U-Net — pure transformer (DiT-based)
- Joint attention between text and image (no separate cross-attention)
- Multiple text encoders (CLIP + T5)
- Flow matching objective instead of DDPM

## Conditioning Methods

How conditions (text, images, etc.) are injected into the model:

| Method | Mechanism | Used In |
|--------|-----------|---------|
| Cross-attention | Q from image, KV from condition | SD 1/2 (U-Net) |
| Joint attention | Concatenate condition + image tokens | SD 3, FLUX (DiT) |
| Concatenation | Concat condition to input channels | img2img, inpainting |
| ControlNet | Parallel encoder adds residuals | Pose/edge conditioning |
| IP-Adapter | Image embeddings as additional KV | Image-conditioned generation |
| T2I-Adapter | Lightweight feature injection | Structural control |

## Text Encoders

| Model | Text Encoder | Embedding Dim | Max Tokens |
|-------|-------------|---------------|------------|
| SD 1.x | CLIP ViT-L/14 | 768 | 77 |
| SD 2.x | OpenCLIP ViT-H/14 | 1024 | 77 |
| SDXL | CLIP-G + CLIP-L (concat) | 1280+768 | 77 |
| SD 3 | CLIP-G + CLIP-L + T5-XXL | 1280+768+4096 | 77+256 |
| FLUX.1 | CLIP-L + T5-XXL | 768+4096 | 77+512 |

Larger text encoders → better text understanding → better prompt following.

## Resources

**Papers**

- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Rombach et al., CVPR 2022. The LDM / Stable Diffusion paper.
- [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — Podell et al., ICLR 2024. Larger U-Net, dual text encoders, refiner.
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Esser et al., ICML 2024. SD3 / MMDiT paper — flow matching + multimodal DiT.
- [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — Zhang et al., ICCV 2023. ControlNet — spatial conditioning via parallel encoder.

**Blogs**

- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) — Jay Alammar. Visual walkthrough of the full Stable Diffusion pipeline.
- [Stable Diffusion Deep Dive](https://huggingface.co/blog/stable_diffusion) — HuggingFace. How to use and understand SD with the diffusers library.

## Related

- [VAE](../visual_encoder/04_VAE.md) — the autoencoder that defines the latent space
- [DiT](04_DiT.md) — transformer architecture replacing U-Net in SD3/FLUX
- [Diffusion Basics](01_Diffusion_Basics.md) — underlying diffusion theory
- [Image Generation](../applications/01_Image_Generation.md) — practical applications
