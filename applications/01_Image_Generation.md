# Image Generation

> Parent: [Applications](00_Applications.md)

## Overview

Image generation covers creating, editing, and manipulating images using multimodal models. This page covers the major application patterns built on top of diffusion models and VLMs.

## Text-to-Image Generation

The core application of diffusion models:

```
"a photo of a golden retriever     →  Stable Diffusion /  →  [Generated Image]
 wearing sunglasses at the beach"      FLUX / DALL-E 3
```

### Current Systems

| System | Model | Approach | Strengths |
|--------|-------|----------|-----------|
| Stable Diffusion 3 | MMDiT | Open, latent diffusion | Customizable, community |
| FLUX.1 | DiT + flow matching | Open (dev/schnell) | Quality, prompt adherence |
| DALL-E 3 | Unknown (diffusion) | Proprietary (OpenAI) | Text rendering, safety |
| Midjourney | Unknown | Proprietary | Aesthetic quality |
| Imagen 3 | Cascaded diffusion | Proprietary (Google) | Photorealism |

## Controlled Generation

### ControlNet

Add spatial control to generation via additional conditions:

```
Control input (edge map, pose, depth)
       │
       ▼
┌──────────────┐
│  ControlNet   │  Parallel copy of U-Net/DiT encoder
│  (trainable)  │  Adds residuals to main model
└──────┬───────┘
       │ add residuals
       ▼
┌──────────────┐
│  Main Model   │  Frozen SD/FLUX model
│  (frozen)     │
└──────────────┘
       │
       ▼
  Generated image (follows control input structure)
```

Control types: Canny edges, depth maps, pose skeletons, segmentation maps, normal maps

### IP-Adapter

Use a reference image to guide style/content:
```
Reference image → CLIP features → additional cross-attention KV → Model → Output
```

### Inpainting

Replace specific regions of an image:
```
Original image + Mask + Text prompt → Model → Image with masked region regenerated
```

The model conditions on the unmasked region and generates content for the masked area.

## Image Editing

### Instruction-based Editing

```
Input image + "make it winter" → InstructPix2Pix / MagicBrush → Edited image
```

### SDEdit (Diffusion-based)

```
Input image → Add noise (partial) → Denoise with new prompt → Edited image

Noise level controls edit strength:
  Low noise  → subtle change (color, style)
  High noise → major change (new content)
```

## Practical Considerations

### Quality Factors

| Factor | Impact | Recommendation |
|--------|--------|---------------|
| Guidance scale (CFG) | Prompt adherence vs quality | 7-9 for photos, 3-5 for art |
| Steps | Quality vs speed | 25-30 (DPM-Solver++) |
| Resolution | Detail level | Native resolution of model |
| Negative prompt | Avoid artifacts | "blurry, low quality, deformed" |
| Seed | Reproducibility | Fix for consistent results |

### Inference Cost

```
Single image generation (512x512):
  SD 1.5:   ~1.5s on A100 (25 steps)
  SDXL:     ~5s on A100 (30 steps)
  FLUX.1:   ~10s on A100 (30 steps)
  FLUX.1 schnell: ~2s on A100 (4 steps, distilled)
```

## Related

- [Latent Diffusion](../diffusion/03_Latent_Diffusion.md) — underlying architecture
- [Sampling](../diffusion/02_Sampling.md) — samplers and guidance
- [DiT](../diffusion/04_DiT.md) — modern backbone for generation
- [Video](02_Video.md) — extension to temporal domain
- [Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving image generation
