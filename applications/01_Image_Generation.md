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

#### Classifier-Free Guidance (CFG)

During training, diffusion models randomly drop the text condition (replace with null/empty) some percentage of the time. This gives the model two capabilities: conditional generation \(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, c)\) and unconditional generation \(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)\).

At inference, CFG steers the denoising direction by amplifying the difference between conditioned and unconditioned predictions:

\[
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, c) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing) + w \left[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)\right]
\]

- \(w = 1.0\): no guidance, model follows its learned distribution as-is
- \(w > 1.0\): overshoots toward the conditioned direction — higher prompt fidelity but eventually saturated colors / artifacts
- \(w = 0.0\): purely unconditional generation, ignores the prompt entirely

This works because \(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)\) points in the direction of "what the condition adds," and scaling it up makes the model follow the prompt more aggressively. It requires **two forward passes per step** (one conditioned, one unconditioned), doubling compute. Distilled models (FLUX schnell, SDXL Turbo) bake guidance into the weights so they skip this.

#### Negative Prompt

LLMs have no equivalent because autoregressive generation is a single forward pass conditioned on one token sequence. In diffusion models, since CFG already runs an unconditional pass, the negative prompt simply **replaces the null condition** \(\varnothing\) with an undesired condition \(c_{\text{neg}}\):

\[
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, c_{\text{pos}}, c_{\text{neg}}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, c_{\text{neg}}) + w \left[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, c_{\text{pos}}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, c_{\text{neg}})\right]
\]

Now the model moves *away from* \(c_{\text{neg}}\) and *toward* \(c_{\text{pos}}\). Typical negative prompts like "blurry, low quality, deformed, extra fingers" steer the model away from common failure modes of the training data. This is essentially free — it reuses the second forward pass that CFG already requires.

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
