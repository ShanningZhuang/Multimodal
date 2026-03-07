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
| **Nano Banana Pro** | **Autoregressive (Gemini 3 Pro)** | **Proprietary (Google)** | **Text rendering, reasoning, up to 4K** |
| **Qwen-Image** | **20B MMDiT (AR + DiT)** | **Open (Apache 2.0)** | **Text rendering (CJK), editing, open-source** |

### The Autoregressive Shift: Nano Banana Pro

Nano Banana Pro (built on Gemini 3 Pro) represents a fundamental architectural departure from diffusion-based generators. Instead of iterative denoising, it generates images **autoregressively** — the same way LLMs generate text.

```
Diffusion approach (SD, FLUX):
  Random noise → denoise step 1 → step 2 → ... → step 50 → Image

Autoregressive approach (Nano Banana Pro):
  Prompt → Gemini 3 Pro (think) → predict token 1 → token 2 → ... → token ~1290 → Decode → Image
```

**How it works:**

1. **Thinking step** — Gemini 3 Pro's reasoning ("thinking") is mandatory and acts as prompt augmentation, orienting the user's intent before generation begins. The model may generate interim images during thinking to test composition.
2. **Token generation** — the text encoder (Gemini 3 Pro) generates ~1,290 autoregressive image tokens for 1 megapixel output, ~1,120 tokens for 4MP, and ~2,000 tokens for 16MP (4K). Each token is sampled from a probability distribution, with randomness at every step.
3. **Image decoding** — tokens are decoded into pixels by an image decoder.

**Why this matters for the field:**

| Property | Diffusion Models | Nano Banana Pro (AR) |
|----------|-----------------|---------------------|
| Generation process | Iterative denoising (20-50 steps) | Sequential token prediction (~1,290 tokens) |
| Negative prompt | Yes (replaces null condition in CFG) | **No** — just a single prompt, like an LLM |
| CFG / guidance scale | Required for quality | **Not needed** — reasoning handles prompt adherence |
| Seed reproducibility | Deterministic (seed → noise → image) | **Non-deterministic** — randomness at every token step; seed cannot control output |
| Text rendering | Historically weak | Excellent — LLM backbone understands text semantics |
| Multi-image grids | Difficult (fixed latent space) | Natural — AR generation is aware of already-generated subimages |
| Prompt understanding | CLIP/T5 text encoder (separate from generator) | **Same model** — the LLM that understands the prompt also generates the image |

The key insight: by unifying the text encoder and the image generator into a single LLM, Nano Banana Pro leverages the model's world knowledge and reasoning for image generation. There is no separate CLIP encoder, no CFG, no negative prompt — just a prompt in, image out, exactly like how LLMs handle text.

**Model variants:**

| Variant | Base LLM | Tokens/Image | Max Resolution | Speed |
|---------|----------|-------------|---------------|-------|
| Nano Banana | Gemini 2.5 Flash | ~1,290 | 1MP (1K) | Fast |
| Nano Banana Pro | Gemini 3 Pro | ~1,120 | 4MP (2K) / 16MP (4K) | 20s-60s+ (thinking) |
| Nano Banana 2 | Gemini 3.1 Flash | — | — | Fast (Pro quality + Flash speed) |

### Open-Source Alternative: Qwen-Image

Qwen-Image is a **20B parameter MMDiT** (Multimodal Diffusion Transformer) from Alibaba's Qwen team, released under Apache 2.0. It follows the **Pattern 1: AR + DiT(Main)** architecture described in [Omni-Modality Serving](04_Omni_Serving.md):

```
                   Qwen-Image Architecture

Text prompt ──► Qwen2.5-VL ──► Text/Image    ──► MMDiT ──► VAE ──► Image
                (AR decoder)    embeddings        (DiT)     Decoder
                                    │              ▲
Input image ──► VAE Encoder ────────┘              │
(for editing)   (appearance)                   Patchify
                                              (latent patches)
```

**Key characteristics:**

- **20B parameters** — one of the largest open-source image generation models
- **Text rendering excellence** — particularly strong at Chinese and English text in images, outperforming most competitors on typography benchmarks
- **Dual-path editing** — for Qwen-Image-Edit, the input image is simultaneously processed by Qwen2.5-VL (semantic understanding) and a VAE encoder (appearance preservation), enabling both semantic edits (style transfer, rotation) and appearance edits (local modifications while keeping other regions unchanged)
- **Multi-task trained** — jointly trained on text-to-image (T2I), text-and-image-to-image (TI2I), and image-to-image (I2I) reconstruction, ensuring latent alignment between the VLM and DiT
- **Practical CFG** — uses `true_cfg_scale` (default 4.0), but negative prompt can be empty (`" "`) and still produce good results. Far less sensitive to negative prompt engineering than SD/FLUX

```python
# Qwen-Image generation — minimal prompt, no negative prompt needed
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
image = pipe(
    prompt="a cat wearing a top hat, watercolor style",
    negative_prompt=" ",          # empty is fine
    num_inference_steps=50,
    true_cfg_scale=4.0,
).images[0]
```

**Qwen-Image-Edit** extends this with instruction-based editing:

```python
# Edit: just describe what you want changed
image = pipe(
    prompt="change the cat's hat to red",
    input_image=original_image,   # dual-path: VL understands, VAE preserves
    num_inference_steps=50,
)
```

The dual-path design is what makes editing precise: Qwen2.5-VL *understands* what to change (semantic path), while the VAE encoder *preserves* what should stay the same (appearance path). This avoids the blunt approach of SDEdit (add noise → re-denoise), giving much finer control.

### Generation Paradigm Comparison

| | Diffusion (SD/FLUX) | AR-only (Nano Banana Pro) | AR + DiT (Qwen-Image) |
|---|---|---|---|
| Architecture | U-Net/DiT denoiser | LLM token predictor + decoder | VLM encoder + MMDiT denoiser |
| Prompt interface | Prompt + negative prompt | Prompt only | Prompt only (negative optional) |
| CFG required | Yes | No | Optional (default 4.0) |
| Generation mechanism | Iterative denoising | Sequential token prediction | Latent diffusion |
| Text rendering | Weak | Excellent | Excellent (esp. CJK) |
| Image editing | Separate models/pipelines | Multi-turn conversation | Built-in dual-path |
| Open-source | Yes (SD, FLUX) | No (proprietary) | Yes (Apache 2.0) |
| Serving via vLLM-Omni | DiT pipeline | N/A | Yes (Pattern 1) |

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

These factors apply primarily to **diffusion-based** models (SD, FLUX, Qwen-Image). Autoregressive models like Nano Banana Pro handle most of these internally through reasoning.

| Factor | Diffusion Models | AR Models (Nano Banana Pro) |
|--------|-----------------|----------------------------|
| Guidance scale (CFG) | 7-9 for photos, 3-5 for art | N/A (no CFG) |
| Steps | 25-50 (quality vs speed) | N/A (token count is fixed) |
| Resolution | Native resolution of model | Select via aspect ratio param |
| Negative prompt | "blurry, low quality, deformed" | N/A (just describe what you want) |
| Seed | Fix for consistent results | Unreliable (randomness at every token) |

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
- [Omni-Modality Serving](04_Omni_Serving.md) — serving Qwen-Image and other multi-stage models via vLLM-Omni
- [Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving image generation
