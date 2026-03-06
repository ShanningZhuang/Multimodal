# Unified Vision-Language Models

> Parent: [Vision-Language Models](00_Vision_Language.md)

## Overview

Unified models handle both **visual understanding** (image → text) and **visual generation** (text → image) within a single model. This is a frontier research area aiming to create models that perceive and create visual content — a step toward more general multimodal intelligence.

## The Unification Challenge

Understanding and generation require different visual representations:

```
Understanding:  Image → semantic features → LLM → "a cat on a mat"
                (needs high-level meaning)

Generation:     "a cat on a mat" → LLM → pixel-level features → Image
                (needs spatial/pixel detail)

Challenge: Semantic encoders (CLIP) are great for understanding but lack pixel detail.
           VAEs preserve pixel detail but lack semantic structure.
```

## Approach 1: Dual Encoder (Janus, BAGEL)

Use separate encoders for understanding and generation:

```
┌─────────────────────────────────────────────────────┐
│                    Janus / BAGEL                     │
│                                                     │
│  Understanding:                                     │
│    Image → SigLIP encoder → projector → ┐          │
│                                          ├→ LLM    │
│  Generation:                             │          │
│    Image → VAE encoder → projector →    ┘          │
│                                          │          │
│  LLM → visual tokens → VAE decoder → Generated img│
│                          or Diffusion               │
└─────────────────────────────────────────────────────┘
```

### Janus (DeepSeek)

- Dual encoder: SigLIP (understanding) + VQ tokenizer (generation)
- Autoregressive image generation via discrete visual tokens
- Simple approach: just different input processing for each task

### BAGEL (ByteDance)

- Dual encoder: SigLIP 2 (understanding) + FLUX VAE (generation)
- Uses diffusion for image generation (not autoregressive)
- Interleaved understanding and generation in one conversation

**Advantage**: Each encoder optimized for its task
**Disadvantage**: Architectural complexity, two visual processing paths

## Approach 2: Unified Encoder

Use a single visual representation for both tasks:

```
┌─────────────────────────────────────────────────┐
│              Unified Approach                    │
│                                                 │
│  Understanding:                                 │
│    Image → Single Encoder → LLM → text answer  │
│                                                 │
│  Generation:                                    │
│    Prompt → LLM → latent features → Decoder → Image│
│                                     (RAE)       │
│                                                 │
│  Same encoder features for both!                │
└─────────────────────────────────────────────────┘
```

### Show-o

- Uses VQ tokenizer for both understanding and generation
- Autoregressive (text) + discrete diffusion (image) in one transformer
- Simpler architecture than dual encoder

### Chameleon (Meta)

- Everything is discrete tokens (text + image via VQ-VAE)
- Purely autoregressive over interleaved image-text token sequences
- True early fusion — no separate encoders

### RAE-based (Emerging)

- Semantic encoder (SigLIP/DINOv2) + RAE decoder
- Diffusion in semantic latent space
- Single encoder handles both tasks effectively
- Simplest architecture, promising results

## Approach Comparison

| Approach | Understanding | Generation | Complexity | Example |
|----------|--------------|------------|------------|---------|
| Dual encoder | Excellent | Excellent | High | Janus, BAGEL |
| VQ unified | Good | Good | Medium | Show-o, Chameleon |
| Semantic + RAE | Very good | Good | Low | Emerging research |
| Separate models | Best-in-class | Best-in-class | Highest | GPT-4V + DALL-E 3 |

## Generation Methods

### Autoregressive (Token-by-token)

```
Text: "a sunset" → LLM → [v1] [v2] [v3] ... [vN] → VQ Decoder → Image

Each visual token generated left-to-right, like text generation.
```

Used by: Janus (VQ), Chameleon, LlamaGen

- Compatible with standard LLM inference
- Quality limited by VQ tokenizer
- Slow for high-resolution (many tokens)

### Diffusion-based

```
Text: "a sunset" → LLM → conditioning → Diffusion denoising → VAE Decoder → Image

Iterative denoising in continuous latent space.
```

Used by: BAGEL, Transfusion

- Higher generation quality
- Requires diffusion inference infrastructure
- Naturally handles continuous visual features

### Hybrid (AR + Diffusion)

```
Text → AR LLM → coarse visual tokens → Diffusion refinement → Image

Autoregressive for structure, diffusion for detail.
```

## Inference Implications

Unified models create unique **serving challenges**:

| Component | Compute Pattern | Memory Pattern |
|-----------|----------------|----------------|
| Text generation | Memory-bound (AR) | KV cache grows |
| Image understanding | Compute-bound (prefill) | Fixed |
| Image generation (AR) | Memory-bound | KV cache grows |
| Image generation (diffusion) | Compute-bound | Fixed per step |

Mixed workloads are hard to batch efficiently — see [Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md).

## Key Models Summary

| Model | Organization | Encoders | Generation | Open |
|-------|-------------|----------|------------|------|
| Janus Pro | DeepSeek | SigLIP + VQ | AR tokens | Yes |
| BAGEL | ByteDance | SigLIP 2 + FLUX VAE | Diffusion | Yes |
| Show-o | Various | VQ unified | Discrete diffusion | Yes |
| Chameleon | Meta | VQ-VAE | AR tokens | Yes |
| Transfusion | Meta | VAE | Diffusion within AR | Research |
| GPT-4o | OpenAI | Unknown | Unknown | No |
| Gemini | Google | Unknown | Unknown | No |

## Related

- [Architecture](01_Architecture.md) — fusion strategies these models build on
- [VAE](../visual_encoder/04_VAE.md) — reconstruction encoder used in dual-encoder models
- [Semantic Encoders](../visual_encoder/03_Semantic_Encoders.md) — understanding encoder, potential unified encoder
- [Diffusion Models](../diffusion/00_Diffusion.md) — generation backbone for diffusion-based models
- [Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving challenges for unified models
