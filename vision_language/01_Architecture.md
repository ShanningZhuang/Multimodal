# VLM Architecture Patterns

> Parent: [Vision-Language Models](00_Vision_Language.md)

## Overview

The central question in VLM design is **how to fuse visual and language information**. Different architectures make different trade-offs between efficiency, capability, and complexity. This page covers the main design patterns used in modern VLMs.

## Fusion Strategies

### Late Fusion (Projection-based)

Most common approach. Process image and text independently, then combine:

```
Image → Visual Encoder → [visual tokens] ──→ Projector ──→ ┐
                                                            ├→ LLM → output
Text  → Tokenizer     → [text tokens]   ────────────────→ ┘
```

The visual tokens are projected into the LLM's embedding space and concatenated with text tokens. The LLM processes them together via self-attention.

Used by: LLaVA, Qwen-VL, InternVL, PaliGemma

**Advantages**: Simple, leverages pretrained LLMs and encoders
**Disadvantages**: No early interaction between modalities

### Cross-Attention Fusion

Insert cross-attention layers into the LLM that attend to visual features:

```
LLM Layer N:    Self-Attention → Cross-Attention → FFN
                     ↑                  ↑
                text tokens      visual features (from encoder)
```

Used by: Flamingo, Llama 3.2 Vision

**Advantages**: More fine-grained visual grounding
**Disadvantages**: Modifies the LLM architecture, harder to leverage pretrained models

### Early Fusion

Process raw image patches directly as input tokens (no separate encoder):

```
Image patches → Linear embedding → ┐
                                    ├→ Transformer → output
Text tokens   → Embedding       → ┘

(Interleaved from the start)
```

Used by: Fuyu, some research models

**Advantages**: End-to-end, no frozen encoder limitations
**Disadvantages**: Very expensive (many tokens), needs training from scratch

## Visual Token Projectors

The projector transforms visual encoder output into the LLM's input space:

### Linear Projection

```python
# Simplest approach — used in LLaVA v1
projector = nn.Linear(vision_dim, llm_dim)  # e.g., 1024 → 4096
```

### MLP Projector

```python
# Used in LLaVA v1.5 — better performance
projector = nn.Sequential(
    nn.Linear(vision_dim, llm_dim),
    nn.GELU(),
    nn.Linear(llm_dim, llm_dim),
)
```

### C-Abstractor / Resampler

Reduce number of visual tokens before feeding to LLM:

```
Visual encoder output: 576 tokens (24x24 patches for ViT-L/14 @ 336px)
                         │
                    Resampler/Q-Former
                         │
                    64 or 128 query tokens → LLM

Reduces compute significantly (576 → 64 tokens)
```

Used by: BLIP-2 (Q-Former), Qwen-VL (cross-attention resampler)

### Token Compression Comparison

| Approach | Visual Tokens | Quality | LLM Compute |
|----------|--------------|---------|-------------|
| No compression | 576+ | Best | High |
| MLP (LLaVA) | 576 | Very good | High |
| Resampler (BLIP-2) | 32-64 | Good | Low |
| Avg pooling 2x2 | 144 | Good | Medium |
| Dynamic (InternVL) | 256-1024 | Very good | Adaptive |

## Resolution Handling

Higher resolution → more visual detail but more tokens:

### Fixed Resolution

Resize all images to fixed size (e.g., 336x336):
- Simple but loses detail for high-res images
- 336px with ViT-L/14 → 576 tokens

### Dynamic Resolution (Tile-based)

Split large images into tiles, process each separately:

```
High-res image (1344x672)
       │
       ▼
┌──┬──┬──┬──┐
│T1│T2│T3│T4│  4 tiles (336x336 each)
└──┴──┴──┴──┘
       +
  [Thumbnail]    1 downscaled overview (336x336)
       │
       ▼
  5 × 576 = 2880 tokens → LLM
```

Used by: LLaVA-NeXT, InternVL 2, Qwen-VL 2

**Trade-off**: More tiles → better detail but significantly more tokens → higher cost.

## Multi-Image & Video Support

### Multi-Image

Concatenate visual tokens from multiple images:
```
Image 1 → encoder → [vis_tokens_1] ──→ ┐
Image 2 → encoder → [vis_tokens_2] ──→ ├→ LLM
Image 3 → encoder → [vis_tokens_3] ──→ ┘
```

### Video

Sample frames, process each as an image:
```
Video → sample N frames → encode each → concatenate → LLM

Challenge: N frames × 576 tokens = thousands of tokens
Solutions: temporal pooling, frame sampling, token compression
```

## Architecture Decision Tree

```
Need maximum understanding quality?
  → Late fusion with MLP projector (LLaVA-style)
  → High-res: dynamic tiling

Need generation + understanding?
  → Dual encoder (Janus, BAGEL) or unified (Show-o)
  → See: Unified Models

Need efficiency / fast inference?
  → Resampler / token compression
  → Fewer tiles

Need to preserve pretrained LLM?
  → Late fusion (don't modify LLM architecture)
  → Cross-attention requires modifying LLM layers
```

## Related

- [Models](02_Models.md) — specific VLM implementations using these patterns
- [Unified Models](03_Unified_Models.md) — architectures that also generate images
- [Training](04_Training.md) — how these architectures are trained
- [Visual Encoders](../visual_encoder/00_Visual_Encoder.md) — encoder choices for VLMs
