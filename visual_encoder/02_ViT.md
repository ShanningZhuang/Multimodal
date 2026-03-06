# Vision Transformer (ViT)

> Parent: [Visual Encoders](00_Visual_Encoder.md)

## Overview

The Vision Transformer (ViT) applies the transformer architecture directly to images by splitting them into patches and treating each patch as a token. ViT demonstrated that pure transformer architectures can match or exceed CNNs on vision tasks when trained at scale, and it became the foundation for modern visual encoders (CLIP, DINOv2, SigLIP).

## Architecture

```
Image (224x224x3)
       │
       ▼
┌──────────────────┐
│  Patch Embedding  │   Split into 16x16 patches → 196 patches
│  (Linear proj.)   │   Each patch: 16x16x3=768 → D-dim vector
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  [CLS] + Patches  │   Prepend learnable [CLS] token → 197 tokens
│  + Pos. Encoding   │   Add learnable positional embeddings
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Transformer      │   L layers of multi-head self-attention + FFN
│  Encoder Blocks   │   (same as language transformer encoder)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  [CLS] output     │   Use as image representation for classification
│  or all tokens     │   Use all tokens for dense prediction
└──────────────────┘
```

## Patch Embedding

The key insight: treat an image as a sequence of patches, analogous to tokens in NLP.

```python
# Patch embedding is just a convolution with kernel_size=stride=patch_size
patch_embed = nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,    # e.g., 768
    kernel_size=patch_size,     # e.g., 16
    stride=patch_size           # non-overlapping
)

# For 224x224 image with patch_size=16:
# Output: (batch, 768, 14, 14) → reshape to (batch, 196, 768)
# 196 = 14 * 14 = (224/16)^2 patches
```

### Patch Size Trade-offs

| Patch Size | Tokens (224x224) | Resolution | Compute |
|------------|------------------|------------|---------|
| 32x32 | 49 | Low | Low |
| 16x16 | 196 | Medium | Medium |
| 14x14 | 256 | Higher | Higher |
| 8x8 | 784 | High | Very high |

Smaller patches = more tokens = higher resolution but quadratic attention cost.

## Positional Encoding

Unlike CNNs, transformers have no built-in notion of spatial position. ViT uses **learnable positional embeddings** — one vector per patch position, added to the patch embeddings.

```
Learned positional embeddings (visualized as 2D cosine similarity):

Position (0,0)  Position (0,1)  Position (1,0) ...
┌───┐           ┌───┐           ┌───┐
│███│ high sim  │██░│           │██░│
│██░│ along     │█░░│           │░░░│
│█░░│ column    │░░░│           │░░░│
└───┘           └───┘           └───┘

→ Model learns 2D spatial structure from 1D position indices
```

Variants:
- **Learnable 1D**: original ViT — works well, learns 2D structure implicitly
- **Sinusoidal 2D**: fixed, based on (row, col) coordinates
- **RoPE 2D**: rotary position embeddings extended to 2D (used in modern VLMs)

## CLS Token

A special learnable token prepended to the patch sequence:
- Attends to all patches via self-attention
- Its output serves as a global image representation
- Used for classification (linear head on CLS output)

Alternative: **global average pooling** over all patch outputs (sometimes works equally well).

## ViT Variants

| Model | Params | Embed Dim | Layers | Heads | Patch Size |
|-------|--------|-----------|--------|-------|------------|
| ViT-S/16 | 22M | 384 | 12 | 6 | 16 |
| ViT-B/16 | 86M | 768 | 12 | 12 | 16 |
| ViT-L/16 | 307M | 1024 | 24 | 16 | 16 |
| ViT-H/14 | 632M | 1280 | 32 | 16 | 14 |
| ViT-G/14 | 1.8B | 1664 | 48 | 16 | 14 |

Naming convention: `ViT-{Size}/{Patch}` (e.g., ViT-L/14 = Large model, 14x14 patches)

## Key Properties

1. **Global receptive field from layer 1**: every patch attends to every other patch (vs. CNN's local-to-global)
2. **Scales with data**: needs large datasets (ImageNet-21k or larger) — poor with small data due to lack of inductive bias
3. **Flexible resolution**: can handle different image sizes by adjusting number of patches (with positional embedding interpolation)
4. **Transfer learning**: pretrained ViT features transfer well to downstream tasks

## ViT as Backbone for Multimodal Models

ViT is the foundation for nearly all modern visual encoders:

```
ViT Backbone
    │
    ├── CLIP ViT ← trained with language supervision (contrastive)
    ├── SigLIP ViT ← improved CLIP training (sigmoid loss)
    ├── DINOv2 ViT ← self-supervised (no language)
    └── WebSSL ViT ← self-supervised at web scale
```

These models use ViT architecture but differ in **training objective** — which determines what the representations capture.

## Related

- [CNN Basics](01_CNN_Basics.md) — predecessor architecture ViT replaces
- [Semantic Encoders](03_Semantic_Encoders.md) — CLIP, SigLIP, DINOv2 built on ViT
- [DiT](../diffusion/04_DiT.md) — applies transformer (similar to ViT) to diffusion
- [LLM KB: Attention Mechanism](../../LLM/transformer/) — same multi-head attention used in ViT
