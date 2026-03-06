# Semantic Visual Encoders

> Parent: [Visual Encoders](00_Visual_Encoder.md)

## Overview

Semantic encoders produce visual representations that capture high-level meaning rather than pixel-level details. They are trained with either language supervision (contrastive learning with text) or self-supervised objectives (no labels needed). These encoders are the visual backbone of modern VLMs and, surprisingly, can also work well for image generation when combined with a suitable decoder (RAE).

## Training Paradigms

```
┌─────────────────────────────────────────────────────────┐
│                Training Approaches                       │
├────────────────────────┬────────────────────────────────┤
│  Language-Supervised    │  Self-Supervised               │
│                        │                                │
│  Image + Text pairs    │  Images only                   │
│  CLIP, SigLIP 2        │  DINOv2, WebSSL                │
│                        │                                │
│  Learns text-aligned   │  Learns visual structure       │
│  representations       │  (no language bias)            │
│                        │                                │
│  Good for zero-shot    │  Good for dense prediction     │
│  classification,       │  (segmentation, depth,         │
│  retrieval, VLMs       │  matching)                     │
└────────────────────────┴────────────────────────────────┘
```

## CLIP (Contrastive Language-Image Pretraining)

### Training Objective

CLIP learns to align images and text in a shared embedding space using **contrastive learning**.

```
                    Text Encoder
                   ┌───────────┐
 "a dog"      ──→  │Transformer│ ──→  T₁  ─┐
 "a cat"      ──→  │           │ ──→  T₂   │
 "a car"      ──→  │           │ ──→  T₃   │  Maximize similarity
                   └───────────┘            │  on diagonal
                                            ▼
                                     ┌─────────────┐
                                     │ T₁  T₂  T₃  │
                                     │ ┌──┬──┬──┐  │
                                I₁   │ │██│  │  │  │  ← match
                                I₂   │ │  │██│  │  │  ← match
                                I₃   │ │  │  │██│  │  ← match
                                     │ └──┴──┴──┘  │
                    Image Encoder    └─────────────┘
                   ┌───────────┐            ▲
 [dog photo]  ──→  │  ViT      │ ──→  I₁  ─┘
 [cat photo]  ──→  │           │ ──→  I₂
 [car photo]  ──→  │           │ ──→  I₃
                   └───────────┘
```

Loss: InfoNCE (softmax cross-entropy over cosine similarities in the batch)

```python
# Simplified CLIP loss
logits = image_features @ text_features.T * temperature
labels = torch.arange(batch_size)
loss = (F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)) / 2
```

### Key Properties
- **Zero-shot classification**: compare image embedding to text embeddings of class names
- **Open vocabulary**: understands arbitrary text concepts without task-specific training
- **Trained on 400M image-text pairs** from the internet (WIT dataset)

## SigLIP 2 (Sigmoid Loss for Language-Image Pretraining)

Improvement over CLIP's contrastive loss:

| Aspect | CLIP (Softmax) | SigLIP (Sigmoid) |
|--------|----------------|-------------------|
| Loss | Softmax over batch | Binary sigmoid per pair |
| Batch dependency | Normalizes across batch | Each pair independent |
| Scaling | Needs large batches | Works with any batch size |
| Negatives | All non-matching pairs | Each pair is pos/neg |

```python
# SigLIP loss — no need for global batch normalization
logits = image_features @ text_features.T * temperature + bias
labels = 2 * torch.eye(batch_size) - 1  # +1 for matches, -1 for non-matches
loss = -F.logsigmoid(labels * logits).mean()
```

**SigLIP 2** adds:
- Larger and more diverse training data
- Multiple resolution support
- Improved training recipes
- State-of-the-art on VLM benchmarks as visual backbone

## DINOv2 (Self-Supervised)

Trained **without any language supervision** using self-distillation:

```
                  ┌──────────────┐
  Augmented       │   Student     │
  View 1    ──→   │   Network     │ ──→  s₁
                  └──────┬───────┘
                         │ minimize distance
                  ┌──────┴───────┐
  Augmented       │   Teacher     │
  View 2    ──→   │   (EMA)       │ ──→  t₂
                  └──────────────┘

Teacher = exponential moving average of student weights
Loss = cross-entropy between student and teacher outputs
```

Key properties:
- **No language bias**: learns purely visual structure
- **Excellent for dense prediction**: segmentation, depth estimation, correspondence
- **Strong linear probing**: features are highly informative even with a linear classifier
- **Trained on LVD-142M**: curated dataset of 142M images

## WebSSL

Self-supervised learning at web scale:
- Uses uncurated web images (larger and more diverse than DINOv2's data)
- Demonstrates that scale of data matters more than curation for SSL
- Achieves strong performance across understanding benchmarks

## Comparison

| Encoder | Training | Data | Strengths | Used In |
|---------|----------|------|-----------|---------|
| CLIP ViT-L/14 | Contrastive (text) | 400M pairs | Zero-shot, VLM backbone | LLaVA, many VLMs |
| SigLIP 2 So400m | Sigmoid (text) | Large-scale | Best VLM backbone, scalable | PaliGemma, Qwen-VL |
| DINOv2-L | Self-distillation | 142M images | Dense prediction, no text bias | Depth estimation, robotics |
| WebSSL-L | Self-supervised | Web-scale | Scale, robustness | Research |

## For Generation: The RAE Connection

Traditional assumption: VAE needed for generation, semantic encoder for understanding.

**Recent finding**: semantic encoders + RAE decoder can do both:

```
Understanding:     Image → SigLIP 2 → features → LLM → text answer
                                  │
Generation:        Text → LLM → features → RAE Decoder → image
                                  │
Same encoder features for both! ─┘
```

This eliminates the need for dual encoders (e.g., Janus's separate CLIP + VAE), simplifying architecture and reducing overhead.

## Related

- [ViT](02_ViT.md) — architecture used by all semantic encoders
- [VAE](04_VAE.md) — alternative encoder type for reconstruction/generation
- [VLM Architecture](../vision_language/01_Architecture.md) — how semantic encoders connect to LLMs
- [Unified Models](../vision_language/03_Unified_Models.md) — Janus, BAGEL use dual vs unified encoders
