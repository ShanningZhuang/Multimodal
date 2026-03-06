# CNN Basics

> Parent: [Visual Encoders](00_Visual_Encoder.md)

## Overview

Convolutional Neural Networks (CNNs) were the dominant architecture for computer vision before Vision Transformers. Understanding CNNs provides essential context for why ViTs and modern encoders were developed, and CNNs remain important as components in many multimodal systems.

## Core Operations

### Convolution

A convolution slides a learned filter (kernel) across the input, computing element-wise products and summing them. This captures **local spatial patterns** like edges, textures, and shapes.

```
Input (5x5)          Kernel (3x3)         Output (3x3)
┌─┬─┬─┬─┬─┐         ┌─┬─┬─┐
│1│0│1│0│1│         │1│0│1│             ┌─┬─┬─┐
├─┼─┼─┼─┼─┤         ├─┼─┼─┤             │4│3│4│
│0│1│0│1│0│    *    │0│1│0│     =      ├─┼─┼─┤
├─┼─┼─┼─┼─┤         ├─┼─┼─┤             │2│4│3│
│1│0│1│0│1│         │1│0│1│             ├─┼─┼─┤
├─┼─┼─┼─┼─┤         └─┴─┴─┘             │4│3│4│
│0│1│0│1│0│                              └─┴─┴─┘
├─┼─┼─┼─┼─┤
│1│0│1│0│1│
└─┴─┴─┴─┴─┘
```

Key properties:
- **Local receptive field**: each output depends on a small input region
- **Parameter sharing**: same kernel applied everywhere (translation equivariance)
- **Sparse connectivity**: far fewer parameters than fully-connected layers

### Pooling

Reduces spatial dimensions while retaining important features:
- **Max pooling**: takes maximum value in each window (most common)
- **Average pooling**: takes mean value
- **Global average pooling**: reduces entire feature map to single value per channel

### Feature Hierarchy

```
Layer 1:  Edges, colors         (3x3 receptive field)
Layer 2:  Textures, corners     (5x5 receptive field)
Layer 3:  Parts, patterns       (larger receptive field)
Layer N:  Objects, scenes       (full image)
```

## Key Architectures

### ResNet (2015)

Introduced **residual connections** (skip connections) that enabled training very deep networks (50-152+ layers).

```
        x
        │
   ┌────┴────┐
   │  Conv    │
   │  BN+ReLU│
   │  Conv    │
   │  BN      │
   └────┬────┘
        │
   x + F(x)  ← residual connection
        │
      ReLU
```

Why it matters:
- Solved vanishing gradient problem for deep networks
- ResNet features are still widely used as backbone representations
- Skip connection idea reappears in U-Net, transformers, diffusion models

### ConvNeXt (2022)

Modernized CNN design using ideas from Vision Transformers:
- Patchify stem (4x4 non-overlapping convolutions, like ViT patches)
- Inverted bottleneck blocks
- Larger kernels (7x7)
- Layer normalization instead of batch normalization

Showed that CNNs can match ViT performance when modernized — the architecture gap was largely about training recipes and design choices, not attention vs. convolution fundamentally.

## CNN vs ViT

| Aspect | CNN | ViT |
|--------|-----|-----|
| Receptive field | Local → grows with depth | Global from layer 1 |
| Inductive bias | Translation equivariance | Minimal (learned) |
| Data efficiency | Better with small data | Needs large-scale data |
| Scalability | Diminishing returns at scale | Scales well with data+compute |
| Positional info | Implicit (spatial structure) | Explicit (positional encoding) |

## Why This Matters for Multimodal

- **Feature extraction**: CNN backbones (ResNet, ConvNeXt) are still used in some VLMs
- **U-Net**: CNN-based architecture used in diffusion models (Stable Diffusion v1/v2)
- **Hybrid designs**: Some models combine CNN stems with transformer blocks
- **Understanding ViT**: Knowing CNN limitations motivates the shift to ViT

## Related

- [Vision Transformer](02_ViT.md) — replaced CNNs as dominant vision architecture
- [Diffusion Models: Latent Diffusion](../diffusion/03_Latent_Diffusion.md) — U-Net (CNN) vs DiT (transformer)
