# Variational Autoencoders (VAE)

> Parent: [Visual Encoders](00_Visual_Encoder.md)

## Overview

Variational Autoencoders (VAEs) learn to encode images into a compact latent space and decode them back to pixels. In the context of multimodal models, VAEs are primarily used to create the **latent space** in which diffusion models operate (Latent Diffusion / Stable Diffusion). They compress images from pixel space to a much smaller latent representation, making diffusion training and inference computationally feasible.

## VAE Architecture

```
Encoder                    Latent Space              Decoder
┌──────────┐          ┌──────────────────┐      ┌──────────┐
│          │          │                  │      │          │
│  Image   │──→ μ ──→│  z = μ + σ·ε     │──→   │  Recon.  │
│ (256x256)│          │  (reparameterize)│      │ (256x256)│
│          │──→ σ ──→│                  │      │          │
│          │          │  z ∈ R^{4×32×32} │      │          │
└──────────┘          └──────────────────┘      └──────────┘

Loss = Reconstruction Loss + KL Divergence
     = ||x - x̂||² + KL(q(z|x) || p(z))
```

### Reparameterization Trick

To backpropagate through sampling:
```
z = μ + σ * ε,  where ε ~ N(0, I)
```
This separates the stochastic part (ε) from the learnable parameters (μ, σ), enabling gradient flow.

### KL Divergence Term

Regularizes the latent space to be close to a standard normal distribution N(0, I). This ensures:
- Latent space is smooth and continuous
- Similar images map to nearby points
- Random samples from N(0, I) produce valid images

## VAE in Latent Diffusion

The primary use of VAEs in modern generative models:

```
Training:
  Image (512x512x3) ──→ VAE Encoder ──→ Latent (64x64x4) ──→ Train diffusion here
                                              │
                                         8x compression per spatial dim
                                         48x total compression

Inference:
  Random noise (64x64x4) ──→ Denoise ──→ Clean latent ──→ VAE Decoder ──→ Image (512x512x3)
```

Key benefit: diffusion operates on **64x64x4 = 16K values** instead of **512x512x3 = 786K values** — a ~48x reduction in the space the diffusion model must handle.

## Specific VAE Models

### SD-VAE (Stable Diffusion VAE)

Used in Stable Diffusion 1.x and 2.x:
- Encoder: convolutional downsampling (4 stages)
- Latent channels: 4
- Spatial compression: 8x (512x512 → 64x64)
- Architecture: ResNet blocks with attention at lower resolutions

```
SD-VAE Encoder:
  512x512x3 → 256x256x128 → 128x128x256 → 64x64x512 → 64x64x4 (μ,σ → z)
  (conv)       (down)         (down)         (down)       (conv)
```

### FLUX VAE

Used in FLUX.1 (Black Forest Labs):
- Higher quality reconstruction than SD-VAE
- Latent channels: 16 (vs 4 in SD-VAE)
- Better preservation of fine details
- Spatial compression: 8x

### SD3 VAE

Stable Diffusion 3:
- 16 latent channels
- Improved training with larger datasets
- Better color accuracy and detail preservation

## VAE vs Semantic Encoders for Generation

| Aspect | VAE | Semantic Encoder (+ RAE) |
|--------|-----|--------------------------|
| What it encodes | Pixel-level details | High-level semantics |
| Reconstruction | Near-perfect | Approximate (through RAE decoder) |
| Latent dim | Low (4-16 channels) | High (768-1024 dim per token) |
| Generation quality | Standard approach | Emerging alternative |
| Understanding | Poor (no semantic structure) | Excellent |
| Dual encoder needed? | Yes (need separate CLIP/SigLIP) | No (unified) |

### Representation Autoencoder (RAE)

RAE bridges the gap between semantic and pixel-faithful representations:

```
Traditional:  Image → VAE Encoder → low-dim latent → Diffusion → VAE Decoder → Image
                      (pixel-faithful)

RAE:          Image → Semantic Encoder → high-dim latent → Diffusion → RAE Decoder → Image
                      (SigLIP/DINOv2)                                  (learned)
```

RAE trains a decoder to reconstruct pixels from semantic encoder features. This means:
- The semantic encoder's features carry enough information for reconstruction
- Diffusion can operate in the semantic encoder's latent space
- One encoder serves both understanding and generation

## Training Considerations

### KL Weight (β-VAE)

The balance between reconstruction and regularization:
- β too low → latent space not smooth, poor generation from random samples
- β too high → blurry reconstructions, information bottleneck
- Latent Diffusion models often train VAE with low β (good reconstruction) since diffusion handles generation

### Perceptual Loss

Modern VAEs add perceptual loss for sharper reconstructions:
```
Loss = ||x - x̂||² + λ_percep * ||φ(x) - φ(x̂)||² + β * KL
       ─────────     ──────────────────────────     ─────
       pixel-level    feature-level (VGG/LPIPS)    regularization
```

### Adversarial Training

Many modern VAEs include a discriminator (GAN loss) to improve sharpness:
- The VAE-GAN approach helps avoid blurry outputs
- SD-VAE uses a PatchGAN discriminator

## Related

- [Latent Diffusion](../diffusion/03_Latent_Diffusion.md) — uses VAE latent space for diffusion
- [Semantic Encoders](03_Semantic_Encoders.md) — alternative approach using CLIP/SigLIP + RAE
- [Unified Models](../vision_language/03_Unified_Models.md) — Janus uses VAE for generation, CLIP for understanding
- [Image Generation](../applications/01_Image_Generation.md) — downstream application of VAE-based pipelines
