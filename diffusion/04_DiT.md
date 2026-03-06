# Diffusion Transformer (DiT)

> Parent: [Diffusion Models](00_Diffusion.md)

## Overview

The Diffusion Transformer (DiT) replaces the U-Net in diffusion models with a transformer architecture. This enables better scaling behavior, simpler architecture, and leverages the same scaling laws observed in LLMs. DiT is the backbone of modern generative models including Stable Diffusion 3, FLUX.1, and Sora.

## Motivation: Why Replace U-Net?

| Aspect | U-Net | DiT |
|--------|-------|-----|
| Architecture | CNN + attention hybrid | Pure transformer |
| Scaling | Diminishing returns at scale | Predictable scaling laws |
| Resolution | Fixed resolution (retrain needed) | Flexible (adjust token count) |
| Training | Complex architecture tuning | Simple, well-understood |
| Hardware | Mixed ops, harder to optimize | Dense ops, GPU-friendly |
| Ecosystem | Requires custom implementation | Leverage LLM tooling |

## DiT Architecture

```
Input: noisy latent z_t (e.g., 64x64x4 from VAE)
       │
       ▼
┌──────────────────┐
│  Patchify         │   Split latent into patches (e.g., 2x2)
│  + Linear embed   │   → sequence of patch tokens
└──────┬───────────┘   64x64 with 2x2 patches → 1024 tokens
       │
       ▼
┌──────────────────┐
│  + Pos. Encoding  │   Learnable or sinusoidal position embeddings
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────────┐
│         DiT Block (×N)               │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ adaLN-Zero                     │  │
│  │ (adaptive Layer Norm from      │  │
│  │  timestep + class embedding)   │  │
│  └────────────┬───────────────────┘  │
│               ▼                      │
│  ┌────────────────────────────────┐  │
│  │ Multi-Head Self-Attention      │  │
│  └────────────┬───────────────────┘  │
│               ▼                      │
│  ┌────────────────────────────────┐  │
│  │ adaLN-Zero                     │  │
│  └────────────┬───────────────────┘  │
│               ▼                      │
│  ┌────────────────────────────────┐  │
│  │ Feed-Forward Network (MLP)     │  │
│  └────────────────────────────────┘  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────┐
│  Linear → Unpatchify      │   Reshape back to spatial latent
│  Predict noise + variance │
└──────────────────────────┘
```

## adaLN-Zero (Adaptive Layer Normalization)

The key conditioning mechanism in DiT. Instead of cross-attention, DiT injects timestep and class information through layer normalization parameters:

```python
# Standard LayerNorm: y = (x - μ) / σ * γ + β   (fixed γ, β)
# adaLN:             y = (x - μ) / σ * γ_c + β_c  (γ, β from conditioning)

# adaLN-Zero: also learns a gating parameter α initialized to zero
class DiTBlock(nn.Module):
    def __init__(self, dim):
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim)
        self.mlp = MLP(dim)
        # Produces 6 parameters: γ1, β1, α1, γ2, β2, α2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

    def forward(self, x, c):  # c = timestep + class embedding
        γ1, β1, α1, γ2, β2, α2 = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention with adaptive normalization + zero-init gating
        x = x + α1 * self.attn(γ1 * self.norm1(x) + β1)
        # FFN with adaptive normalization + zero-init gating
        x = x + α2 * self.mlp(γ2 * self.norm2(x) + β2)
        return x
```

Why adaLN-Zero?
- α initialized to zero → at initialization, each block is an identity function
- Stable training for deep networks (similar motivation to ResNet skip connections)
- More parameter-efficient than cross-attention for simple conditions

## Scaling Behavior

DiT follows clear scaling laws similar to LLMs:

```
Model sizes tested in original DiT paper:

DiT-S/2:   33M params  │████
DiT-B/2:  130M params  │████████████
DiT-L/2:  458M params  │████████████████████████████████████
DiT-XL/2: 675M params  │██████████████████████████████████████████████████

FID improves predictably with:
- More parameters (model size)
- More training compute (steps × batch size)
- More data
```

This predictability makes DiT attractive for scaling to billions of parameters — you can forecast performance before training.

## DiT Variants in Practice

### MMDiT (Stable Diffusion 3)

Extends DiT for multimodal conditioning with **joint attention**:

```
┌──────────────────────────────────────────────┐
│                MMDiT Block                    │
│                                              │
│  Image tokens → separate adaLN → ┐          │
│                                   ├→ Joint   │
│  Text tokens  → separate adaLN → ┘  Attn    │
│                                              │
│  Image tokens → separate FFN                │
│  Text tokens  → separate FFN                │
└──────────────────────────────────────────────┘

- Two streams (image + text) with shared attention
- Separate FFNs and normalization per modality
- Text tokens participate in self-attention alongside image tokens
```

### FLUX.1

Builds on MMDiT with flow matching:
- Flow matching replaces DDPM objective (straighter sampling paths)
- Improved architecture details
- Single-stream blocks in later layers (merge image + text)
- RoPE for position encoding

### Sora (Video DiT)

Extends DiT to video:
- Spacetime patches: 3D patches from video frames
- Spatial + temporal attention
- Variable duration, resolution, aspect ratio

## DiT for Robotics

DiT architecture is also used for **action generation** in robotics:

```
Robot Policy as Diffusion:
  Observation (images + state) → DiT → Denoised action sequence

Why DiT works for robotics:
- Multimodal action distributions (multiple valid actions)
- Smooth, continuous trajectories
- Can condition on diverse observations
```

Examples: Diffusion Policy, π₀ (Physical Intelligence)

## Compute Comparison

```
For 512x512 generation (approximate):

U-Net (SD 1.5):     ~1.5 TFLOPS per step
DiT-XL/2:           ~1.2 TFLOPS per step
MMDiT (SD3):        ~3.5 TFLOPS per step (but fewer steps needed)

Key: DiT's ops are more hardware-friendly (dense matmuls vs mixed CNN+attn)
→ Better GPU utilization despite similar FLOPS
```

## Related

- [Latent Diffusion](03_Latent_Diffusion.md) — DiT replaces U-Net in the LDM framework
- [ViT](../visual_encoder/02_ViT.md) — DiT uses same core transformer architecture as ViT
- [Robotics](../applications/03_Robotics.md) — DiT applied to action generation
- [Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving DiT models
- [LLM KB: Transformer](../../LLM/transformer/) — shared attention mechanism
