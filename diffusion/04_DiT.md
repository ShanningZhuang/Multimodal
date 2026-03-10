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

## Patchify and Unpatchify

### Patchify

The input latent $z_t \in \mathbb{R}^{H \times W \times C}$ is split into non-overlapping patches and linearly embedded, exactly like ViT:

```python
# Patchify = Conv2d with kernel_size=stride=patch_size
patchify = nn.Conv2d(in_channels, hidden_dim, kernel_size=p, stride=p)
# Input: (B, C, H, W)  →  Output: (B, hidden_dim, H/p, W/p)
# Reshape to sequence: (B, N, hidden_dim) where N = (H/p) × (W/p)
```

Token count: $N = \frac{H}{p} \times \frac{W}{p}$

| Latent Size | Patch Size | Tokens | Typical Use |
|-------------|-----------|--------|-------------|
| 32×32 | 2 | 256 | DiT-XL/2 (256px generation) |
| 64×64 | 2 | 1024 | DiT-XL/2 (512px generation) |
| 128×128 | 2 | 4096 | High-resolution (1024px) |

**Resolution flexibility**: unlike U-Net where the architecture encodes a fixed resolution, DiT simply produces more tokens at higher resolution. No retraining is needed — only positional embeddings need interpolation (same as ViT).

### Unpatchify

The final layer reverses patchification: a linear head predicts $p^2 \times C$ values per token (the denoised patch pixels), then reshapes back to the spatial latent:

```python
# Each token predicts a full patch
unpatchify_head = nn.Linear(hidden_dim, p * p * out_channels)
# Output: (B, N, p²·C) → reshape → (B, C, H, W)
```

The original DiT also predicts per-patch variance (for the DDPM loss), so the head actually outputs $p^2 \times 2C$ values.

## Timestep and Class Conditioning

Before entering the transformer blocks, the timestep and class label are converted into a single conditioning vector $c$:

```
Timestep t ──→ Sinusoidal Embedding ──→ MLP ──→ t_emb
                                                  │
Class label y ──→ Embedding Table ──→ class_emb   │
                                          │       │
                                          ▼       ▼
                                     c = t_emb + class_emb
```

$$c = \text{MLP}(\text{sin\_embed}(t)) + \text{Embed}(y)$$

This vector $c$ is fed into every DiT block via adaLN-Zero to modulate the layer normalization parameters.

## adaLN-Zero (Adaptive Layer Normalization)

The key conditioning mechanism in DiT. Instead of cross-attention, DiT injects timestep and class information through layer normalization parameters:

### Math

Each DiT block produces 6 modulation parameters from the conditioning vector:

$$(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2) = \text{MLP}(c)$$

The forward pass applies these as:

$$h = x + \alpha_1 \odot \text{Attn}\bigl(\gamma_1 \odot \text{LN}(x) + \beta_1\bigr)$$
$$\text{out} = h + \alpha_2 \odot \text{FFN}\bigl(\gamma_2 \odot \text{LN}(h) + \beta_2\bigr)$$

where $\gamma$ scales, $\beta$ shifts, and $\alpha$ gates (all initialized to zero).

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
- **Zero initialization**: α starts at zero → at init, each block is an identity function → stable training for deep networks (similar motivation to ResNet skip connections)
- **Parameter efficient**: 6·dim parameters per block vs. a full cross-attention layer
- **Sufficient for simple conditions**: timestep + class label are low-dimensional; cross-attention is overkill

### Conditioning Mechanism Comparison

| Mechanism | How Condition Enters | Parameters/Block | Best For |
|-----------|---------------------|-------------------|----------|
| adaLN-Zero | Modulates LayerNorm γ, β, α | 6 × dim | Simple conditions (t, class) |
| Cross-attention | Q from image, KV from condition | ~4 × dim² | Sequence conditions (text) |
| Joint attention | Concat condition + image tokens | 0 extra (shared QKV) | Rich multimodal interaction |

## Scaling Behavior

DiT follows clear scaling laws similar to LLMs:

```
Model sizes tested in original DiT paper:

DiT-S/2:   33M params  │████                              FID-256: 68.4
DiT-B/2:  130M params  │████████████                      FID-256: 43.5
DiT-L/2:  458M params  │████████████████████████████████████  FID-256: 23.3
DiT-XL/2: 675M params  │██████████████████████████████████████████████████  FID-256: 9.62

FID improves predictably with:
- More parameters (model size)
- More training compute (steps × batch size)
- More data
```

DiT-XL/2 achieves FID 2.27 on ImageNet 256×256 (class-conditional, with CFG), which was state-of-the-art at publication. This predictability makes DiT attractive for scaling to billions of parameters — you can forecast performance before training.

## MMDiT (Stable Diffusion 3) — Deep Dive

MMDiT (Multimodal Diffusion Transformer) extends DiT from class-conditional generation to text-to-image by introducing **dual-stream** processing with **joint attention**.

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        MMDiT Block                                │
│                                                                  │
│  Image tokens x:  [x₁, x₂, ..., x_n]   (from patchified latent)│
│  Text tokens y:   [y₁, y₂, ..., y_m]    (from text encoders)    │
│                                                                  │
│  ┌───────────────────────┐  ┌───────────────────────┐           │
│  │ Image Stream          │  │ Text Stream            │           │
│  │ adaLN(x) → Qx, Kx, Vx│  │ adaLN(y) → Qy, Ky, Vy│           │
│  └───────────┬───────────┘  └───────────┬───────────┘           │
│              │                          │                        │
│              ▼                          ▼                        │
│  ┌──────────────────────────────────────────────────┐           │
│  │            Joint Self-Attention                    │           │
│  │                                                    │           │
│  │  Q = [Qx; Qy]     (concatenate along seq dim)     │           │
│  │  K = [Kx; Ky]                                      │           │
│  │  V = [Vx; Vy]                                      │           │
│  │                                                    │           │
│  │  Attn = softmax(QK^T / √d) V                      │           │
│  │                                                    │           │
│  │  Split output → image_out, text_out                │           │
│  └───────────┬──────────────────────┬───────────────┘           │
│              ▼                      ▼                            │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ Image FFN         │  │ Text FFN          │                     │
│  │ (separate weights)│  │ (separate weights) │                     │
│  └──────────────────┘  └──────────────────┘                     │
│                                                                  │
│  Timestep → adaLN-Zero (modulates both streams)                  │
└──────────────────────────────────────────────────────────────────┘
```

### Joint Attention Math

Each stream computes its own Q, K, V projections, then they are concatenated for a single attention operation:

$$Q = [Q_x; Q_y], \quad K = [K_x; K_y], \quad V = [V_x; V_y]$$

$$\text{Attn} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$

The output is split back: the first $n$ rows go to the image stream, the last $m$ rows go to the text stream. This means:
- Image tokens attend to all text tokens (replaces cross-attention)
- Text tokens attend to all image tokens (bidirectional)
- Image tokens attend to each other (self-attention)
- Text tokens attend to each other (self-attention)

All four interaction patterns happen in a single attention layer — more expressive than separate self + cross attention.

### SD3 Specifics

| Component | Details |
|-----------|---------|
| DiT blocks | 24 MMDiT blocks |
| Parameters | ~2B (text + image streams) |
| Text encoders | CLIP-G (1280d) + CLIP-L (768d) + T5-XXL (4096d) |
| Text pooling | CLIP pooled embeddings → timestep conditioning |
| T5 embeddings | Sequence → concatenated as text stream tokens |
| Latent channels | 16 (vs 4 in SD 1.x) |
| Training objective | Flow matching (rectified flow) |
| Patch size | 2×2 |

The three text encoders serve different roles: CLIP provides aligned image-text semantics, T5-XXL provides rich language understanding. CLIP pooled outputs are added to the timestep embedding for global conditioning; T5 sequence outputs form the text token stream for joint attention.

## FLUX.1 — Deep Dive

FLUX.1 (by Black Forest Labs, the creators of Stable Diffusion) is the successor to SD3's MMDiT with a hybrid dual+single stream architecture and improved flow matching.

### Hybrid Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   FLUX.1 Architecture                     │
│                                                          │
│  Text encoders: CLIP-L (pooled → timestep conditioning)  │
│                 T5-XXL (sequence → text tokens)           │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  19× Dual-Stream Blocks (MMDiT-style)              │  │
│  │  Image stream ←→ Text stream (joint attention)     │  │
│  │  Separate QKV projections, separate FFNs           │  │
│  └────────────────────────┬───────────────────────────┘  │
│                           │                              │
│                  Concatenate image + text tokens          │
│                           │                              │
│  ┌────────────────────────▼───────────────────────────┐  │
│  │  38× Single-Stream Blocks (standard DiT)           │  │
│  │  All tokens in one unified sequence                │  │
│  │  Shared QKV projections, shared FFN                │  │
│  └────────────────────────┬───────────────────────────┘  │
│                           │                              │
│              Extract image tokens → unpatchify           │
└──────────────────────────────────────────────────────────┘
```

The intuition: early layers need separate processing for the two modalities (different statistics, different semantics). Later layers have already aligned the representations, so a single unified stream is more parameter-efficient and allows deeper cross-modal fusion.

### RoPE for 2D Positions

FLUX uses Rotary Position Embeddings (RoPE) extended to 2D spatial positions instead of learnable absolute position embeddings:

```
For each image token at grid position (row, col):
  - Split hidden dim into two halves
  - First half:  apply RoPE with position = row
  - Second half: apply RoPE with position = col

Benefits:
  - Relative position awareness (like RoPE in LLMs)
  - Extrapolates to unseen resolutions better than learned absolute positions
  - No need to interpolate position embeddings at different resolutions
```

### Flow Matching with Velocity Prediction

FLUX uses rectified flow matching (see [01_Diffusion_Basics.md](01_Diffusion_Basics.md)) instead of the DDPM noise-prediction objective:

$$x_t = (1-t) \cdot x_0 + t \cdot \epsilon, \quad t \in [0, 1]$$

The model predicts the velocity $v = \epsilon - x_0$ (the direction from data to noise):

$$\mathcal{L} = \|v_\theta(x_t, t) - (\epsilon - x_0)\|^2$$

At inference, the model integrates this velocity field using an Euler solver to transport noise to data along approximately straight paths. Straighter paths mean fewer steps are needed.

### FLUX.1 Variants

| Variant | Steps | CFG | License | Key Feature |
|---------|-------|-----|---------|-------------|
| FLUX.1-dev | 20-50 | Yes (guidance distilled) | Non-commercial | Full quality, guidance distillation |
| FLUX.1-schnell | 1-4 | No | Apache 2.0 | Distilled for few-step generation |

FLUX.1-schnell is distilled from FLUX.1-dev using progressive distillation, enabling high-quality generation in just 4 steps without classifier-free guidance — making it ~10x faster at inference.

### FLUX.1 Key Specs

| Component | Details |
|-----------|---------|
| Parameters | ~12B |
| Dual-stream blocks | 19 |
| Single-stream blocks | 38 |
| Hidden dim | 3072 |
| Attention heads | 24 |
| Text encoders | CLIP-L + T5-XXL |
| Position encoding | 2D RoPE |
| Training objective | Rectified flow matching |
| Latent channels | 16 |
| VAE | FLUX autoencoder (16ch) |

## Flow Matching + DiT

Flow matching provides an elegant alternative to the DDPM training objective. Here we connect the math from [01_Diffusion_Basics.md](01_Diffusion_Basics.md) to DiT specifically.

### Recap

The interpolation path defines a straight line between data $x_0$ and noise $\epsilon$:

$$x_t = (1-t) \cdot x_0 + t \cdot \epsilon, \quad t \in [0, 1]$$

The velocity (time derivative) along this path:

$$v = \frac{dx_t}{dt} = \epsilon - x_0$$

### Training

The DiT is trained to predict this velocity:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, \epsilon}\bigl[\|v_\theta(x_t, t) - (\epsilon - x_0)\|^2\bigr]$$

Compared to DDPM's noise prediction loss $\|\epsilon_\theta - \epsilon\|^2$, flow matching has:
- Linear interpolation path (vs. the curved DDPM forward process)
- Single time range $t \in [0, 1]$ (vs. discrete $t \in \{1, ..., T\}$)
- Simpler sampler at inference (Euler ODE solver)

### Inference (Euler Solver)

```python
# Flow matching inference with Euler method
x = torch.randn(shape)  # Start from noise (t=1)
timesteps = torch.linspace(1, 0, num_steps + 1)  # 1 → 0

for i in range(num_steps):
    t = timesteps[i]
    dt = timesteps[i + 1] - timesteps[i]  # negative step
    v = model(x, t)          # Predict velocity
    x = x + v * dt           # Euler step
# x is now the generated image (t=0)
```

Models using flow matching + DiT: **SD3**, **FLUX.1**, **Wan2.1**, **CogVideoX**.

## DiT for Video

Video generation extends DiT by adding a temporal dimension to the patch tokenization.

### Spacetime Patches

```
Video: T frames × H × W × C

                    ┌──┐
           frame 0  │  │  ←── spatial patches (like image DiT)
           frame 1  │  │
           frame 2  │  │  ←── temporal patches group F consecutive frames
           frame 3  │  │
                    └──┘

3D patch: (F, p_h, p_w) — groups F frames of p_h × p_w spatial patches
Token count: N = (T/F) × (H/p_h) × (W/p_w)
```

For example, with $F=1, p_h=p_w=2$ on a 16-frame 256×256 video with 4-channel latent: $N = 16 \times 128 \times 128 = 262{,}144$ tokens — attention on this is expensive!

### Factored Attention

To handle the quadratic cost, video DiTs typically factorize attention:

| Strategy | Attention Pattern | Trade-off |
|----------|-------------------|-----------|
| Full 3D | All tokens attend to all tokens | Best quality, O(N²) cost |
| Spatial + Temporal | Alternate spatial-only and temporal-only attention layers | Good quality, much cheaper |
| Windowed | Local spatial + global temporal | Scales to long videos |

### Video DiT Models

| Model | Architecture | Key Innovation |
|-------|-------------|----------------|
| Sora (OpenAI) | Spacetime DiT | Variable duration/resolution/aspect ratio |
| CogVideoX | 3D full attention DiT | Expert-adaptive LayerNorm |
| Wan2.1 / 2.2 | Flow matching DiT | Efficient 3D VAE, open-source |
| Movie Gen (Meta) | Temporal-spatial factored DiT | Joint video+audio generation |

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

## Resources

**Papers**

- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) — Peebles & Xie, ICCV 2023. The original DiT paper — replaces U-Net with transformer, demonstrates scaling laws.
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Esser et al., ICML 2024. SD3 / MMDiT — dual-stream joint attention + flow matching.
- [FLUX.1 Technical Report](https://arxiv.org/abs/2408.06072) — Black Forest Labs, 2024. Hybrid dual+single stream DiT with 2D RoPE.
- [Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models](https://arxiv.org/abs/2402.17177) — Survey of Sora's spacetime DiT architecture and capabilities.
- [CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — Tsinghua/Zhipu, 2024. Full-attention 3D DiT for video.
- [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314) — Alibaba, 2025. Open-source flow-matching video DiT.

**Blogs**

- [The DiT Paper Explained](https://amaarora.github.io/posts/2023-01-05-dit.html) — Aman Arora. Visual walkthrough of DiT architecture, adaLN-Zero, and scaling results.
- [Illustrating Stable Diffusion 3](https://jalammar.github.io/illustrated-stable-diffusion/) — Jay Alammar. Diagrams of the MMDiT architecture and text encoder pipeline.

## Related

- [Latent Diffusion](03_Latent_Diffusion.md) — DiT replaces U-Net in the LDM framework
- [ViT](../visual_encoder/02_ViT.md) — DiT uses same core transformer architecture as ViT
- [Robotics](../applications/03_Robotics.md) — DiT applied to action generation
- [Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving DiT models
- [LLM KB: Transformer](../../LLM/transformer/) — shared attention mechanism
- **Hands-on scripts**: [05_dit_architecture.py](../scripts/05_dit_architecture.py), [06_latent_diffusion_pipeline.py](../scripts/06_latent_diffusion_pipeline.py), [07_flux_dit_blocks.py](../scripts/07_flux_dit_blocks.py), [08_denoising_loop.py](../scripts/08_denoising_loop.py)
