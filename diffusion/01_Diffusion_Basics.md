# Diffusion Basics

> Parent: [Diffusion Models](00_Diffusion.md)

## Overview

Denoising Diffusion Probabilistic Models (DDPM) define a forward process that gradually adds Gaussian noise to data, and a reverse process that learns to denoise. The model is trained to predict the noise added at each step, using a simple MSE loss. Despite this simplicity, diffusion models produce state-of-the-art generative results.

## Forward Process (Diffusion)

Add Gaussian noise gradually over T timesteps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$$

where β_t is the noise schedule (small values, e.g., 0.0001 to 0.02).

Key property — can jump directly to any timestep t:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t} \, x_0, (1-\bar\alpha_t) I)$$

where ᾱ_t = ∏ᵢ₌₁ᵗ (1 - βᵢ)

```python
# Sample x_t directly from x_0 (used during training)
def q_sample(x_0, t, noise):
    alpha_bar_t = alpha_bar[t]
    return sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
```

## Reverse Process (Denoising)

Learn to reverse the forward process:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

The model predicts the mean μ_θ, which is parameterized as predicting the noise ε_θ.

## Training

Training is remarkably simple:

```python
# DDPM Training Loop
for x_0 in dataloader:
    t = torch.randint(0, T, (batch_size,))    # Random timestep
    noise = torch.randn_like(x_0)              # Random noise
    x_t = q_sample(x_0, t, noise)              # Add noise to x_0

    predicted_noise = model(x_t, t)            # Predict added noise
    loss = F.mse_loss(predicted_noise, noise)  # Simple MSE loss

    loss.backward()
    optimizer.step()
```

### Prediction Targets

The model can be trained to predict different targets (mathematically equivalent):

| Target | Predicts | Used By |
|--------|----------|---------|
| ε (noise) | The noise added | DDPM, Stable Diffusion |
| x₀ (clean image) | The original data | Some implementations |
| v (velocity) | v = √ᾱ_t · ε - √(1-ᾱ_t) · x₀ | Improved DDPM, SD 2.x |

v-prediction provides better training stability, especially at high noise levels.

## Noise Schedules

The schedule β₁, β₂, ..., β_T controls how quickly noise is added:

```
Signal-to-noise ratio over timesteps:

SNR
 ▲
 │██████
 │      ███
 │         ████
 │             ████
 │                 ████
 │                     ████████
 └──────────────────────────────→ timestep t
 0                              T

Linear:  β_t increases linearly (original DDPM)
Cosine:  Slower noise addition, better for low-res (Improved DDPM)
Scaled:  Adjusted for different resolutions
```

### Linear vs Cosine Schedule

| Schedule | Behavior | Good For |
|----------|----------|----------|
| Linear | Fast noise increase | Standard, simple |
| Cosine | Gradual noise, more time at low noise | Better image quality |
| Shifted (log-SNR) | Resolution-aware | High-resolution generation |

At high resolution, linear schedules add too much noise too quickly — shifted/cosine schedules preserve more structure.

## Sampling (Inference)

DDPM sampling: iterate from x_T to x_0 (T steps, typically T=1000):

```python
# DDPM Sampling (slow but exact)
x = torch.randn(shape)  # Start from pure noise

for t in reversed(range(T)):
    predicted_noise = model(x, t)

    # Compute mean of p(x_{t-1} | x_t)
    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]
    x = (1/sqrt(alpha_t)) * (x - (1-alpha_t)/sqrt(1-alpha_bar_t) * predicted_noise)

    if t > 0:
        x += sigma_t * torch.randn_like(x)  # Add noise (stochastic)
```

Problem: 1000 steps is very slow. See [02_Sampling.md](02_Sampling.md) for faster methods.

## Conditional Generation

To generate images matching a condition (text, class, image):

### Classifier Guidance

Use a separate classifier p(y|x_t) to guide denoising:

```
Modified noise: ε̂ = ε_θ(x_t, t) - s · ∇_{x_t} log p(y | x_t)
                                    ↑
                              guidance scale
```

Drawback: requires training a separate classifier on noisy images.

### Classifier-Free Guidance (CFG)

Train a single model with and without conditioning. At inference, extrapolate:

```
ε̂ = ε_θ(x_t, t, ∅) + s · (ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
     ─────────────       ──────────────────────────────────────
     unconditional        direction toward condition c

s = guidance scale (typically 7.5 for Stable Diffusion)
```

During training, randomly drop the condition (replace with null embedding ∅) some percentage of the time (e.g., 10%). See more in [02_Sampling.md](02_Sampling.md).

## Flow Matching (Modern Alternative)

Flow matching reformulates diffusion as learning a vector field that transports noise to data along straight paths:

```
Diffusion:     Curved paths from noise to data (many steps needed)
Flow Matching: Straight paths from noise to data (fewer steps)

Noise ──────●────────── Data     (diffusion: curved)
             \
Noise ───────────────── Data     (flow matching: straight)
```

- Used in Stable Diffusion 3, FLUX.1
- Simpler training objective (regress velocity field)
- More efficient sampling (straighter trajectories = fewer steps)

## Resources

**Papers**

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., NeurIPS 2020. The foundational DDPM paper.
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — Nichol & Dhariwal, ICML 2021. Cosine schedule, learned variance, v-prediction.
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Ho & Salimans, NeurIPS 2022 Workshop. CFG — the standard conditioning technique.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., ICLR 2023. Optimal-transport conditional flow matching.
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — Liu et al., ICLR 2023. Rectified flow — straight-path ODE for generation.

**Blogs**

- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — Lilian Weng. Comprehensive math walkthrough of DDPM, score-based models, and connections between them.
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — HuggingFace. Step-by-step PyTorch implementation of DDPM with detailed commentary.
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) — Yang Song. Score-based perspective unifying SMLD and DDPM.

## Related

- [Sampling Methods](02_Sampling.md) — DDIM, DPM-Solver for faster generation
- [Latent Diffusion](03_Latent_Diffusion.md) — applying diffusion in VAE latent space
- [DiT](04_DiT.md) — transformer-based denoising network
