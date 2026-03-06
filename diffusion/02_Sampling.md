# Sampling Methods

> Parent: [Diffusion Models](00_Diffusion.md)

## Overview

DDPM requires ~1000 denoising steps, making generation very slow. Modern samplers reduce this to 20-50 steps (or even 1-4 with distillation) while maintaining quality. This section covers the key sampling algorithms and classifier-free guidance.

## DDIM (Denoising Diffusion Implicit Models)

DDIM reformulates the reverse process as a **deterministic** (non-Markovian) mapping:

```python
# DDIM Sampling — can skip timesteps
# Use a subsequence: e.g., [999, 949, 899, ..., 49, 0] (20 steps instead of 1000)

x = torch.randn(shape)

for i in range(len(timesteps) - 1):
    t = timesteps[i]
    t_prev = timesteps[i + 1]

    predicted_noise = model(x, t)

    # Predict x_0
    x_0_pred = (x - sqrt(1 - alpha_bar[t]) * predicted_noise) / sqrt(alpha_bar[t])

    # DDIM update (deterministic when η=0)
    x = (sqrt(alpha_bar[t_prev]) * x_0_pred +
         sqrt(1 - alpha_bar[t_prev]) * predicted_noise)
```

Key properties:
- **Deterministic**: same noise → same image (when η=0)
- **Skip steps**: use any subset of timesteps
- **Invertible**: can encode images back to noise (useful for editing)
- Typically 20-50 steps for good quality

## DPM-Solver

Treats the diffusion ODE as a math problem and applies high-order ODE solvers:

| Solver | Order | Steps for Good Quality | Key Idea |
|--------|-------|----------------------|----------|
| DDIM | 1st order | 50 | Euler method |
| DPM-Solver-2 | 2nd order | 20 | Midpoint method |
| DPM-Solver++ | 2nd/3rd | 15-20 | Multistep, better stability |

Higher-order solvers take fewer steps because they better approximate the continuous denoising trajectory.

## Euler / Euler Ancestral

Simple ODE solvers commonly used in practice:

- **Euler**: Basic first-order ODE solver, deterministic
- **Euler Ancestral (Euler a)**: Adds noise at each step (stochastic), more creative/varied outputs
- Used in many Stable Diffusion UIs as default sampler

## Classifier-Free Guidance (CFG) — Deep Dive

CFG is the most important technique for controllable generation:

```
At training:
  - Randomly drop condition c → ∅ with probability p_drop (e.g., 10%)
  - Model learns both conditional and unconditional generation

At inference:
  ε_guided = ε_unconditional + s * (ε_conditional - ε_unconditional)

  where s = guidance scale
```

### Guidance Scale Effects

```
s = 1.0:  No guidance (model output as-is)
          → diverse but may not match prompt well

s = 7.5:  Standard (Stable Diffusion default)
          → good balance of quality and prompt adherence

s = 15+:  Strong guidance
          → very prompt-adherent but oversaturated, artifacts

s = 0.0:  Pure unconditional
          → ignores prompt entirely
```

### Computational Cost

CFG requires **two forward passes** per step (conditional + unconditional), doubling inference cost. Approaches to mitigate:
- **Distillation**: train student model without CFG (Consistency models, LCM)
- **Negative prompt**: replace ∅ with a negative condition to steer away from undesired features

## Consistency Models

Learn to map any noisy x_t directly to x_0 in a single step:

```
Traditional:  x_T → x_{T-1} → ... → x_1 → x_0  (many steps)
Consistency:  x_t  ─────────────────────→  x_0    (one step)
```

- Can generate in 1-4 steps
- Quality trade-off vs. multi-step sampling
- Latent Consistency Models (LCM): apply to latent diffusion

## Scheduler/Sampler Summary

| Sampler | Steps | Quality | Speed | Deterministic |
|---------|-------|---------|-------|---------------|
| DDPM | 1000 | Excellent | Very slow | No |
| DDIM | 20-50 | Good | Fast | Yes (η=0) |
| DPM-Solver++ | 15-25 | Very good | Fast | Yes |
| Euler | 20-30 | Good | Fast | Yes |
| Euler Ancestral | 20-30 | Good (varied) | Fast | No |
| LCM | 4-8 | Good | Very fast | Yes |
| Consistency | 1-2 | Decent | Fastest | Yes |

## Practical Sampling Code

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

# Switch to fast sampler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Generate with CFG
image = pipe(
    prompt="a photo of an astronaut riding a horse",
    guidance_scale=7.5,    # CFG strength
    num_inference_steps=25  # Sampling steps
).images[0]
```

## Related

- [Diffusion Basics](01_Diffusion_Basics.md) — DDPM foundation and noise schedules
- [Latent Diffusion](03_Latent_Diffusion.md) — samplers applied in latent space
- [AI_Infra: Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving implications of step count
