# Sampling Methods

> Parent: [Diffusion Models](00_Diffusion.md)

## Overview

DDPM requires ~1000 denoising steps, making generation very slow. Modern samplers reduce this to 20-50 steps (or even 1-4 with distillation) while maintaining quality. This section covers the key sampling algorithms and classifier-free guidance.

## DDIM (Denoising Diffusion Implicit Models)

Recall that the DDPM forward process gives us:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

DDIM reformulates the reverse process as a **deterministic** (non-Markovian) mapping. Instead of modeling \(p(x_{t-1} \mid x_t)\) as a Gaussian like DDPM, DDIM defines a family of non-Markovian processes indexed by \(\eta\).

**Step 1 — Predict \(x_0\)** from \(x_t\) by rearranging the forward equation:

$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

**Step 2 — DDIM update rule** (generalized form with stochasticity parameter \(\eta\)):

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\, \epsilon_\theta(x_t, t) + \sigma_t\, \epsilon$$

where:

$$\sigma_t = \eta \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$$

- When \(\eta = 0\): \(\sigma_t = 0\), the process is **fully deterministic** (pure DDIM)
- When \(\eta = 1\): recovers the DDPM stochastic process

Substituting \(\hat{x}_0\) back, the deterministic (\(\eta = 0\)) update simplifies to:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left(\frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1 - \bar{\alpha}_{t-1}}\, \epsilon_\theta(x_t, t)$$

In code (deterministic DDIM, \(\eta = 0\)):

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
- **Deterministic**: same noise \(\rightarrow\) same image (when \(\eta = 0\))
- **Skip steps**: use any subsequence of timesteps \(\{\tau_1, \tau_2, \ldots, \tau_S\} \subset \{1, \ldots, T\}\) with \(S \ll T\)
- **Invertible**: can encode images back to noise (useful for editing)
- Typically 20-50 steps for good quality

## DPM-Solver

The diffusion process can be described by a continuous-time ODE (the **probability flow ODE**):

$$\frac{dx}{dt} = f(t)\, x + \frac{g^2(t)}{2\sigma_t}\, \epsilon_\theta(x, t)$$

where \(f(t)\) and \(g(t)\) are the drift and diffusion coefficients of the forward SDE, and \(\sigma_t = \sqrt{1 - \bar{\alpha}_t}\).

DPM-Solver introduces a change of variable \(\lambda_t = \log(\sqrt{\bar{\alpha}_t} / \sigma_t)\) (the log signal-to-noise ratio), which simplifies the ODE into an exact solution form:

$$x_s = \frac{\sigma_s}{\sigma_t} x_t - \sigma_s \int_{\lambda_t}^{\lambda_s} e^{-\lambda}\, \hat{x}_0(\lambda)\, d\lambda$$

where \(\hat{x}_0(\lambda)\) is the predicted clean image at log-SNR \(\lambda\). Different orders of Taylor expansion on \(\hat{x}_0(\lambda)\) give different solvers:

| Solver | Order | Steps for Good Quality | Key Idea |
|--------|-------|----------------------|----------|
| DDIM | 1st order | 50 | Euler method (constant approximation of \(\hat{x}_0\)) |
| DPM-Solver-2 | 2nd order | 20 | Midpoint method (linear approximation) |
| DPM-Solver++ | 2nd/3rd | 15-20 | Multistep, better stability |

Higher-order solvers take fewer steps because they better approximate the continuous denoising trajectory.

## Euler / Euler Ancestral

Simple ODE solvers commonly used in practice. Given the probability flow ODE \(dx = f(x, t)\, dt\):

**Euler** (deterministic, first-order): discretize with step size \(\Delta t = t_{i+1} - t_i\):

$$x_{i+1} = x_i + f(x_i, t_i)\, \Delta t$$

**Euler Ancestral** (stochastic): injects noise at each step for diversity:

$$x_{i+1} = x_i + f(x_i, t_i)\, \Delta t + g(t_i)\, \sqrt{|\Delta t|}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

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

In math: the model learns both \(\epsilon_\theta(x_t, t, c)\) and \(\epsilon_\theta(x_t, t, \varnothing)\) by randomly dropping \(c\) with probability \(p_\text{drop}\).

At inference, combine the conditional and unconditional predictions:

$$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + s \cdot \left[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\right]$$

where \(s\) is the **guidance scale**. This can be rewritten as:

$$\tilde{\epsilon}_\theta = (1 - s)\, \epsilon_\theta(x_t, t, \varnothing) + s\, \epsilon_\theta(x_t, t, c)$$

Intuitively, CFG amplifies the direction in noise space that points toward the condition \(c\) and away from the unconditional output. In score function terms:

$$\nabla_{x_t} \log p_s(x_t \mid c) = \nabla_{x_t} \log p(x_t) + s \cdot \nabla_{x_t} \log p(c \mid x_t)$$

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

Learn a **consistency function** \(f_\theta(x_t, t)\) that maps any noisy \(x_t\) on the same ODE trajectory directly to the origin \(x_0\):

```
Traditional:  x_T → x_{T-1} → ... → x_1 → x_0  (many steps)
Consistency:  x_t  ─────────────────────→  x_0    (one step)
```

In math:

$$\text{Traditional:} \quad x_T \to x_{T-1} \to \cdots \to x_1 \to x_0 \quad \text{(many steps)}$$

$$\text{Consistency:} \quad f_\theta(x_t, t) = x_0 \quad \text{(one step, for any } t \text{)}$$

The key **self-consistency property**: for any two points on the same PF-ODE trajectory,

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t') \quad \forall\, t, t' \in [\epsilon, T]$$

with the boundary condition \(f_\theta(x_\epsilon, \epsilon) = x_\epsilon\) (identity at \(t = \epsilon\)).

**Training** minimizes the consistency loss:

$$\mathcal{L} = \mathbb{E}\left[\, d\!\left(f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(\hat{x}_{t_n}, t_n)\right)\,\right]$$

where \(\hat{x}_{t_n}\) is obtained by one ODE step from \(x_{t_{n+1}}\), \(\theta^-\) is an EMA of \(\theta\), and \(d(\cdot, \cdot)\) is a distance metric (e.g., \(\ell_2\) or LPIPS).

- Can generate in 1-4 steps
- Quality trade-off vs. multi-step sampling
- Latent Consistency Models (LCM): apply consistency distillation to latent diffusion

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

## Resources

**Papers**

- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song et al., ICLR 2021. DDIM — deterministic sampling, step skipping, and inversion.
- [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling](https://arxiv.org/abs/2206.00927) — Lu et al., NeurIPS 2022. High-order ODE solver for diffusion.
- [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095) — Lu et al., 2022. Multistep solver with thresholding for guided sampling.
- [Consistency Models](https://arxiv.org/abs/2303.01469) — Song et al., ICML 2023. Direct mapping from any noise level to data in one step.
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378) — Luo et al., 2023. Apply consistency distillation to latent diffusion for 1-4 step generation.

**Blogs**

- [Stable Diffusion Samplers: A Comprehensive Guide](https://stable-diffusion-art.com/samplers/) — Practical comparison of samplers with visual examples and speed benchmarks.
- [How does Stable Diffusion work?](https://stable-diffusion-art.com/how-stable-diffusion-work/) — Accessible explanation of the full pipeline including sampler choices.

## Related

- [Diffusion Basics](01_Diffusion_Basics.md) — DDPM foundation and noise schedules
- [Latent Diffusion](03_Latent_Diffusion.md) — samplers applied in latent space
- [AI_Infra: Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving implications of step count
