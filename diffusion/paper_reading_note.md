# Diffusion — Task Definition

## The Core Task: Generative Modeling

Given a dataset of samples from an unknown distribution $p_{\text{data}}(x)$, learn a model that can produce **new** samples from that distribution.

This is fundamentally hard because $p_{\text{data}}$ lives in extremely high-dimensional space (e.g., 256×256×3 = 196,608 dimensions for images) with complex, multi-modal structure. Directly modeling $p_{\text{data}}$ is intractable.

## How Diffusion Decomposes the Task

Diffusion's key insight: **don't generate in one shot**. Decompose the generation into a sequence of small, tractable denoising steps.

The training task at each step is a **regression problem**:

```
Given:  x_t = signal(x_0, t) + noise(ε, t)    (a noisy mixture at timestep t)
Predict: ε       (noise prediction — DDPM)
     or  x_0     (data prediction)
     or  v = ε - x_0  (velocity prediction — flow matching)
```

Why this works: each individual denoising step is "easy" — the model only needs to make a small correction. Chained together over T steps, these small corrections compose into a powerful generative process that transforms pure Gaussian noise into complex, high-quality data.

- **One big step** (noise → data): impossible to learn directly
- **Many tiny steps** (slightly less noisy → slightly more clean): each one is a simple regression

## The Task Across Modalities

The same diffusion framework applies to any continuous data distribution. What changes is the **data space** $x_0$, the **conditioning signal** $c$, and the **network architecture** that parameterizes the denoiser.

### Image Generation

$$p_{\text{data}}(x) = \text{distribution over images}$$

| Component | Details |
|-----------|---------|
| Data $x_0$ | Image pixels, or (more commonly) VAE latents $z_0 \in \mathbb{R}^{H \times W \times C}$ |
| Condition $c$ | Text prompt, class label, reference image, or unconditional |
| Tokenization | 2D patchify: split latent into $p \times p$ patches → sequence of tokens |
| Architecture | U-Net (SD 1.x/2.x), DiT (SD3, FLUX), MMDiT (SD3) |
| Output | Denoised latent → VAE decode → image |

Task: given text "a cat on a surfboard", sample from $p(\text{image} \mid \text{text})$ by iteratively denoising a random latent.

### Video Generation

$$p_{\text{data}}(x) = \text{distribution over videos (sequences of frames)}$$

| Component | Details |
|-----------|---------|
| Data $x_0$ | Video latents $z_0 \in \mathbb{R}^{T \times H \times W \times C}$ (T = temporal frames) |
| Condition $c$ | Text prompt, first frame (image-to-video), or both |
| Tokenization | 3D patchify: $(F, p_h, p_w)$ spacetime patches → token sequence |
| Architecture | 3D DiT with factored spatial-temporal attention (Sora, CogVideoX, Wan2.1) |
| Output | Denoised video latent → 3D VAE decode → video frames |
| Extra challenge | Temporal consistency — frames must be coherent across time |

The token count explodes: a 16-frame 256×256 video can produce 262K tokens. Factored attention (spatial-only + temporal-only layers) is essential.

### Text Generation (Continuous Diffusion)

$$p_{\text{data}}(x) = \text{distribution over text embeddings}$$

| Component | Details |
|-----------|---------|
| Data $x_0$ | Continuous text embeddings (not discrete tokens) |
| Condition $c$ | Prompt prefix, instructions, or unconditional |
| Tokenization | Word/subword embeddings already form a sequence |
| Architecture | Transformer denoiser operating on embedding space |
| Output | Denoised embeddings → nearest-neighbor decode to tokens |
| Challenge | Text is inherently discrete; diffusion operates on continuous space, so a mapping between discrete tokens and continuous embeddings is needed |

Examples: Diffusion-LM, MDLM, SEDD. Less dominant than autoregressive LLMs for text, but enables parallel generation (all tokens denoised simultaneously vs. one-by-one).

### Action Generation (Diffusion Policy)

$$p_{\text{data}}(x) = \text{distribution over action trajectories}$$

| Component | Details |
|-----------|---------|
| Data $x_0$ | Action sequence $a_{1:H} \in \mathbb{R}^{H \times D_a}$ (H-step horizon, $D_a$-dim actions) |
| Condition $c$ | Robot observation: camera images + proprioceptive state (joint angles, gripper) |
| Tokenization | Action chunks: group H timesteps into the sequence |
| Architecture | 1D temporal DiT or U-Net (1D convolutions over action horizon) |
| Output | Denoised action trajectory → execute first $k$ actions → re-plan |
| Why diffusion | Robot tasks have **multimodal** action distributions — multiple valid ways to grasp an object. Diffusion captures all modes; MSE regression collapses to the mean |

Example: Diffusion Policy — given image observation, denoise a full action trajectory, execute a chunk, observe again, re-plan.

### Vision-Language-Action (VLA) Models

$$p_{\text{data}}(x) = \text{distribution over actions, conditioned on vision + language}$$

| Component | Details |
|-----------|---------|
| Data $x_0$ | Action sequence (same as Diffusion Policy) |
| Condition $c$ | Language instruction + camera images (multi-view) + proprioceptive state |
| Architecture | Large vision-language model backbone (e.g., pre-trained VLM) + diffusion action head |
| Key idea | Leverage VLM's world knowledge for understanding; use diffusion head for precise, multimodal action generation |
| Output | Denoised action trajectory conditioned on "pick up the red cup" + current visual scene |

Examples: π₀ (Physical Intelligence), Octo. The VLM encodes the scene and instruction into a rich conditioning representation; the diffusion head generates the action trajectory. This separates **understanding** (VLM) from **control** (diffusion).

## Unified View

```
All diffusion tasks share the same skeleton:

1. Define data space x_0         (images, video, actions, ...)
2. Define condition c             (text, observations, ...)
3. Forward:  x_t = mix(x_0, ε, t)
4. Train:    minimize ‖f_θ(x_t, t, c) - target‖²
5. Infer:    x_T → x_{T-1} → ... → x_0  (iterative denoising)

What changes:
- Dimensionality and structure of x_0
- How x_0 is tokenized (2D patches, 3D patches, action chunks, ...)
- What c provides and how it enters the network (adaLN, cross-attn, joint-attn)
- Architecture (U-Net, DiT, MMDiT, 3D DiT, ...)
```

## Paper Reading

