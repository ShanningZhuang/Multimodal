# Generation Patterns: How AR and DiT Compose End-to-End

> Parent: [Multimodal Generation Architecture](00_Multimodal_Generation.md)

## Overview

Modern multimodal generation systems combine autoregressive (AR) language models with Diffusion Transformers (DiT) in distinct architectural patterns. Each pattern places the generation bottleneck differently, supports different modalities, and has different serving implications.

This document uses the vLLM-Omni architecture diagram as the central reference:

![Multi-modality model architectures](../images/multi-modality-models.png)

Three primary patterns emerge, plus a fourth fully-autoregressive approach:

| Pattern | Bottleneck | Example Models | Output Modalities |
|---------|-----------|----------------|-------------------|
| [AR + DiT(Main)](#pattern-1-ar--ditmain) | DiT | Qwen-Image, Qwen-Image-Edit | Image, Video |
| [AR(Main) + DiT](#pattern-2-armain--dit) | AR decoder | BAGEL, Hunyuan Image 3.0 | Image + Text (interleaved) |
| [AR + DiT (Omni)](#pattern-3-ar--dit-omni) | Both AR and DiT | Qwen3-Omni, Ming-Omni | Text + Audio |
| [AR-Dominated](#pattern-4-ar-dominated-no-separate-dit) | AR decoder + learned decoder | Nano Banana Pro | Image (no separate DiT) |

---

## Pattern 1: AR + DiT(Main)

**The DiT is the primary generation backbone.** The AR model (a VLM) processes inputs and produces conditioning embeddings, but the heavy lifting of generation happens in the DiT denoising loop.

### Example Models

- **Qwen-Image** — 20B MMDiT, text-to-image
- **Qwen-Image-Edit** — dual-path variant for instruction-based editing

### Data Flow

```
Text-to-Image (Qwen-Image):

Text prompt
    │
    ▼
┌──────────────────┐
│  Qwen2.5-VL      │  AR decoder (understanding)
│  (AR decoder)     │  Processes text, produces semantic embeddings
└────────┬─────────┘
         │ embeddings
         ▼
┌──────────────────┐
│  Patchify         │  Convert embeddings to latent patches
└────────┬─────────┘
         │ latent patches + noise
         ▼
┌──────────────────┐
│  DiT              │  Iterative denoising (20-50 steps)
│  (20B MMDiT)      │  The MAIN generation work happens here
│                   │  adaLN-Zero conditioning, flow matching
└────────┬─────────┘
         │ denoised latents
         ▼
┌──────────────────┐
│  UnPatchify       │  Reconstruct spatial latent from patches
└────────┬─────────┘
         │ spatial latent
         ▼
┌──────────────────┐
│  VAE Decoder      │  Latent → pixel space
└────────┬─────────┘
         │
         ▼
      Output Image
```

### Dual-Path Editing Variant (Qwen-Image-Edit)

For image editing, the input image is processed through **two parallel paths**:

```
                    Qwen-Image-Edit Dual-Path

Text instruction ──► Qwen2.5-VL ──► Semantic embeddings ──┐
("make it winter")   (understands     (what to change)      │
                      the edit)                              ├──► DiT ──► VAE Dec ──► Edited Image
Input image ───────► VAE Encoder ──► Appearance latents ───┘
                     (preserves        (what to keep)
                      pixel detail)
```

- **Semantic path** (Qwen2.5-VL): understands the edit instruction in context of the input image — handles style transfer, object manipulation, spatial reasoning
- **Appearance path** (VAE encoder): preserves pixel-level detail of regions that should remain unchanged
- The DiT receives both signals, enabling fine-grained control over what changes and what stays

### Component Roles

| Component | Role | Learn More |
|-----------|------|------------|
| Qwen2.5-VL (AR decoder) | Text/image understanding, produces conditioning embeddings | [VLM Architecture](../vision_language/01_Architecture.md) |
| Patchify / UnPatchify | Convert between spatial latents and patch sequences for DiT | [DiT](../diffusion/04_DiT.md) |
| DiT (MMDiT) | Main generation backbone — iterative denoising with adaLN-Zero | [DiT](../diffusion/04_DiT.md) |
| VAE Encoder | Encode input images to latent space (editing path) | [VAE](../visual_encoder/04_VAE.md) |
| VAE Decoder | Decode denoised latents to pixel space | [VAE](../visual_encoder/04_VAE.md) |

### Why DiT Is the Bottleneck

The DiT runs 20-50 denoising steps, each a full forward pass through a 20B parameter transformer. The AR decoder runs once (single forward pass for conditioning). This means:
- **Compute**: dominated by DiT (20-50x more forward passes than AR)
- **Serving**: the DiT stage needs its own GPU allocation and scheduler — see [Omni-Modality Serving](../applications/04_Omni_Serving.md) for how vLLM-Omni handles this with `OmniGenerationScheduler`
- **Optimization**: DiT-specific techniques (step distillation, flow matching) directly reduce latency

### Supported Tasks

- Text-to-image (t2i)
- Text-to-video (t2v)
- Image-to-image editing (i2i)

---

## Pattern 2: AR(Main) + DiT

**The AR decoder is the primary generation backbone.** It autoregressively generates both text tokens and image latent tokens in a unified sequence. The DiT serves as an auxiliary refinement module.

### Example Models

- **BAGEL** (ByteDance) — dual encoder (SigLIP 2 + FLUX VAE), diffusion refinement
- **Hunyuan Image 3.0** — AR-first image generation

### Data Flow

```
Text + Image Input (BAGEL):

Text prompt              Input image (optional)
    │                          │
    ▼                          ▼
┌──────────┐          ┌────────────────┐
│  Text     │          │ SigLIP 2       │  Understanding encoder
│  Tokenizer│          │ (semantic)     │
└────┬──────┘          └───────┬────────┘
     │                         │
     │                  ┌──────┴────────┐
     │                  │ FLUX VAE      │  Generation encoder
     │                  │ (appearance)  │
     │                  └───────┬───────┘
     │                          │
     └──────────┬───────────────┘
                │ unified token sequence
                ▼
┌───────────────────────────────────┐
│  AR Decoder (Main)                │
│  Autoregressively generates:      │
│  - Text tokens (answers, CoT)     │
│  - Image latent tokens            │
│  All interleaved in one sequence  │
└────────────┬──────────────────────┘
             │
     ┌───────┴───────┐
     │               │
     ▼               ▼
  Text output   Image latent tokens
  (answers,          │
   chain-of-         ▼
   thought)   ┌──────────────┐
              │  DiT          │  Refines latents (auxiliary)
              │  (FLUX DiT)   │  Fewer steps than Pattern 1
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  VAE Decoder  │  Latent → pixel space
              └──────┬───────┘
                     │
                     ▼
                Output Image
```

### The Dual Encoder Design

BAGEL uses the same dual encoder approach described in [Unified Models](../vision_language/03_Unified_Models.md):

- **SigLIP 2** ([Semantic Encoders](../visual_encoder/03_Semantic_Encoders.md)): high-level understanding — what objects are in the image, their relationships, scene meaning
- **FLUX VAE** ([VAE](../visual_encoder/04_VAE.md)): pixel-level detail — textures, colors, fine spatial structure

This separation lets the AR decoder receive both semantic context (for reasoning about the image) and appearance detail (for faithful reconstruction/editing).

### Interleaved Generation

A key capability of Pattern 2: the AR decoder generates text and image tokens in a **single unified sequence**:

```
AR output stream:
[text] "The image shows a sunset. Let me create a variation..." [CoT tokens] [img_start] [latent_1] [latent_2] ... [latent_N] [img_end]
```

This enables chain-of-thought reasoning *before* image generation — the model can plan its output, explain decisions, then generate the image, all in one autoregressive pass.

### Component Roles

| Component | Role | Learn More |
|-----------|------|------------|
| SigLIP 2 | Semantic understanding of input images | [Semantic Encoders](../visual_encoder/03_Semantic_Encoders.md) |
| FLUX VAE Encoder | Pixel-level appearance encoding of input images | [VAE](../visual_encoder/04_VAE.md) |
| AR Decoder (Main) | Generates text + image latent tokens autoregressively | [VLM Architecture](../vision_language/01_Architecture.md) |
| DiT (FLUX DiT) | Auxiliary refinement of image latents | [DiT](../diffusion/04_DiT.md) |
| VAE Decoder | Decode refined latents to pixels | [VAE](../visual_encoder/04_VAE.md) |

### Why AR Is the Bottleneck

The AR decoder generates potentially thousands of tokens (text + image latents) sequentially. The DiT runs fewer refinement steps since the AR decoder already produces structured latents. This means:
- **Compute**: dominated by AR token generation (sequential, memory-bound)
- **Serving**: AR stage needs KV cache management, efficient batching — see [Omni-Modality Serving](../applications/04_Omni_Serving.md) for how vLLM-Omni uses `OmniARScheduler`
- **Optimization**: speculative decoding, KV cache compression directly reduce latency

### Supported Tasks

- Text-to-image (t2i)
- Image-to-image editing (i2i)
- Image-to-text understanding (i2t) — same model, no DiT needed
- Interleaved text+image generation

---

## Pattern 3: AR + DiT (Omni)

**Both AR and DiT stages are critical.** The AR pipeline handles multimodal understanding and text/audio token generation, while the DiT synthesizes the final output waveform. This pattern supports any-to-any modality conversion.

### Example Models

- **Qwen3-Omni** (Alibaba) — MoE thinker + talker + Code2Wav
- **Ming-Omni**

### Data Flow

```
Any-to-Text+Audio (Qwen3-Omni):

Text         Image/Video        Audio
  │               │                │
  ▼               ▼                ▼
┌────────┐  ┌───────────┐  ┌────────────┐
│  Text   │  │  Visual    │  │  Audio     │
│Tokenizer│  │  Encoder   │  │  Encoder   │
└───┬─────┘  └─────┬─────┘  └─────┬──────┘
    │              │               │
    └──────────────┼───────────────┘
                   │ unified embeddings
                   ▼
    ┌─────────────────────────────┐
    │  Stage 0: Thinker (AR)      │  Processes all inputs
    │  MoE language model         │  Generates text/reasoning tokens
    │  Output: text + hidden      │  Emits text to user AND
    │          states             │  hidden states to next stage
    └─────────────┬───────────────┘
                  │ hidden states + embeddings
                  ▼
    ┌─────────────────────────────┐
    │  Stage 1: Talker (AR)       │  Converts thinker's representations
    │  Embedding → RVQ predictor  │  into audio codec codes
    │  Output: 8-layer RVQ codes  │  (Residual Vector Quantization)
    └─────────────┬───────────────┘
                  │ RVQ codec codes
                  ▼
    ┌─────────────────────────────┐
    │  Stage 2: Code2Wav (DiT)    │  Synthesizes audio waveform
    │  Non-autoregressive         │  from discrete codec codes
    │  Diffusion-based vocoder    │  Output: 24kHz audio
    └─────────────┬───────────────┘
                  │
                  ▼
          Audio Waveform (24kHz)
```

### The Three-Stage Pipeline

Unlike Patterns 1-2 which have two logical stages, Pattern 3 has **three**:

1. **Thinker** — full multimodal understanding. Accepts text, image, video, audio via dedicated encoders. Generates text responses and produces hidden state representations for the Talker.

2. **Talker** — audio token generation. Takes the Thinker's hidden states and converts them into **RVQ codec codes** — discrete tokens that represent audio at multiple levels of detail.

3. **Code2Wav** — waveform synthesis. A DiT-based vocoder that converts RVQ codes into a continuous audio waveform. This is a non-autoregressive diffusion process.

### RVQ (Residual Vector Quantization)

RVQ represents audio as **multiple layers of discrete codes**, each capturing progressively finer detail:

```
Layer 1: ████████████████  Coarse structure (pitch, rhythm)
Layer 2: ░░██░░██░░██░░██  Mid-level detail (timbre)
Layer 3: ·░·░·░·░·░·░·░·░  Fine detail (texture, breath)
  ...
Layer 8: ················  Finest residual detail

Each layer quantizes the RESIDUAL left by previous layers.
Total: 8 layers x sequence_length codes
```

The Talker generates all 8 layers of RVQ codes, which the Code2Wav DiT then synthesizes into a continuous waveform. This is analogous to how a VAE decoder converts image latents to pixels, but for audio.

### Component Roles

| Component | Role | Learn More |
|-----------|------|------------|
| Text Tokenizer | Tokenize text input | [LLM KB](../../LLM/) |
| Visual Encoder | Encode image/video to embeddings | [Semantic Encoders](../visual_encoder/03_Semantic_Encoders.md) |
| Audio Encoder | Encode audio input (Whisper-like) | — |
| Thinker (AR) | Multimodal reasoning, text generation | [VLM Architecture](../vision_language/01_Architecture.md) |
| Talker (AR) | Hidden states → RVQ codec codes | — |
| Code2Wav (DiT) | RVQ codes → audio waveform (diffusion) | [DiT](../diffusion/04_DiT.md) |

### Why Both AR and DiT Are Critical

Unlike Patterns 1-2 where one stage clearly dominates:
- The **Thinker** must process arbitrarily long multimodal inputs and generate text (AR, memory-bound)
- The **Talker** must generate codec codes for the full audio duration (AR, sequential)
- The **Code2Wav** must synthesize high-quality audio (DiT, compute-bound)

No single stage can be optimized away. This is why Pattern 3 models need vLLM-Omni's **fully disaggregated pipeline** — each stage runs as an independent process with its own GPU, scheduler, and batching strategy. See [Omni-Modality Serving](../applications/04_Omni_Serving.md) for implementation details.

### Supported Tasks

- Any-to-text+audio (text, image, video, audio inputs → text + spoken audio output)

---

## Pattern 4: AR-Dominated (No Separate DiT)

**The LLM handles creative generation; a learned decoder handles pixel reconstruction.** Often called "fully autoregressive," but this is misleading — the AR model generates discrete image tokens, not pixels. A non-trivial **image decoder** (tokenizer decoder, possibly with diffusion-based upsampling) converts tokens back to pixel space. The key distinction from Patterns 1-3 is that no separate DiT operates on the latent space during generation.

### Example Models

- **Nano Banana Pro** (Google, built on Gemini 3 Pro) — proprietary
- **Chameleon** (Meta) — VQ-VAE tokenizer + AR generation
- **Janus** (DeepSeek) — VQ tokenizer + AR generation

### Data Flow

```
AR-Dominated (Nano Banana Pro — hypothesized):

Text prompt
    │
    ▼
┌──────────────────────────┐
│  Gemini 3 Pro (LLM)      │
│  1. Thinking/reasoning    │  Plans the image via CoT
│  2. Token generation      │  ~1,290 discrete image tokens for 1MP
│  (autoregressive)         │  Each token sampled from VQ codebook
└───────────┬──────────────┘
            │ discrete image tokens (VQ codes)
            ▼
┌──────────────────────────┐
│  Image Decoder            │  VQ codes → pixels
│  (likely MAGVIT-2 or      │  Deep CNN/Transformer decoder
│   similar VQ decoder)     │  ~28x spatial upsampling
└───────────┬──────────────┘
            │               ┌──────────────────────────┐
            ├──────────────►│  Diffusion Upsampler?     │  (for 4K output)
            │               │  (hypothesized for high-  │
            │               │   res variants)           │
            │               └───────────┬──────────────┘
            │                           │
            ▼                           ▼
        Output Image (1MP)         Output Image (4K)
```

### The "Fully AR" Misconception

The label "fully autoregressive" describes the **token generation** stage, not the full pipeline. Converting ~1,290 discrete tokens to 1 megapixel (1024x1024 = 1,048,576 pixels) requires a powerful learned decoder — each token encodes ~813 pixels worth of information. This decoder does substantial work:

- **VQ codebook lookup** — each token maps to a learned embedding vector
- **Spatial upsampling** — from ~36x36 grid to 1024x1024 (~28x upsampling)
- **Detail synthesis** — the decoder must hallucinate fine-grained textures, edges, and gradients not captured in the discrete codes

### Hypothesized Internals (Nano Banana Pro)

Nano Banana Pro is proprietary, but we can reason about its architecture from observable behavior and Google's published research:

#### Hypothesis A: VQ Tokenizer + Powerful Decoder (Most Likely)

Google developed **[MAGVIT-2](https://arxiv.org/abs/2310.05737)** (Yu et al., ICLR 2024), a state-of-the-art VQ tokenizer specifically for autoregressive image generation. It introduced **Lookup-Free Quantization (LFQ)** — replacing traditional codebook lookup with binary decomposition, enabling a massive vocabulary (2^18 = 262,144 codes) without codebook collapse. Built on the earlier **[MAGVIT](https://arxiv.org/abs/2212.05199)** (Yu et al., CVPR 2023), which introduced 3D VQ tokenization for video:

```
Training:  Image → MAGVIT-2 Encoder → ~1,290 VQ codes → MAGVIT-2 Decoder → Reconstructed image
                    (learns codebook)                     (learns upsampling)

Inference: Prompt → Gemini (AR) → ~1,290 VQ codes → MAGVIT-2 Decoder → Image
```

- ~36x36 spatial grid of codes (36x36 = 1,296 ≈ 1,290)
- Decoder is a deep convolutional network, not a simple lookup
- This is what DALL-E 1, Parti, and Chameleon all do

#### Hypothesis B: AR Tokens + Diffusion Upsampler (Likely for 4K)

Google has a history of cascaded approaches (Imagen used cascaded diffusion). The token count scaling is suspicious:

| Resolution | Tokens | Pixels | Tokens/Pixel Ratio |
|-----------|--------|--------|-------------------|
| 1MP (1K) | ~1,290 | 1M | 1:775 |
| 4MP (2K) | ~1,120 | 4M | 1:3,571 |
| 16MP (4K) | ~2,000 | 16M | 1:8,000 |

If it were purely VQ-decoded, token count should scale roughly linearly with pixels. Instead, going from 1MP to 16MP (16x more pixels) only requires ~1.55x more tokens. This strongly suggests a **diffusion-based super-resolution** stage handles the high-res upsampling:

```
Gemini (AR) → ~2,000 VQ codes → VQ Decoder → Base image (e.g. 512x512)
                                                    │
                                                    ▼
                                              Diffusion SR → 4K image
```

This would make the 4K variant effectively **Pattern 2 (AR Main + DiT)** in disguise.

#### Hypothesis C: Continuous Latent Tokens + VAE Decoder

Instead of discrete VQ codes, the LLM might predict **continuous vectors**:

```
Gemini (AR) → continuous latent vectors → VAE-like Decoder → Image
```

This blurs the line with latent diffusion — the AR model generates a latent representation directly, and a VAE decoder reconstructs pixels. Less likely given Google's investment in discrete tokenization (MAGVIT-2).

### Open-Source AR-Dominated Models

Unlike Nano Banana Pro, these models have known architectures:

| Model | Tokenizer | Decoder | Architecture |
|-------|-----------|---------|-------------|
| Chameleon (Meta) | VQ-VAE (8,192 codebook) | CNN decoder | Purely AR over discrete tokens |
| Janus (DeepSeek) | VQ tokenizer | VQ decoder | AR generation, separate understanding encoder |
| Parti (Google) | ViT-VQGAN (8,192 codebook) | CNN decoder | AR generation, 1,024 tokens for 256x256 |
| MAGVIT-2 (Google) | LFQ (262,144 codebook) | CNN decoder | Tokenizer — first to show AR beats diffusion |

All confirm the pattern: AR generates discrete codes, a learned decoder reconstructs pixels. The decoder quality is critical to output quality.

### Key Differences from Patterns 1-3

- **No diffusion in the main generation loop** — no iterative denoising for the creative decisions (composition, content, structure)
- **No separate text encoder** — the same LLM that understands the prompt generates the image tokens
- **Non-deterministic** — randomness at every token step; seeds cannot reproduce exact outputs
- **Reasoning-integrated** — the model's chain-of-thought reasoning directly informs generation
- **Decoder does heavy lifting** — unlike Patterns 1-3 where the DiT/diffusion is explicit, here the decoder's sophistication is hidden but critical

See [Image Generation](../applications/01_Image_Generation.md) for a detailed comparison with diffusion-based approaches.

---

## Pattern Comparison

| | AR + DiT(Main) | AR(Main) + DiT | AR + DiT (Omni) | AR-Dominated |
|---|---|---|---|---|
| **Bottleneck** | DiT (20-50 denoising steps) | AR decoder (token-by-token) | Both AR and DiT | AR decoder + decoder quality |
| **DiT role** | Main generator | Auxiliary refiner | Waveform synthesizer | None (or hidden in upsampler) |
| **AR role** | Conditioning only | Main generator | Understanding + codec generation | Creative generation (tokens) |
| **Decoder role** | VAE decoder (simple) | VAE decoder (simple) | DiT vocoder | VQ decoder (heavy lifting) |
| **Output modalities** | Image, Video | Image + Text | Text + Audio | Image |
| **Interleaved text+image** | No | Yes (CoT + image) | Yes (text + audio) | Yes (think + image) |
| **Example models** | Qwen-Image | BAGEL, Hunyuan | Qwen3-Omni | Nano Banana Pro, Chameleon |
| **CFG needed** | Optional | Model-dependent | No | No |
| **Serving complexity** | 2-stage (AR + DiT) | 2-stage (AR + DiT) | 3-stage (Thinker + Talker + DiT) | 1-stage (AR + decoder) |
| **Primary optimization** | DiT step distillation | KV cache, speculative decoding | Disaggregated pipeline | KV cache, VQ codebook quality |

### When to Use Which

- **Pattern 1 (AR + DiT Main)**: Best image/video quality from open-source models. DiT handles the hard part of pixel-level generation. Choose when generation quality is the priority and you can afford DiT compute.

- **Pattern 2 (AR Main + DiT)**: Best for mixed text+image tasks. The AR decoder can reason about the image before generating it. Choose when you need interleaved understanding and generation (e.g., visual QA followed by image editing in one conversation).

- **Pattern 3 (AR + DiT Omni)**: Required for any-to-any modality, especially text+audio output. The three-stage pipeline is more complex to serve but handles the full multimodal spectrum. Choose for voice assistants and omni-modal agents.

- **Pattern 4 (AR-Dominated)**: Simplest serving architecture — no explicit DiT pipeline to manage. The LLM handles creative generation; a learned decoder (VQ decoder, possibly with diffusion upsampler for high-res) handles pixel reconstruction. Currently strongest in proprietary models (Gemini). The decoder quality is the hidden bottleneck — MAGVIT-2 class tokenizers are required for competitive output.

## Appendix: vLLM-Omni Model Reference

The [vLLM-Omni](https://github.com/vllm-project/vllm-omni) codebase provides concrete implementations of Patterns 1-3. Each model is defined by a **YAML stage config** that declares the pipeline stages, and **stage input processors** that transform data between stages. This appendix maps every supported model to its generation pattern.

Source: `/root/vllm-omni/vllm_omni/model_executor/`

### Pattern 1: AR + DiT(Main) — Image Generation

These models use the AR stage for conditioning/understanding, then a DiT stage does the main image generation work.

#### GLM-Image

```
Stage 0 (AR)                          Stage 1 (DiT)
┌─────────────────────┐   prior     ┌─────────────────────┐
│ GlmImageForCond...  │   tokens    │ Diffusion pipeline   │
│ worker_type: ar      │ ─────────► │ worker_type: gen     │
│ output: token_ids    │ ar2diffusion│ output: image        │
│ is_comprehension: T  │            │                      │
└─────────────────────┘            └─────────────────────┘
```

- **Stage config**: `stage_configs/glm_image.yaml`
- **Stage input processor**: `stage_input_processors/glm_image.py` — `ar2diffusion()` upsamples prior tokens by 2x (32x→16x downsampling), extracts VQ-VAE token IDs for i2i mode
- **Modalities**: text (+ optional image) → image
- **Notable**: supports both t2i and i2i modes; prior tokens are upsampled before DiT

#### MammothModa2

```
Stage 0 (AR)                          Stage 1 (DiT)
┌─────────────────────┐   hidden    ┌─────────────────────┐
│ MammothModa2AR...    │   states   │ MammothModa2DiT...   │
│ worker_type: ar      │ ─────────► │ worker_type: gen     │
│ output: latent       │  ar2dit    │ output: image        │
│ max_batch: 100       │            │ max_batch: 1         │
└─────────────────────┘            └─────────────────────┘
```

- **Stage config**: `stage_configs/mammoth_moda2.yaml` (full pipeline), `mammoth_moda2_ar.yaml` (AR-only for understanding)
- **Stage input processor**: `stage_input_processors/mammoth_moda2.py` — `ar2dit()` extracts hidden states, creates separate `text_condition_token_mask` and `image_condition_token_mask` attention masks
- **Modalities**: multimodal → image (or text-only in AR-only mode)
- **Notable**: MoE support with separate `und_expert` vs `gen_expert` routing; AR-only config available for pure understanding tasks

#### Hunyuan-Image-3

```
Stage 0 (AR)
┌─────────────────────┐
│ HunyuanImage3For...  │   Only AR stage defined in vLLM-Omni
│ worker_type: ar      │   External DiT pipeline handles image generation
│ output: latent       │
│ tp_size: 8           │
│ is_comprehension: T  │
└─────────────────────┘
```

- **Stage config**: `stage_configs/hunyuan_image_3_moe.yaml`
- **Modalities**: image + text → latent (for external diffusion)
- **Notable**: requires 8x L40S-48G GPUs (TP=8); only the AR comprehension stage is defined in vLLM-Omni, the DiT runs externally

### Pattern 2: AR(Main) + DiT — Interleaved Text + Image

The AR decoder is the main backbone, generating both text and image latent tokens. The DiT refines image latents.

#### BAGEL

```
Stage 0 (AR — Main)                   Stage 1 (DiT — Auxiliary)
┌─────────────────────┐   KV cache  ┌─────────────────────┐
│ AR thinker           │  + latents  │ DiT image generator  │
│ worker_type: ar      │ ─────────► │ worker_type: gen     │
│ output: text         │  collect_   │ output: image        │
│ expand_cfg_prompts   │  cfg_kv_   │                      │
│ (CFG triple-branch)  │  caches    │                      │
└─────────────────────┘            └─────────────────────┘
```

- **Stage config**: `stage_configs/bagel.yaml`, `bagel_multiconnector.yaml` (distributed)
- **Stage input processor**: `stage_input_processors/bagel.py` — `expand_cfg_prompts()` creates triple-branch CFG (gen + cfg_text + cfg_img); `collect_cfg_kv_caches()` transfers KV caches from AR to DiT via `kv_transfer_manager`
- **Modalities**: text (+ optional image) → text + image
- **Notable**: unique CFG implementation — AR stage generates 3 prompt variants (positive, negative-text, negative-image), each producing KV caches that the DiT stage uses for classifier-free guidance. KV caches are transferred via `kv_transfer_manager`, not just embeddings

### Pattern 3: AR + DiT (Omni) — Multi-Stage Audio

These models have AR stages for understanding and codec generation, plus a DiT/generation stage for waveform synthesis.

#### Qwen3-Omni-MoE (3-stage, flagship)

```
Stage 0 (Thinker)          Stage 1 (Talker)          Stage 2 (Code2Wav)
┌──────────────────┐ hidden ┌──────────────────┐ RVQ  ┌──────────────────┐
│ Qwen3OmniMoe...  │ states │ Qwen3OmniMoe...  │ codes│ Qwen3OmniMoe...  │
│ worker: ar       │ ─────► │ worker: ar       │ ───► │ worker: gen      │
│ output: latent   │thinker │ output: latent   │talker│ output: audio    │
│ MoE, GPU 0      │2talker │ GPU 1, 60% mem   │2code │ GPU 1, 10% mem   │
│ 90% mem          │        │                  │2wav  │                  │
│ final: text      │        │ stop: [2150]     │      │                  │
└──────────────────┘        └──────────────────┘      └──────────────────┘
```

- **Stage configs**: `stage_configs/qwen3_omni_moe.yaml` (sync), `qwen3_omni_moe_async_chunk.yaml` (streaming), `qwen3_omni_moe_multiconnector.yaml` (distributed)
- **Stage input processor**: `stage_input_processors/qwen3_omni.py` — `thinker2talker()` extracts embedding layers "0" and "24" plus TTS special tokens (bos/eos/pad); `talker2code2wav()` transposes 8-layer RVQ codes and flattens
- **Modalities**: text + image + video + audio → text + audio
- **Notable**: MoE thinker/talker; async streaming variant chunks codec frames (25 frames/chunk) with sliding context window; Stage 0 emits both text (to user) and hidden states (to Stage 1)

#### Qwen2.5-Omni (3-stage, dense)

```
Stage 0 (Thinker)          Stage 1 (Talker)          Stage 2 (Token2Wav)
┌──────────────────┐ hidden ┌──────────────────┐ codec┌──────────────────┐
│ Qwen2_5Omni...   │ states │ Qwen2_5Omni...   │ codes│ Qwen2_5Omni...   │
│ worker: ar       │ ─────► │ worker: ar       │ ───► │ worker: gen      │
│ output: latent   │thinker │ output: latent   │      │ output: audio    │
│ GPU 0, 80% mem   │2talker │                  │      │                  │
│ final: text      │        │                  │      │                  │
└──────────────────┘        └──────────────────┘      └──────────────────┘
```

- **Stage configs**: `stage_configs/qwen2_5_omni.yaml`, `qwen2_5_omni_multiconnector.yaml` (distributed)
- **Stage input processor**: `stage_input_processors/qwen2_5_omni.py` — `thinker2talker()` splits latent into prompt embeddings + generated tokens, creates talker prompt with special tokens (PAD=8292, START=8293, END=8294)
- **Modalities**: multimodal → text + audio
- **Notable**: dense model (not MoE); predecessor to Qwen3-Omni

#### MiMo-Audio (2-stage, fused thinker+talker)

```
Stage 0 (Fused Thinker+Talker)        Stage 1 (Code2Wav)
┌──────────────────────────┐   codec  ┌──────────────────┐
│ MiMoAudioForCond...      │   codes  │ MiMoAudioToken2..│
│ worker: ar               │ ───────► │ worker: gen      │
│ model_stage:             │ llm2     │ output: audio    │
│   fused_thinker_talker   │ code2wav │                  │
│ output: latent           │          │                  │
└──────────────────────────┘          └──────────────────┘
```

- **Stage configs**: `stage_configs/mimo_audio.yaml` (sync), `mimo_audio_async_chunk.yaml` (streaming)
- **Stage input processor**: `stage_input_processors/mimo_audio.py` — `llm2code2wav()` extracts `code_predictor_codes`, removes zero-padded frames, flattens column-major; async variant accumulates chunks with sliding window
- **Modalities**: text + audio → text + audio
- **Notable**: fuses thinker and talker into a single AR stage (2-stage instead of 3-stage pipeline), simplifying serving at the cost of less stage-level parallelism

#### Qwen3-TTS (2-stage, speech-only)

```
Stage 0 (Talker)                       Stage 1 (Code2Wav)
┌──────────────────────────┐   audio  ┌──────────────────┐
│ Qwen3TTSTalkerFor...     │   codes  │ Qwen3TTSCode2Wav │
│ worker: ar               │ ───────► │ worker: gen      │
│ output: latent           │ talker2  │ output: audio    │
│                          │ code2wav │                  │
└──────────────────────────┘          └──────────────────┘
```

- **Stage configs**: `stage_configs/qwen3_tts.yaml` (async), `qwen3_tts_batch.yaml` (batched, batch=4), `qwen3_tts_no_async_chunk.yaml` (sync)
- **Stage input processor**: `stage_input_processors/qwen3_tts.py` — `talker2code2wav()` extracts 16-layer audio codes, filters EOS frames; async variant has two-phase streaming: initial small chunks for low TTFA (time-to-first-audio), then normal-sized chunks
- **Modalities**: text → audio
- **Notable**: no thinker stage (direct text → speech); 16-layer RVQ codes (vs 8-layer in Qwen3-Omni); batch mode supports concurrent TTS requests

#### CosyVoice3 (2-stage, speech-only)

```
Stage 0 (Talker)                       Stage 1 (Code2Wav)
┌──────────────────────────┐   speech ┌──────────────────┐
│ CosyVoice3Model          │   tokens │ CosyVoice3Model  │
│ worker: ar               │ ───────► │ worker: gen      │
│ output: latent           │ text2flow│ output: audio    │
│ gpu_mem: 0.4             │          │ gpu_mem: 0.2     │
└──────────────────────────┘          └──────────────────┘
```

- **Stage config**: `stage_configs/cosyvoice3.yaml`
- **Stage input processor**: `stage_input_processors/cosyvoice3.py` — `text2flow()` prepends original prompt token IDs to speech token output
- **Modalities**: text → audio
- **Notable**: lightweight pipeline (40% + 20% GPU memory); speech tokens are prepended with original text tokens for alignment

### Summary Table

| Model | Pattern | Stages | AR Worker | Gen Worker | Modalities | Config Variants |
|-------|---------|--------|-----------|------------|------------|----------------|
| GLM-Image | 1 (AR+DiT Main) | 2 | Thinker → prior tokens | DiT → image | text/image → image | single, distributed |
| MammothModa2 | 1 (AR+DiT Main) | 2 (or 1) | AR → hidden states | DiT → image | multimodal → image | full pipeline, AR-only |
| Hunyuan-Image-3 | 1 (AR+DiT Main) | 1 (AR only) | AR → latent | (external) | image+text → latent | MoE, TP=8 |
| BAGEL | 2 (AR Main+DiT) | 2 | AR Main → text + KV caches | DiT refiner → image | text/image → text+image | single, distributed |
| Qwen3-Omni-MoE | 3 (Omni) | 3 | Thinker → Talker → RVQ | Code2Wav → audio | any → text+audio | sync, async, distributed |
| Qwen2.5-Omni | 3 (Omni) | 3 | Thinker → Talker → codec | Token2Wav → audio | multimodal → text+audio | single, distributed |
| MiMo-Audio | 3 (Omni) | 2 | Fused thinker+talker → codec | Code2Wav → audio | text+audio → text+audio | sync, async |
| Qwen3-TTS | 3 (Omni) | 2 | Talker → audio codes | Code2Wav → audio | text → audio | async, batch, sync |
| CosyVoice3 | 3 (Omni) | 2 | Talker → speech tokens | Code2Wav → audio | text → audio | single |

### Key Code Paths

```
vllm_omni/model_executor/
├── stage_configs/              # YAML pipeline definitions (18 files)
│   ├── bagel.yaml              # Pattern 2: AR(Main) + DiT with CFG
│   ├── glm_image.yaml          # Pattern 1: AR + DiT(Main) with prior tokens
│   ├── mammoth_moda2.yaml      # Pattern 1: AR + DiT(Main) with hidden states
│   ├── hunyuan_image_3_moe.yaml # Pattern 1: AR-only (DiT external)
│   ├── qwen3_omni_moe.yaml     # Pattern 3: 3-stage Omni
│   ├── qwen2_5_omni.yaml       # Pattern 3: 3-stage Omni (dense)
│   ├── mimo_audio.yaml         # Pattern 3: 2-stage fused Omni
│   ├── qwen3_tts.yaml          # Pattern 3: 2-stage TTS
│   └── cosyvoice3.yaml         # Pattern 3: 2-stage TTS
├── stage_input_processors/     # Inter-stage data transforms (8 files)
│   ├── bagel.py                # expand_cfg_prompts(), collect_cfg_kv_caches()
│   ├── glm_image.py            # ar2diffusion() — prior token upsampling
│   ├── mammoth_moda2.py        # ar2dit() — hidden state extraction + masking
│   ├── qwen3_omni.py           # thinker2talker(), talker2code2wav() + async
│   ├── qwen2_5_omni.py         # thinker2talker() — embedding split
│   ├── mimo_audio.py           # llm2code2wav() — codec flattening
│   ├── qwen3_tts.py            # talker2code2wav() — 16-layer RVQ + streaming
│   └── cosyvoice3.py           # text2flow() — prefix prepend
└── models/                     # Model implementations (8 directories)
    ├── bagel/                  # AR thinker + DiT refiner
    ├── glm_image/              # AR prior token generator
    ├── mammoth_moda2/          # AR + DiT with MoE routing
    ├── hunyuan_image3/         # AR comprehension (TP=8)
    ├── qwen3_omni/             # MoE thinker + talker + code2wav
    ├── qwen2_5_omni/           # Dense thinker + talker + token2wav
    ├── mimo_audio/             # Fused thinker+talker + code2wav
    ├── qwen3_tts/              # TTS talker + code2wav
    └── cosyvoice3/             # TTS talker + code2wav
```

## Related

- [Unified Models](../vision_language/03_Unified_Models.md) — dual encoder vs unified encoder approaches that map to these patterns
- [DiT](../diffusion/04_DiT.md) — the diffusion transformer used in Patterns 1-3
- [VAE](../visual_encoder/04_VAE.md) — encoder/decoder for latent space conversion
- [Semantic Encoders](../visual_encoder/03_Semantic_Encoders.md) — CLIP/SigLIP used as understanding encoders
- [Sampling](../diffusion/02_Sampling.md) — CFG and sampling algorithms used in DiT stages
- [VLM Architecture](../vision_language/01_Architecture.md) — fusion strategies, projectors
- [Omni-Modality Serving](../applications/04_Omni_Serving.md) — how vLLM-Omni implements disaggregated serving for these patterns
- [Image Generation](../applications/01_Image_Generation.md) — practical application of Patterns 1 and 4
