# Multimodal Models Learning Path

> Goal: Understand how vision and language converge — from visual encoders to diffusion models to unified VLMs

## Overview: The Multimodal Stack (Bottom to Top)

```
┌─────────────────────────────────────────────┐
│           Applications                       │  ← Generation, Robotics, Video
├─────────────────────────────────────────────┤
│       Vision-Language Models                 │  ← VLMs, Unified Models
├─────────────────────────────────────────────┤
│          Diffusion Models                    │  ← DDPM, Latent Diffusion, DiT
├─────────────────────────────────────────────┤
│         Visual Encoders                      │  ← CNN, ViT, CLIP, VAE
└─────────────────────────────────────────────┘
```

---

## Phase 1: Visual Encoders

> How images become representations that models can process

| # | Topic | File | Status |
|---|-------|------|--------|
| 1 | CNN Basics | [visual_encoder/01_CNN_Basics.md](visual_encoder/01_CNN_Basics.md) | [ ] |
| 2 | Vision Transformer (ViT) | [visual_encoder/02_ViT.md](visual_encoder/02_ViT.md) | [ ] |
| 3 | Semantic Encoders (CLIP, SigLIP 2, DINOv2) | [visual_encoder/03_Semantic_Encoders.md](visual_encoder/03_Semantic_Encoders.md) | [ ] |
| 4 | VAE (Variational Autoencoders) | [visual_encoder/04_VAE.md](visual_encoder/04_VAE.md) | [ ] |

**Section index**: [visual_encoder/00_Visual_Encoder.md](visual_encoder/00_Visual_Encoder.md)

---

## Phase 2: Diffusion Models

> How to generate images by learning to reverse a noising process

| # | Topic | File | Status |
|---|-------|------|--------|
| 1 | Diffusion Basics (DDPM) | [diffusion/01_Diffusion_Basics.md](diffusion/01_Diffusion_Basics.md) | [ ] |
| 2 | Sampling (DDIM, DPM-Solver, CFG) | [diffusion/02_Sampling.md](diffusion/02_Sampling.md) | [ ] |
| 3 | Latent Diffusion (Stable Diffusion) | [diffusion/03_Latent_Diffusion.md](diffusion/03_Latent_Diffusion.md) | [ ] |
| 4 | DiT (Diffusion Transformer) | [diffusion/04_DiT.md](diffusion/04_DiT.md) | [ ] |

**Section index**: [diffusion/00_Diffusion.md](diffusion/00_Diffusion.md)

---

## Phase 3: Vision-Language Models

> How to combine vision and language in a single model

| # | Topic | File | Status |
|---|-------|------|--------|
| 1 | VLM Architecture Patterns | [vision_language/01_Architecture.md](vision_language/01_Architecture.md) | [ ] |
| 2 | Key Models (LLaVA, Qwen-VL, InternVL) | [vision_language/02_Models.md](vision_language/02_Models.md) | [ ] |
| 3 | Unified Models (Janus, BAGEL) | [vision_language/03_Unified_Models.md](vision_language/03_Unified_Models.md) | [ ] |
| 4 | Multimodal Training | [vision_language/04_Training.md](vision_language/04_Training.md) | [ ] |

**Section index**: [vision_language/00_Vision_Language.md](vision_language/00_Vision_Language.md)

---

## Phase 3.5: Generation Architecture

> How components compose into end-to-end generation systems

| # | Topic | File | Status |
|---|-------|------|--------|
| 1 | Generation Patterns (AR+DiT composition) | [multimodal_generation/01_Generation_Patterns.md](multimodal_generation/01_Generation_Patterns.md) | [ ] |

**Section index**: [multimodal_generation/00_Multimodal_Generation.md](multimodal_generation/00_Multimodal_Generation.md)

---

## Phase 4: Applications

> Where multimodal models meet the real world

| # | Topic | File | Status |
|---|-------|------|--------|
| 1 | Image Generation | [applications/01_Image_Generation.md](applications/01_Image_Generation.md) | [ ] |
| 2 | Video Generation & Understanding | [applications/02_Video.md](applications/02_Video.md) | [ ] |
| 3 | Robotics & Embodied AI | [applications/03_Robotics.md](applications/03_Robotics.md) | [ ] |

**Section index**: [applications/00_Applications.md](applications/00_Applications.md)

---

## Cross-References

| Related KB | Topic | Link |
|-----------|-------|------|
| LLM | Transformer architecture (attention, FFN) | [LLM KB](../LLM/) |
| AI_Infra | Multimodal inference serving | [AI_Infra/inference/07_Multimodal_Inference.md](../AI_Infra/inference/07_Multimodal_Inference.md) |
| AI_Infra | Inference frameworks (vLLM, SGLang) | [AI_Infra/inference/05_Frameworks.md](../AI_Infra/inference/05_Frameworks.md) |
| AI_Infra | Distributed training systems | [AI_Infra/distributed/00_Distributed.md](../AI_Infra/distributed/00_Distributed.md) |

## Key Papers

| Paper | Year | Contribution |
|-------|------|-------------|
| ViT (Dosovitskiy et al.) | 2020 | Vision Transformer |
| CLIP (Radford et al.) | 2021 | Contrastive language-image pretraining |
| DDPM (Ho et al.) | 2020 | Denoising diffusion probabilistic models |
| Latent Diffusion (Rombach et al.) | 2022 | Stable Diffusion, diffusion in latent space |
| DiT (Peebles & Xie) | 2023 | Transformer replaces U-Net for diffusion |
| [MAGVIT](https://arxiv.org/abs/2212.05199) (Yu et al.) | 2023 | 3D VQ tokenizer for video/image |
| [MAGVIT-2](https://arxiv.org/abs/2310.05737) (Yu et al.) | 2024 | LFQ tokenizer — AR beats diffusion |
| LLaVA (Liu et al.) | 2023 | Simple, effective VLM |
| DINOv2 (Oquab et al.) | 2023 | Self-supervised ViT at scale |
| SigLIP 2 (Tschannen et al.) | 2025 | Improved contrastive visual encoder |
| Janus (DeepSeek) | 2025 | Unified understanding + generation |
| BAGEL (ByteDance) | 2025 | Dual encoder unified model |
