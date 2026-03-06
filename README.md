# Multimodal Models Knowledge Base

A structured collection of notes on multimodal AI models, covering visual encoders, diffusion models, vision-language models, and applications in generation, video, and robotics.

## Folder Structure

```
Multimodal/
├── 00_Multimodal.md          # Root index (learning path)
├── visual_encoder/            # CNN, ViT, CLIP, SigLIP, VAE
│   ├── 00_Visual_Encoder.md
│   ├── 01_CNN_Basics.md
│   ├── 02_ViT.md
│   ├── 03_Semantic_Encoders.md
│   └── 04_VAE.md
├── diffusion/                 # DDPM, samplers, latent diffusion, DiT
│   ├── 00_Diffusion.md
│   ├── 01_Diffusion_Basics.md
│   ├── 02_Sampling.md
│   ├── 03_Latent_Diffusion.md
│   └── 04_DiT.md
├── vision_language/           # VLM architectures, models, training
│   ├── 00_Vision_Language.md
│   ├── 01_Architecture.md
│   ├── 02_Models.md
│   ├── 03_Unified_Models.md
│   └── 04_Training.md
└── applications/              # Image gen, video, robotics
    ├── 00_Applications.md
    ├── 01_Image_Generation.md
    ├── 02_Video.md
    └── 03_Robotics.md
```

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Folder | lowercase_underscores | `visual_encoder/` |
| Index file | 00_TopicName.md | `00_Visual_Encoder.md` |
| Content file | XX_Topic_Name.md | `01_CNN_Basics.md` |
| Acronyms | UPPERCASE | `VAE`, `ViT`, `DiT`, `VLM` |
| Links | relative with .md | `[Link](../00_Parent.md)` |

## Topics Covered

| Topic | Description |
|-------|-------------|
| **Visual Encoders** | CNN, ViT, CLIP, SigLIP 2, DINOv2, VAE, RAE |
| **Diffusion** | DDPM, noise schedules, DDIM, DPM-Solver, CFG, latent diffusion, DiT |
| **Vision-Language** | VLM architectures, LLaVA, Qwen-VL, unified models (Janus, BAGEL) |
| **Applications** | Text-to-image, video generation, robotics/VLA |

## Related Knowledge Bases

- [LLM](../LLM/) — Language model algorithms (transformer, attention)
- [AI_Infra](../AI_Infra/) — Systems and infrastructure (inference, distributed training)

## Generating Content with AI

See [PROMPT.md](PROMPT.md) for a ready-to-use prompt when asking LLMs to help generate or expand this knowledge base.
