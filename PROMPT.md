# Knowledge Base Generation Prompt

Use this prompt when asking LLMs to help generate or expand the Multimodal knowledge base.

---

## System Prompt

```
You are helping me build a structured knowledge base about multimodal models (vision, diffusion, VLMs). Follow these conventions strictly:

### Folder Structure
- Each topic area is a folder with lowercase and underscores: `topic_name/`
- Folders are NOT numbered (e.g., `visual_encoder/`, not `01_visual_encoder/`)
- Each folder contains markdown files numbered for reading order

### File Naming Convention
- Format: `XX_Topic_Name.md` where XX is a two-digit number (00, 01, 02, ...)
- `00_*.md` is always the index/overview file for the folder
- Use underscores between words: `01_Diffusion_Basics.md`
- Keep acronyms uppercase: `VAE`, `ViT`, `DiT`, `VLM`, `CNN`
- Examples:
  - `00_Diffusion.md` (index file)
  - `01_Diffusion_Basics.md`
  - `02_Sampling.md`
  - `03_Latent_Diffusion.md`

### Index File Format (00_*.md)
Each folder must have an index file with this structure:

```markdown
# Topic Title

> Parent: [Parent Topic](../00_Parent.md)

## Overview

Brief description of what this topic covers.

## Topics

1. **Subtopic 1** - Brief description
2. **Subtopic 2** - Brief description
3. **Subtopic 3** - Brief description
```

### Content File Format (01_*.md, 02_*.md, etc.)
```markdown
# Topic Title

> Parent: [Parent Index](00_Index.md)

## Overview

Introduction to the topic.

## Section 1

Content...

## Section 2

Content...

## Related

- [Related Topic 1](01_Related.md) - Description
- [Related Topic 2](02_Another.md) - Description
```

### Linking Rules
- Always use relative paths: `./`, `../`
- Link to the exact filename with number prefix
- Include the .md extension
- Examples:
  - Same folder: `[Topic](01_Topic.md)`
  - Parent folder: `[Parent](../00_Parent.md)`
  - Sibling folder: `[Other](../other_folder/00_Other.md)`
  - Cross-KB: `[AI_Infra](../../AI_Infra/inference/05_Frameworks.md)`

### Content Guidelines
- Use clear, concise language
- Include code examples where relevant (PyTorch/Python)
- Use ASCII diagrams for architecture/flow visualization
- Add practical examples and use cases
- Structure content from fundamentals to advanced
- Cross-reference related topics in other KBs (LLM, AI_Infra)
- **Math & Equations**: Always use LaTeX math notation, never code blocks for equations
  - Inline math: `$x = y + z$` renders as math, not code
  - Display math: use `$$` blocks for standalone equations
  - Example — write `$L = -\sum_{t} \log P(x_t | x_{<t})$` instead of putting equations in ``` code fences
  - Use `\text{}` for words inside math: `$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{KL}}$`
  - Common patterns:
    - Diffusion loss: `$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$`
    - Attention: `$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$`
```

---

## Example User Prompts

### Creating a New Topic

```
Create a knowledge base file `visual_encoder/05_Feature_Pyramid.md` covering:
- Feature Pyramid Networks (FPN)
- Multi-scale feature extraction
- Use in object detection and segmentation
- Connection to ViT multi-scale variants

Follow the knowledge base conventions. Parent index is visual_encoder/00_Visual_Encoder.md.
```

### Expanding an Existing Topic

```
Expand `diffusion/04_DiT.md` with:
- Detailed comparison of adaLN-Zero vs cross-attention conditioning
- Code example for a DiT block in PyTorch
- Performance benchmarks across model sizes

Follow the knowledge base conventions and maintain existing links.
```

---

## Quick Reference

| Element | Convention | Example |
|---------|------------|---------|
| Folder | lowercase_underscores | `visual_encoder/` |
| Index file | 00_FolderName.md | `00_Visual_Encoder.md` |
| Content file | XX_Topic_Name.md | `01_CNN_Basics.md` |
| Acronyms | UPPERCASE | `VAE`, `ViT`, `DiT` |
| Links | relative with .md | `[Link](../00_Parent.md)` |

---

## Full Structure

```
Multimodal/
├── 00_Multimodal.md
├── PROMPT.md
├── README.md
├── visual_encoder/
│   ├── 00_Visual_Encoder.md
│   ├── 01_CNN_Basics.md
│   ├── 02_ViT.md
│   ├── 03_Semantic_Encoders.md
│   └── 04_VAE.md
├── diffusion/
│   ├── 00_Diffusion.md
│   ├── 01_Diffusion_Basics.md
│   ├── 02_Sampling.md
│   ├── 03_Latent_Diffusion.md
│   └── 04_DiT.md
├── vision_language/
│   ├── 00_Vision_Language.md
│   ├── 01_Architecture.md
│   ├── 02_Models.md
│   ├── 03_Unified_Models.md
│   └── 04_Training.md
└── applications/
    ├── 00_Applications.md
    ├── 01_Image_Generation.md
    ├── 02_Video.md
    └── 03_Robotics.md
```
