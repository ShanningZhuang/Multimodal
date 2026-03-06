# Vision-Language Models: Key Models

> Parent: [Vision-Language Models](00_Vision_Language.md)

## Overview

This page covers the most important understanding-focused VLMs — models that take image+text input and produce text output. These represent the current state of the art for visual question answering, image captioning, document understanding, and visual reasoning.

## LLaVA (Large Language and Vision Assistant)

The most influential open-source VLM, known for its simplicity:

### LLaVA Architecture

```
Image → CLIP ViT-L/14 → visual features → MLP Projector → ┐
                         (576 tokens)     (Linear+GELU+Linear) ├→ Vicuna/Llama → Answer
Text  → Tokenizer     → text tokens   ─────────────────────→ ┘
```

### Evolution

| Version | Visual Encoder | Projector | LLM | Resolution | Key Change |
|---------|---------------|-----------|-----|------------|------------|
| LLaVA 1.0 | CLIP ViT-L/14 | Linear | Vicuna 7B/13B | 224 | Proof of concept |
| LLaVA 1.5 | CLIP ViT-L/14 | MLP (2-layer) | Vicuna 7B/13B | 336 | Better projector, more data |
| LLaVA-NeXT | CLIP ViT-L/14 | MLP | Various | Dynamic tiles | High-res, dynamic resolution |
| LLaVA-OneVision | SigLIP | MLP | Qwen2 | Dynamic | Multi-image, video, scaling |

### LLaVA Training Recipe

A 2-stage approach that became standard:

```
Stage 1: Pretraining (Alignment)
  - Freeze: visual encoder + LLM
  - Train: projector only
  - Data: 558K image-caption pairs
  - Goal: align visual features to LLM embedding space

Stage 2: Instruction Tuning
  - Freeze: visual encoder
  - Train: projector + LLM (full finetune or LoRA)
  - Data: 665K visual instruction data (VQA, conversation, reasoning)
  - Goal: follow visual instructions
```

Key insight: simple architecture + good data curation > complex architecture

## Qwen-VL

Alibaba's VLM series, known for strong multilingual and document understanding:

### Qwen-VL 2 Architecture

```
Image → SigLIP ViT → Dynamic resolution tiling → ┐
                      (variable token count)       ├→ Qwen2 LLM → Answer
Text  → Tokenizer  → text tokens ─────────────→ ┘
```

Key features:
- **Dynamic resolution**: adapts tile count based on image size and content
- **Native multi-image**: handles interleaved image-text naturally
- **Strong OCR/document understanding**: trained on large-scale document data
- Available in 2B, 7B, 72B sizes

### Qwen-VL 2.5 Improvements
- Better video understanding
- Improved agentic capabilities (UI grounding, tool use)
- Structured output (JSON, bounding boxes)

## InternVL

Shanghai AI Lab's VLM, notable for scaling the visual encoder:

### InternVL 2 Architecture

```
Image → InternViT-6B → Dynamic tiling → MLP → ┐
        (6B param ViT!)  (448px tiles)          ├→ InternLM2/Llama → Answer
Text  → Tokenizer ──────────────────────────→ ┘
```

Key innovations:
- **Large vision encoder**: InternViT-6B (much larger than typical 300M ViT-L)
- **Progressive training**: pretrain ViT → align → instruction tune
- **Strong benchmark performance**: competitive with proprietary models

| Model | ViT Params | LLM Params | Total |
|-------|-----------|------------|-------|
| InternVL2-1B | 300M | 900M | 1.2B |
| InternVL2-8B | 300M | 7.7B | 8B |
| InternVL2-26B | 6B | 20B | 26B |
| InternVL2-76B | 6B | 70B | 76B |

## PaliGemma

Google's efficient VLM:
- SigLIP visual encoder (400M)
- Gemma LLM (2B)
- Designed for fine-tuning on specific tasks
- Strong transfer learning with compact model

## Model Comparison

| Model | Encoder | LLM | Projector | Strengths |
|-------|---------|-----|-----------|-----------|
| LLaVA-OneVision | SigLIP | Qwen2 | MLP | Simple, open, good baseline |
| Qwen-VL 2.5 | SigLIP | Qwen2.5 | Resampler | Multilingual, document, agent |
| InternVL 2 | InternViT-6B | Various | MLP | Large ViT, benchmark leader |
| PaliGemma | SigLIP | Gemma | Linear | Efficient, fine-tuning friendly |
| GPT-4V/o | Unknown | GPT-4 | Unknown | Best proprietary, reasoning |
| Claude 3.5 | Unknown | Claude | Unknown | Strong reasoning, safety |

## Common Capabilities

| Capability | Description | Example |
|-----------|-------------|---------|
| VQA | Answer questions about images | "What color is the car?" |
| OCR | Read text in images | Document/receipt understanding |
| Grounding | Locate objects (bounding boxes) | "Find the red ball" |
| Reasoning | Multi-step visual reasoning | "Which is heavier?" (needs physics) |
| Description | Describe image content | Detailed captioning |
| Multi-image | Compare/reason across images | "What changed between these?" |
| Video | Understand video content | "Summarize this clip" |

## Related

- [Architecture](01_Architecture.md) — design patterns used by these models
- [Unified Models](03_Unified_Models.md) — models that also generate images
- [Training](04_Training.md) — training recipes and data pipelines
- [Semantic Encoders](../visual_encoder/03_Semantic_Encoders.md) — CLIP/SigLIP used as visual backbone
