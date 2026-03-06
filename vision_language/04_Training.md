# Multimodal Training

> Parent: [Vision-Language Models](00_Vision_Language.md)

## Overview

Training VLMs involves multiple stages: pretraining the visual encoder, aligning visual and language representations, and instruction tuning for task performance. The training recipe and data curation are often more important than architectural choices.

## Training Stages

```
Stage 0: Pretrained Components (frozen, from prior work)
  ├── Visual encoder: CLIP/SigLIP (pretrained on image-text pairs)
  └── LLM: Llama/Qwen/Gemma (pretrained on text)
         │
Stage 1: Alignment Pretraining
  ├── Train: projector only (encoder + LLM frozen)
  ├── Data: image-caption pairs (558K-1M+)
  └── Goal: map visual features into LLM embedding space
         │
Stage 2: Visual Instruction Tuning
  ├── Train: projector + LLM (encoder frozen)
  ├── Data: instruction-following data (665K-1M+)
  └── Goal: follow visual instructions accurately
         │
(Optional) Stage 3: Preference Optimization
  ├── Method: DPO/RLHF on visual tasks
  └── Goal: reduce hallucination, improve quality
```

## Stage 1: Alignment Pretraining

### What It Does

Teaches the projector to translate visual encoder outputs into the LLM's "language":

```
Before alignment:
  Visual features → random projector → LLM sees garbage
  LLM cannot interpret visual tokens

After alignment:
  Visual features → trained projector → LLM sees meaningful tokens
  LLM can describe images based on visual features
```

### Data

Image-caption pairs from datasets like:
- **LAION**: large-scale web-scraped image-text pairs
- **CC3M/CC12M**: Conceptual Captions
- **ShareGPT4V**: GPT-4V generated detailed captions
- **ALLaVA**: synthetic high-quality captions

### Training Details

```python
# Pseudocode for alignment pretraining
for image, caption in alignment_data:
    visual_features = frozen_encoder(image)
    visual_tokens = projector(visual_features)  # ← trainable

    input_tokens = concat(visual_tokens, tokenize(caption))
    loss = cross_entropy(frozen_llm(input_tokens))  # next-token prediction

    loss.backward()  # gradients only flow through projector
    optimizer.step()
```

Typical: 1 epoch, ~hours on 8 GPUs

## Stage 2: Visual Instruction Tuning

### What It Does

Fine-tunes the LLM to follow visual instructions — answer questions, describe, reason about images:

```
Input:  [image tokens] + "What is the dog doing in this image?"
Target: "The dog is playing fetch with a red ball in a park."
```

### Data Types

| Type | Example | Source |
|------|---------|--------|
| VQA | "What color is the car?" → "Red" | VQAv2, GQA |
| Conversation | Multi-turn dialogue about an image | GPT-4V generated |
| Reasoning | "Which object is heavier?" | Custom/synthetic |
| OCR/Document | "What does the sign say?" | TextVQA, DocVQA |
| Grounding | "Where is the cat?" → [bbox] | RefCOCO |
| Description | "Describe this image in detail" | ShareGPT4V |

### LLaVA-Instruct Data Pipeline

```
1. Start with COCO/web images
2. Generate diverse questions using GPT-4 (with image captions + bboxes as context)
3. Generate answers using GPT-4V (directly on images)
4. Filter for quality and accuracy
5. Mix data types: 40% conversation, 30% reasoning, 20% VQA, 10% description
```

### Training Details

- Train both projector and LLM (or LLM with LoRA for efficiency)
- Keep visual encoder frozen (already well-trained)
- Learning rate: lower than LLM pretraining (1e-5 to 2e-5)
- Typical: 1 epoch, ~1 day on 8-64 GPUs

## Data Curation Insights

Key findings from VLM research:

1. **Data quality > quantity**: 665K well-curated samples (LLaVA 1.5) beats millions of noisy samples
2. **Diverse task mixing**: include multiple task types to avoid forgetting
3. **Synthetic data works**: GPT-4V generated data is highly effective
4. **Resolution matters**: high-res images improve OCR and detail understanding
5. **Text data mixing**: include pure text data to prevent language capability degradation

## Training for Unified Models

Unified models (generation + understanding) require additional training:

### Additional Training Stages

```
Stage 2b: Image Generation Training
  ├── Data: text-image pairs (generation direction)
  ├── Method: next-token prediction (AR) or diffusion loss
  └── Challenge: balance understanding and generation quality

Stage 2c: Interleaved Training
  ├── Data: documents with interleaved images and text
  ├── Method: predict both text tokens and image tokens
  └── Key: enables natural multimodal conversation
```

### Balancing Understanding and Generation

```
Loss = λ_text * L_text + λ_understand * L_understand + λ_generate * L_generate

If λ_generate too high → understanding degrades
If λ_generate too low  → poor generation quality

Common solution: curriculum learning
  Phase 1: Understanding only
  Phase 2: Add generation (gradually increase λ_generate)
  Phase 3: Joint optimization
```

## Efficient Training Techniques

| Technique | What It Does | When to Use |
|-----------|-------------|-------------|
| LoRA | Low-rank adapter for LLM | Limited GPU memory |
| Frozen encoder | Don't train ViT | Most cases (it's already good) |
| Gradient checkpointing | Trade compute for memory | Large models |
| DeepSpeed ZeRO | Shard optimizer states | Multi-GPU training |
| Mixed precision (BF16) | Reduce memory, speed up | Always |
| Dynamic resolution | Variable tile count | Diverse image sizes |

## Evaluation Benchmarks

| Benchmark | Task | Key Metric |
|-----------|------|------------|
| MMBench | General VLM capability | Accuracy |
| MMMU | Multi-discipline understanding | Accuracy |
| MathVista | Mathematical reasoning with visuals | Accuracy |
| DocVQA | Document understanding | ANLS |
| TextVQA | OCR and text reading | Accuracy |
| POPE | Hallucination evaluation | F1 |
| RealWorldQA | Real-world visual reasoning | Accuracy |

## Related

- [Architecture](01_Architecture.md) — architectures being trained
- [Models](02_Models.md) — specific models and their training recipes
- [Semantic Encoders](../visual_encoder/03_Semantic_Encoders.md) — pretrained encoders used as backbones
- [AI_Infra: Distributed Training](../../AI_Infra/distributed/00_Distributed.md) — training infrastructure
