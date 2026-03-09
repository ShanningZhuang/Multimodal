# Hands-On Learning Scripts

Run these scripts in order to learn how visual encoders and VLMs work.

## Setup

```bash
cd /root/build-your-knowledge/Multimodal
uv sync
```

This installs all dependencies (torch, transformers, pillow, numpy) from `pyproject.toml`.

## Scripts

| # | Script | What You Learn | VRAM |
|---|--------|---------------|------|
| 1 | `01_clip_encoder.py` | Patch embedding, position encoding, transformer layers, [CLS] token | ~1GB |
| 2 | `02_llava_pipeline.py` | Full VLM pipeline: CLIP → projector → merge → LLM → output | ~14GB |
| 3 | `03_compare_encoders.py` | CLIP vs SigLIP: architecture, features, text-image alignment | ~3GB |
| 4 | `04_projector_and_merge.py` | MLP projector, placeholder tokens, masked_scatter merge | ~1GB |

## How to Run

```bash
cd /root/build-your-knowledge/Multimodal
uv run python scripts/01_clip_encoder.py
uv run python scripts/02_llava_pipeline.py
uv run python scripts/03_compare_encoders.py
uv run python scripts/04_projector_and_merge.py
```

## Interactive Exploration

For deeper exploration, run interactively:

```bash
cd /root/build-your-knowledge/Multimodal
uv run ipython -i scripts/01_clip_encoder.py
# After script finishes, you're in a REPL with all variables loaded
# Try: vision_model.encoder.layers[0]  — inspect a transformer layer
# Try: patch_embeds.shape  — check intermediate tensors
```

## Recommended Learning Path

1. Start with **04** (projector_and_merge) — smallest, no model download, shows the core fusion concept
2. Then **01** (CLIP encoder) — understand visual tokenization
3. Then **03** (compare encoders) — see CLIP vs SigLIP differences
4. Finally **02** (LLaVA pipeline) — full end-to-end VLM with real generation
