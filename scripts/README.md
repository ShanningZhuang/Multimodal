# Hands-On Learning Scripts

Run these scripts in order to learn how visual encoders, VLMs, and diffusion models work.

## Setup

```bash
cd /root/build-your-knowledge/Multimodal
uv sync
```

This installs all dependencies (torch, transformers, diffusers, pillow, numpy) from `pyproject.toml`.

## Scripts

### Visual Encoders & VLMs (01–04)

| # | Script | What You Learn | VRAM |
|---|--------|---------------|------|
| 1 | `01_clip_encoder.py` | Patch embedding, position encoding, transformer layers, [CLS] token | ~1GB |
| 2 | `02_llava_pipeline.py` | Full VLM pipeline: CLIP → projector → merge → LLM → output | ~14GB |
| 3 | `03_compare_encoders.py` | CLIP vs SigLIP: architecture, features, text-image alignment | ~3GB |
| 4 | `04_projector_and_merge.py` | MLP projector, placeholder tokens, masked_scatter merge | ~1GB |

### Diffusion & DiT (05–08)

| # | Script | What You Learn | VRAM |
|---|--------|---------------|------|
| 5 | `05_dit_architecture.py` | DiT patchify, adaLN-Zero conditioning, unpatchify, class conditioning | ~2GB |
| 6 | `06_latent_diffusion_pipeline.py` | Full FLUX.1-schnell pipeline: text encode → denoise → VAE decode → image | ~24GB |
| 7 | `07_flux_dit_blocks.py` | FLUX.1-dev dual-stream vs single-stream blocks, 2D RoPE, attention matrix | ~24GB |
| 8 | `08_denoising_loop.py` | Manual denoising loop, intermediate latent visualization, noise→image progression | ~24GB |

## How to Run

```bash
cd /root/build-your-knowledge/Multimodal

# Visual encoder & VLM scripts
uv run python scripts/01_clip_encoder.py
uv run python scripts/02_llava_pipeline.py
uv run python scripts/03_compare_encoders.py
uv run python scripts/04_projector_and_merge.py

# Diffusion & DiT scripts
uv run python scripts/05_dit_architecture.py
uv run python scripts/06_latent_diffusion_pipeline.py
uv run python scripts/07_flux_dit_blocks.py
uv run python scripts/08_denoising_loop.py
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

**VLM track** (visual understanding):
1. Start with **04** (projector_and_merge) — smallest, no model download, shows the core fusion concept
2. Then **01** (CLIP encoder) — understand visual tokenization
3. Then **03** (compare encoders) — see CLIP vs SigLIP differences
4. Finally **02** (LLaVA pipeline) — full end-to-end VLM with real generation

**Diffusion track** (image generation):
1. Start with **05** (DiT architecture) — lightweight, shows patchify + adaLN-Zero + unpatchify
2. Then **08** (denoising loop) — watch noise become an image step by step
3. Then **06** (latent diffusion pipeline) — full end-to-end FLUX.1-schnell pipeline
4. Finally **07** (FLUX DiT blocks) — deep dive into dual/single-stream architecture
