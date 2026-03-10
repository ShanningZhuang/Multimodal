"""
Lab 6: Latent Diffusion Pipeline — End-to-End with FLUX.1-schnell

This script traces the full DiT-based latent diffusion pipeline:
  Text encoding (CLIP-L + T5) → Noise init → Step-by-step denoising
  → VAE decode → Save image

Uses FLUX.1-schnell (4-step distilled model) to show the entire flow
from text prompt to generated image, printing shapes at every stage.

Run: uv run python scripts/06_latent_diffusion_pipeline.py

Concepts covered:
  - Dual text encoding: CLIP-L for pooled features, T5-XXL for sequence
  - Noise initialization in latent space
  - Step-by-step denoising with a DiT (flow matching / Euler solver)
  - VAE decoding from latent to pixel space
  - Saving the final generated image

Reference docs:
  - diffusion/03_Latent_Diffusion.md (LDM architecture, text encoders)
  - diffusion/04_DiT.md (FLUX.1 architecture)
  - diffusion/01_Diffusion_Basics.md (flow matching)
"""

import os
import torch
from diffusers import FluxPipeline

# ============================================================
# 1. Load FLUX.1-schnell
# ============================================================
print("=" * 70)
print("Loading FLUX.1-schnell — 4-step distilled DiT")
print("=" * 70)

model_id = "black-forest-labs/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

print(f"\nModel: {model_id}")
print(f"\nPipeline components:")
for name, component in pipe.components.items():
    if component is not None and hasattr(component, 'parameters'):
        params = sum(p.numel() for p in component.parameters())
        print(f"  {name}: {component.__class__.__name__} ({params:,} params)")
    elif component is not None:
        print(f"  {name}: {component.__class__.__name__}")

# ============================================================
# 2. Text Encoding — CLIP-L + T5-XXL
# ============================================================
print(f"\n{'=' * 70}")
print("Step 1: Text Encoding (CLIP-L + T5-XXL)")
print("=" * 70)

prompt = "A golden retriever sitting in a field of sunflowers, oil painting style"
print(f"\n  Prompt: '{prompt}'")

# Tokenize with both tokenizers
tokenizer_clip = pipe.tokenizer
tokenizer_t5 = pipe.tokenizer_2

clip_tokens = tokenizer_clip(prompt, return_tensors="pt", padding="max_length",
                              max_length=77, truncation=True)
t5_tokens = tokenizer_t5(prompt, return_tensors="pt", padding="max_length",
                          max_length=512, truncation=True)

print(f"\n  CLIP-L tokenizer:")
print(f"    Token IDs shape: {clip_tokens['input_ids'].shape}")
print(f"    Max length: 77 tokens")

print(f"\n  T5-XXL tokenizer:")
print(f"    Token IDs shape: {t5_tokens['input_ids'].shape}")
print(f"    Max length: 512 tokens")

# Encode with both text encoders
with torch.no_grad():
    clip_input = {k: v.to("cuda") for k, v in clip_tokens.items()}
    clip_output = pipe.text_encoder(**clip_input)
    clip_pooled = clip_output.pooler_output  # Pooled CLS output

    t5_input = {k: v.to("cuda") for k, v in t5_tokens.items()}
    t5_output = pipe.text_encoder_2(**t5_input)
    t5_embeds = t5_output[0]  # Sequence of token embeddings

print(f"\n  CLIP-L output:")
print(f"    Pooled embedding: {clip_pooled.shape}")
print(f"    → Used for timestep conditioning (added to t_emb)")

print(f"\n  T5-XXL output:")
print(f"    Sequence embeddings: {t5_embeds.shape}")
print(f"    → Used as text tokens in joint/single-stream attention")

# ============================================================
# 3. Noise Initialization
# ============================================================
print(f"\n{'=' * 70}")
print("Step 2: Initialize Latent Noise")
print("=" * 70)

# FLUX uses 16-channel latent space
height, width = 1024, 1024
vae_scale_factor = pipe.vae_scale_factor
latent_h = height // vae_scale_factor
latent_w = width // vae_scale_factor
latent_channels = pipe.transformer.config.in_channels

print(f"\n  Target image size: {height}×{width}")
print(f"  VAE scale factor: {vae_scale_factor}× downsampling")
print(f"  Latent size: {latent_h}×{latent_w}×{latent_channels}")

# FLUX packs latents differently — the pipeline handles this
generator = torch.Generator(device="cuda").manual_seed(42)
latents = torch.randn(
    1, latent_channels, latent_h, latent_w,
    device="cuda", dtype=torch.bfloat16, generator=generator,
)
print(f"  Initial noise shape: {latents.shape}")
print(f"  Noise stats: mean={latents.mean():.4f}, std={latents.std():.4f}")

# ============================================================
# 4. Step-by-Step Denoising
# ============================================================
print(f"\n{'=' * 70}")
print("Step 3: Denoising Loop (4 Euler steps)")
print("=" * 70)

num_steps = 4
print(f"\n  FLUX.1-schnell is distilled for {num_steps}-step generation")
print(f"  No classifier-free guidance needed (guidance_scale=0.0)")
print(f"\n  Running pipeline with callback to trace each step...")

step_latents = []

def trace_callback(pipe, step_index, timestep, callback_kwargs):
    """Callback to inspect latents at each denoising step."""
    lat = callback_kwargs["latents"]
    step_latents.append(lat.detach().clone())
    print(f"\n    Step {step_index + 1}/{num_steps}:")
    print(f"      Timestep: {timestep}")
    print(f"      Latent shape: {lat.shape}")
    print(f"      Latent stats: mean={lat.mean():.4f}, std={lat.std():.4f}, "
          f"min={lat.min():.4f}, max={lat.max():.4f}")
    return callback_kwargs

with torch.no_grad():
    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=0.0,
        generator=torch.Generator(device="cuda").manual_seed(42),
        callback_on_step_end=trace_callback,
        output_type="pil",
    )

image = result.images[0]

# ============================================================
# 5. VAE Decode
# ============================================================
print(f"\n{'=' * 70}")
print("Step 4: VAE Decode (latent → pixels)")
print("=" * 70)

print(f"\n  VAE decoder converts latent back to pixel space:")
print(f"    Input: latent {latent_h}×{latent_w}×{latent_channels}")
print(f"    Output: image {height}×{width}×3")
print(f"\n  VAE parameters: {sum(p.numel() for p in pipe.vae.parameters()):,}")
print(f"\n  Generated image size: {image.size} (W×H)")

# ============================================================
# 6. Save Result
# ============================================================
print(f"\n{'=' * 70}")
print("Step 5: Save Generated Image")
print("=" * 70)

output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "06_flux_schnell_output.png")
image.save(output_path)
print(f"\n  Saved to: {output_path}")

# ============================================================
# 7. Summary
# ============================================================
print(f"\n{'=' * 70}")
print("Summary: FLUX.1-schnell Pipeline")
print("=" * 70)
print(f"""
  Prompt: '{prompt}'

  Pipeline stages:
    1. Text Encoding:
       - CLIP-L → pooled embedding ({clip_pooled.shape[-1]}d) → timestep conditioning
       - T5-XXL → sequence embeddings ({t5_embeds.shape[-1]}d) → text tokens for attention

    2. Noise Init: {latent_h}×{latent_w}×{latent_channels} random Gaussian latent

    3. Denoising: {num_steps} Euler steps (flow matching, no CFG)
       - DiT predicts velocity field at each timestep
       - Euler solver: x_{{t+dt}} = x_t + v * dt

    4. VAE Decode: latent → {height}×{width}×3 pixel image

  FLUX.1-schnell achieves high quality in just {num_steps} steps because:
    - Distilled from FLUX.1-dev (which needs 20-50 steps)
    - No CFG needed (guidance distilled into the model)
    - Flow matching provides straighter denoising trajectories
""")
