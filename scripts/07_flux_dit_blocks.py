"""
Lab 7: FLUX.1 DiT Blocks — Dual-Stream vs Single-Stream Anatomy

This script deep-dives into FLUX.1-dev's hybrid transformer architecture:
  - Dual-stream blocks: separate QKV projections, joint attention, separate FFNs
  - Single-stream blocks: unified sequence with shared QKV and FFN
  - 2D RoPE for spatial position encoding
  - Attention matrix structure and shapes

Run: uv run python scripts/07_flux_dit_blocks.py

Concepts covered:
  - MMDiT-style dual-stream block (image + text streams)
  - Joint attention: concatenate Q/K/V from both streams
  - Single-stream block: all tokens in one unified sequence
  - RoPE extended to 2D grid positions
  - How image and text tokens interact in the attention matrix

Reference docs:
  - diffusion/04_DiT.md (MMDiT, FLUX.1 architecture)
  - diffusion/03_Latent_Diffusion.md (text encoders, conditioning)
"""

import torch
from diffusers import FluxTransformer2DModel

# ============================================================
# 1. Load FLUX.1-dev Transformer
# ============================================================
print("=" * 70)
print("Loading FLUX.1-dev Transformer (12B DiT)")
print("=" * 70)

model_id = "black-forest-labs/FLUX.1-dev"
transformer = FluxTransformer2DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
transformer = transformer.to("cuda")
transformer.eval()

total_params = sum(p.numel() for p in transformer.parameters())
print(f"\nModel: {model_id}")
print(f"  Total parameters: {total_params:,} ({total_params/1e9:.1f}B)")

# Print config
config = transformer.config
print(f"\n  Architecture config:")
print(f"    Hidden dim (joint):   {config.joint_attention_dim}")
print(f"    Num attention heads:  {config.num_attention_heads}")
print(f"    Head dim:             {config.joint_attention_dim // config.num_attention_heads}")
print(f"    Dual-stream blocks:   {config.num_layers}")
print(f"    Single-stream blocks: {config.num_single_layers}")
print(f"    In channels:          {config.in_channels}")

# ============================================================
# 2. Architecture Overview
# ============================================================
print(f"\n{'=' * 70}")
print("Step 1: Architecture Overview — Dual + Single Stream")
print("=" * 70)

print(f"""
  FLUX.1-dev hybrid architecture:

  ┌─────────────────────────────────────────────────────┐
  │  {config.num_layers}× Dual-Stream Blocks (MMDiT-style)               │
  │    Image tokens ←──joint attention──→ Text tokens   │
  │    Separate QKV projections, separate FFNs          │
  ├─────────────────────────────────────────────────────┤
  │  Concatenate image + text → unified sequence        │
  ├─────────────────────────────────────────────────────┤
  │  {config.num_single_layers}× Single-Stream Blocks (standard DiT)             │
  │    All tokens in one sequence, shared QKV + FFN     │
  └─────────────────────────────────────────────────────┘
""")

# Count parameters per section
dual_params = 0
single_params = 0
other_params = 0

for name, param in transformer.named_parameters():
    if "transformer_blocks." in name:
        dual_params += param.numel()
    elif "single_transformer_blocks." in name:
        single_params += param.numel()
    else:
        other_params += param.numel()

print(f"  Parameter distribution:")
print(f"    Dual-stream blocks:   {dual_params:>14,} ({dual_params/total_params*100:.1f}%)")
print(f"    Single-stream blocks: {single_params:>14,} ({single_params/total_params*100:.1f}%)")
print(f"    Other (embed, norm):  {other_params:>14,} ({other_params/total_params*100:.1f}%)")

# ============================================================
# 3. Dual-Stream Block Anatomy
# ============================================================
print(f"\n{'=' * 70}")
print("Step 2: Dual-Stream Block Anatomy (MMDiT-style)")
print("=" * 70)

dual_block = transformer.transformer_blocks[0]
print(f"\n  Block 0 components:")
for name, child in dual_block.named_children():
    child_params = sum(p.numel() for p in child.parameters())
    print(f"    {name}: {child.__class__.__name__} ({child_params:,} params)")

# Detailed breakdown of attention
print(f"\n  Joint Attention internals:")
if hasattr(dual_block, 'attn'):
    attn = dual_block.attn
    for name, child in attn.named_children():
        child_params = sum(p.numel() for p in child.parameters())
        print(f"    attn.{name}: {child.__class__.__name__} ({child_params:,} params)")

hidden_dim = config.joint_attention_dim
num_heads = config.num_attention_heads
head_dim = hidden_dim // num_heads

print(f"\n  Dual-stream attention flow:")
print(f"    Image tokens x ({hidden_dim}d) → separate Linear → Qx, Kx, Vx")
print(f"    Text tokens y  ({hidden_dim}d) → separate Linear → Qy, Ky, Vy")
print(f"    Joint: Q=[Qx;Qy], K=[Kx;Ky], V=[Vx;Vy]")
print(f"    Attention = softmax(QK^T / √{head_dim}) V")
print(f"    Split output → image_out, text_out")
print(f"    Image → Image FFN (separate weights)")
print(f"    Text  → Text FFN  (separate weights)")

# ============================================================
# 4. Single-Stream Block Anatomy
# ============================================================
print(f"\n{'=' * 70}")
print("Step 3: Single-Stream Block Anatomy")
print("=" * 70)

single_block = transformer.single_transformer_blocks[0]
print(f"\n  Single block 0 components:")
for name, child in single_block.named_children():
    child_params = sum(p.numel() for p in child.parameters())
    print(f"    {name}: {child.__class__.__name__} ({child_params:,} params)")

print(f"\n  Single-stream flow:")
print(f"    All tokens (image + text concatenated) → unified QKV")
print(f"    Single attention over entire sequence")
print(f"    Single FFN applied to all tokens")
print(f"\n  Why single-stream in later layers?")
print(f"    After {config.num_layers} dual-stream blocks, image and text representations")
print(f"    are already well-aligned. Separate streams are no longer needed,")
print(f"    and a unified stream is more parameter-efficient.")

# ============================================================
# 5. Compare Parameter Counts
# ============================================================
print(f"\n{'=' * 70}")
print("Step 4: Per-Block Parameter Comparison")
print("=" * 70)

dual_block_params = sum(p.numel() for p in dual_block.parameters())
single_block_params = sum(p.numel() for p in single_block.parameters())

print(f"\n  Single dual-stream block:   {dual_block_params:>12,} params")
print(f"  Single single-stream block: {single_block_params:>12,} params")
print(f"  Ratio: dual/single = {dual_block_params/single_block_params:.2f}×")
print(f"\n  Dual blocks have ~2× params because they maintain separate")
print(f"  QKV projections and FFNs for image and text streams.")

# ============================================================
# 6. Attention Matrix Structure
# ============================================================
print(f"\n{'=' * 70}")
print("Step 5: Attention Matrix Structure")
print("=" * 70)

# Simulate token counts for a 1024x1024 image
img_h, img_w = 1024, 1024
vae_scale = 8  # FLUX VAE downsampling
patch_size = 2  # FLUX patch size
latent_h = img_h // vae_scale
latent_w = img_w // vae_scale
num_img_tokens = (latent_h // patch_size) * (latent_w // patch_size)
num_txt_tokens = 512  # T5 max length

print(f"\n  For a {img_h}×{img_w} image:")
print(f"    VAE: {img_h}×{img_w} → {latent_h}×{latent_w} latent")
print(f"    Patchify (p={patch_size}): {latent_h//patch_size}×{latent_w//patch_size} = {num_img_tokens} image tokens")
print(f"    Text tokens: up to {num_txt_tokens}")

total_tokens = num_img_tokens + num_txt_tokens
print(f"\n  Dual-stream block attention matrix:")
print(f"    Q: [{num_img_tokens} img + {num_txt_tokens} txt] = {total_tokens} rows")
print(f"    K: [{num_img_tokens} img + {num_txt_tokens} txt] = {total_tokens} cols")
print(f"    Attention matrix: {total_tokens} × {total_tokens} = {total_tokens**2:,} entries")
print(f"    Per head, per batch")

print(f"""
  Attention pattern (dual-stream):
  ┌────────────────────┬────────────────┐
  │  img→img           │  img→txt        │
  │  (self-attention)  │  (cross-attn)   │   {num_img_tokens} img tokens
  ├────────────────────┼────────────────┤
  │  txt→img           │  txt→txt        │
  │  (cross-attn)      │  (self-attention)│   {num_txt_tokens} txt tokens
  └────────────────────┴────────────────┘
       {num_img_tokens} img                {num_txt_tokens} txt

  All four quadrants are computed in a single attention operation!
  This is more expressive than separate self-attn + cross-attn.
""")

# ============================================================
# 7. 2D RoPE Position Encoding
# ============================================================
print(f"{'=' * 70}")
print("Step 6: 2D RoPE — Rotary Position Embeddings for Images")
print("=" * 70)

print(f"""
  Standard RoPE (LLMs): 1D sequence position
    - Token at position i gets rotation by angle i·θ
    - Encodes relative distance between tokens

  FLUX 2D RoPE: extend to spatial grid (row, col)
    - Split head dimension into two halves
    - First half:  RoPE with position = row
    - Second half: RoPE with position = col

  Example for a {latent_h//patch_size}×{latent_w//patch_size} grid:
    Token at (3, 5): first half rotated by 3·θ, second half by 5·θ
    Token at (3, 6): first half same (3·θ), second half by 6·θ
    → Attention between them captures that they are in the same row,
      1 column apart.

  Advantages over learned absolute position embeddings:
    - Relative position awareness (like in LLMs)
    - Better extrapolation to unseen resolutions
    - No need to interpolate when changing image size
""")

# ============================================================
# 8. Timestep Conditioning in FLUX
# ============================================================
print(f"{'=' * 70}")
print("Step 7: Timestep Conditioning")
print("=" * 70)

print(f"\n  FLUX uses flow matching with t ∈ [0, 1]:")
print(f"    t=0: clean data, t=1: pure noise")
print(f"\n  Conditioning pipeline:")
print(f"    1. Timestep t → sinusoidal embedding")
print(f"    2. + CLIP pooled text embedding (global text features)")
print(f"    3. MLP → conditioning vector c")
print(f"    4. c modulates every block via adaLN")
print(f"\n  This combines WHEN (noise level) with WHAT (text content)")
print(f"  in a single conditioning vector.")

# ============================================================
# 9. Summary
# ============================================================
print(f"\n{'=' * 70}")
print("Summary: FLUX.1-dev Architecture")
print("=" * 70)
print(f"""
  FLUX.1-dev ({total_params/1e9:.1f}B parameters)

  Hybrid architecture:
    {config.num_layers}× Dual-stream blocks (MMDiT-style):
      - Separate QKV projections for image and text
      - Joint attention: all tokens attend to all tokens
      - Separate FFNs per modality
      - {dual_block_params:,} params/block

    {config.num_single_layers}× Single-stream blocks:
      - Concatenated image+text as unified sequence
      - Shared QKV and FFN
      - {single_block_params:,} params/block

  Position encoding: 2D RoPE (row/col in split head dims)
  Text encoders: CLIP-L (pooled → conditioning) + T5-XXL (sequence → tokens)
  Training: Rectified flow matching (velocity prediction)

  Key design choices:
    1. Dual → Single transition: modalities align in early layers,
       then benefit from unified processing
    2. 2D RoPE: resolution-flexible, no learned position embeddings
    3. Flow matching: enables fewer sampling steps than DDPM
""")
