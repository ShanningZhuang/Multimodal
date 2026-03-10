"""
Lab 5: DiT Architecture — How Diffusion Transformers Denoise

This script traces every step of the original DiT-XL/2 architecture:
  Noisy latent → Patchify → Pos embed → Timestep+Class conditioning
  → DiT block (adaLN-Zero) → Full forward → Unpatchify → Predicted noise

Run: uv run python scripts/05_dit_architecture.py

Concepts covered:
  - Patchify: Conv2d turns latent into a sequence of patch tokens
  - Positional embeddings for spatial layout
  - Timestep + class conditioning → adaLN-Zero
  - DiT block internals: adaLN → attention → adaLN → FFN → gating
  - Unpatchify: linear head reshapes tokens back to spatial latent

Reference docs:
  - diffusion/04_DiT.md (DiT architecture, adaLN-Zero math)
  - diffusion/01_Diffusion_Basics.md (DDPM training objective)
"""

import torch

# ============================================================
# 1. Load DiT-XL/2 (675M params, class-conditional ImageNet)
# ============================================================
print("=" * 70)
print("Loading DiT-XL/2-256 — the original Diffusion Transformer")
print("=" * 70)

from diffusers import DiTPipeline

model_id = "facebook/DiT-XL-2-256"
pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

dit = pipe.transformer
dit.eval()

config = dit.config
hidden_size = config.num_attention_heads * config.attention_head_dim

print(f"\nModel: {model_id}")
print(f"  Parameters: {sum(p.numel() for p in dit.parameters()):,}")
print(f"  Hidden dim: {hidden_size} ({config.num_attention_heads} heads × {config.attention_head_dim} head_dim)")
print(f"  Num layers: {config.num_layers}")
print(f"  Num heads:  {config.num_attention_heads}")
print(f"  Patch size: {config.patch_size}")
print(f"  Sample size: {config.sample_size} (latent spatial dim)")
print(f"  In channels: {config.in_channels}")
print(f"  Num classes: {config.num_embeds_ada_norm}")

# ============================================================
# 2. Prepare inputs — noisy latent + timestep + class label
# ============================================================
print(f"\n{'=' * 70}")
print("Step 1: Prepare Inputs")
print("=" * 70)

device = torch.device("cuda")
latent_size = config.sample_size  # 32 for 256px generation
in_channels = config.in_channels  # 4 (VAE latent channels)

# Simulate a noisy latent (as if from the VAE encoder + noise)
noisy_latent = torch.randn(1, in_channels, latent_size, latent_size, device=device)
print(f"\n  Noisy latent z_t: {noisy_latent.shape}")
print(f"  → (batch=1, channels={in_channels}, H={latent_size}, W={latent_size})")

# Timestep (which noise level we're at)
timestep = torch.tensor([500], device=device)  # middle of the schedule
print(f"  Timestep: {timestep.item()} (out of 1000 — medium noise)")

# Class label (ImageNet class 207 = golden retriever)
class_label = torch.tensor([207], device=device)
print(f"  Class label: {class_label.item()} (ImageNet: golden retriever)")

# ============================================================
# 3. Trace Patchify
# ============================================================
print(f"\n{'=' * 70}")
print("Step 2: Patchify (latent → sequence of patch tokens)")
print("=" * 70)

patch_size = config.patch_size

# The patchify layer
pos_embed_layer = dit.pos_embed  # PatchEmbed module
patch_conv = pos_embed_layer.proj  # The Conv2d inside PatchEmbed
print(f"\n  Patchify conv: Conv2d({in_channels}, {hidden_size}, "
      f"kernel={patch_size}, stride={patch_size})")

with torch.no_grad():
    # Manual patchify
    patch_embeds = patch_conv(noisy_latent)
    print(f"\n  Input:  {noisy_latent.shape}  (batch, C, H, W)")
    print(f"  After Conv2d: {patch_embeds.shape}  (batch, hidden, H/p, W/p)")

    grid_h, grid_w = patch_embeds.shape[2], patch_embeds.shape[3]
    num_patches = grid_h * grid_w
    print(f"\n  Grid: {grid_h} × {grid_w} = {num_patches} patches")
    print(f"  Each patch covers {patch_size}×{patch_size} = {patch_size**2} latent pixels")
    print(f"  Token count formula: N = (H/p) × (W/p) = "
          f"({latent_size}/{patch_size}) × ({latent_size}/{patch_size}) = {num_patches}")

    # Flatten to sequence via the full PatchEmbed forward
    patch_tokens = pos_embed_layer(noisy_latent)
    print(f"\n  After PatchEmbed (flatten + pos embed): {patch_tokens.shape}")
    print(f"  → {patch_tokens.shape[1]} tokens, each {patch_tokens.shape[2]}-dim")

# ============================================================
# 4. Trace Timestep + Class Conditioning
# ============================================================
print(f"\n{'=' * 70}")
print("Step 3: Timestep + Class Conditioning")
print("=" * 70)

with torch.no_grad():
    # The conditioning is inside AdaLayerNormZero → CombinedTimestepLabelEmbeddings
    ada_norm = dit.transformer_blocks[0].norm1
    combined_emb = ada_norm.emb  # CombinedTimestepLabelEmbeddings

    print(f"\n  Conditioning pipeline (inside AdaLayerNormZero):")
    print(f"    CombinedTimestepLabelEmbeddings:")
    for name, child in combined_emb.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"      {name}: {child.__class__.__name__} ({params:,} params)")

    print(f"\n  Flow:")
    print(f"    1. Timestep t → sinusoidal embedding → MLP → t_emb ({hidden_size}d)")
    print(f"    2. Class label y → Embedding table ({config.num_embeds_ada_norm}+1 classes) → class_emb ({hidden_size}d)")
    print(f"    3. c = t_emb + class_emb ({hidden_size}d)")
    print(f"    4. c → SiLU → Linear({hidden_size}, {6*hidden_size}) → 6 modulation vectors")

# ============================================================
# 5. Trace a Single DiT Block (adaLN-Zero)
# ============================================================
print(f"\n{'=' * 70}")
print("Step 4: Single DiT Block — adaLN-Zero Internals")
print("=" * 70)

block = dit.transformer_blocks[0]
print(f"\n  DiT Block 0 components:")
for name, child in block.named_children():
    params = sum(p.numel() for p in child.parameters())
    print(f"    {name}: {child.__class__.__name__} ({params:,} params)")

# The adaLN linear layer
ada_linear = block.norm1.linear
print(f"\n  adaLN-Zero modulation layer:")
print(f"    Linear({hidden_size} → {ada_linear.out_features}) = 6 × {hidden_size}")
print(f"    Produces (γ1, β1, α1, γ2, β2, α2) from conditioning c")

print(f"\n  Forward pass through block:")
print(f"    1. Compute c = t_emb + class_emb")
print(f"    2. (γ1, β1, α1, γ2, β2, α2) = Linear(SiLU(c))")
print(f"    3. h = x + α1 ⊙ Attn(γ1 ⊙ LN(x) + β1)")
print(f"    4. out = h + α2 ⊙ FFN(γ2 ⊙ LN(h) + β2)")
print(f"    5. α is initialized to zero → block starts as identity")

# ============================================================
# 6. Full Forward Pass
# ============================================================
print(f"\n{'=' * 70}")
print("Step 5: Full Forward Pass")
print("=" * 70)

with torch.no_grad():
    output = dit(
        noisy_latent,
        timestep=timestep,
        class_labels=class_label,
    )
    predicted = output.sample
    print(f"\n  Input noisy latent:  {noisy_latent.shape}")
    print(f"  Predicted output:    {predicted.shape}")
    print(f"  → Same spatial shape as input! (after unpatchify)")

    print(f"\n  Data flow:")
    print(f"    Noisy latent ({latent_size}×{latent_size}×{in_channels})")
    print(f"    → Patchify: {num_patches} tokens × {hidden_size}d")
    print(f"    → + Position embeddings")
    print(f"    → {config.num_layers}× DiT blocks (each conditioned on t+class)")
    print(f"    → Layer norm + Linear heads")
    print(f"    → Unpatchify: {latent_size}×{latent_size}×{config.out_channels}")

    # Statistics
    print(f"\n  Output statistics:")
    print(f"    Mean: {predicted.mean().item():.4f}")
    print(f"    Std:  {predicted.std().item():.4f}")
    print(f"    Min:  {predicted.min().item():.4f}")
    print(f"    Max:  {predicted.max().item():.4f}")

# ============================================================
# 7. Unpatchify Details
# ============================================================
print(f"\n{'=' * 70}")
print("Step 6: Unpatchify — Tokens Back to Spatial Latent")
print("=" * 70)

print(f"\n  DiT uses two output projection layers:")
print(f"    proj_out_1: Linear({hidden_size} → {dit.proj_out_1.out_features})")
print(f"      = {hidden_size} → 2 × {hidden_size} (expand for gating)")
print(f"    proj_out_2: Linear({hidden_size} → {dit.proj_out_2.out_features})")
print(f"      = {hidden_size} → p² × out_channels = {patch_size**2} × {config.out_channels} = {dit.proj_out_2.out_features}")
print(f"\n  Unpatchify reshapes the per-token predictions back to spatial:")
print(f"    ({num_patches}, {dit.proj_out_2.out_features})")
print(f"    → ({grid_h}, {grid_w}, {patch_size}, {patch_size}, {config.out_channels})")
print(f"    → ({grid_h * patch_size}, {grid_w * patch_size}, {config.out_channels})")
print(f"    = ({latent_size}, {latent_size}, {config.out_channels})")
print(f"\n  out_channels={config.out_channels} = 2 × in_channels={in_channels}")
print(f"  → predicts both noise AND variance (for the learned variance loss)")

# ============================================================
# 8. Compare: Different Classes, Same Noise
# ============================================================
print(f"\n{'=' * 70}")
print("Step 7: Effect of Class Conditioning")
print("=" * 70)

with torch.no_grad():
    classes_to_test = {
        207: "golden retriever",
        980: "volcano",
        88: "macaw",
    }

    noise_for_all = noisy_latent  # same noise input

    print(f"\n  Same noise, different class labels → different predicted noise:")
    predictions = {}
    for cls_id, cls_name in classes_to_test.items():
        cls_tensor = torch.tensor([cls_id], device=device)
        out = dit(noise_for_all, timestep=timestep, class_labels=cls_tensor).sample
        predictions[cls_id] = out
        print(f"    Class {cls_id:>4} ({cls_name:>17}): mean={out.mean():.4f}, std={out.std():.4f}")

    # Compare predictions
    for (id1, name1), (id2, name2) in [
        ((207, "golden retriever"), (980, "volcano")),
        ((207, "golden retriever"), (88, "macaw")),
    ]:
        diff = (predictions[id1] - predictions[id2]).abs().mean()
        print(f"\n    Mean abs difference ({name1} vs {name2}): {diff:.4f}")

    print(f"\n  The class label changes the adaLN-Zero parameters in every block,")
    print(f"  which steers the denoising toward generating that specific class.")

# ============================================================
# 9. Summary
# ============================================================
print(f"\n{'=' * 70}")
print("Summary: DiT Architecture")
print("=" * 70)
print(f"""
  DiT-XL/2-256 ({sum(p.numel() for p in dit.parameters()):,} parameters)

  Architecture:
    1. Patchify: Conv2d({in_channels}→{hidden_size}, k={patch_size}, s={patch_size})
       → {num_patches} patch tokens
    2. + Positional embeddings (learnable, {num_patches}×{hidden_size})
    3. Conditioning: sinusoidal_embed(t) + class_embed(y) → c
    4. {config.num_layers}× DiT blocks, each:
       - adaLN-Zero: c → (γ1, β1, α1, γ2, β2, α2)
       - Self-Attention with adaptive norm
       - FFN with adaptive norm + zero-init gating
    5. Unpatchify: Linear({hidden_size}→{dit.proj_out_2.out_features}) + reshape

  Key insight: adaLN-Zero is a lightweight alternative to cross-attention
  for injecting conditioning. The α gating starts at zero, making each
  block an identity function at initialization → stable deep training.

  For text-conditioned generation (SD3, FLUX), cross-attention or joint
  attention replaces adaLN-Zero because text is a sequence, not a scalar.
""")
