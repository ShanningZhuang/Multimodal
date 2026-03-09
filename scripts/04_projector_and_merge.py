"""
Lab 4: Projector & Embedding Merge — The Fusion Point

This script focuses on the CRITICAL step where vision meets language:
  1. How the MLP projector transforms visual features
  2. How placeholder tokens work in the input sequence
  3. How masked_scatter merges visual embeddings into text embeddings

Run: uv run python Multimodal/scripts/04_projector_and_merge.py

This is the smallest, most focused script — it uses a tiny model to make
the merge mechanism crystal clear without needing a large GPU.

Concepts covered:
  - MLP projector internals (weight shapes, activation)
  - Placeholder token mechanism (<image> → N token IDs)
  - masked_scatter_ operation (the actual fusion)
  - What the LLM "sees" after fusion

Reference docs:
  - vision_language/01_Architecture.md (Appendix: vLLM Code Reference)
"""

import torch
import torch.nn as nn

device = torch.device("cuda")

# ============================================================
# 1. Simulate the MLP Projector
# ============================================================
print("=" * 70)
print("Part 1: MLP Projector — Bridging Vision and Language Spaces")
print("=" * 70)

# Typical dimensions
vision_dim = 1024   # CLIP ViT-L output dim
text_dim = 4096     # Llama-7B hidden dim

# Build a projector (same architecture as LLaVA)
projector = nn.Sequential(
    nn.Linear(vision_dim, text_dim),   # vision_dim → text_dim
    nn.GELU(),                          # non-linear activation
    nn.Linear(text_dim, text_dim),     # text_dim → text_dim
).to(device)

# Count parameters
total_params = sum(p.numel() for p in projector.parameters())
print(f"\n  Architecture: Linear({vision_dim}→{text_dim}) + GELU + Linear({text_dim}→{text_dim})")
print(f"  Parameters: {total_params:,}")
print(f"\n  Weight shapes:")
for name, param in projector.named_parameters():
    print(f"    {name}: {param.shape}")

# Simulate visual features (576 tokens from CLIP)
num_visual_tokens = 576  # 24x24 grid
visual_features = torch.randn(1, num_visual_tokens, vision_dim, device=device)

print(f"\n  Input (from CLIP): {visual_features.shape}  → {num_visual_tokens} tokens x {vision_dim}d")

with torch.no_grad():
    projected = projector(visual_features)
    print(f"  Output (for LLM):  {projected.shape}  → {num_visual_tokens} tokens x {text_dim}d")
    print(f"\n  Same number of tokens, but now in the LLM's embedding space!")

# ============================================================
# 2. Placeholder Token Mechanism
# ============================================================
print(f"\n{'=' * 70}")
print("Part 2: Placeholder Tokens — How <image> Becomes 576 Positions")
print("=" * 70)

# Simulate tokenization
IMAGE_TOKEN_ID = 32000  # LLaVA's image token ID

# Original prompt: "USER: <image>\nWhat is this?\nASSISTANT:"
# After tokenization + placeholder expansion:
text_token_ids = [1, 3148, 1001, 29901]  # "USER:"
image_token_ids = [IMAGE_TOKEN_ID] * num_visual_tokens  # 576 placeholders
rest_token_ids = [13, 5618, 338, 445, 29973, 13, 22933, 9047, 13566, 29901]  # "\nWhat is this?\nASSISTANT:"

full_token_ids = text_token_ids + image_token_ids + rest_token_ids
input_ids = torch.tensor([full_token_ids], device=device)

print(f"\n  Prompt: 'USER: <image>\\nWhat is this?\\nASSISTANT:'")
print(f"\n  Tokenized sequence ({len(full_token_ids)} tokens):")
print(f"    [{0}..{len(text_token_ids)-1}] Text tokens (USER:)          = {len(text_token_ids)} tokens")
print(f"    [{len(text_token_ids)}..{len(text_token_ids)+num_visual_tokens-1}] Image placeholders (ID={IMAGE_TOKEN_ID}) = {num_visual_tokens} tokens")
print(f"    [{len(text_token_ids)+num_visual_tokens}..{len(full_token_ids)-1}] Text tokens (What is this?...) = {len(rest_token_ids)} tokens")
print(f"    Total: {len(full_token_ids)} tokens")

print(f"\n  Why 576 placeholders?")
print(f"    → CLIP ViT with patch_size=14 on 336x336 image → 24x24 = 576 patches")
print(f"    → Each placeholder will be REPLACED by a visual embedding")
print(f"    → This reserves the right number of KV cache slots in the LLM")

# ============================================================
# 3. The Merge Operation
# ============================================================
print(f"\n{'=' * 70}")
print("Part 3: Embedding Merge — masked_scatter_ in Action")
print("=" * 70)

# Create fake text embeddings (as if from LLM's embed_tokens)
seq_len = len(full_token_ids)
text_embeddings = torch.randn(1, seq_len, text_dim, device=device) * 0.1  # small random values
# Mark text positions distinctively
for i in range(len(text_token_ids)):
    text_embeddings[0, i, 0] = 1.0  # text tokens have 1.0 in first dim
for i in range(len(text_token_ids) + num_visual_tokens, seq_len):
    text_embeddings[0, i, 0] = 1.0

# Create the image token mask
is_image = (input_ids == IMAGE_TOKEN_ID)

print(f"\n  text_embeddings: {text_embeddings.shape}  (from LLM's embed_tokens layer)")
print(f"  projected_visual: {projected.shape}  (from MLP projector)")
print(f"  is_image mask: {is_image.shape}, sum={is_image.sum().item()} True positions")

# Show state before merge
print(f"\n  BEFORE merge:")
print(f"    Position 0 (text 'USER'):     embed[0][:4] = {text_embeddings[0, 0, :4].tolist()}")
print(f"    Position {len(text_token_ids)} (image placeholder): embed[0][:4] = {text_embeddings[0, len(text_token_ids), :4].tolist()}")
print(f"    Position {seq_len-1} (text ':'):       embed[0][:4] = {text_embeddings[0, -1, :4].tolist()}")

# Do the merge (this is what _merge_multimodal_embeddings does)
merged = text_embeddings.clone()

# Flatten projected features for scatter
visual_flat = projected.reshape(-1, text_dim)

# masked_scatter_: replace positions where is_image=True with visual features
merged.masked_scatter_(
    is_image.unsqueeze(-1).expand_as(merged),
    visual_flat,
)

print(f"\n  AFTER merge (masked_scatter_):")
print(f"    Position 0 (text 'USER'):     embed[0][:4] = {merged[0, 0, :4].tolist()}")
print(f"    Position {len(text_token_ids)} (NOW visual):     embed[0][:4] = {merged[0, len(text_token_ids), :4].tolist()}")
print(f"    Position {seq_len-1} (text ':'):       embed[0][:4] = {merged[0, -1, :4].tolist()}")

# Verify text positions are unchanged
text_unchanged = torch.allclose(
    text_embeddings[0, 0],
    merged[0, 0],
)
visual_changed = not torch.allclose(
    text_embeddings[0, len(text_token_ids)],
    merged[0, len(text_token_ids)],
)
print(f"\n  Text positions unchanged: {text_unchanged}")
print(f"  Image positions replaced:  {visual_changed}")

# ============================================================
# 4. What the LLM Sees
# ============================================================
print(f"\n{'=' * 70}")
print("Part 4: What the LLM Sees After Merge")
print("=" * 70)

print(f"""
  The LLM's self-attention now operates on this unified sequence:

  Position:  [0  1  2  3 | 4  5  6 ... 579 | 580 581 ... 589]
  Content:   [U  S  E  R | v1 v2 v3 ... v576| W   h   ...  : ]
  Type:       text tokens  visual tokens      text tokens

  Key insight: The LLM doesn't know these are "visual" embeddings.
  It just sees a sequence of vectors in its embedding space.
  The projector's job was to make the visual vectors "look like"
  plausible text embeddings so the LLM can attend to them naturally.

  This is LATE FUSION — vision and language only meet at the
  embedding level, not inside the transformer layers.
""")

# ============================================================
# 5. Projector Variants
# ============================================================
print(f"{'=' * 70}")
print("Part 5: Projector Variants")
print("=" * 70)

# Linear (LLaVA v1)
linear_proj = nn.Linear(vision_dim, text_dim).to(device)

# MLP (LLaVA v1.5) — what we built above
mlp_proj = projector

# Resampler (Qwen-VL, BLIP-2) — reduces token count
# Simplified version
class SimpleResampler(nn.Module):
    def __init__(self, vision_dim, text_dim, num_queries=64):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, text_dim))
        self.cross_attn = nn.MultiheadAttention(text_dim, num_heads=8, batch_first=True)
        self.proj = nn.Linear(vision_dim, text_dim)

    def forward(self, visual_features):
        # Project visual features to text dim first
        projected = self.proj(visual_features)
        # Cross-attention: queries attend to visual features
        queries = self.queries.unsqueeze(0).expand(visual_features.shape[0], -1, -1)
        output, _ = self.cross_attn(queries, projected, projected)
        return output

resampler = SimpleResampler(vision_dim, text_dim, num_queries=64).to(device)

with torch.no_grad():
    linear_out = linear_proj(visual_features)
    mlp_out = mlp_proj(visual_features)
    resampler_out = resampler(visual_features)

print(f"\n  Input: {visual_features.shape}  ({num_visual_tokens} tokens)")
print(f"\n  Linear projection (LLaVA v1):")
print(f"    Output: {linear_out.shape}  → same token count, just projected")
print(f"    Params: {sum(p.numel() for p in linear_proj.parameters()):,}")
print(f"\n  MLP projection (LLaVA v1.5):")
print(f"    Output: {mlp_out.shape}  → same token count, better quality")
print(f"    Params: {sum(p.numel() for p in mlp_proj.parameters()):,}")
print(f"\n  Resampler (Qwen-VL, BLIP-2):")
print(f"    Output: {resampler_out.shape}  → REDUCED to 64 tokens!")
print(f"    Params: {sum(p.numel() for p in resampler.parameters()):,}")
print(f"\n  Token compression: {num_visual_tokens} → 64 = {num_visual_tokens/64:.0f}x fewer tokens for the LLM")
print(f"  This dramatically reduces LLM compute but may lose spatial detail")

print(f"\n{'=' * 70}")
print("Summary")
print("=" * 70)
print(f"""
  The projector + merge is the SIMPLEST part of a VLM, but the most important:

  1. Projector: transforms visual features into the LLM's embedding space
     - Linear: fast, simple, used in LLaVA v1
     - MLP: better quality, used in LLaVA v1.5+
     - Resampler: compresses tokens, used in Qwen-VL, BLIP-2

  2. Placeholder: <image> → N token IDs reserves KV cache space

  3. masked_scatter_: replaces placeholder embeddings with projected visual features

  After this, the LLM sees a unified sequence and generates text normally.
  The "magic" is that the projector learns to make visual features look like
  meaningful tokens to the LLM.
""")
