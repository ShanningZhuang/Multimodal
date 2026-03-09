"""
Lab 1: CLIP Vision Encoder — How Images Become Tokens

This script traces every step of the CLIP vision encoder:
  Raw pixels → Patch embedding → Position encoding → Transformer layers → Visual tokens

Run: uv run python Multimodal/scripts/01_clip_encoder.py

Concepts covered:
  - Patch embedding (Conv2d with kernel_size=stride=patch_size)
  - [CLS] token prepend
  - Positional embeddings
  - Transformer encoder layers
  - Feature selection (which layer's output to use)

Reference docs:
  - visual_encoder/02_ViT.md (ViT architecture)
  - visual_encoder/03_Semantic_Encoders.md (CLIP training objective)
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ============================================================
# 1. Load CLIP (lightweight — runs on CPU or any GPU)
# ============================================================
print("=" * 70)
print("Loading CLIP ViT-L/14 (336px) — the visual encoder used in LLaVA")
print("=" * 70)

model_id = "openai/clip-vit-large-patch14-336"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).cuda()
model.eval()
device = torch.device("cuda")

vision_model = model.vision_model
print(f"\nModel loaded: {model_id}")
print(f"  Image size: {vision_model.config.image_size}px")
print(f"  Patch size: {vision_model.config.patch_size}px")
print(f"  Hidden dim: {vision_model.config.hidden_size}")
print(f"  Num layers: {vision_model.config.num_hidden_layers}")
print(f"  Num heads:  {vision_model.config.num_attention_heads}")

# ============================================================
# 2. Prepare an image
# ============================================================
# Create a simple test image (red-to-blue gradient)
import numpy as np

img_array = np.zeros((400, 600, 3), dtype=np.uint8)
for x in range(600):
    img_array[:, x, 0] = int(255 * (1 - x / 600))  # red decreases
    img_array[:, x, 2] = int(255 * (x / 600))       # blue increases
image = Image.fromarray(img_array)

inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

print(f"\n{'=' * 70}")
print("Step 1: Image Preprocessing (done by CLIPProcessor)")
print("=" * 70)
print(f"  Original image size: {image.size} (W x H)")
print(f"  After preprocessing: {pixel_values.shape}")
print(f"  → (batch=1, channels=3, H={pixel_values.shape[2]}, W={pixel_values.shape[3]})")
print(f"  Resized to {pixel_values.shape[2]}x{pixel_values.shape[3]} and normalized")

# ============================================================
# 3. Trace Patch Embedding
# ============================================================
print(f"\n{'=' * 70}")
print("Step 2: Patch Embedding (Conv2d)")
print("=" * 70)

embeddings_layer = vision_model.embeddings

# The patch embedding is a Conv2d with kernel_size=stride=patch_size
patch_conv = embeddings_layer.patch_embedding
patch_size = vision_model.config.patch_size
print(f"  Patch embedding: Conv2d(3, {vision_model.config.hidden_size}, "
      f"kernel={patch_size}, stride={patch_size})")
print(f"  This is NON-OVERLAPPING — each {patch_size}x{patch_size} pixel region → 1 token")

# Run patch embedding manually
with torch.no_grad():
    patch_embeds = patch_conv(pixel_values)
    print(f"\n  Input:  {pixel_values.shape}  (batch, 3, H, W)")
    print(f"  Output: {patch_embeds.shape}  (batch, embed_dim, grid_h, grid_w)")

    grid_h, grid_w = patch_embeds.shape[2], patch_embeds.shape[3]
    print(f"\n  Grid: {grid_h} x {grid_w} = {grid_h * grid_w} patches")
    print(f"  Each patch covers {patch_size}x{patch_size} = {patch_size**2} pixels")

    # Flatten to sequence
    patch_embeds_flat = patch_embeds.flatten(2).transpose(1, 2)
    print(f"\n  After flatten + transpose: {patch_embeds_flat.shape}")
    print(f"  → Now a SEQUENCE of {patch_embeds_flat.shape[1]} patch tokens, each {patch_embeds_flat.shape[2]}-dim")

# ============================================================
# 4. Trace CLS Token + Position Embedding
# ============================================================
print(f"\n{'=' * 70}")
print("Step 3: [CLS] Token + Positional Embeddings")
print("=" * 70)

with torch.no_grad():
    cls_token = embeddings_layer.class_embedding
    print(f"  [CLS] token shape: {cls_token.shape}  (learnable parameter)")

    # Prepend CLS
    batch_size = pixel_values.shape[0]
    cls_expanded = cls_token.unsqueeze(0).expand(batch_size, 1, -1)
    with_cls = torch.cat([cls_expanded, patch_embeds_flat], dim=1)
    print(f"  After prepend [CLS]: {with_cls.shape}")
    print(f"  → {patch_embeds_flat.shape[1]} patches + 1 CLS = {with_cls.shape[1]} tokens")

    # Add position embeddings
    pos_embed = embeddings_layer.position_embedding
    print(f"\n  Position embedding: {pos_embed.weight.shape}  (num_positions, embed_dim)")
    print(f"  → One learnable vector per position (including CLS)")

    position_ids = embeddings_layer.position_ids
    final_embeds = with_cls + pos_embed(position_ids)
    print(f"  After adding position embeddings: {final_embeds.shape}")

# ============================================================
# 5. Run Full Vision Encoder
# ============================================================
print(f"\n{'=' * 70}")
print("Step 4: Transformer Encoder (self-attention layers)")
print("=" * 70)

with torch.no_grad():
    # Run full model with hidden states output
    outputs = vision_model(pixel_values, output_hidden_states=True)

    print(f"  Number of transformer layers: {len(vision_model.encoder.layers)}")
    print(f"  Number of hidden states: {len(outputs.hidden_states)} (input + {len(outputs.hidden_states)-1} layers)")
    print(f"\n  Hidden state shapes through layers:")
    for i, hs in enumerate(outputs.hidden_states):
        label = "input" if i == 0 else f"layer {i}"
        print(f"    [{label:>8}]: {hs.shape}")

    # Show what LLaVA typically uses
    print(f"\n  LLaVA uses layer -2 (second to last): {outputs.hidden_states[-2].shape}")
    print(f"  Then drops the [CLS] token: {outputs.hidden_states[-2][:, 1:, :].shape}")

    last_hidden = outputs.last_hidden_state
    print(f"\n  Final output (last_hidden_state): {last_hidden.shape}")
    print(f"  [CLS] token embedding: {last_hidden[:, 0, :].shape}  ← used for image-level tasks")
    print(f"  Patch tokens: {last_hidden[:, 1:, :].shape}  ← used for detailed understanding")

# ============================================================
# 6. Visualize attention (which patches attend to which)
# ============================================================
print(f"\n{'=' * 70}")
print("Step 5: Understanding the Output")
print("=" * 70)

with torch.no_grad():
    # Get the features LLaVA would use
    selected_layer = outputs.hidden_states[-2]  # layer -2
    visual_tokens = selected_layer[:, 1:, :]    # drop CLS

    print(f"  Visual tokens for LLM: {visual_tokens.shape}")
    print(f"  → {visual_tokens.shape[1]} tokens, each {visual_tokens.shape[2]}-dim")
    print(f"  → These represent a {grid_h}x{grid_w} spatial grid of the image")
    print(f"  → Token [0] = top-left patch, token [{grid_h*grid_w-1}] = bottom-right patch")

    # Show that different patches have different embeddings
    top_left = visual_tokens[0, 0, :]
    bottom_right = visual_tokens[0, -1, :]
    cosine_sim = torch.cosine_similarity(top_left, bottom_right, dim=0)
    print(f"\n  Cosine similarity between top-left and bottom-right patches: {cosine_sim:.4f}")
    print(f"  (Our gradient image has red on left, blue on right — should be low similarity)")

    # Neighboring patches should be more similar
    top_left_neighbor = visual_tokens[0, 1, :]
    cosine_sim_neighbor = torch.cosine_similarity(top_left, top_left_neighbor, dim=0)
    print(f"  Cosine similarity between adjacent patches: {cosine_sim_neighbor:.4f}")
    print(f"  (Adjacent patches should be more similar)")

# ============================================================
# 7. Bonus: CLIP's text-image alignment
# ============================================================
print(f"\n{'=' * 70}")
print("Bonus: CLIP Contrastive Alignment")
print("=" * 70)

with torch.no_grad():
    # Encode image and text
    texts = ["a red and blue gradient", "a photo of a cat", "a sunset over ocean"]
    text_inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in text_inputs.items()}

    outputs = model(**text_inputs)

    # Cosine similarity between image and each text
    image_embeds = outputs.image_embeds  # (1, 768) — projected CLS token
    text_embeds = outputs.text_embeds    # (3, 768) — projected [EOS] tokens

    print(f"  Image embedding: {image_embeds.shape}")
    print(f"  Text embeddings: {text_embeds.shape}")

    sims = torch.cosine_similarity(image_embeds, text_embeds, dim=-1)
    print(f"\n  Text-Image similarities:")
    for text, sim in zip(texts, sims):
        print(f"    '{text}': {sim:.4f}")
    print(f"\n  CLIP learned to align images and text in the SAME embedding space")
    print(f"  This is why it works as a VLM visual encoder — its features are")
    print(f"  already semantically meaningful and text-aligned")

print(f"\n{'=' * 70}")
print("Summary: What happened")
print("=" * 70)
print(f"""
  Image ({image.size[0]}x{image.size[1]})
    → Resize to {pixel_values.shape[2]}x{pixel_values.shape[3]}
    → Patch embed: Conv2d({patch_size}x{patch_size}) → {grid_h*grid_w} patches
    → Prepend [CLS] → {grid_h*grid_w + 1} tokens
    → Add position embeddings
    → {len(vision_model.encoder.layers)} transformer layers
    → Select layer -2, drop [CLS]
    → {grid_h*grid_w} visual tokens of dim {vision_model.config.hidden_size}

  These {grid_h*grid_w} tokens are what LLaVA feeds into its MLP projector,
  then into the LLM. Each token represents a {patch_size}x{patch_size} pixel
  region of the original image.
""")
