"""
Lab 3: Compare CLIP vs SigLIP Visual Encoders

This script runs the same image through CLIP and SigLIP to compare:
  - Architecture differences (CLS token, patch sizes)
  - Output shapes and feature quality
  - Similarity structures in the embedding space

Run: uv run python Multimodal/scripts/03_compare_encoders.py

Concepts covered:
  - CLIP: softmax contrastive loss, [CLS] token, 14x14 patches
  - SigLIP: sigmoid loss, NO [CLS] token, variable patch sizes
  - Why SigLIP replaced CLIP in modern VLMs (PaliGemma, LLaVA-OneVision)

Reference docs:
  - visual_encoder/03_Semantic_Encoders.md (CLIP vs SigLIP training)
"""

import torch
import numpy as np
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    SiglipModel, SiglipProcessor,
)

# ============================================================
# 1. Create test images
# ============================================================
def make_test_images():
    """Create simple, recognizable test images."""
    images = {}

    # Solid red
    img = np.full((300, 300, 3), [220, 30, 30], dtype=np.uint8)
    images["red square"] = Image.fromarray(img)

    # Solid blue
    img = np.full((300, 300, 3), [30, 30, 220], dtype=np.uint8)
    images["blue square"] = Image.fromarray(img)

    # Horizontal stripes
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    for y in range(300):
        if (y // 30) % 2 == 0:
            img[y, :] = [200, 200, 200]
    images["stripes"] = Image.fromarray(img)

    # Gradient
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    for x in range(300):
        img[:, x] = [int(255 * x / 300)] * 3
    images["gradient"] = Image.fromarray(img)

    return images

images = make_test_images()
# Use first image for detailed analysis
test_image = images["red square"]

# ============================================================
# 2. Load both models
# ============================================================
print("=" * 70)
print("Loading CLIP and SigLIP")
print("=" * 70)

clip_id = "openai/clip-vit-large-patch14-336"
siglip_id = "google/siglip-so400m-patch14-384"

print(f"  CLIP:   {clip_id}")
print(f"  SigLIP: {siglip_id}")

clip_processor = CLIPProcessor.from_pretrained(clip_id)
clip_model = CLIPModel.from_pretrained(clip_id).cuda()
clip_model.eval()

siglip_processor = SiglipProcessor.from_pretrained(siglip_id)
siglip_model = SiglipModel.from_pretrained(siglip_id).cuda()
siglip_model.eval()

device = torch.device("cuda")

# ============================================================
# 3. Architecture Comparison
# ============================================================
print(f"\n{'=' * 70}")
print("Architecture Comparison")
print("=" * 70)

clip_vc = clip_model.vision_model.config
siglip_vc = siglip_model.vision_model.config

comparison = [
    ("Image size", f"{clip_vc.image_size}px", f"{siglip_vc.image_size}px"),
    ("Patch size", f"{clip_vc.patch_size}px", f"{siglip_vc.patch_size}px"),
    ("Grid", f"{clip_vc.image_size // clip_vc.patch_size}x{clip_vc.image_size // clip_vc.patch_size}",
             f"{siglip_vc.image_size // siglip_vc.patch_size}x{siglip_vc.image_size // siglip_vc.patch_size}"),
    ("Num patches", str((clip_vc.image_size // clip_vc.patch_size) ** 2),
                    str((siglip_vc.image_size // siglip_vc.patch_size) ** 2)),
    ("Has [CLS]", "Yes (+1 token)", "No"),
    ("Total tokens", str((clip_vc.image_size // clip_vc.patch_size) ** 2 + 1),
                     str((siglip_vc.image_size // siglip_vc.patch_size) ** 2)),
    ("Hidden dim", str(clip_vc.hidden_size), str(siglip_vc.hidden_size)),
    ("Num layers", str(clip_vc.num_hidden_layers), str(siglip_vc.num_hidden_layers)),
    ("Num heads", str(clip_vc.num_attention_heads), str(siglip_vc.num_attention_heads)),
    ("Loss", "Softmax (InfoNCE)", "Sigmoid (binary)"),
]

print(f"\n  {'Property':<20} {'CLIP':<25} {'SigLIP':<25}")
print(f"  {'─' * 20} {'─' * 25} {'─' * 25}")
for prop, clip_val, siglip_val in comparison:
    print(f"  {prop:<20} {clip_val:<25} {siglip_val:<25}")

# ============================================================
# 4. Run both encoders on the same image
# ============================================================
print(f"\n{'=' * 70}")
print("Running both encoders on the same image")
print("=" * 70)

with torch.no_grad():
    # CLIP
    clip_inputs = clip_processor(images=test_image, return_tensors="pt")
    clip_pixel = clip_inputs["pixel_values"].to(device)
    clip_outputs = clip_model.vision_model(clip_pixel, output_hidden_states=True)

    print(f"\n  CLIP:")
    print(f"    Preprocessed pixels: {clip_pixel.shape}")
    print(f"    Output (last hidden): {clip_outputs.last_hidden_state.shape}")
    print(f"    → Token 0 is [CLS], tokens 1..{clip_outputs.last_hidden_state.shape[1]-1} are patches")

    # SigLIP
    siglip_inputs = siglip_processor(images=test_image, return_tensors="pt")
    siglip_pixel = siglip_inputs["pixel_values"].to(device)
    siglip_outputs = siglip_model.vision_model(siglip_pixel, output_hidden_states=True)

    print(f"\n  SigLIP:")
    print(f"    Preprocessed pixels: {siglip_pixel.shape}")
    print(f"    Output (last hidden): {siglip_outputs.last_hidden_state.shape}")
    print(f"    → ALL tokens are patches (no [CLS])")

# ============================================================
# 5. Feature quality comparison
# ============================================================
print(f"\n{'=' * 70}")
print("Feature Analysis: Spatial Structure")
print("=" * 70)

with torch.no_grad():
    # For CLIP: drop CLS token
    clip_patches = clip_outputs.last_hidden_state[:, 1:, :]  # (1, 576, 1024)
    siglip_patches = siglip_outputs.last_hidden_state         # (1, 729, 1152)

    print(f"\n  Patch tokens — CLIP: {clip_patches.shape}, SigLIP: {siglip_patches.shape}")

    # Self-similarity: how similar are patches to each other?
    for name, patches in [("CLIP", clip_patches[0]), ("SigLIP", siglip_patches[0])]:
        # Normalize
        patches_norm = patches / patches.norm(dim=-1, keepdim=True)
        sim_matrix = patches_norm @ patches_norm.T

        # Stats
        n = patches.shape[0]
        # Exclude diagonal (self-similarity = 1.0)
        mask = ~torch.eye(n, dtype=torch.bool)
        off_diag = sim_matrix[mask]

        print(f"\n  {name} patch self-similarity:")
        print(f"    Mean: {off_diag.mean():.4f}")
        print(f"    Std:  {off_diag.std():.4f}")
        print(f"    Min:  {off_diag.min():.4f}, Max: {off_diag.max():.4f}")

    print(f"\n  Lower mean similarity = more diverse/informative features")
    print(f"  Higher std = more variation between similar and dissimilar patches")

# ============================================================
# 6. Text-Image alignment comparison
# ============================================================
print(f"\n{'=' * 70}")
print("Text-Image Alignment: Zero-Shot Classification")
print("=" * 70)

texts = [
    "a red colored image",
    "a blue colored image",
    "a striped pattern",
    "a gradient from dark to light",
]

with torch.no_grad():
    print(f"\n  Testing {len(images)} images against {len(texts)} text descriptions\n")

    # CLIP
    print("  CLIP similarities:")
    for img_name, img in images.items():
        clip_in = clip_processor(text=texts, images=img, return_tensors="pt", padding=True)
        clip_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in clip_in.items()}
        clip_out = clip_model(**clip_in)
        sims = torch.cosine_similarity(
            clip_out.image_embeds.unsqueeze(1),
            clip_out.text_embeds.unsqueeze(0),
            dim=-1
        )[0]
        best_idx = sims.argmax().item()
        print(f"    '{img_name:12s}' → best match: '{texts[best_idx]}' ({sims[best_idx]:.3f})")

    # SigLIP
    print(f"\n  SigLIP similarities:")
    for img_name, img in images.items():
        siglip_in = siglip_processor(text=texts, images=img, return_tensors="pt", padding="max_length")
        siglip_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in siglip_in.items()}
        siglip_out = siglip_model(**siglip_in)
        sims = torch.cosine_similarity(
            siglip_out.image_embeds.unsqueeze(1),
            siglip_out.text_embeds.unsqueeze(0),
            dim=-1
        )[0]
        best_idx = sims.argmax().item()
        print(f"    '{img_name:12s}' → best match: '{texts[best_idx]}' ({sims[best_idx]:.3f})")

# ============================================================
# 7. Key Takeaways
# ============================================================
print(f"\n{'=' * 70}")
print("Key Takeaways")
print("=" * 70)
print(f"""
  1. SigLIP has NO [CLS] token — all output tokens are patch tokens
     CLIP: {clip_outputs.last_hidden_state.shape[1]} tokens = 1 CLS + {clip_patches.shape[1]} patches
     SigLIP: {siglip_outputs.last_hidden_state.shape[1]} tokens = {siglip_patches.shape[1]} patches (all patches)

  2. SigLIP uses sigmoid loss (per-pair) instead of softmax (whole-batch)
     → Scales better, doesn't need huge batch sizes
     → See 03_Semantic_Encoders.md for the math

  3. SigLIP has become the preferred encoder for modern VLMs:
     → PaliGemma, LLaVA-OneVision, BAGEL all use SigLIP variants
     → Larger hidden dim ({siglip_vc.hidden_size} vs {clip_vc.hidden_size}) = richer features

  4. Both produce spatial token grids that map 1:1 to image regions
     → This spatial structure is why VLMs can do object localization
""")
