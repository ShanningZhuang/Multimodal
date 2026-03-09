"""
Lab 2: LLaVA Pipeline — Step-by-Step VLM Execution

This script traces the FULL LLaVA pipeline, calling each component manually:
  Image → CLIP encoder → MLP projector → Merge with text → LLM → Output

Run: uv run python Multimodal/scripts/02_llava_pipeline.py

Concepts covered:
  - Late fusion (projection-based) architecture
  - MLP projector (vision_dim → text_dim)
  - Embedding merge via placeholder replacement
  - How the LLM sees a unified sequence of visual + text tokens

Reference docs:
  - vision_language/01_Architecture.md (fusion strategies)

Requires: ~14GB VRAM (fp16).
"""

import argparse

import torch
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default=None, help="Path to an image file")
args = parser.parse_args()

device_map = "auto"
dtype = torch.float16

from transformers import AutoProcessor, LlavaForConditionalGeneration

# ============================================================
# 1. Load Model
# ============================================================
model_id = "llava-hf/llava-1.5-7b-hf"
print(f"Loading {model_id} (dtype={dtype}, device_map={device_map})...")
print("This may take a minute on first run (downloading ~14GB).\n")

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=dtype, device_map=device_map,
)
model.eval()

# ============================================================
# 2. Inspect Model Architecture
# ============================================================
print("=" * 70)
print("Model Architecture")
print("=" * 70)

print(f"\n  model.vision_tower  → {type(model.vision_tower).__name__}")
print(f"  model.multi_modal_projector → {type(model.multi_modal_projector).__name__}")
print(f"  model.language_model → {type(model.language_model).__name__}")

print(f"\nProjector weights:")
for name, param in model.multi_modal_projector.named_parameters():
    print(f"  {name}: {param.shape}")

vision_dim = model.config.vision_config.hidden_size
text_dim = model.config.text_config.hidden_size
print(f"\n  Vision encoder output dim: {vision_dim}")
print(f"  LLM hidden dim: {text_dim}")
print(f"  Projector: {vision_dim} → {text_dim} → {text_dim}  (2-layer MLP with GELU)")

# ============================================================
# 3. Prepare Input
# ============================================================
print(f"\n{'=' * 70}")
print("Preparing Input")
print("=" * 70)

if args.image:
    image = Image.open(args.image).convert("RGB")
    print(f"  Loaded image: {args.image} ({image.size})")
else:
    # Create a test image with recognizable content
    img_array = np.zeros((300, 400, 3), dtype=np.uint8)
    # Red rectangle on left
    img_array[50:250, 30:180, 0] = 220
    # Blue rectangle on right
    img_array[50:250, 220:370, 2] = 220
    # Green bar at bottom
    img_array[260:290, 30:370, 1] = 200
    image = Image.fromarray(img_array)
    print("  Using synthetic test image (red rect, blue rect, green bar)")

prompt = "USER: <image>\nDescribe what you see in this image in detail.\nASSISTANT:"
print(f"  Prompt: {prompt[:60]}...")

inputs = processor(text=prompt, images=image, return_tensors="pt")
# Move to same device as model
device = next(model.parameters()).device
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

pixel_values = inputs["pixel_values"]
input_ids = inputs["input_ids"]

print(f"\n  pixel_values: {pixel_values.shape}  (batch, channels, H, W)")
print(f"  input_ids:    {input_ids.shape}  (batch, seq_len)")

# ============================================================
# 4. Step-by-Step Forward Pass
# ============================================================

with torch.no_grad():
    # ----- STEP A: Vision Encoder -----
    print(f"\n{'=' * 70}")
    print("Step A: Vision Encoder (CLIP ViT)")
    print("=" * 70)

    vision_outputs = model.vision_tower(
        pixel_values,
        output_hidden_states=True,
    )

    print(f"  Input pixels: {pixel_values.shape}")
    print(f"  Num layers: {len(vision_outputs.hidden_states) - 1}")

    # LLaVA selects a specific layer (usually -2)
    feature_layer = model.config.vision_feature_layer
    select_strategy = model.config.vision_feature_select_strategy
    print(f"  vision_feature_layer: {feature_layer}")
    print(f"  vision_feature_select_strategy: '{select_strategy}'")

    selected = vision_outputs.hidden_states[feature_layer]
    print(f"  Selected hidden state: {selected.shape}")

    if select_strategy == "default":
        # Drop [CLS] token
        image_features = selected[:, 1:, :]
        print(f"  After dropping [CLS]: {image_features.shape}")
    else:
        image_features = selected
        print(f"  Using all tokens (including [CLS]): {image_features.shape}")

    num_visual_tokens = image_features.shape[1]
    print(f"\n  Result: {num_visual_tokens} visual tokens of dim {image_features.shape[2]}")

    # ----- STEP B: MLP Projector -----
    print(f"\n{'=' * 70}")
    print("Step B: MLP Projector (Vision → Language space)")
    print("=" * 70)

    # Trace through each layer of the projector
    proj = model.multi_modal_projector
    x = image_features

    print(f"  Input: {x.shape}  (dim={x.shape[-1]} = vision encoder dim)")

    # Linear 1
    x_after_linear1 = proj.linear_1(x)
    print(f"  After linear_1: {x_after_linear1.shape}  (dim={x_after_linear1.shape[-1]} = LLM dim)")

    # Activation
    x_after_act = proj.act(x_after_linear1)
    print(f"  After GELU activation: {x_after_act.shape}")

    # Linear 2
    projected_features = proj.linear_2(x_after_act)
    print(f"  After linear_2: {projected_features.shape}")

    # Verify it matches the LLM's embedding dim
    print(f"\n  Projected dim ({projected_features.shape[-1]}) == LLM hidden dim ({text_dim}): "
          f"{projected_features.shape[-1] == text_dim}")

    # ----- STEP C: Text Tokenization -----
    print(f"\n{'=' * 70}")
    print("Step C: Text Tokenization & Placeholder Analysis")
    print("=" * 70)

    image_token_id = model.config.image_token_index
    print(f"  Image token ID: {image_token_id}")
    print(f"  Input IDs: {input_ids.shape}")

    # Show token-by-token
    tokens = input_ids[0].tolist()
    print(f"\n  Token sequence ({len(tokens)} tokens):")
    is_image_mask = (input_ids[0] == image_token_id)
    image_positions = torch.where(is_image_mask)[0]

    # Show first few, image region, and last few
    decoded_tokens = []
    for i, tid in enumerate(tokens):
        if tid == image_token_id:
            decoded_tokens.append(f"[IMG:{tid}]")
        else:
            text = processor.tokenizer.decode([tid])
            decoded_tokens.append(f"'{text}'({tid})")

    if len(decoded_tokens) <= 20:
        for i, dt in enumerate(decoded_tokens):
            print(f"    [{i:3d}] {dt}")
    else:
        # Show first 5, ..., image region summary, ..., last 5
        for i in range(5):
            print(f"    [{i:3d}] {decoded_tokens[i]}")
        if len(image_positions) > 0:
            print(f"    ... ({len(image_positions)} image placeholder tokens) ...")
        for i in range(max(5, len(tokens) - 5), len(tokens)):
            print(f"    [{i:3d}] {decoded_tokens[i]}")

    print(f"\n  Image placeholder count: {is_image_mask.sum().item()}")
    print(f"  Visual tokens from encoder: {num_visual_tokens}")
    print(f"  Match: {is_image_mask.sum().item() == num_visual_tokens}")

    # ----- STEP D: Embedding Merge -----
    print(f"\n{'=' * 70}")
    print("Step D: Embedding Merge (the key fusion step)")
    print("=" * 70)

    # Get text embeddings
    text_embeddings = model.language_model.model.embed_tokens(input_ids)
    print(f"  Text embeddings: {text_embeddings.shape}")
    print(f"  Image placeholder positions: [{image_positions[0]}..{image_positions[-1]}]")

    # Merge: replace placeholder embeddings with projected visual features
    merged = text_embeddings.clone()
    merged[0, image_positions] = projected_features[0]

    print(f"\n  Before merge: text_embeddings[0, {image_positions[0].item()}] = "
          f"{text_embeddings[0, image_positions[0], :3].tolist()}")
    print(f"  After merge:  merged[0, {image_positions[0].item()}]          = "
          f"{merged[0, image_positions[0], :3].tolist()}")
    print(f"\n  The LLM now sees a UNIFIED sequence:")
    print(f"    [text tokens] [576 visual tokens] [text tokens]")
    print(f"    Total: {merged.shape[1]} tokens, all in the same {text_dim}-dim space")

    # ----- STEP E: Generate -----
    print(f"\n{'=' * 70}")
    print("Step E: LLM Generation")
    print("=" * 70)

    print(f"  Running full generation (this calls all steps above internally)...")
    output_ids = model.generate(**inputs, max_new_tokens=128)

    # Decode
    generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
    # Extract just the assistant's response
    if "ASSISTANT:" in generated_text:
        answer = generated_text.split("ASSISTANT:")[-1].strip()
    else:
        answer = generated_text
    print(f"\n  Generated: {answer}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 70}")
print("Complete Pipeline Summary")
print("=" * 70)
print(f"""
  Image ({image.size[0]}x{image.size[1]})
    │
    ▼ Resize + normalize (CLIPProcessor)
  pixel_values {tuple(pixel_values.shape)}
    │
    ▼ CLIP ViT ({len(vision_outputs.hidden_states)-1} layers, select layer {feature_layer})
  image_features {tuple(image_features.shape)}  ({num_visual_tokens} tokens x {vision_dim}d)
    │
    ▼ MLP Projector: Linear({vision_dim}→{text_dim}) + GELU + Linear({text_dim}→{text_dim})
  projected_features {tuple(projected_features.shape)}  ({num_visual_tokens} tokens x {text_dim}d)
    │
    ▼ Replace {is_image_mask.sum().item()} <image> placeholder tokens
  merged_embeddings {tuple(merged.shape)}  (text + visual, unified sequence)
    │
    ▼ Llama LLM (autoregressive generation)
  "{answer[:80]}..."
""")
