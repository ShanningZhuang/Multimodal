# Hands-On Lab: Tracing VLM Execution

> Parent: [Visual Encoders](00_Visual_Encoder.md)

## Overview

This lab teaches you how visual encoders and VLM pipelines work by **running real models and tracing every step**: how an image becomes patches, how patches become embeddings, how embeddings merge with text, and how the LLM produces output.

## Runnable Scripts (Start Here)

Self-contained Python scripts in `../scripts/` — run them and read the annotated output:

| # | Script | What You Learn | GPU Required |
|---|--------|---------------|-------------|
| 1 | [01_clip_encoder.py](../scripts/01_clip_encoder.py) | Patch embedding, [CLS] token, transformer layers, feature selection | No |
| 2 | [02_llava_pipeline.py](../scripts/02_llava_pipeline.py) | Full VLM: CLIP → MLP projector → merge → Llama → output | Yes (~14GB) or `--cpu` |
| 3 | [03_compare_encoders.py](../scripts/03_compare_encoders.py) | CLIP vs SigLIP architecture, features, zero-shot alignment | No |
| 4 | [04_projector_and_merge.py](../scripts/04_projector_and_merge.py) | Projector variants, placeholder tokens, masked_scatter merge | No (simulated) |

```bash
cd /root/build-your-knowledge/Multimodal
uv sync
uv run python scripts/01_clip_encoder.py
```

**Recommended order**: Start with **04** (projector, no download needed) → **01** (CLIP encoder) → **03** (compare encoders) → **02** (full LLaVA pipeline).

For interactive exploration, use `uv run ipython -i Multimodal/scripts/01_clip_encoder.py` — you get a REPL with all variables loaded after the script finishes.

---

## Advanced: Instrumenting vLLM Source Code

The scripts above use HuggingFace Transformers (single-threaded, debugger-friendly). For tracing through vLLM's production serving code, you can add logging directly to vLLM source files.

**Source code**: `/root/vllm/vllm/model_executor/models/`

### Prerequisites

```bash
# Install vLLM in editable mode so your code changes take effect
cd /root/vllm
uv pip install -e .
```

## Lab 1: CLIP Vision Encoder — How Images Become Tokens

**Goal**: Understand how a raw image (H x W x 3) becomes a sequence of embedding vectors that an LLM can process.

**File to instrument**: `vllm/model_executor/models/clip.py`

### Step 1: Trace Patch Embedding

The patch embedding converts raw pixels to a sequence of patch vectors. This is the ViT equivalent of tokenization.

Add logging to `CLIPVisionEmbeddings.forward()` (line 340):

```python
# clip.py, class CLIPVisionEmbeddings, def forward()
def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    batch_size = pixel_values.shape[0]
    target_dtype = self.patch_embedding.weight.dtype

    # === LAB LOGGING START ===
    print(f"\n{'='*60}")
    print(f"[CLIP Patch Embedding]")
    print(f"  Input pixel_values shape: {pixel_values.shape}")
    print(f"  → (batch={batch_size}, channels=3, H={pixel_values.shape[2]}, W={pixel_values.shape[3]})")
    print(f"  Patch size: {self.patch_size}x{self.patch_size}")
    print(f"  Grid: {self.image_size // self.patch_size}x{self.image_size // self.patch_size} = {self.num_patches} patches")
    # === LAB LOGGING END ===

    patch_embeds = self.patch_embedding(
        pixel_values.to(dtype=target_dtype)
    )  # shape = [*, width, grid, grid]

    # === LAB LOGGING START ===
    print(f"  After Conv2d: {patch_embeds.shape}  (batch, embed_dim, grid_h, grid_w)")
    # === LAB LOGGING END ===

    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    # === LAB LOGGING START ===
    print(f"  After flatten+transpose: {patch_embeds.shape}  (batch, num_patches, embed_dim)")
    # === LAB LOGGING END ===

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

    # === LAB LOGGING START ===
    print(f"  After prepend [CLS]: {embeddings.shape}  (batch, 1+num_patches, embed_dim)")
    # === LAB LOGGING END ===

    embeddings = embeddings + self.position_embedding(self.position_ids)

    # === LAB LOGGING START ===
    print(f"  After + position embedding: {embeddings.shape}")
    print(f"  Position IDs range: 0..{self.num_positions-1}")
    print(f"  Output: {embeddings.shape[1]} tokens of dim {embeddings.shape[2]}")
    print(f"{'='*60}\n")
    # === LAB LOGGING END ===

    return embeddings
```

**What to observe**:
- Input: `(1, 3, 336, 336)` — one 336x336 RGB image
- After Conv2d: `(1, 1024, 21, 21)` — each 16x16 patch projected to 1024-dim
- After flatten: `(1, 441, 1024)` — 441 patches (21x21) as a sequence
- After [CLS]: `(1, 442, 1024)` — CLS token prepended
- This is exactly the ViT architecture from `02_ViT.md`, running on real data

### Step 2: Trace the Full Vision Transformer

Add logging to `CLIPVisionTransformer.forward()` (line 697):

```python
# clip.py, class CLIPVisionTransformer, def forward()
def forward(self, pixel_values, *, select_layers=None, feature_select_strategy=None):
    hidden_states = self.embeddings(pixel_values)
    hidden_states = self.pre_layrnorm(hidden_states)

    # === LAB LOGGING START ===
    print(f"[CLIP Vision Transformer]")
    print(f"  After embeddings + pre_layernorm: {hidden_states.shape}")
    print(f"  Num transformer layers: {len(self.encoder.layers)}")
    print(f"  select_layers: {select_layers}")
    print(f"  feature_select_strategy: {feature_select_strategy}")
    # === LAB LOGGING END ===

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        return_all_hidden_states=select_layers is not None,
    )

    encoder_outputs = resolve_visual_encoder_outputs(
        encoder_outputs,
        self.post_layernorm,
        select_layers=select_layers,
        max_possible_layers=self.config.num_hidden_layers,
        feature_select_strategy=feature_select_strategy,
    )

    # === LAB LOGGING START ===
    if isinstance(encoder_outputs, torch.Tensor):
        print(f"  Final encoder output: {encoder_outputs.shape}")
    else:
        print(f"  Final encoder output type: {type(encoder_outputs)}")
    print(f"  feature_select_strategy controls whether [CLS] is included or dropped")
    # === LAB LOGGING END ===

    return encoder_outputs
```

**What to observe**:
- `select_layers` — LLaVA typically selects a specific layer's output (e.g., layer -2)
- `feature_select_strategy` — "default" keeps [CLS] + patches, "full" keeps all tokens
- The encoder is just standard transformer blocks (same as LLM, but without causal mask)

---

## Lab 2: LLaVA — The Complete VLM Pipeline

**Goal**: Trace the full path from image + text → visual encoder → projector → embedding merge → LLM output.

**File to instrument**: `vllm/model_executor/models/llava.py`

### Step 3: Trace the MLP Projector

The projector bridges vision and language embedding spaces. Add logging to `LlavaMultiModalProjector.forward()` (line 157):

```python
# llava.py, class LlavaMultiModalProjector, def forward()
def forward(self, image_features: torch.Tensor) -> torch.Tensor:
    # === LAB LOGGING START ===
    print(f"\n{'='*60}")
    print(f"[LLaVA MLP Projector]")
    print(f"  Input (from CLIP): {image_features.shape}")
    print(f"  → (num_tokens, vision_hidden_size={image_features.shape[-1]})")
    # === LAB LOGGING END ===

    hidden_states, _ = self.linear_1(image_features)

    # === LAB LOGGING START ===
    print(f"  After linear_1: {hidden_states.shape}  (vision_dim → text_dim)")
    # === LAB LOGGING END ===

    hidden_states = self.act(hidden_states)
    hidden_states, _ = self.linear_2(hidden_states)

    # === LAB LOGGING START ===
    print(f"  After GELU + linear_2: {hidden_states.shape}  (text_dim → text_dim)")
    print(f"  Now compatible with LLM embedding space!")
    print(f"{'='*60}\n")
    # === LAB LOGGING END ===

    return hidden_states
```

**What to observe**:
- Input: `(576, 1024)` — 576 visual tokens from CLIP (24x24 patches, no [CLS])
- After linear_1: `(576, 4096)` — projected to LLM's hidden size
- After linear_2: `(576, 4096)` — these embeddings can now replace `<image>` tokens

### Step 4: Trace `_process_image_input` — The Composition Point

This is where visual encoder + projector are called together. Add logging to `_process_image_input()` (line 641):

```python
# llava.py, class LlavaForConditionalGeneration
def _process_image_input(self, image_input):
    # === LAB LOGGING START ===
    print(f"\n{'='*60}")
    print(f"[LLaVA _process_image_input]")
    print(f"  Input type: {image_input['type']}")
    # === LAB LOGGING END ===

    if image_input["type"] == "image_embeds":
        return image_input["data"]

    image_features = self._process_image_pixels(image_input)

    # === LAB LOGGING START ===
    if isinstance(image_features, torch.Tensor):
        print(f"  After vision encoder: {image_features.shape}")
    else:
        print(f"  After vision encoder: {len(image_features)} feature tensors")
        for i, f in enumerate(image_features):
            print(f"    [{i}]: {f.shape}")
    # === LAB LOGGING END ===

    if isinstance(image_features, torch.Tensor):
        result = self.multi_modal_projector(image_features)
        # === LAB LOGGING START ===
        print(f"  After projector: {result.shape}")
        print(f"  These embeddings will replace <image> placeholder tokens")
        print(f"{'='*60}\n")
        # === LAB LOGGING END ===
        return result

    feature_sizes = [f.shape[0] for f in image_features]
    image_embeds = self.multi_modal_projector(torch.cat(image_features))
    image_embeds = torch.split(image_embeds, feature_sizes)
    return image_embeds
```

### Step 5: Trace the Embedding Merge — Where Visual Meets Language

This is the most important function to understand. Add logging to `_merge_multimodal_embeddings()` in `vllm/model_executor/models/utils.py` (line 443):

```python
# utils.py
def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    # === LAB LOGGING START ===
    print(f"\n{'='*60}")
    print(f"[MERGE: Visual embeddings into text embeddings]")
    print(f"  inputs_embeds (text): {inputs_embeds.shape}")
    print(f"  is_multimodal mask: {is_multimodal.shape}, {is_multimodal.sum().item()} placeholder positions")
    # === LAB LOGGING END ===

    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    # === LAB LOGGING START ===
    print(f"  multimodal_embeddings flattened: {mm_embeds_flat.shape}")
    print(f"  → {mm_embeds_flat.shape[0]} visual tokens will replace {is_multimodal.sum().item()} placeholders")
    print(f"  Before merge - text embed[0][:5]: {inputs_embeds[0, :5].tolist()}")
    # === LAB LOGGING END ===

    try:
        inputs_embeds.masked_scatter_(
            is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype)
        )
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()
        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)
            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e
        raise ValueError("Error during masked scatter operation") from e

    # === LAB LOGGING START ===
    print(f"  After merge - visual embeddings injected at placeholder positions")
    print(f"  Result: {inputs_embeds.shape} — ready for LLM forward pass")
    print(f"{'='*60}\n")
    # === LAB LOGGING END ===

    return inputs_embeds
```

**What to observe**:
- `inputs_embeds`: `(seq_len, 4096)` — text token embeddings with `<image>` placeholders
- `is_multimodal`: boolean mask showing which positions are `<image>` tokens (576 of them)
- `mm_embeds_flat`: `(576, 4096)` — the projected visual features
- `masked_scatter_` replaces the 576 placeholder positions with visual features **in-place**
- After this, the LLM sees a unified sequence of text + visual embeddings

### Running the Instrumented Code

After adding the logging, run LLaVA inference:

```bash
cd /root/vllm
uv run python examples/offline_inference/vision_language.py --model llava
```

Or write a minimal script:

```python
# trace_llava.py
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    max_model_len=4096,
    max_num_seqs=1,
    limit_mm_per_prompt={"image": 1},
)

image = ImageAsset("cherry_blossom").pil_image
prompt = "<image>\nWhat do you see in this image?"

output = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    },
    sampling_params=SamplingParams(max_tokens=64, temperature=0),
)
print(f"\nAnswer: {output[0].outputs[0].text}")
```

```bash
uv run python trace_llava.py
```

### Expected Output Flow

```
============================================================
[CLIP Patch Embedding]
  Input pixel_values shape: torch.Size([1, 3, 336, 336])
  → (batch=1, channels=3, H=336, W=336)
  Patch size: 14x14
  Grid: 24x24 = 576 patches
  After Conv2d: torch.Size([1, 1024, 24, 24])
  After flatten+transpose: torch.Size([1, 576, 1024])
  After prepend [CLS]: torch.Size([1, 577, 1024])
  After + position embedding: torch.Size([1, 577, 1024])
  Output: 577 tokens of dim 1024
============================================================

[CLIP Vision Transformer]
  After embeddings + pre_layernorm: torch.Size([1, 577, 1024])
  Num transformer layers: 23
  select_layers: [-2]
  Final encoder output: torch.Size([1, 576, 1024])

============================================================
[LLaVA _process_image_input]
  Input type: pixel_values
  After vision encoder: torch.Size([576, 1024])

============================================================
[LLaVA MLP Projector]
  Input (from CLIP): torch.Size([576, 1024])
  After linear_1: torch.Size([576, 4096])
  After GELU + linear_2: torch.Size([576, 4096])
  Now compatible with LLM embedding space!
============================================================

  After projector: torch.Size([576, 4096])
  These embeddings will replace <image> placeholder tokens

============================================================
[MERGE: Visual embeddings into text embeddings]
  inputs_embeds (text): torch.Size([600, 4096])
  is_multimodal mask: torch.Size([600]), 576 placeholder positions
  multimodal_embeddings flattened: torch.Size([576, 4096])
  → 576 visual tokens will replace 576 placeholders
  After merge - visual embeddings injected at placeholder positions
  Result: torch.Size([600, 4096]) — ready for LLM forward pass
============================================================

Answer: The image shows a beautiful cherry blossom tree...
```

---

## Lab 3: Compare Different Encoders

After understanding LLaVA's CLIP encoder, trace other architectures to see how they differ.

### SigLIP (used by PaliGemma, LLaVA-OneVision)

File: `vllm/model_executor/models/siglip.py`

Key difference from CLIP: **no [CLS] token**.

Add the same patch embedding logging as Lab 1 to `SiglipVisionEmbeddings.forward()`. You'll see:
- Output: `(1, 576, 1152)` — 576 patches, no +1 for CLS
- SigLIP uses sigmoid loss instead of softmax (see `03_Semantic_Encoders.md`)

### Qwen2-VL Custom ViT (Dynamic Resolution)

File: `vllm/model_executor/models/qwen2_vl.py`

Key difference: **dynamic resolution with patch merging**.

Instrument `Qwen2VisionPatchEmbed.forward()` (line 447) and `Qwen2VisionPatchMerger.forward()` (line 476):

```python
# qwen2_vl.py, class Qwen2VisionPatchEmbed
def forward(self, pixel_values, grid_thw):
    # === LAB LOGGING START ===
    print(f"\n[Qwen2-VL Patch Embed]")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  grid_thw: {grid_thw}")  # (temporal, height_patches, width_patches)
    # === LAB LOGGING END ===
    ...
```

You'll see:
- Variable-sized inputs (not fixed 336x336)
- `grid_thw` encoding temporal/spatial dimensions
- Patch merging reduces token count after embedding

### InternVL (Dynamic Tiling)

File: `vllm/model_executor/models/internvl.py`

Key difference: **tile-based processing** — splits large images into multiple 448x448 tiles.

Instrument the processing info class to see how tiles are computed, then trace through the vision encoder to see multiple tiles processed.

---

## Lab 4: Trace the Multimodal Registry

**Goal**: Understand how vLLM knows which processor/encoder to use for which model.

Add logging to `MultiModalRegistry.create_processor()` in `vllm/multimodal/registry.py`:

```python
# registry.py, class MultiModalRegistry
def create_processor(self, model_config, tokenizer, ...):
    # === LAB LOGGING START ===
    model_cls = ...  # however the class is resolved
    print(f"\n[MultiModalRegistry.create_processor]")
    print(f"  Model: {model_config.model}")
    print(f"  Has _processor_factory: {hasattr(model_cls, '_processor_factory')}")
    # === LAB LOGGING END ===
    ...
```

This shows you the factory pattern: each VLM class is decorated with `@MULTIMODAL_REGISTRY.register_processor()`, which stores a processor factory, info factory, and dummy inputs builder.

---

## Lab 5: Trace the Prompt Replacement Pipeline

**Goal**: See how `<image>` in the text prompt gets replaced with the right number of visual tokens.

File: `vllm/multimodal/processing/processor.py`

The key concept: before the model runs, vLLM's input processor counts how many visual tokens the encoder will produce, then inserts that many placeholder token IDs into the prompt. This ensures the KV cache has the right number of slots.

From LLaVA's `forward()` docstring (line 674):

```
Original prompt: "USER: <image>\nWhat's the content?"
Tokenized:       [1, 3148, 1001, 29901, 29871, 32000, 29871, 13, ...]
                                                 ^^^^^
                                          single <image> token (32000)

After input processing (576 placeholder tokens inserted):
[1, 3148, 1001, 29901, 29871, 32000, 32000, ...(x576)..., 32000, 29871, 13, ...]

At forward time: these 576 positions are replaced by visual embeddings
via _merge_multimodal_embeddings()
```

Add logging to `BaseMultiModalProcessor.apply()` to trace this replacement.

---

## Concept Map: What You're Tracing

```
Lab 1: CLIP Encoder
  Image (336x336x3)
    │ Conv2d(kernel=14, stride=14)
    ▼
  Patch embeddings (576 x 1024)
    │ + [CLS] token + position embeddings
    ▼
  577 tokens (1024-dim)
    │ 23 transformer layers (self-attention + FFN)
    ▼
  577 tokens (1024-dim), select layer -2, drop [CLS]
    │
    ▼
  576 visual tokens (1024-dim)  ← output of vision encoder

Lab 2: LLaVA Pipeline
  576 tokens (1024-dim)  ← from CLIP
    │ Linear(1024 → 4096) + GELU + Linear(4096 → 4096)
    ▼
  576 tokens (4096-dim)  ← projected to LLM space

  Text: "USER: <image>\nWhat's in this?"
    │ Tokenizer → [text_tokens] with 576 placeholder positions
    ▼
  inputs_embeds (seq_len x 4096)  ← text embeddings

  MERGE: masked_scatter_(visual_embeds, at placeholder positions)
    │
    ▼
  Unified sequence (seq_len x 4096)  ← visual + text mixed
    │ LLM forward pass (Llama/Mistral)
    ▼
  Output tokens → "A cherry blossom tree..."
```

## Debugging Environment: vLLM vs HuggingFace Transformers

### The Problem with vLLM

vLLM's `LLM` class runs model execution in a background process/thread (`AsyncLLM` → `EngineCore`), making interactive debugging (pdb, breakpoints) difficult. CUDA graph compilation also obscures the forward pass.

### Option A: vLLM with `enforce_eager` (Logging-Based Tracing)

Disable CUDA graphs so every forward pass runs eagerly (your print statements work):

```python
llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    enforce_eager=True,       # disable CUDA graph capture
    max_num_seqs=1,           # one request at a time for clarity
    max_model_len=4096,
)
```

This works well for `print()`-based tracing (Labs 1-5 above). You'll see all your logging in the terminal. However, `breakpoint()`/pdb won't work well because the model runs in a subprocess.

### Option B: HuggingFace Transformers (Best for Interactive Debugging)

For interactive step-through debugging, use HuggingFace Transformers directly. Everything runs in a single thread in the main process — `breakpoint()` and pdb work perfectly.

```python
# trace_llava_hf.py — single-threaded, fully debuggable
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

# Prepare input
image = Image.open("path/to/image.jpg")  # or use any PIL image
prompt = "USER: <image>\nWhat do you see?\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"pixel_values shape: {inputs['pixel_values'].shape}")

# Step through the forward pass
with torch.no_grad():
    # 1. Vision encoder
    vision_outputs = model.vision_tower(inputs["pixel_values"])
    print(f"Vision encoder output: {vision_outputs.shape}")

    # 2. Feature selection (drop CLS, select layer)
    # In HF LLaVA, this happens inside the model
    image_features = vision_outputs  # simplified

    # 3. Projector
    projected = model.multi_modal_projector(image_features)
    print(f"After projector: {projected.shape}")

    # 4. Full forward (does merge + LLM internally)
    output = model.generate(**inputs, max_new_tokens=64)
    print(processor.decode(output[0], skip_special_tokens=True))
```

To debug interactively, add `breakpoint()` anywhere:

```python
# Inside transformers source code (e.g., modeling_llava.py)
def forward(self, ...):
    breakpoint()  # pdb drops in here — inspect all tensors
    ...
```

Find HF source location:
```bash
uv run python -c "import transformers; print(transformers.__file__)"
# Then edit: {transformers_path}/models/llava/modeling_llava.py
```

### Option C: HuggingFace Manual Step-Through (Maximum Control)

For the deepest understanding, call each component manually:

```python
# trace_step_by_step.py — call each VLM component individually
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    CLIPVisionModel,
    LlavaForConditionalGeneration,
)

model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

image = Image.open("path/to/image.jpg")
prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    # ===== STEP 1: Patch Embedding =====
    pixel_values = inputs["pixel_values"]
    print(f"\n=== Step 1: Patch Embedding ===")
    print(f"Raw image pixels: {pixel_values.shape}")  # (1, 3, 336, 336)

    vision_model = model.vision_tower.vision_model
    patch_embeds = vision_model.embeddings.patch_embedding(pixel_values)
    print(f"After Conv2d: {patch_embeds.shape}")       # (1, 1024, 24, 24)
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
    print(f"Flattened patches: {patch_embeds.shape}")  # (1, 576, 1024)

    # ===== STEP 2: Full Vision Encoder =====
    print(f"\n=== Step 2: Vision Transformer ===")
    vision_outputs = model.vision_tower(
        pixel_values,
        output_hidden_states=True,
    )
    # LLaVA uses second-to-last layer
    selected = vision_outputs.hidden_states[-2]
    print(f"Selected layer output: {selected.shape}")  # (1, 577, 1024)
    # Remove CLS token
    image_features = selected[:, 1:, :]
    print(f"After CLS removal: {image_features.shape}")  # (1, 576, 1024)

    # ===== STEP 3: MLP Projector =====
    print(f"\n=== Step 3: MLP Projector ===")
    projected = model.multi_modal_projector(image_features)
    print(f"Projected features: {projected.shape}")    # (1, 576, 4096)
    print(f"LLM hidden size: {model.config.text_config.hidden_size}")

    # ===== STEP 4: Text Embedding =====
    print(f"\n=== Step 4: Text Embeddings ===")
    input_ids = inputs["input_ids"]
    print(f"Token IDs: {input_ids.shape}")
    image_token_id = model.config.image_token_index
    is_image = (input_ids == image_token_id)
    print(f"Image token positions: {is_image.sum().item()} tokens at ID={image_token_id}")

    text_embeds = model.get_input_embeddings()(input_ids)
    print(f"Text embeddings: {text_embeds.shape}")

    # ===== STEP 5: Merge (conceptual) =====
    print(f"\n=== Step 5: Merge ===")
    print(f"Text sequence length: {text_embeds.shape[1]}")
    print(f"Visual tokens to insert: {projected.shape[1]}")
    print(f"After merge, total sequence: ~{text_embeds.shape[1]} tokens")
    print(f"These go into the Llama decoder for autoregressive generation")

    # ===== STEP 6: Generate =====
    print(f"\n=== Step 6: Generate ===")
    output = model.generate(**inputs, max_new_tokens=64)
    answer = processor.decode(output[0], skip_special_tokens=True)
    print(f"Output: {answer}")
```

### Which to Use?

| Method | Debugging | Threading | Best For |
|--------|-----------|-----------|----------|
| vLLM + `enforce_eager` | print() only | Multi-process | Production-like tracing, Labs 1-5 |
| HF Transformers `.generate()` | breakpoint() works | Single thread | Interactive exploration |
| HF manual step-through | Full control | Single thread | Deep understanding of each component |

**Recommended path**: Start with **Option C** (HF manual step-through) to build intuition about shapes and data flow, then use **Option A** (vLLM + logging) to see how the same concepts are implemented in a production serving system.

## Tips for Effective Code Tracing

1. **Start with shapes** — always print `.shape` of tensors. Shape changes tell you exactly what each operation does.

2. **Use `VLLM_LOGGING_LEVEL=DEBUG`** for built-in vLLM logging:
   ```bash
   VLLM_LOGGING_LEVEL=DEBUG uv run python trace_llava.py 2>&1 | head -200
   ```

3. **One model at a time** — start with LLaVA (simplest), then move to Qwen2-VL (dynamic resolution), then InternVL (tiling).

4. **Use `breakpoint()`** for interactive exploration:
   ```python
   # Add to any function you want to inspect interactively
   breakpoint()  # drops into pdb — use p tensor.shape, p tensor[:5], etc.
   ```

5. **Check weight shapes** to understand architecture without running inference:
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("llava-hf/llava-1.5-7b-hf")
   for name, param in model.named_parameters():
       if "vision" in name or "projector" in name:
           print(f"{name}: {param.shape}")
   ```

6. **Remove logging when done** — use `git diff` to review your changes, then `git checkout -- .` to revert.

## Key Files Quick Reference

| What | File | Line | Function |
|------|------|------|----------|
| Patch embedding (CLIP) | `models/clip.py` | 340 | `CLIPVisionEmbeddings.forward()` |
| Vision transformer (CLIP) | `models/clip.py` | 697 | `CLIPVisionTransformer.forward()` |
| MLP projector (LLaVA) | `models/llava.py` | 157 | `LlavaMultiModalProjector.forward()` |
| Image processing (LLaVA) | `models/llava.py` | 641 | `_process_image_input()` |
| Embedding merge | `models/utils.py` | 443 | `_merge_multimodal_embeddings()` |
| LLaVA forward pass | `models/llava.py` | 666 | `LlavaForConditionalGeneration.forward()` |
| Multimodal registry | `multimodal/registry.py` | 98 | `MultiModalRegistry` |
| Prompt replacement | `multimodal/processing/processor.py` | — | `BaseMultiModalProcessor.apply()` |
| Resampler (Qwen-VL) | `layers/resampler.py` | — | `Resampler2` |
| Patch embed (Qwen2-VL) | `models/qwen2_vl.py` | 447 | `Qwen2VisionPatchEmbed.forward()` |
| Patch merger (Qwen2-VL) | `models/qwen2_vl.py` | 476 | `Qwen2VisionPatchMerger.forward()` |
| SigLIP encoder | `models/siglip.py` | — | `SiglipVisionEmbeddings.forward()` |

## Related

- [ViT Architecture](02_ViT.md) — the theory behind what CLIP implements
- [Semantic Encoders](03_Semantic_Encoders.md) — CLIP vs SigLIP vs DINOv2 training objectives
- [VLM Architecture](../vision_language/01_Architecture.md) — fusion strategies + full vLLM code reference
