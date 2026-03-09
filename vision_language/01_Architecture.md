# VLM Architecture Patterns

> Parent: [Vision-Language Models](00_Vision_Language.md)

## Overview

The central question in VLM design is **how to fuse visual and language information**. Different architectures make different trade-offs between efficiency, capability, and complexity. This page covers the main design patterns used in modern VLMs.

## Fusion Strategies

### Late Fusion (Projection-based)

Most common approach. Process image and text independently, then combine:

```
Image → Visual Encoder → [visual tokens] ──→ Projector ──→ ┐
                                                            ├→ LLM → output
Text  → Tokenizer     → [text tokens]   ────────────────→ ┘
```

The visual tokens are projected into the LLM's embedding space and concatenated with text tokens. The LLM processes them together via self-attention.

Used by: LLaVA, Qwen-VL, InternVL, PaliGemma

**Advantages**: Simple, leverages pretrained LLMs and encoders
**Disadvantages**: No early interaction between modalities

### Cross-Attention Fusion

Insert cross-attention layers into the LLM that attend to visual features:

```
LLM Layer N:    Self-Attention → Cross-Attention → FFN
                     ↑                  ↑
                text tokens      visual features (from encoder)
```

Used by: Flamingo, Llama 3.2 Vision

**Advantages**: More fine-grained visual grounding
**Disadvantages**: Modifies the LLM architecture, harder to leverage pretrained models

### Early Fusion

Process raw image patches directly as input tokens (no separate encoder):

```
Image patches → Linear embedding → ┐
                                    ├→ Transformer → output
Text tokens   → Embedding       → ┘

(Interleaved from the start)
```

Used by: Fuyu, some research models

**Advantages**: End-to-end, no frozen encoder limitations
**Disadvantages**: Very expensive (many tokens), needs training from scratch

## Visual Token Projectors

The projector transforms visual encoder output into the LLM's input space:

### Linear Projection

```python
# Simplest approach — used in LLaVA v1
projector = nn.Linear(vision_dim, llm_dim)  # e.g., 1024 → 4096
```

### MLP Projector

```python
# Used in LLaVA v1.5 — better performance
projector = nn.Sequential(
    nn.Linear(vision_dim, llm_dim),
    nn.GELU(),
    nn.Linear(llm_dim, llm_dim),
)
```

### C-Abstractor / Resampler

Reduce number of visual tokens before feeding to LLM:

```
Visual encoder output: 576 tokens (24x24 patches for ViT-L/14 @ 336px)
                         │
                    Resampler/Q-Former
                         │
                    64 or 128 query tokens → LLM

Reduces compute significantly (576 → 64 tokens)
```

Used by: BLIP-2 (Q-Former), Qwen-VL (cross-attention resampler)

### Token Compression Comparison

| Approach | Visual Tokens | Quality | LLM Compute |
|----------|--------------|---------|-------------|
| No compression | 576+ | Best | High |
| MLP (LLaVA) | 576 | Very good | High |
| Resampler (BLIP-2) | 32-64 | Good | Low |
| Avg pooling 2x2 | 144 | Good | Medium |
| Dynamic (InternVL) | 256-1024 | Very good | Adaptive |

## Resolution Handling

Higher resolution → more visual detail but more tokens:

### Fixed Resolution

Resize all images to fixed size (e.g., 336x336):
- Simple but loses detail for high-res images
- 336px with ViT-L/14 → 576 tokens

### Dynamic Resolution (Tile-based)

Split large images into tiles, process each separately:

```
High-res image (1344x672)
       │
       ▼
┌──┬──┬──┬──┐
│T1│T2│T3│T4│  4 tiles (336x336 each)
└──┴──┴──┴──┘
       +
  [Thumbnail]    1 downscaled overview (336x336)
       │
       ▼
  5 × 576 = 2880 tokens → LLM
```

Used by: LLaVA-NeXT, InternVL 2, Qwen-VL 2

**Trade-off**: More tiles → better detail but significantly more tokens → higher cost.

## Multi-Image & Video Support

### Multi-Image

Concatenate visual tokens from multiple images:
```
Image 1 → encoder → [vis_tokens_1] ──→ ┐
Image 2 → encoder → [vis_tokens_2] ──→ ├→ LLM
Image 3 → encoder → [vis_tokens_3] ──→ ┘
```

### Video

Sample frames, process each as an image:
```
Video → sample N frames → encode each → concatenate → LLM

Challenge: N frames × 576 tokens = thousands of tokens
Solutions: temporal pooling, frame sampling, token compression
```

## Architecture Decision Tree

```
Need maximum understanding quality?
  → Late fusion with MLP projector (LLaVA-style)
  → High-res: dynamic tiling

Need generation + understanding?
  → Dual encoder (Janus, BAGEL) or unified (Show-o)
  → See: Unified Models

Need efficiency / fast inference?
  → Resampler / token compression
  → Fewer tiles

Need to preserve pretrained LLM?
  → Late fusion (don't modify LLM architecture)
  → Cross-attention requires modifying LLM layers
```

## Appendix: vLLM Code Reference

The [vLLM](https://github.com/vllm-project/vllm) codebase implements every architectural pattern described above. This section maps concepts to concrete code paths for hands-on study.

Source: `/root/vllm/vllm/`

### End-to-End Processing Pipeline

```
User Input: "Describe this image" + [photo.jpg]
    │
    ▼
[1] INPUT PARSING (multimodal/parse.py)
    ImageProcessorItems converts PIL images to processor-ready format
    │
    ▼
[2] PROCESSOR LOOKUP (multimodal/registry.py)
    MULTIMODAL_REGISTRY.create_processor() → model-specific processor
    │
    ▼
[3] PLACEHOLDER REPLACEMENT (multimodal/processing/processor.py)
    PromptReplacement: "<image>" token → N visual token IDs
    │
    ▼
[4] VISUAL ENCODING (model_executor/models/clip.py, siglip.py, etc.)
    Raw pixels → patch embeddings → ViT transformer → visual features
    │
    ▼
[5] PROJECTION (model-specific projector: MLP, resampler, etc.)
    Visual features (vision_dim) → LLM embedding space (text_dim)
    │
    ▼
[6] EMBEDDING MERGE (model_executor/models/utils.py)
    _merge_multimodal_embeddings(): masked_scatter visual into text embeddings
    │
    ▼
[7] LLM FORWARD PASS
    Merged embeddings → self-attention → output tokens
```

### Core Multimodal Framework

#### `SupportsMultiModal` Protocol — The Interface Every VLM Implements

All VLMs in vLLM implement this protocol (`model_executor/models/interfaces.py:88`):

```python
class SupportsMultiModal(Protocol):
    supports_multimodal: ClassVar[Literal[True]] = True

    # Set by @MULTIMODAL_REGISTRY.register_processor()
    _processor_factory: ClassVar[_ProcessorFactories]

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Get placeholder text for the i-th modality item (e.g. '<image>')"""
        ...

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        """Produce visual embeddings to merge into text embeddings"""
        ...

    def get_language_model(self) -> VllmModel:
        """Return the underlying LLM component"""
        ...
```

#### `_merge_multimodal_embeddings()` — Where Visual Meets Language

The central fusion function (`model_executor/models/utils.py:443`):

```python
def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,       # (B, L, D) text embeddings
    multimodal_embeddings: NestedTensors, # visual features from projector
    is_multimodal: torch.Tensor,       # boolean mask of placeholder positions
) -> torch.Tensor:
    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    # In-place replacement at placeholder positions
    inputs_embeds.masked_scatter_(
        is_multimodal.unsqueeze(-1),
        mm_embeds_flat.to(dtype=inputs_embeds.dtype)
    )
    return inputs_embeds
```

Every VLM calls this function — it's the concrete implementation of late fusion (visual tokens projected into text space and inserted at `<image>` placeholder positions).

#### Multimodal Registry

```python
# multimodal/registry.py — singleton manages all VLM processors
MULTIMODAL_REGISTRY = MultiModalRegistry()

# Each VLM registers via decorator:
@MULTIMODAL_REGISTRY.register_processor(
    processor=LlavaMultiModalProcessor,
    info=LlavaProcessingInfo,
    dummy_inputs=LlavaDummyInputsBuilder
)
class LlavaForConditionalGeneration(nn.Module, SupportsMultiModal):
    ...
```

### Fusion Strategy Implementations

#### Late Fusion / MLP Projector (LLaVA)

`model_executor/models/llava.py:129` — the classic 2-layer MLP:

```python
class LlavaMultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_size, text_hidden_size, projector_hidden_act, ...):
        self.linear_1 = ColumnParallelLinear(vision_hidden_size, text_hidden_size, ...)
        self.act = get_act_fn(projector_hidden_act)    # GELU
        self.linear_2 = RowParallelLinear(text_hidden_size, text_hidden_size, ...)

    def forward(self, image_features):
        hidden_states, _ = self.linear_1(image_features)  # vision_dim → text_dim
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)    # text_dim → text_dim
        return hidden_states
```

Note the use of `ColumnParallelLinear`/`RowParallelLinear` — even the projector supports tensor parallelism.

#### Perceiver Resampler (Qwen-VL, MiniCPM-V)

`model_executor/layers/resampler.py` — reduces visual token count via cross-attention with learnable queries:

```
576 visual tokens from ViT
        │
        ▼
┌───────────────────┐
│  Resampler2        │  Learnable queries (64-128) attend to visual tokens
│  - 2D sincos pos  │  via cross-attention layers
│  - Cross-Attention │
│  - LayerNorm      │
└───────────────────┘
        │
        ▼
64-128 compressed tokens → LLM
```

`MiniCPM-V` extends this with `Resampler2_5` and `Resampler4_5` (`model_executor/models/minicpmv.py:45`) for multi-resolution pooling strategies.

#### Cross-Attention Fusion (Aria)

`model_executor/models/aria.py` — inserts cross-attention layers into the LLM:

```
LLM Layer N:  Self-Attention → AriaCrossAttention → FFN
                   ↑                    ↑
              text tokens         visual features
```

### Vision Encoder Implementations

#### `VisionEncoderInfo` — Abstract Base (`model_executor/models/vision.py:32`)

```python
class VisionEncoderInfo(ABC, Generic[_C]):
    def get_num_image_tokens(self, *, image_width, image_height) -> int: ...
    def get_image_size(self) -> int: ...
    def get_patch_size(self) -> int: ...
    def get_patch_grid_length(self) -> int: ...
```

`get_vision_encoder_info()` (line 65) dispatches to the correct encoder based on config type: `CLIPVisionConfig` → `CLIPEncoderInfo`, `SiglipVisionConfig` → `SiglipEncoderInfo`, `PixtralVisionConfig` → `PixtralHFEncoderInfo`.

#### Encoder Implementations

| Encoder | File | Info Class | Token Formula |
|---------|------|-----------|---------------|
| CLIP | `models/clip.py` | `CLIPEncoderInfo` | `patch_grid_length^2 + 1` (CLS token) |
| SigLIP | `models/siglip.py` | `SiglipEncoderInfo` | `patch_grid_length^2` (no CLS) |
| Pixtral | `models/pixtral.py` | `PixtralHFEncoderInfo` | Variable (per-image grid) |
| InternViT | `models/intern_vit.py` | (custom) | Dynamic with bilinear interpolation |
| Qwen2-VL ViT | `models/qwen2_vl.py:525` | (custom) | Adaptive multi-resolution |

### VLM Model Implementations

Each model file is a complete implementation of a VLM. Study these to see how the architecture patterns compose in practice:

| Model | File | Visual Encoder | Projector | Fusion | Resolution |
|-------|------|---------------|-----------|--------|------------|
| **LLaVA** | `models/llava.py` | CLIP/SigLIP | 2-layer MLP (line 129) | Late fusion | Fixed 336px |
| **LLaVA-NeXT** | `models/llava_next.py` | CLIP/SigLIP | MLP | Late fusion | Dynamic tiling |
| **LLaVA-OneVision** | `models/llava_onevision.py` | SigLIP | MLP | Late fusion | Dynamic |
| **Qwen2-VL** | `models/qwen2_vl.py` | Custom ViT (line 525) | Linear | Late fusion | Dynamic (patch merger, line 476) |
| **Qwen3-VL** | `models/qwen3_vl.py` | Custom ViT | Linear | Late fusion | Dynamic |
| **InternVL** | `models/internvl.py` | InternViT (`intern_vit.py`) | Linear | Late fusion | Dynamic tiling |
| **Pixtral** | `models/pixtral.py` | Pixtral ViT | Adaptive | Late fusion | Variable per-image |
| **MiniCPM-V** | `models/minicpmv.py` | Idefics2 ViT | Resampler2_5/4_5 (line 45) | Late fusion + resampler | Multi-resolution slices |
| **Phi-3-Vision** | `models/phi3v.py` | CLIP | MLP | Late fusion | Fixed |
| **Aria** | `models/aria.py` | Custom | Cross-attention | Cross-attention fusion | Variable |
| **Molmo** | `models/molmo.py` | CLIP | MLP | Late fusion | Fixed |
| **DeepSeek-VL2** | `models/deepseek_vl2.py` | Custom | MLP | Late fusion | Dynamic |

### Recommended Reading Order

For studying how VLM architectures work in code:

1. **Start simple — LLaVA** (`models/llava.py`)
   - Clean late fusion: CLIP encoder → MLP projector → merge into LLM embeddings
   - Follow `embed_multimodal()` → `_process_image_input()` → projector → `_merge_multimodal_embeddings()`

2. **Add dynamic resolution — LLaVA-NeXT** (`models/llava_next.py`)
   - Same base as LLaVA but adds tile-based high-resolution handling
   - See how tiles are generated and token counts scale

3. **Custom ViT — Qwen2-VL** (`models/qwen2_vl.py`)
   - Custom vision transformer with `Qwen2VisionPatchEmbed` (line 447), `Qwen2VisionPatchMerger` (line 476), `Qwen2VisionBlock` (line 394)
   - Shows how to build a ViT from scratch inside a VLM

4. **Token compression — MiniCPM-V** (`models/minicpmv.py`)
   - Resampler reduces 576+ tokens to 64-128
   - Compare `Resampler2_5` vs `Resampler4_5` for different compression strategies

5. **Cross-attention — Aria** (`models/aria.py`)
   - Different fusion paradigm: visual features attended via cross-attention layers inserted into LLM

### Key Code Paths

```
vllm/
├── multimodal/                         # Multimodal framework
│   ├── __init__.py                     # MULTIMODAL_REGISTRY singleton
│   ├── registry.py                     # MultiModalRegistry — processor lookup
│   ├── inputs.py                       # ImageItem, VideoItem, AudioItem types
│   ├── parse.py                        # Raw input → processor-ready format
│   ├── cache.py                        # Preprocessed embedding cache
│   └── processing/
│       ├── processor.py                # BaseMultiModalProcessor — template method
│       └── context.py                  # BaseProcessingInfo — config/tokenizer access
├── model_executor/
│   ├── models/
│   │   ├── interfaces.py               # SupportsMultiModal protocol (line 88)
│   │   ├── utils.py                    # _merge_multimodal_embeddings() (line 443)
│   │   ├── vision.py                   # VisionEncoderInfo base class (line 32)
│   │   ├── clip.py                     # CLIP encoder + CLIPEncoderInfo
│   │   ├── siglip.py                   # SigLIP encoder + SiglipEncoderInfo
│   │   ├── intern_vit.py              # InternVL vision transformer
│   │   ├── llava.py                    # LLaVA — reference VLM implementation
│   │   ├── llava_next.py              # LLaVA-NeXT — dynamic resolution
│   │   ├── qwen2_vl.py               # Qwen2-VL — custom ViT + patch merger
│   │   ├── qwen3_vl.py               # Qwen3-VL
│   │   ├── internvl.py               # InternVL
│   │   ├── pixtral.py                # Pixtral — variable image sizes
│   │   ├── minicpmv.py               # MiniCPM-V — resampler compression
│   │   ├── phi3v.py                   # Phi-3-Vision
│   │   └── aria.py                    # Aria — cross-attention fusion
│   └── layers/
│       └── resampler.py               # Perceiver resampler (Qwen-VL, MiniCPM-V)
└── config/
    └── multimodal.py                  # MultiModalConfig
```

## Related

- [Models](02_Models.md) — specific VLM implementations using these patterns
- [Unified Models](03_Unified_Models.md) — architectures that also generate images
- [Training](04_Training.md) — how these architectures are trained
- [Visual Encoders](../visual_encoder/00_Visual_Encoder.md) — encoder choices for VLMs
