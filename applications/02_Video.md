# Video Generation & Understanding

> Parent: [Applications](00_Applications.md)

## Overview

Video models extend image-based architectures to handle the temporal dimension. Video generation creates new video content from text or images, while video understanding analyzes existing video. Both present unique challenges due to the massive data volume and temporal consistency requirements.

## Video Generation

### Architecture Approaches

```
Approach 1: Extend Image DiT to 3D (Sora-style)
  Video → 3D patches (space + time) → Transformer → Denoised video
  - Spacetime attention over all patches
  - Handles variable duration and resolution

Approach 2: Temporal layers added to image model
  Image model backbone + temporal attention/conv layers
  - Can leverage pretrained image models
  - Used by AnimateDiff, SVD

Approach 3: Autoregressive frame generation
  Generate frame-by-frame, conditioning on previous frames
  - Good temporal coherence
  - Slow for long videos
```

### Key Systems

| System | Approach | Duration | Resolution | Open |
|--------|----------|----------|------------|------|
| Sora | 3D DiT | Minutes | Up to 1080p | No |
| Kling | DiT-based | ~2 min | 1080p | API |
| Runway Gen-3 | Diffusion | ~10s | 1080p | API |
| Stable Video Diffusion | Temporal layers on SD | ~4s | 576p | Yes |
| CogVideo | 3D transformer | ~6s | 720p | Yes |
| Open-Sora | 3D DiT (Sora reproduction) | Variable | Variable | Yes |

### Challenges

```
Data volume:    1 second of 1080p 30fps = 1920×1080×3×30 ≈ 186M values
                vs single image:       1920×1080×3        ≈  6.2M values
                → 30x more data per second

Temporal consistency:  Objects must maintain identity across frames
                      Camera motion must be smooth
                      Physics must be plausible

Compute:        Training Sora-class models requires thousands of GPUs
                Inference: minutes per video clip
```

## Video Understanding

### VLMs for Video

Most video understanding uses frame sampling + VLM:

```
Video (30fps, 10s = 300 frames)
       │
  Sample N frames (e.g., 8-32)
       │
  Encode each frame with ViT
       │
  Concatenate/pool visual tokens
       │
  Feed to LLM with text query
       │
  Answer: "The person is cooking pasta"
```

### Token Efficiency Challenge

```
8 frames × 576 tokens/frame = 4,608 visual tokens
32 frames × 576 tokens/frame = 18,432 visual tokens (!)

Solutions:
- Temporal pooling (merge tokens across frames)
- Sparse frame sampling
- Token compression (Q-Former / Resampler)
- SlowFast: few high-res + many low-res frames
```

### Video Understanding Models

| Model | Approach | Strengths |
|-------|----------|-----------|
| GPT-4o | Frame sampling + LLM | Best overall quality |
| Gemini 1.5 | Native long context | Handles long videos |
| Qwen-VL 2.5 | Frame sampling | Good open-source option |
| VideoLLaVA | Video-specific training | Specialized for video |
| LLaVA-Video | Dynamic frame selection | Efficient |

## Related

- [DiT](../diffusion/04_DiT.md) — backbone architecture for video generation
- [Image Generation](01_Image_Generation.md) — foundation that video extends
- [VLM Models](../vision_language/02_Models.md) — models used for video understanding
- [Multimodal Inference](../../AI_Infra/inference/07_Multimodal_Inference.md) — serving video models
