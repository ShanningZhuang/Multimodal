# Robotics & Embodied AI

> Parent: [Applications](00_Applications.md)

## Overview

Multimodal models are transforming robotics by enabling robots to understand visual scenes and generate actions. Key innovations include Vision-Language-Action (VLA) models that combine perception with action, and diffusion-based policies that generate smooth, multimodal action trajectories.

## Vision-Language-Action (VLA) Models

VLAs extend VLMs to output robot actions instead of (or in addition to) text:

```
┌──────────────┐    ┌──────────┐    ┌──────────────┐
│ Camera image │──→ │  Visual  │──→ │              │──→ Robot actions
│ + Robot state│    │ Encoder  │    │   LLM/VLM    │    (x, y, z, gripper)
│              │    │(ViT/CLIP)│    │              │
│ "Pick up the │──→ │          │──→ │              │
│  red block"  │    └──────────┘    └──────────────┘
└──────────────┘
```

### Key Models

| Model | Base | Action Output | Key Innovation |
|-------|------|---------------|----------------|
| RT-2 (Google) | PaLI-X VLM | Tokenized actions | Actions as text tokens |
| OpenVLA | Llama + SigLIP | Tokenized actions | Open-source VLA |
| Octo | Transformer | Continuous actions | Diffusion action head |
| π₀ (Physical Intelligence) | VLM + DiT | Flow matching | Scalable, general-purpose |

### RT-2: Actions as Language

```
Input:  [image] "pick up the green block"
Output: "1 128 91 241 5 101 127"  ← discretized action tokens
         ↑  ↑   ↑  ↑   ↑  ↑   ↑
         x  y   z  rx  ry rz grip

Treat robot actions as text tokens → leverage LLM training
```

## Diffusion Policy

Use diffusion models to generate action trajectories:

```
Observation (images + state)
       │
       ▼
┌────────────────────────────────┐
│  Diffusion Policy               │
│                                │
│  1. Encode observation         │
│  2. Start from noise actions   │
│  3. Iteratively denoise        │
│  4. Output: action sequence    │
│     [a₁, a₂, ..., a_H]        │
│     (H-step action horizon)    │
└────────────────────────────────┘
       │
       ▼
  Execute actions on robot
```

### Why Diffusion for Actions?

```
Problem with regression (MSE loss):
  Multiple valid actions → model averages them → bad action

  Valid action 1: go left
  Valid action 2: go right
  MSE average:   go straight into wall!

Diffusion handles multimodal distributions:
  Can represent multiple valid action modes
  Samples one coherent trajectory
```

### DiT for Robotics (π₀)

```
Observation → Visual encoder → ┐
                                ├→ DiT backbone → Denoised actions
Noisy actions → ───────────── ┘
                                    ↑
                           Timestep embedding

Uses flow matching for fast inference (~10 steps)
Real-time control at 5-10 Hz
```

## Simulation to Real Transfer (Sim2Real)

```
1. Train in simulation (fast, safe, unlimited data)
   ├── Physics engine (MuJoCo, Isaac Sim)
   ├── Domain randomization (vary textures, lighting, physics)
   └── Millions of episodes

2. Transfer to real robot
   ├── Visual gap: simulation looks different → domain randomization helps
   ├── Physics gap: sim physics imperfect → careful sim tuning
   └── Fine-tune on small real-world dataset
```

## Key Challenges

| Challenge | Description | Current Solutions |
|-----------|-------------|-------------------|
| Data scarcity | Real robot data is expensive to collect | Sim2Real, teleoperation, scaling laws |
| Real-time control | Must generate actions fast enough | Fast diffusion samplers, action chunking |
| Generalization | Handle novel objects/tasks | Foundation models, language conditioning |
| Safety | Robot actions affect physical world | Conservative policies, human oversight |
| Multimodal input | Combine vision, touch, proprioception | Multi-encoder architectures |

## The Vision

```
Current: Task-specific robot → trained on "pick up cup" → can only pick up cups
         │
         ▼
VLA:     General robot brain → understands language + vision → any task
         "Pick up the cup" ✓
         "Open the drawer" ✓
         "Sort the laundry" ✓ (with enough training data)
```

## Related

- [DiT](../diffusion/04_DiT.md) — backbone architecture for diffusion policy
- [Diffusion Basics](../diffusion/01_Diffusion_Basics.md) — how diffusion generates actions
- [VLM Architecture](../vision_language/01_Architecture.md) — perception backbone for VLAs
- [Semantic Encoders](../visual_encoder/03_Semantic_Encoders.md) — visual representation for robotics
- [AI_Infra: Agent Infrastructure](../../AI_Infra/agent/00_Agent.md) — related agent systems
