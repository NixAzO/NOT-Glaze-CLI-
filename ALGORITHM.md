# How glz Works - Algorithm Breakdown

## Overview

`glz` protects artwork from AI style mimicry by adding invisible perturbations that confuse AI models while remaining imperceptible to humans.

## The Problem

AI models like Stable Diffusion and Midjourney can learn an artist's style from just a few images. They extract "style features" using vision models like CLIP, then replicate that style.

## The Solution

We exploit how AI "sees" images differently than humans:

```
Human Vision: Focuses on shapes, colors, composition
AI Vision:    Extracts high-dimensional feature vectors
```

By adding tiny pixel changes that maximally disrupt AI feature extraction, we can make AI see a completely different style while humans see the same image.

## Algorithm Steps

### 1. Load CLIP Model
```python
model = clip.load("ViT-B/32")
```
CLIP (Contrastive Language-Image Pre-training) is the backbone of most AI art generators. If we fool CLIP, we fool them all.

### 2. Extract Original Features
```python
original_features = model.encode_image(image)
```
This 512-dimensional vector represents how AI perceives the image's style.

### 3. Initialize Perturbation
```python
delta = torch.zeros_like(image, requires_grad=True)
```
Start with zero perturbation, then optimize.

### 4. Adversarial Optimization
```python
for step in range(steps):
    perturbed = image + delta
    features = model.encode_image(perturbed)
    
    # Maximize distance from original
    loss = cosine_similarity(features, original_features)
    
    loss.backward()
    optimizer.step()
    
    # Keep perturbation small (invisible)
    delta = clamp(delta, -epsilon, epsilon)
```

The key insight: we're doing **gradient ascent** on the feature distance while constraining the pixel distance.

### 5. Apply to Full Resolution
```python
delta_fullres = interpolate(delta, original_size)
protected = original + delta_fullres
```

## Visual Explanation

```
┌─────────────────────────────────────────────────────────┐
│                    ORIGINAL IMAGE                        │
│                         │                                │
│                         ▼                                │
│              ┌─────────────────────┐                     │
│              │    CLIP Encoder     │                     │
│              └─────────────────────┘                     │
│                         │                                │
│                         ▼                                │
│              [0.23, -0.45, 0.12, ...]  ← Style Vector    │
│                                                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   PROTECTED IMAGE                        │
│            (looks identical to human)                    │
│                         │                                │
│                         ▼                                │
│              ┌─────────────────────┐                     │
│              │    CLIP Encoder     │                     │
│              └─────────────────────┘                     │
│                         │                                │
│                         ▼                                │
│              [-0.67, 0.89, -0.34, ...]  ← DIFFERENT!     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Parameters

| Parameter | Description | Trade-off |
|-----------|-------------|-----------|
| `epsilon` | Max pixel change (default: 8/255) | Higher = more protection, more visible |
| `steps` | Optimization iterations (default: 50) | More = better protection, slower |
| `lr` | Learning rate (default: 0.01) | Higher = faster but less stable |

## Intensity Presets

- **Low**: `epsilon=4/255, steps=30` - Minimal changes, basic protection
- **Medium**: `epsilon=8/255, steps=50` - Balanced (recommended)
- **High**: `epsilon=16/255, steps=100` - Maximum protection, may be slightly visible

## Limitations

1. **Not 100% foolproof** - Determined attackers with access to your original work may still extract style
2. **Works best against CLIP-based models** - May be less effective against future architectures
3. **Lossy formats** - JPEG compression may reduce effectiveness; use PNG when possible

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - OpenAI's vision-language model
- [Glaze Paper](https://arxiv.org/abs/2302.04222) - University of Chicago's original research
- [Adversarial Examples](https://arxiv.org/abs/1412.6572) - Foundational work on fooling neural networks
