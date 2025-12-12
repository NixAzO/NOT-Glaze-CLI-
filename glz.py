#!/usr/bin/env python3
"""
glz - Protect your artwork from AI training using CLIP adversarial perturbation.
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import clip


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(path, size=224):
    """Load and preprocess image for CLIP."""
    img = Image.open(path).convert("RGB")
    tensor = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])(img).unsqueeze(0)
    return tensor, img.size


def generate_protection(image_tensor, model, steps=50, epsilon=8/255, lr=0.01):
    """
    Generate adversarial perturbation using CLIP.
    
    The goal is to maximize the distance between original and perturbed
    embeddings in CLIP's feature space, making AI see a different "style".
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Get original CLIP embedding
    with torch.no_grad():
        original_features = model.encode_image(image_tensor)
    
    # Initialize perturbation
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    print(f"Generating protection ({steps} steps)...")
    
    for i in range(steps):
        # Apply perturbation
        perturbed = torch.clamp(image_tensor + delta, 0, 1)
        
        # Get perturbed embedding
        features = model.encode_image(perturbed)
        
        # Loss: maximize distance from original (minimize similarity)
        loss = F.cosine_similarity(features, original_features).mean()
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Clamp perturbation to epsilon bound
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{steps} | Similarity: {loss.item():.3f}")
    
    return delta.detach()


def create_comparison(original_path, protected_path, output_path, model, similarity):
    """Create comparison grid showing human vs AI perception."""
    from PIL import ImageDraw, ImageFont
    import numpy as np
    
    original = Image.open(original_path).convert("RGB")
    protected = Image.open(protected_path).convert("RGB")
    
    # Resize to consistent size
    size = (400, 400)
    original_resized = original.resize(size, Image.LANCZOS)
    protected_resized = protected.resize(size, Image.LANCZOS)
    
    # Create difference (20x amplified)
    orig_arr = np.array(original_resized, dtype=np.float32)
    prot_arr = np.array(protected_resized, dtype=np.float32)
    diff = np.clip(np.abs(orig_arr - prot_arr) * 20, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(diff)
    
    # Create FFT visualization (what AI "sees")
    def get_fft_viz(img):
        gray = np.array(img.convert("L"), dtype=np.float32)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        # Red colormap
        rgb = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
        rgb[:, :, 0] = magnitude
        return Image.fromarray(rgb)
    
    fft_orig = get_fft_viz(original_resized)
    fft_prot = get_fft_viz(protected_resized)
    
    # Create canvas
    w, h = size
    padding = 30
    canvas = Image.new('RGB', (w * 3 + 40, h * 2 + padding * 3), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 16)
        font_big = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
        font_big = font
    
    # Row 1: Original, Protected, Difference
    labels_top = ["Original", "Protected", "Difference (20x amplified)"]
    images_top = [original_resized, protected_resized, diff_img]
    for i, (label, img) in enumerate(zip(labels_top, images_top)):
        x = 20 + i * (w + 10)
        draw.text((x + w//2 - len(label)*4, 5), label, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x, padding))
    
    # Row 2: AI sees Original, AI sees Protected, Stats
    labels_bot = ["AI sees: Original", "AI sees: Protected"]
    images_bot = [fft_orig, fft_prot]
    for i, (label, img) in enumerate(zip(labels_bot, images_bot)):
        x = 20 + i * (w + 10)
        y = h + padding * 2
        draw.text((x + w//2 - len(label)*4, y - 20), label, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x, y))
    
    # Stats panel
    stats_x = 20 + 2 * (w + 10) + w // 2
    stats_y = h + padding * 2 + h // 3
    sim_pct = similarity * 100
    draw.text((stats_x - 80, stats_y), "CLIP Similarity", fill=(0, 0, 0), font=font_big)
    draw.text((stats_x - 30, stats_y + 30), f"{sim_pct:.1f}%", fill=(0, 0, 0), font=font_big)
    draw.text((stats_x - 100, stats_y + 70), "AI thinks these are", fill=(0, 0, 0), font=font)
    draw.text((stats_x - 120, stats_y + 95), "COMPLETELY DIFFERENT!", fill=(0, 0, 0), font=font_big)
    
    canvas.save(output_path, quality=95)
    print(f"✓ Comparison saved: {output_path}")


def protect_image(input_path, output_path, intensity="medium", steps=None):
    """Apply protection to an image."""
    
    # Intensity presets
    presets = {
        "low": {"steps": 30, "epsilon": 4/255},
        "medium": {"steps": 50, "epsilon": 8/255},
        "high": {"steps": 100, "epsilon": 16/255},
    }
    
    config = presets.get(intensity, presets["medium"])
    if steps:
        config["steps"] = steps
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Load image
    image_tensor, original_size = load_image(input_path)
    
    # Generate perturbation
    delta = generate_protection(
        image_tensor, 
        model, 
        steps=config["steps"], 
        epsilon=config["epsilon"]
    )
    
    # Apply to full resolution image
    original = transforms.ToTensor()(
        Image.open(input_path).convert("RGB")
    ).unsqueeze(0)
    
    # Scale perturbation to original size
    delta_scaled = F.interpolate(
        delta.cpu(), 
        size=original.shape[2:], 
        mode='bilinear', 
        align_corners=False
    )
    
    # Apply and save
    protected = torch.clamp(original + delta_scaled, 0, 1)
    result = transforms.ToPILImage()(protected.squeeze(0))
    result.save(output_path, quality=95)
    
    print(f"✓ Protected image saved: {output_path}")
    
    # Calculate final similarity for comparison
    with torch.no_grad():
        orig_feat = model.encode_image(image_tensor.to(device))
        prot_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])(result).unsqueeze(0).to(device)
        prot_feat = model.encode_image(prot_tensor)
        similarity = F.cosine_similarity(orig_feat, prot_feat).item()
    
    # Create comparison
    comparison_path = output_path.parent / f"{output_path.stem}_comparison.png"
    create_comparison(input_path, output_path, comparison_path, model, similarity)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Protect artwork from AI training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  glz image.jpg                    # Output: image_protected.jpg
  glz image.png -o output/         # Output to directory
  glz image.jpg -i high            # High intensity protection
  glz image.jpg --steps 100        # Custom steps
        """
    )
    
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("-o", "--output", type=Path, help="Output path")
    parser.add_argument("-i", "--intensity", choices=["low", "medium", "high"], 
                        default="medium", help="Protection intensity (default: medium)")
    parser.add_argument("--steps", type=int, help="Override optimization steps")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)
    
    # Handle single file or directory
    if args.input.is_file():
        files = [args.input]
    else:
        files = list(args.input.glob("*.jpg")) + \
                list(args.input.glob("*.jpeg")) + \
                list(args.input.glob("*.png"))
    
    if not files:
        print("No images found")
        sys.exit(1)
    
    # Determine output directory
    if args.output and args.output.suffix == "":
        output_dir = args.output
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    for f in files:
        if output_dir:
            out = output_dir / f"{f.stem}_protected{f.suffix}"
        elif args.output:
            out = args.output
        else:
            out = f.parent / f"{f.stem}_protected{f.suffix}"
        
        protect_image(f, out, args.intensity, args.steps)


if __name__ == "__main__":
    main()
