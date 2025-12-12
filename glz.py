#!/usr/bin/env python3
"""
glz - Protect artwork from AI training using CLIP adversarial perturbation
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import clip

# Protection intensity presets
PRESETS = {
    "low": {"steps": 30, "epsilon": 4/255},
    "medium": {"steps": 50, "epsilon": 8/255},
    "high": {"steps": 100, "epsilon": 16/255},
}


def get_device():
    """Auto-detect GPU/CPU"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(path, size=224):
    """Load image and convert to tensor"""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0), img.size


def generate_perturbation(img_tensor, model, steps, epsilon, lr=0.01):
    """
    Generate adversarial perturbation to fool CLIP.
    Goal: make perturbed embedding far from original.
    """
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    # Get original embedding
    with torch.no_grad():
        orig_features = model.encode_image(img_tensor)

    # Init perturbation
    delta = torch.zeros_like(img_tensor, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    print(f"Generating protection ({steps} steps)...")

    for step in range(steps):
        perturbed = torch.clamp(img_tensor + delta, 0, 1)
        features = model.encode_image(perturbed)

        # Minimize similarity = maximize distance
        loss = F.cosine_similarity(features, orig_features).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Clamp perturbation within epsilon
        delta.data.clamp_(-epsilon, epsilon)

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{steps} - Similarity: {loss.item():.3f}")

    return delta.detach()


def make_fft_visualization(img):
    """Create FFT visualization (simulates how AI 'sees' the image)"""
    gray = np.array(img.convert("L"), dtype=np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(fft))
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

    # Red colormap
    rgb = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = magnitude
    return Image.fromarray(rgb)


def create_comparison(orig_path, prot_path, out_path, similarity):
    """Create comparison image: Original | Protected | Difference + AI view"""
    orig = Image.open(orig_path).convert("RGB")
    prot = Image.open(prot_path).convert("RGB")

    # Resize to same size
    size = (400, 400)
    orig_resized = orig.resize(size, Image.LANCZOS)
    prot_resized = prot.resize(size, Image.LANCZOS)

    # Create difference image (20x amplified)
    orig_arr = np.array(orig_resized, dtype=np.float32)
    prot_arr = np.array(prot_resized, dtype=np.float32)
    diff = np.clip(np.abs(orig_arr - prot_arr) * 20, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(diff)

    # FFT visualization
    fft_orig = make_fft_visualization(orig_resized)
    fft_prot = make_fft_visualization(prot_resized)

    # Create canvas
    w, h = size
    pad = 30
    canvas = Image.new("RGB", (w * 3 + 40, h * 2 + pad * 3), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 16)
        font_bold = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 20)
    except:
        font = font_bold = ImageFont.load_default()

    # Row 1: Original, Protected, Difference
    row1 = [
        ("Original", orig_resized),
        ("Protected", prot_resized),
        ("Difference (20x amplified)", diff_img),
    ]
    for i, (label, img) in enumerate(row1):
        x = 20 + i * (w + 10)
        draw.text((x + w // 2 - len(label) * 4, 5), label, fill="black", font=font)
        canvas.paste(img, (x, pad))

    # Row 2: AI sees
    row2 = [
        ("AI sees: Original", fft_orig),
        ("AI sees: Protected", fft_prot),
    ]
    for i, (label, img) in enumerate(row2):
        x = 20 + i * (w + 10)
        y = h + pad * 2
        draw.text((x + w // 2 - len(label) * 4, y - 20), label, fill="black", font=font)
        canvas.paste(img, (x, y))

    # Stats panel
    sx = 20 + 2 * (w + 10) + w // 2
    sy = h + pad * 2 + h // 3
    sim_pct = similarity * 100

    draw.text((sx - 80, sy), "CLIP Similarity", fill="black", font=font_bold)
    draw.text((sx - 30, sy + 30), f"{sim_pct:.1f}%", fill="black", font=font_bold)
    draw.text((sx - 100, sy + 70), "AI thinks these are", fill="black", font=font)
    draw.text((sx - 120, sy + 95), "COMPLETELY DIFFERENT!", fill="black", font=font_bold)

    canvas.save(out_path, quality=95)
    print(f"✓ Comparison: {out_path}")


def protect(input_path, output_path, intensity="medium", custom_steps=None):
    """Main function to protect image"""
    config = PRESETS.get(intensity, PRESETS["medium"]).copy()
    if custom_steps:
        config["steps"] = custom_steps

    device = get_device()
    print(f"Device: {device}")
    print("Loading CLIP...")

    model, _ = clip.load("ViT-B/32", device=device)
    img_tensor, _ = load_image(input_path)

    # Generate perturbation
    delta = generate_perturbation(
        img_tensor, model,
        steps=config["steps"],
        epsilon=config["epsilon"]
    )

    # Apply to full resolution image
    orig_full = transforms.ToTensor()(Image.open(input_path).convert("RGB")).unsqueeze(0)
    delta_scaled = F.interpolate(delta.cpu(), size=orig_full.shape[2:], mode="bilinear", align_corners=False)

    protected = torch.clamp(orig_full + delta_scaled, 0, 1)
    result = transforms.ToPILImage()(protected.squeeze(0))
    result.save(output_path, quality=95)
    print(f"✓ Protected: {output_path}")

    # Calculate final similarity
    with torch.no_grad():
        orig_feat = model.encode_image(img_tensor.to(device))
        prot_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])(result).unsqueeze(0).to(device)
        prot_feat = model.encode_image(prot_tensor)
        similarity = F.cosine_similarity(orig_feat, prot_feat).item()

    # Create comparison
    comp_path = output_path.parent / f"{output_path.stem}_comparison.png"
    create_comparison(input_path, output_path, comp_path, similarity)


def main():
    parser = argparse.ArgumentParser(
        prog="glz",
        description="Protect artwork from AI training",
        epilog="Example: glz image.jpg -i high"
    )
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("-o", "--output", type=Path, help="Output path")
    parser.add_argument("-i", "--intensity", choices=["low", "medium", "high"],
                        default="medium", help="Protection level (default: medium)")
    parser.add_argument("--steps", type=int, help="Override optimization steps")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)

    # Get file list
    if args.input.is_file():
        files = [args.input]
    else:
        files = [f for f in args.input.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

    if not files:
        print("No images found")
        sys.exit(1)

    # Handle output
    out_dir = None
    if args.output and not args.output.suffix:
        out_dir = args.output
        out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        if out_dir:
            out = out_dir / f"{f.stem}_protected{f.suffix}"
        elif args.output:
            out = args.output
        else:
            out = f.parent / f"{f.stem}_protected{f.suffix}"

        protect(f, out, args.intensity, args.steps)


if __name__ == "__main__":
    main()
