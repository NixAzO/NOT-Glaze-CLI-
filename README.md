# glz 🎨🛡️

Protect your artwork from AI style mimicry.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## What it does

`glz` adds invisible perturbations to your images that confuse AI models (Stable Diffusion, Midjourney, etc.) while looking identical to humans.

```
Human sees:  Your beautiful artwork
AI sees:     Completely different style
```

## Installation

### Windows / macOS / Linux

```bash
# Clone
git clone https://github.com/yourusername/glz.git
cd glz

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python glz image.jpg

# Output to specific directory
python glz image.png -o protected/

# High intensity protection
python glz image.jpg -i high

# Process entire folder
python glz ./my_artworks/
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output path or directory |
| `-i, --intensity` | `low`, `medium` (default), `high` |
| `--steps` | Custom optimization steps |

## Examples

```bash
# Single image
python glz artwork.png
# → artwork_protected.png

# Multiple images
python glz ./portfolio/ -o ./protected/

# Maximum protection
python glz commission.jpg -i high --steps 150
```

## How it works

Uses CLIP adversarial perturbation to maximize the distance between original and protected image embeddings, making AI perceive a completely different style.

See [ALGORITHM.md](ALGORITHM.md) for technical details.

## Requirements

- Python 3.8+
- PyTorch
- CLIP
- ~2GB RAM (CPU) or ~2GB VRAM (GPU)

GPU is recommended but not required.

## License

MIT

## Disclaimer

This tool provides a layer of protection but is not 100% foolproof. For maximum protection, consider also using [Glaze](https://glaze.cs.uchicago.edu/) from University of Chicago.
