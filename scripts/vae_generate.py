"""
Script to generate and save a reconstructed image using only the VAE (no diffusion).
Usage: python vae_generate.py --checkpoint-dir ./checkpoints --encoder-backbone strainer --output-path ./vae_recon.png
"""

import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from liteldm.data import build_slice_loader, resolve_paths
from liteldm.models import VAE, resolve_device

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save a reconstructed image using only the VAE.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory containing VAE checkpoint")
    parser.add_argument("--encoder-backbone", choices=["conv", "strainer"], default="conv", help="VAE encoder backbone")
    parser.add_argument("--strainer-hidden-features", type=int, default=256)
    parser.add_argument("--strainer-total-layers", type=int, default=6)
    parser.add_argument("--strainer-shared-encoder-layers", type=int, default=5)
    parser.add_argument("--strainer-num-train-decoders", type=int, default=10)
    parser.add_argument("--strainer-encoder-weights", default=None, help="Optional path to pre-trained STRAINER encoder weights (.pth)")
    parser.add_argument("--output-path", default="./vae_recon.png", help="Path to save the reconstructed image")
    parser.add_argument("--local", default=None, help="Override LOCAL storage path")
    return parser.parse_args()

def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    vae_ckpt = ckpt_dir / "vae_ckpt.pt"
    device = resolve_device()
    print(f"Using device: {device}")
    print(f"Looking for VAE checkpoint at: {vae_ckpt.resolve()}")
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt.resolve()}")

    # Load data
    paths = resolve_paths(args.local)
    dataset, loader = build_slice_loader(paths.processed_dir, min_content_ratio=0.01)
    print(f"Dataset loaded: {len(dataset)} slices")

    # Load VAE
    vae = VAE(
        latent_ch=512,
        encoder_backbone=args.encoder_backbone,
        strainer_hidden_features=args.strainer_hidden_features,
        strainer_total_layers=args.strainer_total_layers,
        strainer_shared_encoder_layers=args.strainer_shared_encoder_layers,
        strainer_num_train_decoders=args.strainer_num_train_decoders,
        strainer_encoder_weights=args.strainer_encoder_weights,
    ).to(device)
    vae.load_state_dict(torch.load(str(vae_ckpt), map_location=device))
    vae.eval()
    print(f"VAE loaded from {vae_ckpt}")

    # Get a batch of images
    batch = next(iter(loader))
    x = batch.to(device)
    with torch.no_grad():
        recon, mu, logvar = vae(x)
    # Clamp to [0,1] for visualization
    recon = recon.clamp(0, 1)
    # Save the first reconstructed image
    save_image(recon[0], args.output_path)
    print(f"Saved reconstructed image to {args.output_path}")

if __name__ == "__main__":
    main()
