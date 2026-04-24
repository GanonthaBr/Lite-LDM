"""
Compute PSNR and MS-SSIM for VAE reconstructions from a checkpoint.
Usage: python compute_metrics.py --checkpoint-dir ./checkpoints --num-batches 5
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from liteldm.data import build_slice_loader, resolve_paths
from liteldm.models import VAE, resolve_device
from liteldm.train import vae_loss

try:
    from torchmetrics.functional import peak_signal_noise_ratio as compute_psnr
    from torchmetrics.functional import multiscale_structural_similarity_index_measure as compute_ms_ssim
except ImportError:
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim
    compute_ms_ssim = None

def parse_args():
    parser = argparse.ArgumentParser(description="Compute PSNR and MS-SSIM for VAE reconstructions")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory containing VAE checkpoint")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches to evaluate")
    parser.add_argument("--output-dir", default="./metrics_output", help="Directory to save metrics")
    parser.add_argument(
        "--encoder-backbone",
        choices=["conv", "strainer"],
        default="conv",
        help="VAE encoder backbone",
    )
    parser.add_argument("--strainer-hidden-features", type=int, default=256)
    parser.add_argument("--strainer-total-layers", type=int, default=6)
    parser.add_argument("--strainer-shared-encoder-layers", type=int, default=5)
    parser.add_argument("--strainer-num-train-decoders", type=int, default=10)
    parser.add_argument(
        "--strainer-encoder-weights",
        default=None,
        help="Optional path to pre-trained STRAINER encoder weights (.pth)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    vae_ckpt = ckpt_dir / "vae_ckpt.pt"
    device = resolve_device()
    print(f"Using device: {device}")

    # Check checkpoint exists
    print(f"Looking for VAE checkpoint at: {vae_ckpt.resolve()}")
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt.resolve()}")

    # Load data
    paths = resolve_paths(None)
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

    psnr_scores = []
    ms_ssim_scores = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            x = batch.to(device)
            recon, mu, logvar = vae(x)
            # Clamp to [0,1] if needed
            x_clamped = x.clamp(0, 1)
            recon_clamped = recon.clamp(0, 1)
            # PSNR
            if 'torchmetrics' in globals() or 'compute_psnr' in globals():
                psnr = compute_psnr(recon_clamped.cpu(), x_clamped.cpu(), data_range=1.0)
                # Dynamically set MS-SSIM kernel_size and betas for small images
                ms_ssim = None
                if compute_ms_ssim:
                    _, _, h, w = x_clamped.shape
                    # Use smaller kernel and fewer scales for small images
                    if min(h, w) < 160:
                        ms_ssim = compute_ms_ssim(
                            recon_clamped.cpu(), x_clamped.cpu(), data_range=1.0, kernel_size=7, betas=(0.44, 0.285, 0.300)
                        )
                    else:
                        ms_ssim = compute_ms_ssim(recon_clamped.cpu(), x_clamped.cpu(), data_range=1.0)
            else:
                psnr = compute_psnr(x_clamped.cpu().numpy(), recon_clamped.cpu().numpy(), data_range=1.0)
                ms_ssim = None
                for i in range(x_clamped.shape[0]):
                    ssim_val = compute_ssim(
                        x_clamped[i].cpu().numpy().transpose(1,2,0),
                        recon_clamped[i].cpu().numpy().transpose(1,2,0),
                        data_range=1.0,
                        multichannel=True
                    )
                    ms_ssim = ms_ssim or []
                    ms_ssim.append(ssim_val)
            psnr_scores.append(psnr.mean().item() if hasattr(psnr, 'mean') else float(psnr))
            if ms_ssim is not None:
                ms_ssim_scores.append(ms_ssim.mean().item() if hasattr(ms_ssim, 'mean') else float(ms_ssim))
            print(f"Batch {batch_idx+1}: PSNR={psnr_scores[-1]:.4f}, MS-SSIM={ms_ssim_scores[-1] if ms_ssim_scores else 'N/A'}")

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ms_ssim = sum(ms_ssim_scores) / len(ms_ssim_scores) if ms_ssim_scores else None
    print(f"\nAverage PSNR: {avg_psnr:.4f}")
    if avg_ms_ssim is not None:
        print(f"Average MS-SSIM: {avg_ms_ssim:.4f}")

    # Save metrics
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vae_metrics.txt", "w") as f:
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        if avg_ms_ssim is not None:
            f.write(f"Average MS-SSIM: {avg_ms_ssim:.4f}\n")
    print(f"Metrics saved to {out_dir / 'vae_metrics.txt'}")

if __name__ == "__main__":
    main()
