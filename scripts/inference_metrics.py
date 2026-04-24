"""
Inference and metrics computation from pre-trained checkpoints.
Does not require re-training — loads VAE + diffusion models and evaluates.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from liteldm.data import build_slice_loader, resolve_paths
from liteldm.diffusion import DDPMScheduler
from liteldm.models import DiffusionUNet, VAE, resolve_device
from liteldm.train import encode_dataset_to_latents, vae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference and metrics computation from checkpoints")
    parser.add_argument("--local", default=None, help="Override LOCAL storage path")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory containing model checkpoints")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for metric outputs")
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
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches to evaluate on")
    return parser.parse_args()


def eval_vae_reconstruction(vae: VAE, loader: DataLoader, device: torch.device, num_batches: int = 5):
    """Evaluate VAE reconstruction on a subset of the dataset."""
    vae.eval()
    metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= num_batches:
                break
            x = batch.to(device)
            recon, mu, logvar = vae(x)
            
            # Compute losses
            recon_loss = F.mse_loss(recon, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + 1e-4 * kl_loss
            
            metrics.append({
                "batch": batch_idx + 1,
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": total_loss.item(),
            })
            print(f"  Batch {batch_idx + 1}: recon={recon_loss.item():.4f}, kl={kl_loss.item():.6f}, total={total_loss.item():.4f}")
    
    avg_metrics = {
        "avg_recon_loss": sum(m["recon_loss"] for m in metrics) / len(metrics),
        "avg_kl_loss": sum(m["kl_loss"] for m in metrics) / len(metrics),
        "avg_total_loss": sum(m["total_loss"] for m in metrics) / len(metrics),
    }
    return metrics, avg_metrics


def eval_diffusion_noise(
    unet: DiffusionUNet,
    vae: VAE,
    loader: DataLoader,
    device: torch.device,
    ddpm: DDPMScheduler,
    num_batches: int = 5,
):
    """Evaluate diffusion model on noise prediction."""
    unet.eval()
    vae.eval()
    metrics = []
    
    with torch.no_grad():
        # Encode dataset to latents
        all_latents = encode_dataset_to_latents(vae, loader, device)
        
        # Sample from latent distribution
        num_samples = min(num_batches * 8, all_latents.shape[0])
        z0 = all_latents[:num_samples].to(device)
        
        # Evaluate on random timesteps
        for step_idx in range(num_batches):
            # Random timestep
            t = torch.randint(0, ddpm.T, (z0.shape[0],), device=device)
            z_t, eps = ddpm.add_noise(z0, t)
            eps_pred = unet(z_t, t)
            loss = F.mse_loss(eps_pred, eps)
            
            metrics.append({
                "step": step_idx + 1,
                "noise_mse": loss.item(),
            })
            print(f"  Step {step_idx + 1}: noise_mse={loss.item():.5f}")
    
    avg_noise_mse = sum(m["noise_mse"] for m in metrics) / len(metrics)
    return metrics, {"avg_noise_mse": avg_noise_mse}


def generate_samples(unet: DiffusionUNet, vae: VAE, device: torch.device, ddpm: DDPMScheduler, n: int = 4):
    """Generate sample images."""
    unet.eval()
    vae.eval()
    
    with torch.no_grad():
        # Start from noise in latent space
        z = torch.randn(n, 512, 16, 16, device=device)
        
        # Reverse diffusion
        for t in reversed(range(ddpm.T)):
            z = ddpm.sample_step(unet, z, t)
        
        # Decode to image space
        imgs = vae.decode(z)
    
    return imgs.cpu()


def main() -> None:
    args = parse_args()
    
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vae_ckpt = str(ckpt_dir / "vae_ckpt.pt")
    diff_ckpt = str(ckpt_dir / "diffusion_ckpt.pt")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check checkpoints exist
    if not Path(vae_ckpt).exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")
    if not Path(diff_ckpt).exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diff_ckpt}")
    
    # Setup
    device = resolve_device()
    print(f"Using device: {device}")
    
    # Load data
    paths = resolve_paths(args.local)
    dataset, loader = build_slice_loader(paths.processed_dir, min_content_ratio=0.01)
    print(f"Dataset loaded: {len(dataset)} slices")
    
    # Load models
    vae = VAE(
        latent_ch=512,
        encoder_backbone=args.encoder_backbone,
        strainer_hidden_features=args.strainer_hidden_features,
        strainer_total_layers=args.strainer_total_layers,
        strainer_shared_encoder_layers=args.strainer_shared_encoder_layers,
        strainer_num_train_decoders=args.strainer_num_train_decoders,
        strainer_encoder_weights=args.strainer_encoder_weights,
    ).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    print(f"VAE loaded from {vae_ckpt}")
    
    unet = DiffusionUNet(latent_ch=512).to(device)
    unet.load_state_dict(torch.load(diff_ckpt, map_location=device))
    print(f"DiffusionUNet loaded from {diff_ckpt}")
    
    ddpm = DDPMScheduler(T=1000, device=str(device))
    
    # Evaluate VAE
    print("\n=== VAE Reconstruction Metrics ===")
    vae_batch_metrics, vae_avg_metrics = eval_vae_reconstruction(vae, loader, device, num_batches=args.num_batches)
    print(f"Average VAE metrics: {vae_avg_metrics}")
    
    # Evaluate Diffusion
    print("\n=== Diffusion Noise Prediction Metrics ===")
    diff_batch_metrics, diff_avg_metrics = eval_diffusion_noise(
        unet, vae, loader, device, ddpm, num_batches=args.num_batches
    )
    print(f"Average diffusion metrics: {diff_avg_metrics}")
    
    # Generate samples
    print("\n=== Generating Samples ===")
    generated = generate_samples(unet, vae, device, ddpm, n=args.num_samples)
    print(f"Generated samples: {generated.shape}")
    
    # Save metrics
    all_metrics = {
        "timestamp": run_id,
        "checkpoint_dir": str(ckpt_dir),
        "encoder_backbone": args.encoder_backbone,
        "vae_metrics": {
            "batch_metrics": vae_batch_metrics,
            "average_metrics": vae_avg_metrics,
        },
        "diffusion_metrics": {
            "batch_metrics": diff_batch_metrics,
            "average_metrics": diff_avg_metrics,
        },
        "generated_samples": {
            "count": args.num_samples,
            "shape": list(generated.shape),
        },
    }
    
    metrics_path = out_dir / f"inference_metrics_{run_id}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✓ Saved metrics to: {metrics_path}")
    
    # Save generated samples as image
    from liteldm.generate import plot_generated
    gen_img_path = str(out_dir / f"generated_samples_{run_id}.png")
    plot_generated(generated, save_path=gen_img_path, show=False)
    print(f"✓ Saved generated image grid: {gen_img_path}")


if __name__ == "__main__":
    main()
