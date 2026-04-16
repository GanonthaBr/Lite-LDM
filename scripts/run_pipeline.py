import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from huggingface_hub import login

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from liteldm.data import (
    Niftify,
    build_slice_loader,
    download_dataset_zip,
    list_nifti_files,
    preprocess_nifti,
    resolve_paths,
)
from liteldm.diffusion import DDPMScheduler
from liteldm.generate import generate_slices, plot_generated, reconstruction_quality_plot, save_generated_tensors
from liteldm.models import DiffusionUNet, VAE, resolve_device
from liteldm.preflight import run_preflight
from liteldm.train import build_latent_loader, encode_dataset_to_latents, train_diffusion, train_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiteLDM notebook pipeline as scripts")
    parser.add_argument(
        "--stage",
        choices=["preflight", "data", "train-vae", "train-diffusion", "generate", "recon-check", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument("--local", default=None, help="Override LOCAL storage path")
    parser.add_argument("--token-env", default="huggingface_token", help="Environment variable containing HuggingFace token")
    parser.add_argument("--vae-epochs", type=int, default=50)
    parser.add_argument("--diff-epochs", type=int, default=100)
    parser.add_argument("--num-generate", type=int, default=4)
    parser.add_argument(
        "--min-content-ratio",
        type=float,
        default=0.01,
        help="Slice mean-intensity threshold for keeping axial slices",
    )
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory to save/read model checkpoints")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for generated images and artifacts")
    parser.add_argument("--show-plots", action="store_true", help="Display matplotlib windows (off by default for VM/headless)")
    parser.add_argument("--save-tensors", action="store_true", help="Save generated tensor batch as .pt")
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


def maybe_login(token_env: str) -> None:
    load_dotenv()
    hf_token = os.environ.get(token_env)
    if hf_token:
        login(hf_token)
    else:
        print(f"No HuggingFace token found in env var: {token_env}")


def run_data_stage(local_override: Optional[str]):
    paths = resolve_paths(local_override)
    zip_path = download_dataset_zip(paths.local)
    print(f"Downloaded dataset zip: {zip_path}")

    niftify = Niftify(zip_path, paths.dicom_dir, paths.nifti_dir)
    niftify.run()

    dataset_files = list_nifti_files(paths.nifti_dir)
    print(f"NIfTI files found: {len(dataset_files)}")
    if not dataset_files:
        raise RuntimeError(
            "No NIfTI files were found after DICOM conversion. "
            "Check dataset download/conversion and confirm input DICOM files exist."
        )
    preprocess_nifti(dataset_files, paths.nifti_dir, paths.processed_dir)
    processed_files = list_nifti_files(paths.processed_dir)
    print(f"Processed files found: {len(processed_files)}")
    if not processed_files:
        raise RuntimeError(
            "Preprocessing produced no files in processed_dir. "
            "Check conversion output format and preprocessing logs."
        )
    print(f"Preprocessing complete. Files in: {paths.processed_dir}")
    return paths


def build_training_objects(paths, args):
    dataset, loader = build_slice_loader(paths.processed_dir, min_content_ratio=args.min_content_ratio)
    sample = next(iter(loader))
    print(f"Batch shape: {sample.shape}, dtype: {sample.dtype}, range: [{sample.min():.2f}, {sample.max():.2f}]")

    device = resolve_device()
    print(f"Using device: {device}")

    vae = VAE(
        latent_ch=512,
        encoder_backbone=args.encoder_backbone,
        strainer_hidden_features=args.strainer_hidden_features,
        strainer_total_layers=args.strainer_total_layers,
        strainer_shared_encoder_layers=args.strainer_shared_encoder_layers,
        strainer_num_train_decoders=args.strainer_num_train_decoders,
        strainer_encoder_weights=args.strainer_encoder_weights,
    ).to(device)
    unet = DiffusionUNet(latent_ch=512).to(device)
    ddpm = DDPMScheduler(T=1000, device=str(device))
    return dataset, loader, device, vae, unet, ddpm


def build_inference_objects(args):
    device = resolve_device()
    print(f"Using device: {device}")
    vae = VAE(
        latent_ch=512,
        encoder_backbone=args.encoder_backbone,
        strainer_hidden_features=args.strainer_hidden_features,
        strainer_total_layers=args.strainer_total_layers,
        strainer_shared_encoder_layers=args.strainer_shared_encoder_layers,
        strainer_num_train_decoders=args.strainer_num_train_decoders,
        strainer_encoder_weights=args.strainer_encoder_weights,
    ).to(device)
    unet = DiffusionUNet(latent_ch=512).to(device)
    ddpm = DDPMScheduler(T=1000, device=str(device))
    return device, vae, unet, ddpm


def require_checkpoint(path: str, missing_hint: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}. {missing_hint}"
        )


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae_ckpt = str(ckpt_dir / "vae_ckpt.pt")
    diff_ckpt = str(ckpt_dir / "diffusion_ckpt.pt")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.stage in {"preflight", "all"}:
        run_preflight()

    if args.stage in {"data", "all"}:
        maybe_login(args.token_env)
        paths = run_data_stage(args.local)
    else:
        paths = resolve_paths(args.local)

    needs_loader = args.stage in {"train-vae", "train-diffusion", "recon-check", "all"}
    needs_models_only = args.stage in {"generate"}

    if needs_loader:
        dataset, loader, device, vae, unet, ddpm = build_training_objects(paths, args)
    elif needs_models_only:
        device, vae, unet, ddpm = build_inference_objects(args)
    else:
        return

    if args.stage in {"train-vae", "all"}:
        train_vae(vae, loader, device, epochs=args.vae_epochs, lr=1e-4, ckpt_path=vae_ckpt)

    if args.stage in {"train-diffusion", "all"}:
        require_checkpoint(vae_ckpt, "Run --stage train-vae first.")
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        vae.eval()
        latents = encode_dataset_to_latents(vae, loader, device)
        print(f"Latent bank: {latents.shape}")
        latent_loader = build_latent_loader(latents, batch_size=8)
        train_diffusion(unet, ddpm, latent_loader, device, epochs=args.diff_epochs, lr=2e-4, ckpt_path=diff_ckpt)

    if args.stage in {"generate", "all"}:
        require_checkpoint(vae_ckpt, "Run --stage train-vae first.")
        require_checkpoint(diff_ckpt, "Run --stage train-diffusion first.")
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        unet.load_state_dict(torch.load(diff_ckpt, map_location=device))
        generated = generate_slices(unet, vae, ddpm, device, n=args.num_generate, timesteps=1000)
        gen_img_path = str(out_dir / f"generated_{run_id}.png")
        plot_generated(generated, save_path=gen_img_path, show=args.show_plots)
        print(f"Saved generated image grid: {gen_img_path}")
        if args.save_tensors:
            tensor_path = save_generated_tensors(generated, str(out_dir), prefix=f"generated_{run_id}")
            print(f"Saved generated tensor batch: {tensor_path}")

    if args.stage in {"recon-check", "all"}:
        require_checkpoint(vae_ckpt, "Run --stage train-vae first.")
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        recon_img_path = str(out_dir / f"reconstruction_{run_id}.png")
        reconstruction_quality_plot(vae, loader, device, n=4, save_path=recon_img_path, show=args.show_plots)
        print(f"Saved reconstruction comparison: {recon_img_path}")


if __name__ == "__main__":
    main()
