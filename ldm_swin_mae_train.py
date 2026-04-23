import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchio as tio
import matplotlib.pyplot as plt
import wandb

from monai.networks.schedulers import DDIMScheduler
from monai.metrics import SSIMMetric
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWIN_DIR = os.path.join(SCRIPT_DIR, "src", "encoders", "Swin-MAE")
if SWIN_DIR not in sys.path:
    sys.path.append(SWIN_DIR)

import swin_mae
from utils.dataset import NiftiDataset



def _timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, returns [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.act(self.conv2(h))
        return h + self.skip(x)


class UNet1DDenoiser(nn.Module):
    def __init__(self, latent_len: int, base_ch: int = 64, time_dim: int = 128):
        super().__init__()
        self.latent_len = latent_len
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.enc1 = ResBlock1D(1, base_ch, time_dim)
        self.enc2 = ResBlock1D(base_ch, base_ch * 2, time_dim)
        self.enc3 = ResBlock1D(base_ch * 2, base_ch * 4, time_dim)

        self.down1 = nn.AvgPool1d(2, ceil_mode=True)
        self.down2 = nn.AvgPool1d(2, ceil_mode=True)

        self.mid = ResBlock1D(base_ch * 4, base_ch * 4, time_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ResBlock1D(base_ch * 4 + base_ch * 2, base_ch * 2, time_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ResBlock1D(base_ch * 2 + base_ch, base_ch, time_dim)

        self.out = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """x: [B, L], timesteps: [B]. Returns predicted noise [B, L]."""
        t_emb = _timestep_embedding(timesteps, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h0 = x.unsqueeze(1)
        e1 = self.enc1(h0, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)

        m = self.mid(e3, t_emb)

        u2 = self.up2(m)
        if u2.shape[-1] != e2.shape[-1]:
            u2 = F.interpolate(u2, size=e2.shape[-1], mode="nearest")
        d2 = self.dec2(torch.cat([u2, e2], dim=1), t_emb)

        u1 = self.up1(d2)
        if u1.shape[-1] != e1.shape[-1]:
            u1 = F.interpolate(u1, size=e1.shape[-1], mode="nearest")
        d1 = self.dec1(torch.cat([u1, e1], dim=1), t_emb)

        out = self.out(d1).squeeze(1)
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out.unsqueeze(1), size=x.shape[-1], mode="nearest").squeeze(1)
        return out


class LatentNormalizer(nn.Module):
    def __init__(self, latent_len: int, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(1, latent_len))
        self.register_buffer("std", torch.ones(1, latent_len))
        self.register_buffer("initialized", torch.tensor(False))

    def update(self, latents: torch.Tensor) -> None:
        batch_mean = latents.mean(dim=0, keepdim=True)
        batch_std = latents.std(dim=0, keepdim=True).clamp_min(self.eps)
        if not bool(self.initialized):
            self.mean.copy_(batch_mean)
            self.std.copy_(batch_std)
            self.initialized.fill_(True)
            return
        self.mean.mul_(self.momentum).add_(batch_mean * (1 - self.momentum))
        self.std.mul_(self.momentum).add_(batch_std * (1 - self.momentum))

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.mean) / self.std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.std + self.mean


class SwinMAELatentAdapter(nn.Module):
    def __init__(self, model: nn.Module, image_size: int, latent_len: int | None, device: torch.device):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.device = device

        self.encoder_shape = self._infer_encoder_shape()
        self.full_latent_len = int(np.prod(self.encoder_shape[1:]))
        self.latent_len = latent_len or self.full_latent_len

        if self.latent_len != self.full_latent_len:
            self.encoder_proj = nn.Linear(self.full_latent_len, self.latent_len)
            self.decoder_proj = nn.Linear(self.latent_len, self.full_latent_len)
        else:
            self.encoder_proj = nn.Identity()
            self.decoder_proj = nn.Identity()

    def _infer_encoder_shape(self) -> tuple[int, int, int, int, int]:
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.model.in_chans,
                self.image_size,
                self.image_size,
                self.image_size,
                device=self.device,
            )
            features, _ = self.model.forward_encoder(dummy)
        return tuple(features.shape)

    def encode(self, x: torch.Tensor, no_grad_backbone: bool = False) -> torch.Tensor:
        if no_grad_backbone:
            with torch.no_grad():
                features, _ = self.model.forward_encoder(x)
        else:
            features, _ = self.model.forward_encoder(x)
        flat = features.reshape(features.shape[0], -1)
        return self.encoder_proj(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        full = self.decoder_proj(z)
        b = full.shape[0]
        d, h, w, c = self.encoder_shape[1:]
        features = full.reshape(b, d, h, w, c)
        pred = self.model.forward_decoder(features)
        return self.model.unpatchify(pred)


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Swin-MAE Latent Diffusion", add_help=False)
    parser.add_argument("--swin_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ldm_output")
    parser.add_argument("--log_dir", type=str, default="./ldm_output")
    parser.add_argument("--env_path", type=str, default="", help="Path to .env file for WandB auth")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--in_chans", type=int, default=1)

    parser.add_argument("--model", type=str, default="swin_mae")
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument("--latent_len", type=int, default=0)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=2)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--freeze_swin", action="store_true")
    parser.set_defaults(freeze_swin=True)
    return parser


def build_swin(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.model == "swin_mae" and args.image_size != 224:
        print("Warning: swin_mae default image_size is 224; ensure checkpoints match your input size.")
    model = swin_mae.__dict__[args.model](
        norm_pix_loss=False,
        mask_ratio=args.mask_ratio,
    )
    checkpoint = torch.load(args.swin_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    if args.freeze_swin:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model


def save_checkpoint(
    path: str,
    epoch: int,
    denoiser: nn.Module,
    adapter: SwinMAELatentAdapter,
    normalizer: LatentNormalizer,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    state = {
        "epoch": epoch,
        "denoiser": denoiser.state_dict(),
        "adapter": adapter.state_dict(),
        "normalizer": normalizer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    denoiser: nn.Module,
    adapter: SwinMAELatentAdapter,
    normalizer: LatentNormalizer,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    denoiser.load_state_dict(checkpoint.get("denoiser", {}), strict=False)
    adapter.load_state_dict(checkpoint.get("adapter", {}), strict=False)
    normalizer.load_state_dict(checkpoint.get("normalizer", {}), strict=False)
    optimizer.load_state_dict(checkpoint.get("optimizer", {}))
    return int(checkpoint.get("epoch", 0))


def sample_latents(
    denoiser: nn.Module,
    scheduler: DDIMScheduler,
    num_samples: int,
    latent_len: int,
    device: torch.device,
    steps: int,
) -> torch.Tensor:
    denoiser.eval()
    scheduler.set_timesteps(steps, device=device)
    latents = torch.randn(num_samples, latent_len, device=device)
    for t in scheduler.timesteps:
        ts = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = denoiser(latents, ts)
        step_out = scheduler.step(noise_pred, t, latents)
        latents = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]
    return latents


def main() -> None:
    args = get_args().parse_args()

    if args.env_path:
        load_dotenv(args.env_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform_train = tio.Compose(
        [
            tio.Resample((1.0, 1.0, 1.0)),
            tio.CropOrPad((args.image_size, args.image_size, args.image_size)),
            tio.RandomFlip(axes=(0, 1, 2)),
            tio.ZNormalization(),
        ]
    )

    dataset_train = NiftiDataset(args.data_path, transform=transform_train)
    data_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    wandb_run = None
    if args.env_path:
        wandb_run = wandb.init(project="swinmae_ldm", config=vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    swin = build_swin(args, device)
    adapter = SwinMAELatentAdapter(
        model=swin,
        image_size=args.image_size,
        latent_len=args.latent_len if args.latent_len > 0 else None,
        device=device,
    ).to(device)

    denoiser = UNet1DDenoiser(latent_len=adapter.latent_len).to(device)
    normalizer = LatentNormalizer(latent_len=adapter.latent_len).to(device)
    optim_params = [p for p in list(denoiser.parameters()) + list(adapter.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scheduler = DDIMScheduler(
        num_train_timesteps=args.timesteps,
    )

    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)

    global_step = 0
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, denoiser, adapter, normalizer, optimizer)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs):
        denoiser.train()
        epoch_loss = 0.0
        last_images = None

        for step, batch in enumerate(data_loader):
            images = batch["image"].to(device)
            last_images = images

            latents = adapter.encode(images, no_grad_backbone=args.freeze_swin)
            normalizer.update(latents)
            latents = normalizer.normalize(latents)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy = scheduler.add_noise(latents, noise, timesteps)
            pred_noise = denoiser(noisy, timesteps)

            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if log_writer is not None and global_step % args.log_freq == 0:
                log_writer.add_scalar("train/loss", loss.item(), global_step)
            if wandb_run is not None and global_step % args.log_freq == 0:
                wandb.log({"train/loss": loss.item(), "step": global_step})

        avg_loss = epoch_loss / max(len(data_loader), 1)
        lr_scheduler.step()
        print(f"Epoch {epoch + 1:03d} | Loss {avg_loss:.6f}")

        if log_writer is not None:
            log_writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
        if wandb_run is not None:
            wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f"ldm_checkpoint_{epoch + 1}.pth")
            save_checkpoint(checkpoint_path, epoch + 1, denoiser, adapter, normalizer, optimizer, args)

        if (epoch + 1) % args.save_freq == 0:
            with torch.no_grad():
                sample_latent = sample_latents(
                    denoiser=denoiser,
                    scheduler=scheduler,
                    num_samples=1,
                    latent_len=adapter.latent_len,
                    device=device,
                    steps=args.sample_steps,
                )
                recon = adapter.decode(normalizer.denormalize(sample_latent)).clamp(0, 1)
                target = images[:1].clamp(0, 1)
                ssim_val = ssim_metric(recon, target).mean().item()
                if log_writer is not None:
                    log_writer.add_scalar("eval/ssim", ssim_val, epoch + 1)
                if wandb_run is not None:
                    wandb.log({"eval/ssim": ssim_val, "epoch": epoch + 1})

                mid = recon.shape[2] // 2
                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                ax[0].imshow(target[0, 0, mid].cpu(), cmap="gray")
                ax[0].set_title("Target")
                ax[0].axis("off")
                ax[1].imshow(recon[0, 0, mid].cpu(), cmap="gray")
                ax[1].set_title("Sample")
                ax[1].axis("off")
                fig.tight_layout()
                fig.savefig(os.path.join(args.output_dir, f"sample_epoch_{epoch + 1}.png"), dpi=150)
                plt.close(fig)

        if wandb_run is not None and last_images is not None:
            with torch.no_grad():
                sample_latent = sample_latents(
                    denoiser=denoiser,
                    scheduler=scheduler,
                    num_samples=1,
                    latent_len=adapter.latent_len,
                    device=device,
                    steps=args.sample_steps,
                )
                recon = adapter.decode(normalizer.denormalize(sample_latent)).clamp(0, 1)
                target = last_images[:1].clamp(0, 1)

                mid = recon.shape[2] // 2
                recon_slice = recon[0, 0, mid].cpu().numpy()
                target_slice = target[0, 0, mid].cpu().numpy()
                wandb.log(
                    {
                        "samples/recon_slice": wandb.Image(recon_slice, caption=f"epoch {epoch + 1} recon"),
                        "samples/target_slice": wandb.Image(target_slice, caption=f"epoch {epoch + 1} target"),
                        "epoch": epoch + 1,
                    }
                )

    if wandb_run is not None:
        wandb.finish()

        if args.output_dir:
            log_stats = {
                "epoch": epoch + 1,
                "loss": avg_loss,
            }
            with open(os.path.join(args.output_dir, "log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    main()
