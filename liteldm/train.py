import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 1e-4):
    recon_loss = F.mse_loss(recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


def train_vae(vae, loader: DataLoader, device: torch.device, epochs: int = 50, lr: float = 1e-4, ckpt_path: str = "./vae_ckpt.pt"):
    vae_opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_opt, T_max=epochs)

    history = []
    vae.train()
    for epoch in range(epochs):
        total_loss = recon_l = kl_l = 0.0
        for batch in loader:
            x = batch.to(device)
            recon, mu, logvar = vae(x)
            loss, rl, kll = vae_loss(recon, x, mu, logvar)

            vae_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            vae_opt.step()

            total_loss += loss.item()
            recon_l += rl.item()
            kl_l += kll.item()

        scheduler.step()
        n = len(loader)
        epoch_loss = total_loss / n
        epoch_recon = recon_l / n
        epoch_kl = kl_l / n
        print(
            f"[VAE] Epoch {epoch + 1:>3}/{epochs} "
            f"loss={epoch_loss:.4f} recon={epoch_recon:.4f} kl={epoch_kl:.6f}"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "recon_loss": epoch_recon,
                "kl_loss": epoch_kl,
            }
        )

    torch.save(vae.state_dict(), ckpt_path)
    print(f"VAE saved -> {ckpt_path}")
    return history


def encode_dataset_to_latents(vae, loader: DataLoader, device: torch.device) -> torch.Tensor:
    vae.eval()
    all_latents = []
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            z = vae.encode(x)
            all_latents.append(z.cpu())
    return torch.cat(all_latents, dim=0)


def build_latent_loader(all_latents: torch.Tensor, batch_size: int = 8) -> DataLoader:
    latent_ds = TensorDataset(all_latents)
    return DataLoader(latent_ds, batch_size=batch_size, shuffle=True, pin_memory=True)


def train_diffusion(
    unet,
    scheduler,
    latent_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 2e-4,
    ckpt_path: str = "./diffusion_ckpt.pt",
):
    diff_opt = torch.optim.AdamW(unet.parameters(), lr=lr)
    diff_sched = torch.optim.lr_scheduler.CosineAnnealingLR(diff_opt, T_max=epochs)

    unet.train()
    history = []
    for epoch in range(epochs):
        total = 0.0
        for (z0,) in latent_loader:
            z0 = z0.to(device)
            batch_size = z0.shape[0]

            t = torch.randint(0, scheduler.T, (batch_size,), device=device)
            z_t, eps = scheduler.add_noise(z0, t)
            eps_pred = unet(z_t, t)
            loss = F.mse_loss(eps_pred, eps)

            diff_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            diff_opt.step()
            total += loss.item()

        diff_sched.step()
        epoch_noise_mse = total / len(latent_loader)
        print(f"[DIFF] Epoch {epoch + 1:>3}/{epochs} noise_mse={epoch_noise_mse:.5f}")
        history.append(
            {
                "epoch": epoch + 1,
                "noise_mse": epoch_noise_mse,
            }
        )

    torch.save(unet.state_dict(), ckpt_path)
    print(f"Diffusion model saved -> {ckpt_path}")
    return history
