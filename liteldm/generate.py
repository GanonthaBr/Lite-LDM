from pathlib import Path
from typing import Optional

import torch


@torch.no_grad()
def generate_slices(unet, vae, scheduler, device: torch.device, n: int = 4, timesteps: int = 1000):
    unet.eval()
    vae.eval()

    z = torch.randn(n, 512, 16, 16, device=device)
    for t in reversed(range(timesteps)):
        z = scheduler.sample_step(unet, z, t)

    imgs = vae.decode(z)
    return imgs.cpu()


def _ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _get_pyplot(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_generated(generated: torch.Tensor, save_path: Optional[str] = None, show: bool = False) -> None:
    plt = _get_pyplot(show)
    n = generated.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(generated[i, 0], cmap="gray")
        ax.set_title(f"Generated #{i + 1}")
        ax.axis("off")
    plt.suptitle("LDM - Generated CT Slices (512-dim latent space)", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_generated_tensors(generated: torch.Tensor, output_dir: str, prefix: str = "generated") -> str:
    out_dir = _ensure_dir(output_dir)
    tensor_path = out_dir / f"{prefix}.pt"
    torch.save(generated, tensor_path)
    return str(tensor_path)


@torch.no_grad()
def reconstruction_quality_plot(
    vae,
    loader,
    device: torch.device,
    n: int = 4,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    plt = _get_pyplot(show)
    vae.eval()
    real_batch = next(iter(loader))[:n].to(device)
    recon_batch, _, _ = vae(real_batch)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for i in range(n):
        axes[0, i].imshow(real_batch[i, 0].cpu(), cmap="gray")
        axes[0, i].set_title(f"Real #{i + 1}")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_batch[i, 0].cpu(), cmap="gray")
        axes[1, i].set_title(f"Recon #{i + 1}")
        axes[1, i].axis("off")

    plt.suptitle("VAE: Real vs Reconstruction", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
