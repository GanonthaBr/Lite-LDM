from dataclasses import dataclass


@dataclass
class Paths:
    local: str
    dicom_dir: str
    nifti_dir: str
    processed_dir: str


@dataclass
class TrainConfig:
    vae_epochs: int = 50
    vae_lr: float = 1e-4
    vae_ckpt: str = "./vae_ckpt.pt"
    diff_epochs: int = 100
    diff_lr: float = 2e-4
    diff_ckpt: str = "./diffusion_ckpt.pt"
    latent_ch: int = 512
    image_size: int = 256
    timesteps: int = 1000
