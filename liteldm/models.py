import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VAEEncoder(nn.Module):
    def __init__(self, latent_ch: int = 512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            ResBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            ResBlock(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            ResBlock(256),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            ResBlock(512),
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            ResBlock(512),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
        )
        self.to_mu = nn.Conv2d(512, latent_ch, 1)
        self.to_logvar = nn.Conv2d(512, latent_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.to_mu(h), self.to_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_ch: int = 512):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(latent_ch, 512, 3, padding=1),
            ResBlock(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            ResBlock(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            ResBlock(32),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)


class VAE(nn.Module):
    def __init__(self, latent_ch: int = 512):
        super().__init__()
        self.encoder = VAEEncoder(latent_ch)
        self.decoder = VAEDecoder(latent_ch)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimestepBlock(nn.Module):
    def __init__(self, ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, ch * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.t_proj(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.conv1(F.silu(self.norm1(x)))
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class DiffusionUNet(nn.Module):
    def __init__(self, latent_ch: int = 512, t_dim: int = 256):
        super().__init__()
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        self.enc1 = nn.Conv2d(latent_ch, 512, 3, padding=1)
        self.enc1_tb = TimestepBlock(512, t_dim)

        self.down1 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
        self.enc2_tb = TimestepBlock(512, t_dim)

        self.bot1 = TimestepBlock(512, t_dim)
        self.bot2 = TimestepBlock(512, t_dim)

        self.up1 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.dec1_tb = TimestepBlock(1024, t_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, 1024),
            nn.SiLU(),
            nn.Conv2d(1024, latent_ch, 3, padding=1),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_emb(t)

        e1 = self.enc1_tb(self.enc1(z), te)
        e2 = self.enc2_tb(self.down1(e1), te)

        b = self.bot2(self.bot1(e2, te), te)

        d1 = self.up1(b)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_tb(d1, te)
        return self.out(d1)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
