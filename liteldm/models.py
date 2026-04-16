import math
from copy import deepcopy
from typing import Optional

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


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.in_features, 1.0 / self.in_features)
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class INR(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        layers = []
        if hidden_layers > 0:
            layers.append(
                SineLayer(
                    in_features,
                    hidden_features,
                    is_first=True,
                    omega_0=first_omega_0,
                )
            )
            n_middle = hidden_layers - 1
            if outermost_linear:
                n_middle -= 1
            for _ in range(max(0, n_middle)):
                layers.append(
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        is_first=False,
                        omega_0=hidden_omega_0,
                    )
                )
        if outermost_linear or hidden_layers == 0:
            final = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / max(hidden_omega_0, 1e-12)
                final.weight.uniform_(-bound, bound)
            layers.append(final)
        else:
            layers.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedINR(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        shared_encoder_layers: int = 5,
        num_decoders: int = 10,
    ):
        super().__init__()
        if hidden_layers <= shared_encoder_layers:
            raise ValueError("hidden_layers must be greater than shared_encoder_layers")
        self.shared_encoder_layers = shared_encoder_layers
        self.num_decoders = num_decoders
        self.encoderINR = INR(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=shared_encoder_layers - 1,
            out_features=hidden_features,
            outermost_linear=False,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )
        num_decoder_layers = hidden_layers - shared_encoder_layers
        self.decoderINRs = nn.ModuleList(
            [
                INR(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    hidden_layers=num_decoder_layers - 1,
                    out_features=out_features,
                    outermost_linear=outermost_linear,
                    first_omega_0=first_omega_0,
                    hidden_omega_0=hidden_omega_0,
                )
                for _ in range(num_decoders)
            ]
        )

    def forward(self, coords: torch.Tensor):
        features = self.encoderINR(coords)
        return [decoder(features) for decoder in self.decoderINRs]

    def load_encoder_weights_from(self, other_model) -> None:
        self.encoderINR.load_state_dict(deepcopy(other_model.encoderINR.state_dict()))


class STRAINERVAEEncoder(nn.Module):
    def __init__(
        self,
        latent_ch: int = 512,
        hidden_features: int = 256,
        total_layers: int = 6,
        shared_encoder_layers: int = 5,
        num_train_decoders: int = 10,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.latent_ch = latent_ch
        self.strainer = SharedINR(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=total_layers,
            out_features=1,
            shared_encoder_layers=shared_encoder_layers,
            num_decoders=num_train_decoders,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )
        self.pixel_projection = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, latent_ch * 2),
        )
        self.intensity_scale = nn.Parameter(torch.tensor(1.0))

    def _get_coords(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        xs = torch.linspace(-1, 1, w, device=device)
        ys = torch.linspace(-1, 1, h, device=device)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)
        return coords.unsqueeze(0)

    def load_encoder_weights(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        if "encoder_weights" in state:
            self.strainer.encoderINR.load_state_dict(state["encoder_weights"])
        else:
            self.strainer.encoderINR.load_state_dict(state)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, h, w = x.shape
        coords = self._get_coords(h, w, x.device).expand(bsz, -1, -1)
        features = self.strainer.encoderINR(coords)

        pixel_weights = x.mean(dim=1).reshape(bsz, -1, 1)
        features = features * (1.0 + self.intensity_scale * pixel_weights)

        mu_logvar_px = self.pixel_projection(features)
        mu_px, logvar_px = mu_logvar_px.chunk(2, dim=-1)

        mu_map = mu_px.transpose(1, 2).reshape(bsz, self.latent_ch, h, w)
        logvar_map = logvar_px.transpose(1, 2).reshape(bsz, self.latent_ch, h, w)

        latent_h = max(1, h // 16)
        latent_w = max(1, w // 16)
        mu = F.adaptive_avg_pool2d(mu_map, (latent_h, latent_w))
        logvar = F.adaptive_avg_pool2d(logvar_map, (latent_h, latent_w))
        return mu, logvar


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
    def __init__(
        self,
        latent_ch: int = 512,
        encoder_backbone: str = "conv",
        strainer_hidden_features: int = 256,
        strainer_total_layers: int = 6,
        strainer_shared_encoder_layers: int = 5,
        strainer_num_train_decoders: int = 10,
        strainer_encoder_weights: Optional[str] = None,
    ):
        super().__init__()
        if encoder_backbone == "strainer":
            self.encoder = STRAINERVAEEncoder(
                latent_ch=latent_ch,
                hidden_features=strainer_hidden_features,
                total_layers=strainer_total_layers,
                shared_encoder_layers=strainer_shared_encoder_layers,
                num_train_decoders=strainer_num_train_decoders,
            )
            if strainer_encoder_weights:
                self.encoder.load_encoder_weights(strainer_encoder_weights)
        elif encoder_backbone == "conv":
            self.encoder = VAEEncoder(latent_ch)
        else:
            raise ValueError("encoder_backbone must be either 'conv' or 'strainer'")
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
