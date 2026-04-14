import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import glob
import nibabel as nib
from skimage.transform import resize

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

# SineLayer
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0     = omega_0
        self.is_first    = is_first
        self.in_features = in_features
        self.linear      = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                     1 / self.in_features)
            else:
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

# INR
class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        layers = []
        if hidden_layers > 0:
            layers.append(SineLayer(in_features, hidden_features,
                                    is_first=True, omega_0=first_omega_0))
            n_middle = hidden_layers - 1
            if outermost_linear:
                n_middle -= 1
            for _ in range(n_middle):
                layers.append(SineLayer(hidden_features, hidden_features,
                                        is_first=False, omega_0=hidden_omega_0))
        if outermost_linear or hidden_layers == 0:
            final = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / max(hidden_omega_0, 1e-12)
                final.weight.uniform_(-bound, bound)
            layers.append(final)
        else:
            layers.append(SineLayer(hidden_features, out_features,
                                    is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# SharedINR (STRAINER) — 3D version
class SharedINR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.,
                 shared_encoder_layers=5, num_decoders=10):
        super().__init__()
        assert hidden_layers > shared_encoder_layers
        self.shared_encoder_layers = shared_encoder_layers
        self.num_decoders          = num_decoders
        self.encoderINR = INR(
            in_features=in_features, hidden_features=hidden_features,
            hidden_layers=shared_encoder_layers - 1,
            out_features=hidden_features, outermost_linear=False,
            first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0)
        num_decoder_layers = hidden_layers - shared_encoder_layers
        self.decoderINRs = nn.ModuleList([
            INR(in_features=hidden_features, hidden_features=hidden_features,
                hidden_layers=num_decoder_layers - 1, out_features=out_features,
                outermost_linear=outermost_linear,
                first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0)
            for _ in range(num_decoders)])

    def forward(self, coords):
        features = self.encoderINR(coords)
        return [decoder(features) for decoder in self.decoderINRs]

    def load_encoder_weights_from(self, other_model):
        self.encoderINR.load_state_dict(
            deepcopy(other_model.encoderINR.state_dict()))

# STRAINEREncoder — 3D version
class STRAINEREncoder(nn.Module):
    def __init__(self, hidden_features=256, shared_encoder_layers=5,
                 total_layers=6, in_channels=1, latent_dim=512,
                 num_train_decoders=10):
        super().__init__()
        self.latent_dim = latent_dim
        # 3D: input is (x, y, z) so in_features=3
        self.strainer = SharedINR(
            in_features=3, hidden_features=hidden_features,
            hidden_layers=total_layers, out_features=in_channels,
            shared_encoder_layers=shared_encoder_layers,
            num_decoders=num_train_decoders)
        self.projection = nn.Sequential(
            nn.Linear(hidden_features, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim))

    def _get_coords_3d(self, D, H, W, device):
        # Build 3D voxel coordinate grid in [-1, 1]
        xs = torch.linspace(-1, 1, W)
        ys = torch.linspace(-1, 1, H)
        zs = torch.linspace(-1, 1, D)
        Z, Y, X = torch.meshgrid(zs, ys, xs, indexing='ij')
        coords = torch.stack([
            X.reshape(-1),
            Y.reshape(-1),
            Z.reshape(-1)
        ], dim=-1)
        return coords.unsqueeze(0).to(device)

    def encode(self, x):
        # x shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        device  = x.device
        coords  = self._get_coords_3d(D, H, W, device)
        latents = []
        for i in range(B):
            features = self.strainer.encoderINR(coords)
            pooled   = features.mean(dim=1)
            z_i      = self.projection(pooled)
            latents.append(z_i)
        return torch.cat(latents, dim=0)

    def forward(self, x):
        return self.encode(x)

    def save_encoder(self, path):
        torch.save({'encoder_weights':
                    self.strainer.encoderINR.state_dict()}, path)
        print(f"Encoder saved to {path}")

# 3D coordinate grid
def get_coords_3d(D, H, W, device):
    xs = torch.linspace(-1, 1, W, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    zs = torch.linspace(-1, 1, D, device=device)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing='ij')
    coords = torch.stack([
        X.reshape(-1),
        Y.reshape(-1),
        Z.reshape(-1)
    ], dim=-1)
    return coords.unsqueeze(0)

# Load NIfTI volumes — following Kushal MetaSeg approach
def load_nifti_volumes(nifti_dir, num_volumes=10, target_shape=(80, 80, 100)):
    print(f"\nSearching for NIfTI files in {nifti_dir}...")
    nii_files = sorted(glob.glob(
        os.path.join(nifti_dir, '**', '*.nii.gz'), recursive=True
    ))
    print(f"Found {len(nii_files)} NIfTI volumes total")

    indices  = np.linspace(0, len(nii_files) - 1, num_volumes, dtype=int)
    selected = [nii_files[i] for i in indices]

    volumes = []
    for i, f in enumerate(selected):
        img  = nib.load(f)
        data = img.get_fdata().astype(np.float32)

        # Normalize to [0, 1] using CT windowing
        data = np.clip(data, -1000, 1000)
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)

        # Downsample to target shape (following MetaSeg: 80x80x100)
        data = resize(data, target_shape, anti_aliasing=True).astype(np.float32)

        # Shape: (D, H, W, 1)
        data = data[:, :, :, np.newaxis]
        volumes.append(torch.from_numpy(data).float())
        print(f"  Volume {i+1:2d}: {os.path.basename(f)} -> shape {volumes[-1].shape}")

    return volumes

# Paths
NIFTI_DIR = "/ocean/projects/cis250019p/thierryh/strainer_project/nifti_converted"
SAVE_DIR  = "/ocean/projects/cis250019p/thierryh/strainer_project"

# Load 3D volumes
ct_volumes = load_nifti_volumes(NIFTI_DIR, num_volumes=10, target_shape=(80, 80, 100))
print(f"\nLoaded {len(ct_volumes)} real 3D CT volumes")

D, H, W, C = ct_volumes[0].shape
print(f"Volume size: {D}x{H}x{W}, channels: {C}")
print(f"Total voxels per volume: {D*H*W:,}")

# Build 3D STRAINER model
model = SharedINR(
    in_features=3,
    hidden_features=256,
    hidden_layers=6,
    out_features=C,
    shared_encoder_layers=5,
    num_decoders=len(ct_volumes)
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
targets   = [v.reshape(1, -1, C).to(device) for v in ct_volumes]
coords    = get_coords_3d(D, H, W, device)

# Training loop
print(f"\nStarting 3D STRAINER pre-training on {len(ct_volumes)} NIfTI CT volumes...")
print(f"Iterations: 5000 | LR: 1e-4 | Device: {device}")
print(f"Volume shape: {D}x{H}x{W} (following MetaSeg 80x80x100 approach)")
print("-" * 60)

psnr_log = []
for step in range(5000):
    outputs = model(coords)
    preds   = torch.stack(outputs, dim=0)
    gts     = torch.stack(targets, dim=0)
    loss    = ((preds - gts) ** 2).mean()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    psnr = -10 * torch.log10(loss.detach()).item()
    psnr_log.append(psnr)

    if step % 500 == 0 or step == 4999:
        print(f"  Step {step:5d}/5000 | Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB")

print("-" * 60)
print("Pre-training complete!")

# Save encoder
encoder = STRAINEREncoder(
    hidden_features=256, shared_encoder_layers=5,
    total_layers=6, in_channels=C,
    latent_dim=512, num_train_decoders=len(ct_volumes)
).to(device)

encoder.strainer.load_encoder_weights_from(model)
encoder.save_encoder(os.path.join(SAVE_DIR, "strainer_encoder_3d_weights.pth"))

# Plot PSNR curve
plt.figure(figsize=(10, 5))
plt.plot(psnr_log)
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("3D STRAINER Pre-training on Real LIDC-IDRI NIfTI CT Volumes")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "psnr_curve_3d.png"))
print("PSNR curve saved")
print("All done! 3D encoder weights ready for the team.")
