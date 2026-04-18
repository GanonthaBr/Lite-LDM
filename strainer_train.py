import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import glob
import pydicom
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

# SharedINR (STRAINER)
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

# STRAINEREncoder
class STRAINEREncoder(nn.Module):
    def __init__(self, hidden_features=256, shared_encoder_layers=5,
                 total_layers=6, in_channels=1, latent_dim=512,
                 num_train_decoders=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.strainer = SharedINR(
            in_features=2, hidden_features=hidden_features,
            hidden_layers=total_layers, out_features=in_channels,
            shared_encoder_layers=shared_encoder_layers,
            num_decoders=num_train_decoders)
        self.projection = nn.Sequential(
            nn.Linear(hidden_features, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim))

    def _get_coords(self, H, W, device):
        xs = torch.linspace(-1, 1, W)
        ys = torch.linspace(-1, 1, H)
        Y, X = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
        return coords.unsqueeze(0).to(device)

    def encode(self, x):
        B, C, H, W = x.shape
        device  = x.device
        coords  = self._get_coords(H, W, device)
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

# Coordinate grid
def get_coords(H, W, device):
    xs = torch.linspace(-1, 1, W, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    return coords.unsqueeze(0)

# Load real DICOM CT slices
def load_dicom_slices(data_dir, num_slices=10):
    print(f"\nSearching for DICOM files in {data_dir}...")
    dcm_files = sorted(glob.glob(
        os.path.join(data_dir, '**', '*.dcm'), recursive=True
    ))
    print(f"Found {len(dcm_files)} DICOM files total")

    indices  = np.linspace(0, len(dcm_files) - 1, num_slices, dtype=int)
    selected = [dcm_files[i] for i in indices]

    slices = []
    for i, f in enumerate(selected):
        dcm = pydicom.dcmread(f)
        img = dcm.pixel_array.astype(np.float32)

        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Resize to 128x128
        img = resize(img, (128, 128), anti_aliasing=True)

        # Shape: (H, W, 1)
        img = img[:, :, np.newaxis]
        slices.append(torch.from_numpy(img).float())
        print(f"  Slice {i+1:2d}: {os.path.basename(f)} shape {slices[-1].shape}")

    return slices

# Paths
DATA_DIR = "/jet/home/thierryh/strainer_project/data"
SAVE_DIR = "/ocean/projects/cis250019p/thierryh/strainer_project"

# Load data
ct_slices = load_dicom_slices(DATA_DIR, num_slices=10)
print(f"\nLoaded {len(ct_slices)} real CT slices")

H, W, C = ct_slices[0].shape
print(f"Image size: {H}x{W}, channels: {C}")

# Build model
model = SharedINR(
    in_features=2, hidden_features=256,
    hidden_layers=6, out_features=C,
    shared_encoder_layers=5,
    num_decoders=len(ct_slices)
).to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
targets   = [s.reshape(1, -1, C).to(device) for s in ct_slices]
coords    = get_coords(H, W, device)

# Training loop
print(f"\nStarting STRAINER pre-training on {len(ct_slices)} real CT slices...")
print(f"Iterations: 5000 | LR: 1e-4 | Device: {device}")
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
    latent_dim=512, num_train_decoders=len(ct_slices)
).to(device)

encoder.strainer.load_encoder_weights_from(model)
encoder.save_encoder(os.path.join(SAVE_DIR, "strainer_encoder_weights.pth"))

# Plot PSNR curve
plt.figure(figsize=(10, 5))
plt.plot(psnr_log)
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("STRAINER Pre-training on Real LIDC-IDRI CT Scans")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "psnr_curve.png"))
print("PSNR curve saved")
print("All done! Encoder weights ready for the team.")
