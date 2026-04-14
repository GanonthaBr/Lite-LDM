import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import glob
import os
import nibabel as nib
from skimage.transform import resize

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# SharedINR
class SharedINR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.,
                 shared_encoder_layers=5, num_decoders=1):
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

    def load_encoder_weights_from_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        self.encoderINR.load_state_dict(state['encoder_weights'])
        print(f"Encoder weights loaded from {checkpoint_path}")

# Coordinate grid
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

# Paths
NIFTI_DIR    = "/ocean/projects/cis250019p/thierryh/strainer_project/nifti_converted"
WEIGHTS_PATH = "/ocean/projects/cis250019p/thierryh/strainer_project/strainer_encoder_3d_weights.pth"
SAVE_DIR     = "/ocean/projects/cis250019p/thierryh/strainer_project"

TARGET_SHAPE = (80, 80, 100)
D, H, W      = TARGET_SHAPE

# Load one real CT volume as ground truth
nii_files = sorted(glob.glob(os.path.join(NIFTI_DIR, '**', '*.nii.gz'), recursive=True))
print(f"\nLoading ground truth volume: {os.path.basename(nii_files[0])}")

img  = nib.load(nii_files[0])
data = img.get_fdata().astype(np.float32)
data = np.clip(data, -1000, 1000)
data = (data - data.min()) / (data.max() - data.min() + 1e-8)
data = resize(data, TARGET_SHAPE, anti_aliasing=True).astype(np.float32)
gt   = torch.from_numpy(data).float()
print(f"Ground truth shape: {gt.shape}")

# Build model and load encoder weights
model = SharedINR(
    in_features=3, hidden_features=256,
    hidden_layers=6, out_features=1,
    shared_encoder_layers=5, num_decoders=1
).to(device)

model.load_encoder_weights_from_checkpoint(WEIGHTS_PATH)

# Fine-tune decoder on this volume
target  = gt.reshape(1, -1, 1).to(device)
coords  = get_coords_3d(D, H, W, device)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

print("\nFine-tuning decoder on test volume for 1000 iterations...")
print("-" * 50)

for step in range(1000):
    outputs = model(coords)
    pred    = outputs[0]
    loss    = ((pred - target) ** 2).mean()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if step % 200 == 0 or step == 999:
        psnr = -10 * torch.log10(loss.detach()).item()
        print(f"  Step {step:4d}/1000 | Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB")

print("-" * 50)

# Reconstruct volume
print("\nReconstructing 3D volume...")
model.eval()
with torch.no_grad():
    outputs = model(coords)
    pred    = outputs[0]

pred_vol = pred.reshape(D, H, W).cpu().numpy()
gt_vol   = gt.numpy()

# Plot 3 views: axial, coronal, sagittal
mid_d = D // 2
mid_h = H // 2
mid_w = W // 2

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("STRAINER 3D CT Reconstruction — NIfTI LIDC-IDRI", fontsize=14)

# Ground truth
axes[0, 0].imshow(gt_vol[mid_d, :, :], cmap='gray')
axes[0, 0].set_title("GT — Axial slice")
axes[0, 0].axis('off')

axes[0, 1].imshow(gt_vol[:, mid_h, :], cmap='gray')
axes[0, 1].set_title("GT — Coronal slice")
axes[0, 1].axis('off')

axes[0, 2].imshow(gt_vol[:, :, mid_w], cmap='gray')
axes[0, 2].set_title("GT — Sagittal slice")
axes[0, 2].axis('off')

# Reconstruction
axes[1, 0].imshow(pred_vol[mid_d, :, :], cmap='gray')
axes[1, 0].set_title("STRAINER — Axial slice")
axes[1, 0].axis('off')

axes[1, 1].imshow(pred_vol[:, mid_h, :], cmap='gray')
axes[1, 1].set_title("STRAINER — Coronal slice")
axes[1, 1].axis('off')

axes[1, 2].imshow(pred_vol[:, :, mid_w], cmap='gray')
axes[1, 2].set_title("STRAINER — Sagittal slice")
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "reconstruction_3d.png"), dpi=150)
print(f"\nVisualization saved to {SAVE_DIR}/reconstruction_3d.png")
print("Done!")
