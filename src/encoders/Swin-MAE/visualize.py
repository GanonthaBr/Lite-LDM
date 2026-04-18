import sys
import os
import numpy as np

import torch
import matplotlib.pyplot as plt
import torchio as tio

import swin_mae
import argparse

sys.path.append('..')

def get_args():

    parser = argparse.ArgumentParser('MAE Visualizer', add_help=False)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--image_size', default=224, type=int)

    return parser

def show_slices(volume, title=''):
    """
    volume is (D, H, W) — a single-channel 3D array.
    Shows the middle slice along each of the three axes.
    """
    d, h, w = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[d // 2, :, :], cmap='gray', origin='lower')
    axes[0].set_title(f'{title}\naxial (D={d // 2})')
    axes[1].imshow(volume[:, h // 2, :], cmap='gray', origin='lower')
    axes[1].set_title(f'{title}\ncoronal (H={h // 2})')
    axes[2].imshow(volume[:, :, w // 2], cmap='gray', origin='lower')
    axes[2].set_title(f'{title}\nsagittal (W={w // 2})')
    for ax in axes:
        ax.axis('off')
    return fig


def prepare_model(chkpt_dir_, arch='swin_mae'):
    model = getattr(swin_mae, arch)()
    checkpoint = torch.load(chkpt_dir_, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(x, model, save_path="result.png"):
    """
    x: numpy array of shape (D, H, W) or (D, H, W, C)
    """
    x = torch.tensor(x).float()

    # Ensure (D, H, W, C) then rearrange to (N, C, D, H, W)
    if x.ndim == 3:
        x = x.unsqueeze(-1)                          # (D, H, W, 1)
    x = x.unsqueeze(0)                               # (1, D, H, W, C)
    x = torch.einsum('ndhwc->ncdhw', x)              # (1, C, D, H, W)

    # Run MAE
    loss, y, mask = model(x)
    y = model.unpatchify(y)                           # (1, C, D, H, W)
    y = torch.einsum('ncdhw->ndhwc', y).detach().cpu()

    # Expand mask to match patchified volume then unpatchify
    p = model.patch_size
    c = x.shape[1]
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, p ** 3 * c)   # (N, D*H*W, p^3*C)
    mask = model.unpatchify(mask)                          # (1, C, D, H, W)
    mask = torch.einsum('ncdhw->ndhwc', mask).detach().cpu()

    x = torch.einsum('ncdhw->ndhwc', x).detach().cpu()

    # Masked input, reconstruction, and composite
    im_masked = x * (1 - mask)
    im_recon  = y * mask
    im_paste  = x * (1 - mask) + y * mask

    # Squeeze channel dim for display (single-channel volumes)
    def to_vol(t):
        return t[0, ..., 0].numpy()   # (D, H, W)

    fig, axes_grid = plt.subplots(4, 3, figsize=(12, 16))
    titles = ['original', 'masked', 'reconstruction', 'reconstruction + visible']
    vols   = [to_vol(x), to_vol(im_masked), to_vol(im_recon), to_vol(im_paste)]

    for row, (vol, title) in enumerate(zip(vols, titles)):
        d, h, w = vol.shape
        axes_grid[row][0].imshow(vol[d // 2, :, :], cmap='gray', origin='lower')
        axes_grid[row][0].set_title(f'{title} — axial')
        axes_grid[row][1].imshow(vol[:, h // 2, :], cmap='gray', origin='lower')
        axes_grid[row][1].set_title(f'{title} — coronal')
        axes_grid[row][2].imshow(vol[:, :, w // 2], cmap='gray', origin='lower')
        axes_grid[row][2].set_title(f'{title} — sagittal')
        for ax in axes_grid[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


if __name__ == '__main__':
    # Load a .nii.gz volume with TorchIO and preprocess to match training
    parser = get_args()
    args = parser.parse_args()
    img_path = args.source_path
    input_size = args.image_size
    checkpoint_dir = args.checkpoint_path

    transform = tio.Compose([
        tio.Resample((1.0, 1.0, 1.0)),
        tio.CropOrPad((input_size, input_size, input_size)),
        tio.ZNormalization(),
    ])

    subject = tio.Subject(image=tio.ScalarImage(img_path))
    subject = transform(subject)
    # (C, D, H, W) -> (D, H, W, C) for run_one_image
    img = subject['image'].data.permute(1, 2, 3, 0).numpy()
    assert img.shape == (input_size, input_size, input_size, 1)

    model_mae = prepare_model(checkpoint_dir, 'swin_mae')
    print('Model loaded.')

    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    run_one_image(img, model_mae)