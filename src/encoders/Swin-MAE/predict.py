import sys
import os
import torch
from torch.utils.data import DataLoader
import torchio as tio
import nibabel as nib
from tqdm import tqdm

import swin_mae
from utils.dataset import NiftiDataset

import argparse

def get_args():
    parser = argparse.ArgumentParser('MAE Prediction Generator')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--source_path', type=str, required=True, help='Path to test images')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser

def prepare_model(chkpt_dir_, device, arch='swin_mae'):
    model = getattr(swin_mae, arch)()
    checkpoint = torch.load(chkpt_dir_, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def main():
    parser = get_args()
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Transform (Matches your Training/Visualization)
    transform = tio.Compose([
        tio.Resample((1.0, 1.0, 1.0)),
        tio.CropOrPad((args.image_size, args.image_size, args.image_size)),
        tio.ZNormalization(),
    ])

    # 2. Initialize NiftiDataset
    # Assuming NiftiDataset takes source_path and transform
    dataset = NiftiDataset(args.source_path, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False
    )

    # 3. Load Model
    model = prepare_model(args.checkpoint_path, device)
    print(f"Model loaded. Processing {len(dataset)} volumes...")
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count {parameter_count}")

    # 4. Inference Loop
    for i, batch in enumerate(tqdm(dataloader, desc="Reconstructing")):
        # Note: NiftiDataset usually returns a dictionary or a tensor.
        # We assume 'image' key based on standard monai/torchio patterns.
        images = batch['image'].to(device) 
        
        # Get filenames to save with original names
        # TorchIO/NiftiDataset usually stores the path in 'path' or 'image_path'
        paths = batch['path']

        # Run Model
        # Swin-MAE returns (loss, pred, mask)
        _, preds, _ = model(images)
        
        # Unpatchify to 3D volume: (B, L, D) -> (B, C, D, H, W)
        recon_volumes = model.unpatchify(preds)

        # 5. Save outputs
        for j in range(recon_volumes.shape[0]):
            recon_tensor = recon_volumes[j].cpu()
            original_path = paths[j]
            filename = os.path.basename(original_path)
            
            # Use original affine to ensure the NIfTI orientation is correct
            # batch['image']['affine'] contains the 4x4 matrix
            affine = batch['affine'][j].numpy()
            
            output_path = os.path.join(args.output_dir, filename)
            
            # Save using nibabel or torchio
            output_img = nib.Nifti1Image(recon_tensor.squeeze().numpy(), affine)
            nib.save(output_img, output_path)

if __name__ == '__main__':
    main()