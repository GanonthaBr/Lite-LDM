import torch
import torchio as tio
import glob
import os

class NiftiDataset(torch.utils.data.Dataset):
    """
    Unlabeled dataset for .nii.gz files. Recursively finds all NIFTI files
    under `root_dir` and applies the given TorchIO transform.
    """

    def __init__(self, root_dir: str, transform: tio.Transform | None = None):
        self.transform = transform
        self.file_paths = sorted(glob.glob(
            os.path.join(root_dir, '**', '*.nii.gz'), recursive=True
        ))
        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .nii.gz files found under: {root_dir}")
        print(f"Found {len(self.file_paths)} NIFTI files in {root_dir}")

    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        subject = tio.Subject(
            image=tio.ScalarImage(path)
        )
        if self.transform is not None:
            subject = self.transform(subject)

        # Return a dict so we have the metadata for saving later
        return {
            'image': subject['image'].data,      # Tensor (C, D, H, W)
            'path': path,                        # String
            'affine': subject['image'].affine    # 4x4 Numpy Array
        }
