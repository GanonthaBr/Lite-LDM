import os
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import dicom2nifti
import numpy as np
import torch
import torchio as tio
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataPaths:
    local: str
    dicom_dir: str
    nifti_dir: str
    processed_dir: str


def resolve_paths(local: Optional[str] = None) -> DataPaths:
    local_root = local or os.environ.get("LOCAL", "./local_storage")
    dicom_dir = os.path.join(local_root, "dataset", "dicom")
    nifti_dir = os.path.join(local_root, "dataset", "nifti")
    processed_dir = os.path.join(local_root, "dataset", "processed")
    os.makedirs(dicom_dir, exist_ok=True)
    os.makedirs(nifti_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    return DataPaths(local_root, dicom_dir, nifti_dir, processed_dir)


def download_dataset_zip(local: str, filename: str = "LIDC-IDRI-0050.zip") -> str:
    return hf_hub_download(
        repo_id="deeplearningresearchproject/dataset_project",
        filename=filename,
        repo_type="dataset",
        local_dir=os.path.join(local, "dataset"),
    )


class Niftify:
    def __init__(self, zip_path: str, extract_to: str, nifti_path: str):
        self.zip_path = zip_path
        self.extract_to = extract_to
        self.nifti_path = nifti_path

    def unzip_file(self) -> None:
        if not os.path.isfile(self.zip_path):
            raise FileNotFoundError(f"ZIP file not found: {self.zip_path}")
        os.makedirs(self.extract_to, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.extract_to)
            print(f"Extracted {len(zip_ref.namelist())} files to '{self.extract_to}'")

    def to_nifti(self) -> None:
        if not os.path.isdir(self.extract_to):
            raise FileNotFoundError(f"Missing DICOM directory: {self.extract_to}")
        os.makedirs(self.nifti_path, exist_ok=True)
        dicom2nifti.convert_directory(self.extract_to, self.nifti_path)

    def run(self) -> None:
        self.unzip_file()
        self.to_nifti()


def list_nifti_files(path: str) -> List[str]:
    return [
        fname
        for fname in os.listdir(path)
        if fname.endswith(".nii") or fname.endswith(".nii.gz")
    ]


def preprocess_nifti(
    dataset_files: List[str], nifti_dir: str, processed_dir: str, target_shape: Tuple[int, int, int] = (256, 256, 128)
) -> None:
    pipeline = tio.Compose([
        tio.Resample(1.0),
        tio.CropOrPad(target_shape),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])

    for i, fname in enumerate(dataset_files):
        fpath = os.path.join(nifti_dir, fname)
        out_path = os.path.join(processed_dir, fname)
        if os.path.exists(out_path):
            print(f"[{i + 1}/{len(dataset_files)}] {fname}: already processed, skipping.")
            continue
        subj = tio.Subject(chest_ct=tio.ScalarImage(fpath))
        print(f"[{i + 1}/{len(dataset_files)}] {fname}: {subj.chest_ct.shape}", flush=True)
        trans = pipeline(subj)
        trans.chest_ct.save(out_path)


class CTSliceDataset(Dataset):
    """
    Yields individual 2D axial slices from preprocessed 3D CT volumes.
    Each slice shape: (1, 256, 256), values in [0, 1].
    """

    def __init__(self, processed_dir: str, axis: int = 0, min_content_ratio: float = 0.01):
        self.slices = []
        fnames = list_nifti_files(processed_dir)
        for fname in fnames:
            subj = tio.Subject(ct=tio.ScalarImage(os.path.join(processed_dir, fname)))
            vol = subj.ct.data.squeeze().numpy()
            n = vol.shape[axis]
            for i in range(n):
                slc = np.take(vol, i, axis=axis).astype(np.float32)
                if slc.mean() > min_content_ratio:
                    self.slices.append(slc)
        print(f"Total slices loaded: {len(self.slices)}")

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        slc = self.slices[idx]
        return torch.tensor(slc).unsqueeze(0)


def build_slice_loader(
    processed_dir: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 2,
    min_content_ratio: float = 0.01,
) -> tuple[CTSliceDataset, DataLoader]:
    dataset = CTSliceDataset(processed_dir, min_content_ratio=min_content_ratio)
    if len(dataset) == 0:
        raise RuntimeError(
            "No training slices were extracted from processed volumes. "
            "Verify that processed_dir contains valid .nii/.nii.gz files and "
            "consider lowering min_content_ratio in CTSliceDataset if volumes are mostly empty."
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader
