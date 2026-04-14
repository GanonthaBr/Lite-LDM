# Rad-DIFF-Efficient-Reliability-Aware-Diffusion-Models-for-Synthetic-CT-Image-Generation




## Abstract
Data scarcity is a serious problem that hinders generalizability of deep learning
models on medical tasks. Diffusion models present a unique opportunity to bridge
this gap by generating high fidelity synthetic images that can be used for data
augmentation. However, many of the current diffusion models are slow and
computationally intensive. Moreover, they act as a black box, making them less
trustworthy on high risk tasks. We propose a latent diffusion solution that uses a
lightweight encoder to represent an image into a lower dimensional representation,
thereby reducing the size of the network. We also introduce uncertainty scoring by
Monte Carlo dropout to offer insights into where the model is performing well and
where it needs to improve. This work directly contributes to the advancement of
clinically relevant deep learning solutions

## Script-Based Pipeline

The notebook pipeline has been split into reusable Python modules.

### Project Layout

- `liteldm/preflight.py`: environment validation checks.
- `liteldm/data.py`: data download, DICOM-to-NIfTI conversion, preprocessing, and dataset loading.
- `liteldm/models.py`: VAE + latent diffusion U-Net models.
- `liteldm/diffusion.py`: DDPM scheduler implementation.
- `liteldm/train.py`: VAE and diffusion training loops.
- `liteldm/generate.py`: sample generation and reconstruction visualization.
- `scripts/run_pipeline.py`: command-line entrypoint for running stages.

### Usage

Run all stages:

```bash
python scripts/run_pipeline.py --stage all
```

Run an individual stage:

```bash
python scripts/run_pipeline.py --stage preflight
python scripts/run_pipeline.py --stage data
python scripts/run_pipeline.py --stage train-vae --vae-epochs 50
python scripts/run_pipeline.py --stage train-diffusion --diff-epochs 100
python scripts/run_pipeline.py --stage generate --num-generate 4
python scripts/run_pipeline.py --stage recon-check
```

Optional arguments:

- `--local`: override LOCAL storage path.
- `--token-env`: env var name for HuggingFace token (default: `huggingface_token`).
- `--checkpoint-dir`: where model checkpoints are saved/read (default: `./checkpoints`).
- `--output-dir`: where generated artifacts are saved (default: `./outputs`).
- `--save-tensors`: save generated tensor batch as `.pt`.
- `--show-plots`: display plot windows (off by default for remote/headless VMs).

### Remote GPU VM Setup

Upload the whole folder to your VM, then run:

```bash
chmod +x scripts/setup_vm.sh
./scripts/setup_vm.sh
source .venv/bin/activate
```

Set your HuggingFace token (if needed):

```bash
export huggingface_token="<your_token>"
```

Recommended VM run (headless-safe, with saved outputs):

```bash
python scripts/run_pipeline.py --stage all --checkpoint-dir ./checkpoints --output-dir ./outputs --save-tensors
```

### Saved Results

- Generated image grids are saved as `./outputs/generated_YYYYMMDD_HHMMSS.png`.
- Reconstruction comparison plots are saved as `./outputs/reconstruction_YYYYMMDD_HHMMSS.png`.
- Optional generated tensors are saved as `./outputs/generated_YYYYMMDD_HHMMSS.pt` when `--save-tensors` is used.
- Model checkpoints are saved as `./checkpoints/vae_ckpt.pt` and `./checkpoints/diffusion_ckpt.pt`.



