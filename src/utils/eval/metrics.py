import os
import torch
import nibabel as nib
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchmetrics import Metric
from monai.metrics import SSIMMetric

# --- Metric Classes provided by user ---

class PSNR3D(Metric):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.add_state("sum_psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.data_range = data_range

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.clamp((preds + 1.0) / 2.0, 0.0, 1.0)
        target = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
        mse = torch.mean((preds - target) ** 2, dim=(1, 2, 3, 4))
        psnr = 20 * torch.log10(self.data_range / torch.sqrt(mse + 1e-8))
        self.sum_psnr += torch.sum(psnr)
        self.total += preds.shape[0]

    def compute(self):
        return self.sum_psnr / self.total

class SSIM3D(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_ssim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.ms_ssim_func = SSIMMetric(spatial_dims=3, data_range=1.0, reduction="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.clamp((preds + 1.0) / 2.0, 0.0, 1.0)
        target = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
        score = self.ms_ssim_func(y_pred=preds, y=target)
        self.sum_ssim += score.item() * preds.shape[0]
        self.total += preds.shape[0]

    def compute(self):
        return self.sum_ssim / self.total

# --- Helper Function for Execution ---

import torchio as tio

def evaluate_reconstructions(target_dir, pred_dir, device, input_size=224):
    """
    Evaluates 3D reconstructions with on-the-fly preprocessing for target images.
    """
    psnr_calc = PSNR3D().to(device)
    ssim_calc = SSIM3D().to(device)
    results_map = {}
    
    # Define the same spatial and intensity pipeline used in training
    # Note: We apply this to the target to match the model's output space
    preprocess_pipeline = tio.Compose([
        tio.Resample((1.0, 1.0, 1.0)),
        tio.CropOrPad((input_size, input_size, input_size)),
        tio.ZNormalization(), 
    ])

    filenames = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.nii', '.nii.gz'))])

    for fname in tqdm(filenames, desc="Preprocessing & Evaluating"):
        t_path = os.path.join(target_dir, fname)
        p_path = os.path.join(pred_dir, fname)

        if not os.path.exists(t_path):
            continue

        target_subject = tio.Subject(image=tio.ScalarImage(t_path))
        transformed_target = preprocess_pipeline(target_subject)
        t_vol = transformed_target.image.data.unsqueeze(0).to(device) # [1, 1, D, H, W]

        pred_subject = tio.Subject(image=tio.ScalarImage(p_path))
        p_vol = pred_subject.image.data.unsqueeze(0).to(device) # [1, 1, D, H, W]

        def min_max_scale(tensor):
            t_min, t_max = tensor.min(), tensor.max()
            return (tensor - t_min) / (t_max - t_min + 1e-8)

        p_vol_scaled = min_max_scale(p_vol)
        t_vol_scaled = min_max_scale(t_vol)

        with torch.no_grad():
            psnr_calc.update(p_vol_scaled, t_vol_scaled)
            ssim_calc.update(p_vol_scaled, t_vol_scaled)
            
            current_ssim = ssim_calc.ms_ssim_func(y_pred=p_vol_scaled, y=t_vol_scaled).item()
            mse = torch.mean((p_vol_scaled - t_vol_scaled)**2).item()
            current_psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))

            results_map[fname] = {"PSNR": current_psnr, "SSIM": current_ssim}

    return results_map
# --- Analysis Function provided by user ---

def summarize_evaluation(results_map):
    df = pd.DataFrame.from_dict(results_map, orient='index')
    mean_ssim = df['SSIM'].mean()
    std_ssim = df['SSIM'].std()
    mean_psnr = df['PSNR'].mean()
    std_psnr = df['PSNR'].std()
    
    top_5_keys = df.sort_values(by='SSIM', ascending=False).head(5).index.tolist()
    bottom_5_keys = df.sort_values(by='SSIM', ascending=True).head(5).index.tolist()
    
    print("\n--- MS-SSIM Statistical Summary ---")
    print(f"Mean: {mean_ssim:.4f} ± {std_ssim:.4f}")
    return {
        "mean_ssim": mean_ssim,
        "std_ssim": std_ssim,
        "mean_psnr": mean_psnr,
        "std_psnr": std_psnr,
        "top5_files": top_5_keys,
        "bottom_5_keys": bottom_5_keys
    }

# --- Main Entry Point ---

def main(pred_path, target_path):
    assert os.path.isdir(pred_path), f"{pred_path} must be a directory"
    assert os.path.isdir(target_path), f"{target_path} must be a directory"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting evaluation on {device}...")

    # Run core evaluation
    results_map = evaluate_reconstructions(target_path, pred_path, device)

    # Analyze and print summary
    summary = summarize_evaluation(results_map)
    
    return results_map, summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Evaluate', add_help=False)
    parser.add_argument('--pred_path', type=str)
    parser.add_argument('--target_dir', type=str)
    args = parser.parse_args()
    pred_path = args.pred_path
    target_path = args.target_dir
    results, stats = main(pred_path, target_path)
    print(stats)
    