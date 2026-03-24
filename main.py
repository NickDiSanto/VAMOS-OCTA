import argparse
import os
import random

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader

from data.dataset import (
    OCTAInpaintingDataset,
    get_kfold_splits,
    load_volume_triplets,
    normalize_slice_mask,
)
from evaluation.evaluate_bscans import evaluate_bscans
from evaluation.evaluate_mip import evaluate_projection
from models.unet2p5d import UNet2p5D
from scripts.inpaint_test_volume import inpaint_volume_with_model
from scripts.train import train
from scripts.vamos_loss import VAMOS_Loss
from utils.logging import log
from utils.visualizations import (
    visualize_ncc_slice_stacked,
    visualize_slice_panel,
    visualize_ssim_slice_stacked,
)


def set_global_seed(seed: int = 42):
    """Set the repository's global random-state configuration."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2.5D Inpainting Pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to root directory containing corrupted/gt/mask volumes")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--stack_size", type=int, default=9, help="Number of slices to stack for 2.5D input")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weighted_mse", type=float, default=1.0, help="Weight for the vessel-weighted MSE term")
    parser.add_argument("--axial_mip", type=float, default=3.0, help="Weight for axial MIP loss")
    parser.add_argument("--lateral_mip", type=float, default=3.0, help="Weight for lateral MIP loss")
    parser.add_argument("--axial_aip", type=float, default=3.0, help="Weight for axial AIP loss")
    parser.add_argument("--lateral_aip", type=float, default=3.0, help="Weight for lateral AIP loss")
    parser.add_argument("--wl_alpha", type=float, default=100.0, help="Alpha parameter for vessel-weighted loss shaping")
    parser.add_argument("--wl_gamma", type=float, default=1 / 3, help="Gamma parameter for vessel-weighted loss shaping")
    parser.add_argument("--disable_wl_weighting", action="store_true", help="Disable vessel weighting for the MSE term")
    parser.add_argument("--features", type=int, nargs="+", default=[64, 128, 256, 512], help="Feature channels for U-Net layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (0 disables dropout)")
    parser.add_argument("--enable_3d_input", action="store_true", help="Enable initial 3D preprocessing before the 2D U-Net")
    parser.add_argument("--static_corruptions", action="store_true", help="Use pre-generated corrupted volumes for training instead of online corruption")
    parser.add_argument("--stride", type=int, default=2, help="Stride for dynamic slice sampling in online corruption mode")
    parser.add_argument("--kfold", action="store_true", help="Run full k-fold cross-validation")
    parser.add_argument("--fold_idx", type=int, default=1, help="If not using --kfold, zero-based fold index to run")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only run inference on the test set")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debugging logs")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of times to repeat training/evaluation; outputs are kept separate when > 1")
    parser.add_argument("--output_dir", type=str, default="output/inpainted", help="Directory to save inpainted outputs")
    return parser.parse_args()


def _log_pipeline_arguments(args):
    log("Pipeline arguments:")
    for arg, value in vars(args).items():
        log(f"  {arg}: {value}")


def _log_volume_group(title, volume_triplets):
    log(title)
    for corrupted_path, _, _ in volume_triplets:
        log(f" - {os.path.basename(corrupted_path)}")


def _build_data_loaders(train_vols, val_vols, args, fold_idx):
    train_dataset = OCTAInpaintingDataset(
        train_vols,
        stack_size=args.stack_size,
        static_corruptions=args.static_corruptions,
        stride=args.stride,
    )
    val_dataset = OCTAInpaintingDataset(
        val_vols,
        stack_size=args.stack_size,
        static_corruptions=True,
    )
    # DataLoader shuffling from an explicit generator so batch order stays reproducible.
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    if len(train_dataset) == 0:
        raise ValueError(f"Fold {fold_idx + 1} produced an empty training dataset. Adjust the split or dataset contents.")
    if len(val_dataset) == 0:
        raise ValueError(f"Fold {fold_idx + 1} produced an empty validation dataset. Check the validation masks and stack size.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def _build_model_and_loss(args, device):
    log("Initializing model...")
    model = UNet2p5D(
        in_channels=args.stack_size,
        out_channels=1,
        features=args.features,
        dropout_rate=args.dropout,
        enable_3d_input=args.enable_3d_input,
    ).to(device)

    criterion = VAMOS_Loss(
        lambda_weighted_mse=args.weighted_mse,
        lambda_mip_axial=args.axial_mip,
        lambda_mip_lateral=args.lateral_mip,
        lambda_aip_axial=args.axial_aip,
        lambda_aip_lateral=args.lateral_aip,
        wl_alpha=args.wl_alpha,
        wl_gamma=args.wl_gamma,
        disable_wl_weighting=args.disable_wl_weighting,
    )
    return model, criterion


def _build_output_path(test_corrupted_path, output_dir, run_suffix):
    base_name = os.path.basename(test_corrupted_path).replace("_corrupted.tif", "")
    output_filename = f"{base_name}_VAMOS-OCTA{run_suffix}.tif" if run_suffix else f"{base_name}_VAMOS-OCTA.tif"
    return os.path.join(output_dir, output_filename)


def main(args, device):
    log("Starting Inpainting Pipeline")
    log(f"Device: {device}")
    _log_pipeline_arguments(args)

    log(f"Loading datasets from {args.data_dir}")

    volume_triplets = load_volume_triplets(args.data_dir)
    if len(volume_triplets) < 3:
        raise ValueError("At least 3 volume triplets are required to build train/validation/test splits.")

    folds = get_kfold_splits(volume_triplets, k=len(volume_triplets))
    if args.kfold:
        fold_range = range(len(folds))
    else:
        fold_range = [args.fold_idx]

    for run_idx in range(args.num_runs):
        log(f"\n===== Run {run_idx + 1} of {args.num_runs} =====")

        for fold_idx in fold_range:
            train_vols, val_vols, test_vols = folds[fold_idx]
            log(f"\n========== Fold {fold_idx + 1} ==========")
            _log_volume_group("Training volumes:", train_vols)
            _log_volume_group("Validation volume(s):", val_vols)
            _log_volume_group("Test volume(s):", test_vols)

            log(f"Using {len(train_vols)} volumes for training, {len(val_vols)} for validation, {len(test_vols)} for testing")

            train_loader, val_loader = _build_data_loaders(train_vols, val_vols, args, fold_idx)
            model, criterion = _build_model_and_loss(args, device)

            run_suffix = f"_run{run_idx + 1}" if args.num_runs > 1 else ""
            best_model_path = f"output/best_model{run_suffix}_fold{fold_idx + 1}.pth"

            if not args.skip_train:
                train(model, train_loader, val_loader, criterion, best_model_path, device, args)

            state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)

            for test_idx, (test_corrupted_path, test_gt_path, test_mask_path) in enumerate(test_vols):
                log(f"\nEvaluating test volume {test_idx + 1}/{len(test_vols)}: {os.path.basename(test_corrupted_path)}")

                gt_volume = tiff.imread(test_gt_path)
                corrupted_volume = tiff.imread(test_corrupted_path)
                mask = normalize_slice_mask(tiff.imread(test_mask_path))

                log("Running single-pass inpainting...")
                inpainted = inpaint_volume_with_model(
                    model, gt_volume, corrupted_volume, device, args.stack_size, args.debug, args
                )
                inpainted = inpainted[0] if isinstance(inpainted, tuple) else inpainted

                os.makedirs(args.output_dir, exist_ok=True)
                output_path = _build_output_path(test_corrupted_path, args.output_dir, run_suffix)
                tiff.imwrite(output_path, inpainted.astype(np.uint16))
                log(f"Saved inpainted volume to: {output_path}")
                # Metrics are reported in normalized float space, while saved outputs stay uint16 on disk
                gt_max = gt_volume.max() + 1e-5
                gt_volume = gt_volume.astype(np.float32) / gt_max
                inpainted = inpainted.astype(np.float32) / gt_max

                log("Evaluating single-pass inpainting:")
                evaluate_bscans(gt_volume, inpainted, mask)
                evaluate_projection(gt_volume, inpainted)

                if args.debug:
                    masked_indices = np.where(mask)[0]
                    if len(masked_indices) == 0:
                        log("[DEBUG] No masked slices found; skipping visualization.")
                    else:
                        slice_idx = masked_indices[0]
                        visualize_ncc_slice_stacked(gt_volume, inpainted, slice_idx=slice_idx, window_sizes=[11, 17, 23])
                        visualize_ssim_slice_stacked(gt_volume, inpainted, slice_idx=slice_idx, window_sizes=[7, 11, 17])
                        visualize_slice_panel(gt_volume, inpainted, mask, slice_indices=masked_indices[:3], ncols=3)


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args, device)
