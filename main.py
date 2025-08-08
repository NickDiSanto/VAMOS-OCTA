import os
import torch
from torch.utils.data import DataLoader
import tifffile as tiff
import numpy as np
import argparse
import random

from data.dataset import load_volume_triplets, get_kfold_splits, OCTAInpaintingDataset
from models.unet2p5d import UNet2p5D
from scripts.train import train
from scripts.inpaint_test_volume import evaluate_model_on_test, inpaint_volume_with_model
from scripts.vamos_loss import VAMOS_Loss
from evaluation.evaluate_bscans import evaluate_bscans
from evaluation.evaluate_mip import evaluate_projection
from utils.logging import log
from utils.visualizations import visualize_ncc_slice_stacked, visualize_ssim_slice_stacked, visualize_slice_panel


def set_global_seed(seed: int = 42):
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to root directory containing corrupted/gt/mask volumes')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--stack_size', type=int, default=9, help='Number of slices to stack for 2.5D input')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for AdamW optimizer')
    parser.add_argument('--wl_alpha', type=float, default=100.0, help='Weight for WL loss')
    parser.add_argument('--wl_gamma', type=float, default=1/3, help='Gamma for WL loss')
    parser.add_argument('--disable_wl_weighting', action='store_true', help='Disable weighting of WL loss')
    parser.add_argument('--features', type=int, nargs='+', default=[64, 128, 256, 512], help='Feature channels for UNet layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (0 to disable)')
    parser.add_argument('--enable_3d_input', action='store_true', help='Enable 3D input processing (otherwise, use 2D slices only)')
    parser.add_argument('--static_corruptions', action='store_true', help='Use static offline corruptions for training (default: online corruptions)')
    parser.add_argument('--stride', type=int, default=2, help='Stride for dynamic slicing (default: 1)')
    parser.add_argument('--detect_bright_outliers', action='store_true', help='Automatically detect and inpaint bright slices')
    parser.add_argument('--kfold', action='store_true', help='Run full k-fold cross-validation')
    parser.add_argument('--fold_idx', type=int, default=1, help='If not kfold mode, which fold to run (default: 0)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 6 epochs)')
    parser.add_argument('--skip_train', action='store_true', help='Skip training and only run inference on the test set')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debugging logs')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of times to repeat training for averaging metrics')
    parser.add_argument('--output_dir', type=str, default='output/inpainted', help='Directory to save inpainted outputs')
    return parser.parse_args()


def main(args, device):
    log("Starting Inpainting Pipeline")
    log(f"Device: {device}")

    # Print all arguments
    log("Pipeline arguments:")
    for arg, value in vars(args).items():
        log(f"  {arg}: {value}")

    # Load Dataset
    log("Loading datasets...")
    # Load and split volumes
    volume_triplets = load_volume_triplets(args.data_dir)
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
            log("Training volumes:")
            for v in train_vols: log(f" - {os.path.basename(v[0])}")
            log("Validation volume:")
            for v in val_vols: log(f" - {os.path.basename(v[0])}")
            log("Test volume:")
            for v in test_vols: log(f" - {os.path.basename(v[0])}")

            log(f"Using {len(train_vols)} volumes for training, {len(val_vols)} for validation, {len(test_vols)} for testing")

            # Create dynamic training dataset
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

            test_dataset = OCTAInpaintingDataset(
                test_vols,
                stack_size=args.stack_size,
                static_corruptions=True
            )

            g = torch.Generator()
            g.manual_seed(args.seed)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            # Initialize Model
            log("Initializing model...")
            model = UNet2p5D(
                in_channels=args.stack_size,
                out_channels=1,
                features=args.features,
                dropout_rate=args.dropout,
                enable_3d_input=args.enable_3d_input
            ).to(device)

            # Vessel-Weighted MSE + Axial MIP/AIP Loss + Coronal MIP/AIP Loss
            criterion = VAMOS_Loss(
                lambda_weighted_mse=1.0,
                lambda_mip_axial=3.0,
                lambda_mip_coronal=9.0,
                lambda_aip_axial=3.0,
                lambda_aip_coronal=3.0,
                wl_alpha=args.wl_alpha,
                wl_gamma=args.wl_gamma,
                disable_wl_weighting=args.disable_wl_weighting
            )

            best_model_path = f"output/best_model_fold{fold_idx + 1}.pth"

            # If skip training, load the best model directly
            if not args.skip_train:
                train(model, train_loader, val_loader, criterion, best_model_path, device, args)


            # Evaluate on Held-Out Test Volume
            log("Evaluating on held-out test volume...")
            model.load_state_dict(torch.load(best_model_path))
            test_loss = evaluate_model_on_test(model, test_loader, criterion, device)
            log(f"Final test loss: {test_loss:.4f}")
            
            for test_idx, (test_corrupted_path, test_gt_path, test_mask_path) in enumerate(test_vols):
                log(f"\nEvaluating test volume {test_idx + 1}/{len(test_vols)}: {os.path.basename(test_corrupted_path)}")

                gt_volume = tiff.imread(test_gt_path)
                corrupted_volume = tiff.imread(test_corrupted_path)
                mask = tiff.imread(test_mask_path)

                # Convert mask to 1D binary array if needed
                if mask.ndim == 3:
                    if np.all((mask == 0) | (mask == 1)):
                        mask = mask[:, 0, 0]
                    else:
                        raise ValueError("Unexpected mask format: expected binary 0/1 per slice")
                elif mask.ndim != 1:
                    raise ValueError("Unsupported mask dimensionality")

                # -- Single-pass inpainting
                log("Running singe-pass inpainting...")
                inpainted = inpaint_volume_with_model(
                    model, gt_volume, corrupted_volume, args.detect_bright_outliers, device, args.stack_size, args.debug, args
                )
                inpainted = inpainted[0] if isinstance(inpainted, tuple) else inpainted


                # Save output
                base_name = os.path.basename(test_corrupted_path).replace("_corrupted.tif", "")
                output_path = os.path.join(
                    args.output_dir,
                    f"{base_name}_VAMOS-OCTA.tif"
                )

                os.makedirs(args.output_dir, exist_ok=True)
                tiff.imwrite(output_path, inpainted.astype(np.uint16))
                log(f"Saved inpainted volume to: {output_path}")

                # Normalize volumes for evaluation
                gt_max = gt_volume.max() + 1e-5
                gt_volume = gt_volume.astype(np.float32) / gt_max
                inpainted = inpainted.astype(np.float32) / gt_max

                log("Evaluating single-pass inpainting:")
                evaluate_bscans(gt_volume, inpainted, mask)
                evaluate_projection(gt_volume, inpainted)


    if args.debug:
        # Use a specific corrupted slice
        slice_idx = np.where(mask)[0][0]

        visualize_ncc_slice_stacked(gt_volume, inpainted, slice_idx=slice_idx, window_sizes=[11, 17, 23])
        visualize_ssim_slice_stacked(gt_volume, inpainted, slice_idx=slice_idx, window_sizes=[7, 11, 17])
        visualize_slice_panel(gt_volume, inpainted, mask, slice_indices=np.where(mask)[0][:3], ncols=3)


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)  # Set a global seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args, device)
