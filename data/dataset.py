import os
import random

import numpy as np
import tifffile as tiff
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from utils.logging import log


def load_volume_triplets(data_dir):
    """
    Returns list of (corrupted, gt, mask) triplets from directory.
    Assumes filenames follow pattern: {name}_corrupted.tif, {name}_gt.tif, {name}_mask.tif
    """
    triplets = []
    for f in os.listdir(data_dir):
        if f.endswith("_corrupted.tif"):
            name = f.replace("_corrupted.tif", "")
            corrupted = os.path.join(data_dir, f)
            gt = os.path.join(data_dir, f"{name}_gt.tif")
            mask = os.path.join(data_dir, f"{name}_mask.tif")
            if os.path.exists(gt) and os.path.exists(mask):
                triplets.append((corrupted, gt, mask))
    return sorted(triplets)


def normalize_slice_mask(mask):
    """Convert a stored mask to the 1D slice-wise binary format used throughout the repo."""
    if mask.ndim == 3:
        if np.all((mask == 0) | (mask == 1)):
            return mask[:, 0, 0]
        raise ValueError("Unexpected mask format: expected binary 0/1 per slice")
    if mask.ndim != 1:
        raise ValueError("Unsupported mask dimensionality")
    return mask


def get_kfold_splits(triplets, k=5, seed=42):
    """Build deterministic volume-level train/validation/test folds."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(triplets)):
        trainval_triplets = [triplets[i] for i in trainval_idx]

        # Reproducible shuffle for each fold
        rng = random.Random(seed + fold_idx)
        rng.shuffle(trainval_triplets)

        if len(trainval_triplets) <= 1:
            train_triplets = trainval_triplets
            val_triplets = []
        else:
            val_split = max(1, int(0.2 * len(trainval_triplets)))
            val_split = min(val_split, len(trainval_triplets) - 1)
            val_triplets = trainval_triplets[:val_split]
            train_triplets = trainval_triplets[val_split:]
        test_triplets = [triplets[i] for i in test_idx]

        folds.append((train_triplets, val_triplets, test_triplets))

    return folds


class OCTAInpaintingDataset(Dataset):
    def __init__(self, volume_triples, stack_size=5, static_corruptions=False, stride=1, seed=42):
        """
        Args:
            volume_triples (list): List of tuples [(corrupted_path, clean_path, mask_path)].
            stack_size (int): Number of slices in input stack (must be odd).
            static_corruptions (bool): If False, use online random dropouts (training mode).
                            If True, use fixed pre-corrupted data (validation/test mode).
            stride (int): Step size between consecutive target slices in online corruption mode.
                            (Default 1 for full overlap.)
        """
        assert stack_size % 2 == 1, "Stack size must be odd"
        self.stack_size = stack_size
        self.pad = stack_size // 2
        self.static_corruptions = static_corruptions
        self.stride = stride
        self.rng_np = np.random.default_rng(seed)
        self.rng_py = random.Random(seed)

        if static_corruptions:
            self.data = []
            self._build_static_samples(volume_triples)
        else:
            self.clean_volumes = []
            self.padded_volumes = []
            self.indices = []
            self._build_online_indices(volume_triples)

    def _build_static_samples(self, volume_triples):
        """Precompute fixed corrupted stacks used for validation and static evaluation."""
        for corrupted_path, clean_path, mask_path in volume_triples:
            corrupted = tiff.imread(corrupted_path).astype(np.float32)
            clean = tiff.imread(clean_path).astype(np.float32)

            corrupted = corrupted / (clean.max() + 1e-5)
            clean = clean / (clean.max() + 1e-5)
            mask = tiff.imread(mask_path)

            log(f"[INIT] Loaded volume: {os.path.basename(clean_path)}")

            assert corrupted.shape == clean.shape, "Corrupted and clean volumes must match shape"
            assert corrupted.shape[0] == mask.shape[0], "Number of slices in volume and mask must match"
            mask = normalize_slice_mask(mask)

            padded = np.pad(corrupted, ((self.pad, self.pad), (0, 0), (0, 0)), mode="edge")

            for idx in np.where(mask == 1)[0]:
                if idx < self.pad or idx >= corrupted.shape[0] - self.pad:
                    continue

                stack = padded[idx: idx + self.stack_size]
                target = clean[idx]
                self.data.append((stack, target))

    def _build_online_indices(self, volume_triples):
        """Record stack centers used for on-the-fly synthetic corruption during training."""
        for vol_idx, (_, clean_path, _) in enumerate(volume_triples):
            clean = tiff.imread(clean_path).astype(np.float32)
            clean = clean / (clean.max() + 1e-5)
            orig_len = clean.shape[0]

            self.clean_volumes.append(clean)
            padded = np.pad(clean, ((self.pad, self.pad), (0, 0), (0, 0)), mode="edge")
            self.padded_volumes.append(padded)

            for idx in range(self.pad, orig_len - self.pad, self.stride):
                self.indices.append((vol_idx, idx))

    def __len__(self):
        return len(self.data) if self.static_corruptions else len(self.indices)

    def sample_block_size(self, min_size, max_size, prob=0.4):
        """
        Sample block size from a geometric distribution with rejection sampling
        to avoid clumping at max_size. This matches the offline static corruption behavior.
        """
        while True:
            size = min_size + self.rng_np.geometric(prob) - 1
            if size <= max_size:
                return size

    def __getitem__(self, idx):
        if self.static_corruptions:
            stack, target = self.data[idx]
            stack = torch.from_numpy(stack).float()
            target = torch.from_numpy(target).float().unsqueeze(0)
        else:
            vol_idx, center_idx = self.indices[idx]
            padded_vol = self.padded_volumes[vol_idx]

            stack = padded_vol[center_idx: center_idx + self.stack_size].copy()
            target = padded_vol[center_idx + self.pad].copy()

            stack = stack.astype(np.float32)
            target = target.astype(np.float32)

            stack[self.pad] = 0.0

            block_size = self.sample_block_size(min_size=1, max_size=6, prob=0.4)

            neighbors = list(range(self.stack_size))
            neighbors.remove(self.pad)
            num_neighbors = len(neighbors)

            max_start_idx = num_neighbors - block_size
            if max_start_idx >= 0:
                start_idx = self.rng_py.randint(0, max_start_idx)
                block_indices = neighbors[start_idx: start_idx + block_size]
                for block_idx in block_indices:
                    stack[block_idx] = 0.0

            target = np.expand_dims(target, axis=0)

            stack = torch.from_numpy(stack).float()
            target = torch.from_numpy(target).float()

        valid_mask_stack = (stack.sum(dim=(1, 2)) > 1e-3).float()

        return stack, target, valid_mask_stack
