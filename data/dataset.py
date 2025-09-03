import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import tifffile as tiff
import numpy as np
import random
import os

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


def get_kfold_splits(triplets, k=5, seed=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(triplets)):
        trainval_triplets = [triplets[i] for i in trainval_idx]

        # Reproducible shuffle for each fold
        rng = random.Random(seed + fold_idx)
        rng.shuffle(trainval_triplets)

        val_split = max(1, int(0.2 * len(trainval_triplets)))
        val_triplets = trainval_triplets[:val_split]
        train_triplets = trainval_triplets[val_split:]
        test_triplets = [triplets[i] for i in test_idx]

        folds.append((train_triplets, val_triplets, test_triplets))

    return folds


class OCTAInpaintingDataset(Dataset):
    # def __init__(self, volume_triples: list, stack_size=5, static_corruptions=False, stride=1):
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
            # --- Static mode: use pre-generated corrupted volumes and masks ---
            self.data = []  # will hold (stack, target) pairs

            for corrupted_path, clean_path, mask_path in volume_triples:
                corrupted = tiff.imread(corrupted_path).astype(np.float32)
                clean = tiff.imread(clean_path).astype(np.float32)

                # corrupted = corrupted / (corrupted.max() + 1e-5)
                corrupted = corrupted / (clean.max() + 1e-5)
                clean = clean / (clean.max() + 1e-5)
                mask = tiff.imread(mask_path)

                log(f"[INIT] Loaded volume: {os.path.basename(clean_path)}")

                assert corrupted.shape == clean.shape, "Corrupted and clean volumes must match shape"
                assert corrupted.shape[0] == mask.shape[0], "Number of slices in volume and mask must match"

                # Convert mask to 1D binary array if needed
                if mask.ndim == 3:
                    if np.all((mask == 0) | (mask == 1)):
                        mask = mask[:, 0, 0]
                    else:
                        raise ValueError("Unexpected mask format: expected binary 0/1 per slice")
                elif mask.ndim != 1:
                    raise ValueError("Unsupported mask dimensionality")

                # Pad the corrupted volume on the slice dimension for context at edges
                padded = np.pad(corrupted, ((self.pad, self.pad), (0, 0), (0, 0)), mode='edge')

                # Build stacks only for slices marked as missing (mask==1)
                for idx in np.where(mask == 1)[0]:
                    # Skip slices too close to volume boundary (invalid context)
                    if idx < self.pad or idx >= corrupted.shape[0] - self.pad:
                        continue

                    # Extract stack around the target slice (centered)
                    stack = padded[idx: idx + stack_size]  # shape: (stack_size, H, W)
                    target = clean[idx]                    # shape: (H, W)
                    self.data.append((stack, target))

        else:
            # --- Online corruption mode: build indices for on-the-fly random dropouts ---
            self.clean_volumes = []   # list of clean volumes (np.float32 arrays)
            self.padded_volumes = []  # list of padded clean volumes for easy indexing
            self.indices = []         # list of (volume_index, center_slice_idx) pairs

            for vol_idx, (_, clean_path, _) in enumerate(volume_triples):
                clean = tiff.imread(clean_path).astype(np.float32)
                clean = clean / (clean.max() + 1e-5)
                orig_len = clean.shape[0]

                # Store the clean volume (for clarity) and also a padded copy for slicing
                self.clean_volumes.append(clean)
                padded = np.pad(clean, ((self.pad, self.pad), (0, 0), (0, 0)), mode='edge')
                self.padded_volumes.append(padded)

                # For every valid center slice (skipping edges for full context), 
                # step by 'stride' to create overlapping stacks
                for idx in range(self.pad, orig_len - self.pad, self.stride):
                    self.indices.append((vol_idx, idx))


    def __len__(self):
        if self.static_corruptions:
            return len(self.data)
        else:
            return len(self.indices)
        

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
            # --- Static corruption mode: return precomputed (stack, target) pair ---
            stack, target = self.data[idx]
            stack = torch.from_numpy(stack).float()
            target = torch.from_numpy(target).float().unsqueeze(0)
        else:
            # --- Online corruption mode: generate stack and apply random dropouts ---
            vol_idx, center_idx = self.indices[idx]
            padded_vol = self.padded_volumes[vol_idx]
            
            # Extract the full stack around center_idx
            stack = padded_vol[center_idx: center_idx + self.stack_size].copy()  # shape: (stack_size, H, W)
            # The true target slice is the center slice (no dropout)
            # Note: padded_vol[center_idx + self.pad] corresponds to the original clean slice
            target = padded_vol[center_idx + self.pad].copy()  # shape: (H, W)

            stack = stack.astype(np.float32)
            target = target.astype(np.float32)

            # Always drop the center slice
            stack[self.pad] = 0.0

            # Sample a block size from the same distribution as static corruptions
            block_size = self.sample_block_size(min_size=1, max_size=6, prob=0.4)


            # Choose a starting position for the block among the neighbors
            # The block must fit entirely within the neighbor slices
            neighbors = list(range(self.stack_size))
            neighbors.remove(self.pad)  # exclude center
            num_neighbors = len(neighbors)

            # Compute possible start positions so block doesn't exceed neighbor bounds
            max_start_idx = num_neighbors - block_size
            if max_start_idx >= 0:
                start_idx = self.rng_py.randint(0, max_start_idx)
                block_indices = neighbors[start_idx: start_idx + block_size]
                # Drop the contiguous block of neighbors
                for idx in block_indices:
                    stack[idx] = 0.0

            target = np.expand_dims(target, axis=0)  # (1, H, W)

            # Convert to tensors
            stack = torch.from_numpy(stack).float()  # (stack_size, H, W)
            target = torch.from_numpy(target).float()  # (1, H, W)


        # Validity mask: 1 if slice has non-zero content, else 0
        valid_mask_stack = (stack.sum(dim=(1, 2)) > 1e-3).float()  # (stack_size,)

        return stack, target, valid_mask_stack
    