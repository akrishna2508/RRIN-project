"""
src/datasets.py
===============
PyTorch Dataset classes — the bridge between raw image files on disk
and mini-batches the training loop consumes.

THREE TIERS:
  Tier 1 — RetinaRestorationDataset
      Loads a pseudo-clean image, synthetically degrades it on-the-fly,
      and returns (degraded_4ch_input, clean_3ch_target) pairs.
      This is the primary training data.

  Tier 2 — Noise2NoiseDuplicateCaptureDataset
      Two real, independently-degraded captures of the same eye.
      Used for self-supervised fine-tuning.

  Tier 3 — UnpairedRealDegradedDataset
      Real degraded images with NO paired clean counterpart.
      Used in the CycleGAN domain-adaptation stage.

HOW PYTORCH DATASETS WORK (for beginners):
  A Dataset is like a list of examples. The DataLoader pulls batches
  from it automatically during training. You only need to implement:
    __len__()     → how many items are in the dataset
    __getitem__(i) → return the i-th (input, target) pair
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A

from src.config import CROP_SIZE, RANDOM_SEED
from src.degradation import compose_random_degradation_pipeline
from src.quality_scoring import extract_field_of_view_mask
from src.utils.image_utils import (
    build_four_channel_input_tensor,
    center_crop_pair,
    center_crop_single,
    load_image_as_float_array,
    normalize_and_to_tensor,
    resize_image,
)
from src.config import IMAGE_SIZE


# ---- Augmentation pipelines --------------------------------

def get_synchronized_geometric_augmentations(crop_size: int) -> A.Compose:
    """
    Build an albumentations pipeline that applies the SAME random geometric
    transforms to BOTH the degraded input and the clean target simultaneously.

    This is critical: if we flip the input but not the target, the AI gets
    confused about what it's supposed to produce.

    Only geometric transforms here (flips, rotations, crops).
    Colour/intensity transforms are NOT applied to the target.

    params: crop_size — patch size after random crop
    returns: albumentations.Compose pipeline
    side effects: none
    """
    return A.Compose(
        [
            A.RandomCrop(height=crop_size, width=crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"target_image": "image"},
    )


def get_input_only_appearance_augmentation() -> A.Compose:
    """
    Build a pipeline for mild intensity/colour augmentation applied ONLY
    to the degraded input, NEVER to the clean target.

    This teaches the network to handle varying camera contrast and colour
    without ever changing what it is trying to predict.

    params: none
    returns: albumentations.Compose pipeline
    side effects: none
    """
    return A.Compose(
        [
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.3),
        ]
    )


# ---- Tier 1: Primary synthetic-paired dataset ---------------

class RetinaRestorationDataset(Dataset):
    """
    Primary Tier-1 dataset.

    For every image in the pseudo-clean pool assigned to `split_name`:
      1. Load the pseudo-clean image from disk
      2. Extract the field-of-view mask
      3. Randomly degrade it (fresh random seed every __getitem__ call!)
      4. Apply synchronized geometric augmentation (training only)
      5. Apply input-only colour augmentation (training only)
      6. Normalise to [-1, 1] and build tensors
      7. Return (4-channel degraded input, 3-channel clean target)

    Because the degradation is re-randomised on every call to __getitem__,
    each "epoch" effectively sees a different corrupted version of each
    source image — infinite effective diversity from a finite set of clean images.
    """

    def __init__(
        self,
        metadata_dataframe: pd.DataFrame,
        split_name: str,
        is_training: bool,
    ) -> None:
        """
        params:
            metadata_dataframe — full DataFrame with split_assignment and is_pseudo_clean
            split_name         — "train", "val", or "test"
            is_training        — True for training (enables augmentation and degradation)
        returns: None
        side effects: stores a filtered subset of the DataFrame as an instance attribute
                      (no images are loaded from disk yet — loading is deferred to __getitem__)
        """
        mask = (
            (metadata_dataframe["split_assignment"] == split_name)
            & (metadata_dataframe["is_pseudo_clean"] == True)
        )
        self.rows = metadata_dataframe[mask].reset_index(drop=True)
        self.is_training   = is_training
        self.split_name    = split_name

        self.geo_aug    = get_synchronized_geometric_augmentations(CROP_SIZE) if is_training else None
        self.appear_aug = get_input_only_appearance_augmentation() if is_training else None

        print(f"  Dataset [{split_name}]: {len(self.rows)} pseudo-clean images")

    def __len__(self) -> int:
        """Returns the number of pseudo-clean source images in this split."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Load image at `index`, degrade it, augment it, and return tensors.

        params: index — integer index into self.rows
        returns:
            input_tensor  — (4, CROP_SIZE, CROP_SIZE) in [-1, 1]
            target_tensor — (3, CROP_SIZE, CROP_SIZE) in [-1, 1]
        side effects: disk I/O to load the source image
        """
        row   = self.rows.iloc[index]
        path  = row["file_path"]

        # Load pseudo-clean image
        pseudo_clean = load_image_as_float_array(path)

        # Resize to a consistent size before cropping
        pseudo_clean = resize_image(pseudo_clean, IMAGE_SIZE)

        # Extract field-of-view mask
        uint8_img = (pseudo_clean * 255).astype(np.uint8)
        fov_mask  = extract_field_of_view_mask(uint8_img)

        if self.is_training:
            # Generate a FRESH synthetic degradation every time
            degraded = compose_random_degradation_pipeline(pseudo_clean, fov_mask)

            # Apply IDENTICAL geometric transforms to both images
            augmented      = self.geo_aug(image=degraded, target_image=pseudo_clean)
            degraded       = augmented["image"]
            pseudo_clean   = augmented["target_image"]

            # Apply colour augmentation to INPUT ONLY (never the target)
            degraded = self.appear_aug(image=degraded)["image"]
        else:
            # For val/test: generate one deterministic degradation (centre-crop only)
            # We still generate synthetic degradation so val metrics are comparable
            np.random.seed(index)  # deterministic for val/test
            degraded = compose_random_degradation_pipeline(pseudo_clean, fov_mask)
            np.random.seed(None)   # restore randomness for other operations
            degraded, pseudo_clean = center_crop_pair(degraded, pseudo_clean, CROP_SIZE)

        input_tensor  = build_four_channel_input_tensor(degraded)      # (4, H, W)
        target_tensor = normalize_and_to_tensor(pseudo_clean)           # (3, H, W)
        return input_tensor, target_tensor


# ---- Tier 2: Noise2Noise duplicate-capture dataset ----------

class Noise2NoiseDuplicateCaptureDataset(Dataset):
    """
    Tier-2 self-supervised dataset using real duplicate captures.

    When two independent captures of the SAME eye exist (common in
    clinical workflow where a technician retakes a blurry shot), we can
    train the network to map one noisy capture to the other.

    Neither image is a "clean" ground truth — but mathematically,
    training this way converges to the same solution as training on
    true clean targets (Lehtinen et al. 2018, Noise2Noise).

    DataFrame must have columns: file_path_capture_a, file_path_capture_b
    """

    def __init__(self, duplicate_capture_pairs_dataframe: pd.DataFrame) -> None:
        self.rows = duplicate_capture_pairs_dataframe.reset_index(drop=True)
        print(f"  Noise2Noise dataset: {len(self.rows)} duplicate capture pairs")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Load both independent captures of the same eye visit.

        params: index — integer
        returns:
            capture_a_tensor — (4, CROP_SIZE, CROP_SIZE) — the input
            capture_b_tensor — (3, CROP_SIZE, CROP_SIZE) — the Noise2Noise target
        side effects: disk I/O
        """
        row = self.rows.iloc[index]
        cap_a = load_image_as_float_array(row["file_path_capture_a"])
        cap_b = load_image_as_float_array(row["file_path_capture_b"])

        cap_a = resize_image(cap_a, IMAGE_SIZE)
        cap_b = resize_image(cap_b, IMAGE_SIZE)

        cap_a, cap_b = center_crop_pair(cap_a, cap_b, CROP_SIZE)

        return build_four_channel_input_tensor(cap_a), normalize_and_to_tensor(cap_b)


# ---- Tier 3: Unpaired real-degraded dataset ----------------

class UnpairedRealDegradedDataset(Dataset):
    """
    Tier-3 domain-adaptation dataset.

    Real degraded images with NO paired clean counterpart.
    Used in the CycleGAN-style adversarial domain-adaptation stage
    to close the gap between "what our synthetic degradation produces"
    and "what real camera artefacts look like."

    No target image is needed — these go into the discriminator only.
    """

    def __init__(self, real_degraded_metadata_dataframe: pd.DataFrame) -> None:
        self.rows = real_degraded_metadata_dataframe.reset_index(drop=True)
        print(f"  Tier-3 unpaired dataset: {len(self.rows)} real degraded images")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.FloatTensor:
        """
        Load one real degraded image.

        params: index — integer
        returns: real_degraded_tensor — (4, CROP_SIZE, CROP_SIZE)
        side effects: disk I/O
        """
        row  = self.rows.iloc[index]
        img  = load_image_as_float_array(row["file_path"])
        img  = resize_image(img, IMAGE_SIZE)
        img  = center_crop_single(img, CROP_SIZE)
        return build_four_channel_input_tensor(img)
