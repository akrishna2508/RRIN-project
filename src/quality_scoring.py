"""
src/quality_scoring.py
=======================
Computes four image quality sub-scores that together identify which
fundus photographs are "clean enough" to serve as Tier-1 pseudo-clean
training targets.

WHY WE NEED THIS (for beginners):
  We cannot just use any retina photo as a training target. If we train
  the AI to reproduce a blurry or reflection-covered image, it will LEARN
  to produce blurry, reflection-covered outputs.
  These scoring functions automatically filter out low-quality images.
"""

import cv2
import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm

from src.database import update_quality_scores_in_database
from src.utils.image_utils import load_image_as_float_array
from src.config import QUALITY_QUANTILE_THRESHOLD


# ---- FOV mask extraction -----------------------------------

def keep_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected white region in a binary (0/255) mask.
    Used to isolate the circular fundus disc from noise in the green channel.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    # Component 0 is always the background; start from 1
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8) * 255


def extract_field_of_view_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a binary boolean mask marking the circular fundus disc region.

    params: image_rgb — numpy array, shape (H, W, 3), dtype uint8 [0..255]
    returns: fov_mask — numpy array, shape (H, W), dtype bool

    HOW IT WORKS:
      1. Look at the green channel (best contrast for the disc vs. black border)
      2. Threshold at intensity=10 to find all non-black pixels
      3. Morphological closing fills small holes inside the disc
      4. Keep only the largest connected region (removes stray bright spots)

    side effects: none — pure function
    """
    green_channel = image_rgb[:, :, 1]
    _, binary_mask = cv2.threshold(green_channel, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((25, 25), dtype=np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    largest = keep_largest_connected_component(closed_mask)
    return largest.astype(bool)


# ---- Individual quality sub-scores -------------------------

def compute_sharpness_score(image_rgb: np.ndarray, fov_mask: np.ndarray) -> float:
    """
    Score = variance of the Laplacian of the green channel inside the FOV mask.

    HIGH score ↔ sharp image (lots of high-frequency edge content).
    LOW score  ↔ blurry image (smooth, low-frequency — the Laplacian is nearly zero).

    params:
        image_rgb — uint8 numpy array (H, W, 3)
        fov_mask  — bool mask (H, W)
    returns: sharpness_score (float, higher = sharper)
    side effects: none
    """
    green_channel    = image_rgb[:, :, 1]
    laplacian        = cv2.Laplacian(green_channel, cv2.CV_64F)
    return float(np.var(laplacian[fov_mask]))


def compute_illumination_uniformity_score(
    image_rgb: np.ndarray,
    fov_mask: np.ndarray,
) -> float:
    """
    Score = standard deviation of luminance on a coarse 32×32 downsampled grid.

    HIGH score ↔ uneven illumination (vignetting, hot-spots) — BAD.
    LOW score  ↔ uniform illumination — GOOD.

    NOTE: This score is INVERTED before adding to the composite (lower raw value = better).

    params: image_rgb (uint8), fov_mask (bool)
    returns: illumination_uniformity_score (float, LOWER is better)
    side effects: none
    """
    lab        = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    luminance  = lab[:, :, 0].astype(np.float32)
    downsampled = cv2.resize(luminance, (32, 32), interpolation=cv2.INTER_AREA)
    return float(np.std(downsampled))


def compute_reflection_coverage_score(
    image_rgb: np.ndarray,
    fov_mask: np.ndarray,
) -> float:
    """
    Score = fraction of FOV pixels that are both VERY BRIGHT and DESATURATED.
    Specular highlights (corneal reflections) are near-white (bright + no colour).

    HIGH score ↔ lots of reflections — BAD.
    LOW score  ↔ clean image — GOOD.

    NOTE: This score is INVERTED before adding to the composite.

    params: image_rgb (uint8), fov_mask (bool)
    returns: reflection_coverage_score (float, fraction 0..1, LOWER is better)
    side effects: none
    """
    hsv               = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    value_channel     = hsv[:, :, 2]   # brightness (V in HSV)
    saturation_channel = hsv[:, :, 1]  # colour saturation (S in HSV)

    # Threshold: top 5% brightest pixels AND saturation < 40/255 (near-white)
    brightness_threshold = np.percentile(value_channel[fov_mask], 95)
    reflection_pixels    = (
        (value_channel > brightness_threshold)
        & (saturation_channel < 40)
        & fov_mask
    )
    return float(np.sum(reflection_pixels)) / float(np.sum(fov_mask) + 1e-8)


def compute_fov_completeness_score(
    fov_mask: np.ndarray,
    expected_fraction: float = 0.65,
) -> float:
    """
    Score = absolute deviation of the FOV disc area from the expected fraction.

    A perfectly-framed fundus image fills ~65% of the frame.
    Severely cropped or off-centre captures deviate from this.

    LOW score ↔ well-framed image — GOOD.
    HIGH score ↔ severely cropped or off-centre — BAD.

    NOTE: This score is INVERTED before adding to the composite.

    params: fov_mask (bool), expected_fraction (float, default 0.65)
    returns: fov_completeness_score (float, absolute deviation, LOWER is better)
    side effects: none
    """
    actual_fraction = float(np.sum(fov_mask)) / float(fov_mask.size + 1e-8)
    return abs(actual_fraction - expected_fraction)


def compute_composite_quality_score(
    sharpness_score: float,
    illumination_uniformity_score: float,
    reflection_coverage_score: float,
    fov_completeness_score: float,
) -> float:
    """
    Combine the four sub-scores (already normalised to [0, 1] by the caller)
    into a single scalar. Higher = better quality.

    Weights: sharpness 0.3, illumination 0.3, reflection 0.3, framing 0.1
    The illumination, reflection, and framing scores are INVERTED
    (because lower raw value = better for those three).

    params: all four pool-normalised float sub-scores
    returns: composite_quality_score (float, higher is better)
    side effects: none
    """
    return (
        0.3 * sharpness_score
        + 0.3 * (1.0 - illumination_uniformity_score)
        + 0.3 * (1.0 - reflection_coverage_score)
        + 0.1 * (1.0 - fov_completeness_score)
    )


# ---- Batch quality scoring ----------------------------------

def compute_and_attach_all_quality_scores(
    metadata_dataframe: pd.DataFrame,
    connection: sqlite3.Connection,
    cursor: sqlite3.Cursor,
) -> pd.DataFrame:
    """
    Compute quality scores for every image in the metadata DataFrame,
    write them to the database, and return the DataFrame with score columns added.

    This function:
      1. Iterates over all images (with a progress bar)
      2. Computes all four sub-scores per image
      3. Normalises each sub-score across the whole pool (min-max to [0, 1])
      4. Computes the composite score
      5. Writes everything to the database
      6. Returns the DataFrame with new score columns

    params:
        metadata_dataframe — pandas DataFrame with a 'file_path' column
        connection, cursor — sqlite3 handles
    returns: metadata_dataframe with added quality-score columns
    side effects: updates the database with scores for every image
    """
    raw_scores = {
        "sharpness":      [],
        "illumination":   [],
        "reflection":     [],
        "fov_completeness": [],
    }
    valid_indices = []

    print("Computing image quality scores (this runs once and is then cached)...")
    for idx, row in tqdm(metadata_dataframe.iterrows(), total=len(metadata_dataframe)):
        try:
            img_float = load_image_as_float_array(row["file_path"])
            img_uint8 = (img_float * 255).astype(np.uint8)
            fov_mask  = extract_field_of_view_mask(img_uint8)

            # Skip images where FOV mask is too small (likely not a fundus photo)
            if np.sum(fov_mask) < (fov_mask.size * 0.1):
                continue

            raw_scores["sharpness"].append(
                (idx, compute_sharpness_score(img_uint8, fov_mask))
            )
            raw_scores["illumination"].append(
                (idx, compute_illumination_uniformity_score(img_uint8, fov_mask))
            )
            raw_scores["reflection"].append(
                (idx, compute_reflection_coverage_score(img_uint8, fov_mask))
            )
            raw_scores["fov_completeness"].append(
                (idx, compute_fov_completeness_score(fov_mask))
            )
            valid_indices.append(idx)
        except Exception as e:
            print(f"  Warning: skipping {row['file_path']} — {e}")

    if not valid_indices:
        print("  No valid images found!")
        return metadata_dataframe

    # Min-max normalise each sub-score across the valid pool
    def _normalise(score_list: list) -> dict:
        indices, values = zip(*score_list)
        arr = np.array(values, dtype=np.float64)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min < 1e-9:
            normalised = np.zeros_like(arr)
        else:
            normalised = (arr - arr_min) / (arr_max - arr_min)
        return dict(zip(indices, normalised))

    norm_sharpness    = _normalise(raw_scores["sharpness"])
    norm_illumination = _normalise(raw_scores["illumination"])
    norm_reflection   = _normalise(raw_scores["reflection"])
    norm_fov          = _normalise(raw_scores["fov_completeness"])

    # Add score columns to the DataFrame and persist to DB
    for idx in valid_indices:
        s  = float(norm_sharpness.get(idx, 0.0))
        il = float(norm_illumination.get(idx, 1.0))
        r  = float(norm_reflection.get(idx, 1.0))
        f  = float(norm_fov.get(idx, 1.0))
        c  = compute_composite_quality_score(s, il, r, f)

        metadata_dataframe.at[idx, "sharpness_score"]              = s
        metadata_dataframe.at[idx, "illumination_uniformity_score"] = il
        metadata_dataframe.at[idx, "reflection_coverage_score"]    = r
        metadata_dataframe.at[idx, "fov_completeness_score"]       = f
        metadata_dataframe.at[idx, "composite_quality_score"]      = c

        update_quality_scores_in_database(
            connection, cursor, metadata_dataframe.at[idx, "image_id"],
            {"sharpness_score": s, "illumination_uniformity_score": il,
             "reflection_coverage_score": r, "fov_completeness_score": f,
             "composite_quality_score": c}
        )

    return metadata_dataframe


def select_pseudo_clean_pool(
    metadata_dataframe: pd.DataFrame,
    quantile_threshold: float = QUALITY_QUANTILE_THRESHOLD,
) -> pd.DataFrame:
    """
    Mark every image whose composite_quality_score exceeds the given
    quantile threshold as eligible to serve as a Tier-1 pseudo-clean target.

    Only the top (1 - quantile_threshold) fraction of images are selected.
    With quantile_threshold=0.75 this is the top 25%.

    params:
        metadata_dataframe — DataFrame with a composite_quality_score column
        quantile_threshold — float, default 0.75 (top quarter)
    returns: DataFrame with a new boolean is_pseudo_clean column
    side effects: none — does NOT mutate in place, returns a copy
    """
    if "composite_quality_score" not in metadata_dataframe.columns:
        metadata_dataframe["is_pseudo_clean"] = False
        return metadata_dataframe

    score_threshold = metadata_dataframe["composite_quality_score"].quantile(
        quantile_threshold
    )
    metadata_dataframe = metadata_dataframe.copy()
    metadata_dataframe["is_pseudo_clean"] = (
        metadata_dataframe["composite_quality_score"] >= score_threshold
    )
    n_clean = metadata_dataframe["is_pseudo_clean"].sum()
    print(f"Pseudo-clean pool: {n_clean} / {len(metadata_dataframe)} images "
          f"({100 * n_clean / max(1, len(metadata_dataframe)):.1f}%)")
    return metadata_dataframe
