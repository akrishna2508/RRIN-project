"""
src/degradation.py
==================
The synthetic degradation pipeline: takes a clean retina image and
produces a corrupted version by randomly applying a combination of:
  1. Uneven illumination / vignetting (Perlin noise multiplicative field)
  2. Specular reflections / lens flare (soft elliptical highlights)
  3. Haze / cataract scatter (low-frequency additive overlay)
  4. Defocus or motion blur (Gaussian / directional kernel)
  5. Sensor noise (Poisson shot noise + Gaussian read noise)
  6. JPEG compression artefacts (encode/decode round-trip)

WHY IS EVERYTHING RANDOM? (for beginners)
  If we always applied the exact same corruption, the AI would just memorise
  that specific pattern and learn to remove only THAT corruption.
  By randomising everything on every training step, the AI is forced to
  learn the general principles of restoration, not a specific recipe.
"""

import random
import cv2
import numpy as np
from typing import Optional

try:
    from noise import pnoise2   # Perlin noise library
    _HAS_NOISE_LIB = True
except ImportError:
    _HAS_NOISE_LIB = False

from src.utils.image_utils import rescale_array_to_range


# ---- Perlin illumination field ----------------------------

def generate_perlin_illumination_field(
    height: int,
    width: int,
    octaves_range: tuple = (2, 4),
    gain_range: tuple = (0.55, 1.35),
) -> np.ndarray:
    """
    Generate a smooth, spatially-varying multiplicative gain field that
    simulates vignetting and uneven flash illumination.

    params:
        height, width   — output array dimensions
        octaves_range   — (min, max) number of Perlin noise octaves to use
        gain_range      — (min_gain, max_gain) range the field is rescaled to
    returns: illumination_field — (H, W) float32 array, values in gain_range
    side effects: none

    If the `noise` library is unavailable, falls back to a simple Gaussian field.
    """
    if _HAS_NOISE_LIB:
        octave_count = random.randint(*octaves_range)
        seed         = random.randint(0, 1_000_000)
        raw_field    = np.array(
            [[pnoise2(y / height, x / width,
                      octaves=octave_count,
                      persistence=0.5,
                      lacunarity=2.0,
                      base=seed)
              for x in range(width)]
             for y in range(height)],
            dtype=np.float32,
        )
    else:
        # Fallback: large Gaussian blur of random noise ≈ low-frequency field
        raw_field = np.random.randn(height, width).astype(np.float32)
        sigma = random.uniform(height * 0.1, height * 0.4)
        raw_field = cv2.GaussianBlur(raw_field, (0, 0), sigma)

    return rescale_array_to_range(raw_field, gain_range[0], gain_range[1]).astype(np.float32)


# ---- Specular reflection helpers --------------------------

def sample_biased_reflection_center(fov_mask: np.ndarray) -> tuple[int, int]:
    """
    Sample a position for a specular reflection blob, biased toward
    the centre (optic disc region) and the frame periphery —
    matching where real corneal reflections actually appear.

    params: fov_mask — (H, W) bool array
    returns: (center_y, center_x) in pixel coordinates
    side effects: none (uses random)
    """
    h, w = fov_mask.shape
    if random.random() < 0.5:
        # Centre bias: within ±20% of image centre
        cy = int(h * 0.5 + h * random.uniform(-0.2, 0.2))
        cx = int(w * 0.5 + w * random.uniform(-0.2, 0.2))
    else:
        # Periphery bias: near the FOV edge
        # Sample from FOV boundary pixels
        fov_uint8 = fov_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(fov_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            pts = contours[0].reshape(-1, 2)
            chosen = pts[random.randint(0, len(pts) - 1)]
            cx, cy = int(chosen[0]), int(chosen[1])
        else:
            cy = random.randint(0, h - 1)
            cx = random.randint(0, w - 1)

    return max(0, min(cy, h - 1)), max(0, min(cx, w - 1))


def render_soft_elliptical_mask(
    shape: tuple,
    center_y: int,
    center_x: int,
    radius_y: float,
    radius_x: float,
) -> np.ndarray:
    """
    Render a soft-edged (Gaussian-falloff) elliptical mask.
    Values range from 1.0 at the centre to 0.0 at the edges.

    params: shape — (H, W), center coordinates, radii
    returns: (H, W) float32 array of values in [0, 1]
    side effects: none
    """
    h, w = shape
    ys   = np.arange(h, dtype=np.float32).reshape(-1, 1)
    xs   = np.arange(w, dtype=np.float32).reshape(1, -1)

    # Squared normalised distance from the centre
    dist_sq = ((ys - center_y) / (radius_y + 1e-6)) ** 2 + \
              ((xs - center_x) / (radius_x + 1e-6)) ** 2
    return np.exp(-dist_sq).astype(np.float32)


# ---- Degradation operators --------------------------------

def simulate_specular_reflections(
    image_rgb: np.ndarray,
    fov_mask: np.ndarray,
    num_reflections_range: tuple = (1, 4),
) -> np.ndarray:
    """
    Composite 1–4 soft-edged elliptical specular highlight blobs onto
    the image using a SCREEN blend — so they only ever BRIGHTEN pixels.

    Screen blend formula: result = 1 - (1 - image) × (1 - highlight)
    This matches how real specular reflections work: they add light.

    params:
        image_rgb             — (H, W, 3) float32 array in [0, 1]
        fov_mask              — (H, W) bool array
        num_reflections_range — (min, max) number of blobs to add
    returns: degraded image, same shape and dtype as input
    side effects: none — operates on a copy
    """
    output = image_rgb.copy()
    num_reflections = random.randint(*num_reflections_range)

    for _ in range(num_reflections):
        cy, cx   = sample_biased_reflection_center(fov_mask)
        radius_y = random.uniform(15, 60)
        radius_x = random.uniform(15, 60)
        intensity = random.uniform(0.5, 0.95)

        highlight = render_soft_elliptical_mask(image_rgb.shape[:2], cy, cx, radius_y, radius_x)
        # Screen blend: 1 - (1-img) * (1 - intensity * highlight)
        output = 1.0 - (1.0 - output) * (1.0 - intensity * highlight[..., None])

    return output


def simulate_haze_overlay(
    image_rgb: np.ndarray,
    opacity_range: tuple = (0.0, 0.35),
) -> np.ndarray:
    """
    Add a low-frequency, near-neutral haze layer simulating cataract scatter.

    A random opacity and a near-white haze colour are blended additively
    with the image weighted by a smooth low-frequency field.

    params:
        image_rgb     — (H, W, 3) float32 in [0, 1]
        opacity_range — (min, max) blend strength
    returns: degraded image, same shape
    side effects: none
    """
    h, w   = image_rgb.shape[:2]
    opacity = random.uniform(*opacity_range)
    if opacity < 0.01:
        return image_rgb.copy()

    haze_field = generate_perlin_illumination_field(h, w, octaves_range=(1, 2), gain_range=(0.0, 1.0))

    # Near-neutral haze colour with mild random hue jitter
    base_brightness = random.uniform(0.7, 1.0)
    hue_jitter = random.uniform(-0.05, 0.05)
    haze_colour = np.array([
        base_brightness + hue_jitter,
        base_brightness,
        base_brightness - hue_jitter,
    ], dtype=np.float32).clip(0.0, 1.0)

    haze_weight = (opacity * haze_field)[..., None]   # broadcast over channels
    output = image_rgb * (1.0 - haze_weight) + haze_colour * haze_weight
    return output.clip(0.0, 1.0)


def build_motion_blur_kernel(length: int, angle_degrees: float) -> np.ndarray:
    """
    Build a 1D directional motion-blur convolution kernel of a given
    length and angle (in degrees, measured from horizontal).

    params: length — kernel size in pixels (odd number preferred)
            angle_degrees — direction of motion blur
    returns: (length, length) float32 kernel summing to 1.0
    side effects: none
    """
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    kernel[center, :] = 1.0   # horizontal line

    # Rotate the kernel to the desired angle
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle_degrees, 1.0)
    rotated = cv2.warpAffine(kernel, rotation_matrix, (length, length))
    total = rotated.sum()
    if total > 1e-8:
        rotated /= total
    return rotated


def apply_defocus_or_motion_blur(
    image_rgb: np.ndarray,
    blur_probability: float = 0.6,
) -> np.ndarray:
    """
    With probability `blur_probability`, apply either isotropic Gaussian
    defocus blur or a directional motion-blur kernel (chosen randomly).

    params:
        image_rgb        — (H, W, 3) float32 in [0, 1]
        blur_probability — chance of applying any blur at all
    returns: possibly-blurred image, same shape
    side effects: none
    """
    if random.random() > blur_probability:
        return image_rgb.copy()

    if random.random() < 0.5:
        # Isotropic Gaussian defocus blur
        sigma = random.uniform(0.8, 3.0)
        return cv2.GaussianBlur(image_rgb, ksize=(0, 0), sigmaX=sigma)
    else:
        # Directional motion blur
        kernel_length = random.randint(5, 15)
        kernel_angle  = random.uniform(0.0, 180.0)
        kernel = build_motion_blur_kernel(kernel_length, kernel_angle)
        return cv2.filter2D(image_rgb, -1, kernel)


def add_poisson_gaussian_sensor_noise(
    image_rgb: np.ndarray,
    shot_noise_scale_range: tuple = (0.001, 0.01),
    read_noise_std_range: tuple = (0.001, 0.02),
) -> np.ndarray:
    """
    Add signal-dependent Poisson shot noise AND signal-independent
    Gaussian read noise, modelling a real camera sensor.

    Poisson noise scales with local brightness (brighter pixels →
    more photons → more shot noise). Gaussian noise is constant.

    params:
        image_rgb              — (H, W, 3) float32 in [0, 1]
        shot_noise_scale_range — (min, max) Poisson intensity scale
        read_noise_std_range   — (min, max) Gaussian std deviation
    returns: noisy image clipped to [0, 1]
    side effects: none
    """
    shot_scale = random.uniform(*shot_noise_scale_range)
    read_std   = random.uniform(*read_noise_std_range)

    # Poisson sampling on a scaled-up intensity axis
    clipped     = np.clip(image_rgb, 1e-6, 1.0)
    poisson_lam = clipped / shot_scale
    shot_noisy  = np.random.poisson(poisson_lam).astype(np.float32) * shot_scale

    # Add Gaussian read noise
    read_noisy  = shot_noisy + np.random.normal(0, read_std, size=image_rgb.shape).astype(np.float32)
    return np.clip(read_noisy, 0.0, 1.0)


def apply_jpeg_recompression_artifact(
    image_rgb: np.ndarray,
    quality_range: tuple = (35, 80),
) -> np.ndarray:
    """
    Simulate JPEG compression artefacts by encode/decode round-tripping
    the image in memory at a random quality level.

    Lower quality → stronger blocking and ringing artefacts.
    This simulates legacy PACS/EHR systems that store images as lossy JPEG.

    params:
        image_rgb     — (H, W, 3) float32 in [0, 1]
        quality_range — (min_quality, max_quality) for JPEG encoder (1–100)
    returns: degraded image with compression artefacts
    side effects: none — encode/decode happens entirely in memory
    """
    quality  = random.randint(*quality_range)
    uint8_img = (np.clip(image_rgb, 0.0, 1.0) * 255).astype(np.uint8)

    # Encode to JPEG buffer (in-memory, no temp file written)
    success, encoded_buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(uint8_img, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        return image_rgb.copy()

    # Decode back
    decoded_bgr = cv2.imdecode(encoded_buffer, cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return decoded_rgb.astype(np.float32) / 255.0


# ---- Compose pipeline -------------------------------------

def compose_random_degradation_pipeline(
    pseudo_clean_image_rgb: np.ndarray,
    fov_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply a randomly-ordered, randomly-selected subset of degradation
    operators to a single pseudo-clean image, producing one synthetic
    training input.

    EVERY call re-samples both:
      - which subset of operators to apply (always ≥ 2)
      - all internal parameters of each chosen operator
    so no two calls ever produce identical results.

    params:
        pseudo_clean_image_rgb — (H, W, 3) float32 in [0, 1]
        fov_mask               — (H, W) bool
    returns: degraded_image — same shape, values in [0, 1]
    side effects: none — returns a new array, never mutates the input
    """
    # Step 1: Always apply the illumination field (it multiplicatively
    #         modifies the image before other operators run)
    h, w = pseudo_clean_image_rgb.shape[:2]
    illumination_field = generate_perlin_illumination_field(h, w)
    working_image = pseudo_clean_image_rgb * illumination_field[..., None]
    working_image = np.clip(working_image, 0.0, 1.0)

    # Step 2: Randomly shuffle and apply a subset of the remaining operators
    # Note: simulate_specular_reflections needs fov_mask as an argument
    operator_pool = [
        lambda img: simulate_specular_reflections(img, fov_mask),
        simulate_haze_overlay,
        apply_defocus_or_motion_blur,
        add_poisson_gaussian_sensor_noise,
        apply_jpeg_recompression_artifact,
    ]

    random.shuffle(operator_pool)
    num_to_apply = random.randint(2, len(operator_pool))
    selected_operators = operator_pool[:num_to_apply]

    for op in selected_operators:
        working_image = op(working_image)

    return np.clip(working_image, 0.0, 1.0).astype(np.float32)
