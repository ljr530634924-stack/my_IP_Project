import cv2
import numpy as np


def adjust_ch01_image(
    ch01_path,
    save_path,
    brightness=10,
    exposure=0.3,
    contrast_gain=0.2,
    stretch_low=1.0,
    stretch_high=99.0,
    do_stretch=True,
):
    """
    Adjust channel 01 image while preserving the original bit depth.

    - brightness: additive offset (use 16-bit scale if the source is 16-bit)
    - exposure: multiplicative gain
    - contrast_gain: contrast scaling around mid-gray
    - stretch_low / stretch_high: percentiles for auto contrast stretch (0-100)
    - do_stretch: toggle auto contrast stretch (set False to preserve numeric values)
    """
    # Read preserving source bit depth
    img = cv2.imread(ch01_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {ch01_path}")

    # Track original dtype/limits to preserve 8-bit or 16-bit correctly
    if img.dtype == np.uint16:
        max_val = 65535.0
        mid_val = 32768.0
    else:
        # Fallback to 8-bit path
        max_val = 255.0
        mid_val = 128.0

    img = img.astype(np.float32)

    # Exposure (multiplicative)
    img *= exposure

    # Brightness (additive)
    img += brightness

    # Contrast around mid-gray
    img = (img - mid_val) * contrast_gain + mid_val

    # Auto contrast stretch to full dynamic range for visibility (optional)
    if do_stretch:
        lo = max(0.0, min(100.0, float(stretch_low)))
        hi = max(0.0, min(100.0, float(stretch_high)))
        if hi <= lo:
            lo, hi = 1.0, 99.0  # fallback defaults

        p_lo, p_hi = np.percentile(img, (lo, hi))
        if p_hi > p_lo:
            img = (img - p_lo) * (max_val / (p_hi - p_lo))
        # If p_hi == p_lo, skip scaling to avoid divide by zero

    # Clip to valid range and restore dtype
    img = np.clip(img, 0, max_val)
    if max_val > 255:
        img = img.astype(np.uint16)
    else:
        img = img.astype(np.uint8)

    cv2.imwrite(save_path, img)
    print(f"Saved adjusted image to: {save_path}")
