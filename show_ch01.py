import argparse

import cv2
import numpy as np


def adjust_ch01_image(
    ch01_path,
    save_path,
    brightness=0.0,
    exposure=12.0,
    contrast_gain=3.0,
    stretch_low=1.0,
    stretch_high=92,
    do_stretch=True,
    stretch_mode="stddev",
    std_lo_sigma=1.0,
    std_hi_sigma=3.0,
    manual_high=None,
    use_median_blur=True,
    median_ksize=3,
    use_bg_subtract=False,
    bg_sigma=70.0,
    use_petri_mask=False,
    petri_dp=1.2,
    petri_param1=100.0,
    petri_param2=30.0,
    petri_min_radius=None,
    petri_max_radius=None,
):
    """
    Adjust channel 01 image while preserving the original bit depth.

    - brightness: additive offset (use 16-bit scale if the source is 16-bit)
    - exposure: multiplicative gain
    - contrast_gain: contrast scaling around mid-gray
    - stretch_low / stretch_high: percentiles for auto contrast stretch (0-100)
    - do_stretch: toggle auto contrast stretch (set False to preserve numeric values)
    - stretch_mode: "percentile", "stddev", or "manual"
    - std_lo_sigma / std_hi_sigma: std-dev stretch bounds around mean
    - manual_high: manual upper bound for stretch (requires stretch_mode="manual")
    - use_median_blur: remove isolated hot pixels before stretching
    - use_bg_subtract: subtract a large-scale background before stretching
    - bg_sigma: Gaussian sigma for background estimation (pixels)
    - use_petri_mask: detect petri dish circle and zero-out everything outside
    - petri_dp/param1/param2/min_radius/max_radius: HoughCircles parameters
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

    if use_median_blur:
        # Median blur expects uint8 or uint16
        img_uint = img.astype(np.uint16 if max_val > 255 else np.uint8)
        img_uint = cv2.medianBlur(img_uint, int(median_ksize))
        img = img_uint.astype(np.float32)
    else:
        img = img.astype(np.float32)

    # Exposure (multiplicative)
    img *= float(exposure)

    # Brightness (additive)
    img += float(brightness)

    # Contrast around mid-gray
    img = (img - mid_val) * float(contrast_gain) + mid_val

    if use_bg_subtract:
        # Estimate smooth background and subtract it to suppress large-scale gradients
        sigma = max(1.0, float(bg_sigma))
        bg = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        img = img - bg
        img = np.clip(img, 0, None)

    if use_petri_mask:
        # Detect the dish on a downscaled 8-bit preview to avoid sensitivity to outliers
        preview = img.copy()
        preview = np.clip(preview, 0, max_val)
        preview = (preview / max_val * 255.0).astype(np.uint8)
        preview = cv2.GaussianBlur(preview, (0, 0), sigmaX=2.0, sigmaY=2.0)

        h, w = preview.shape[:2]
        min_r = petri_min_radius
        max_r = petri_max_radius
        if min_r is None:
            min_r = int(0.40 * min(h, w))
        if max_r is None:
            max_r = int(0.55 * min(h, w))

        circles = cv2.HoughCircles(
            preview,
            cv2.HOUGH_GRADIENT,
            dp=float(petri_dp),
            minDist=float(min(h, w)) / 2.0,
            param1=float(petri_param1),
            param2=float(petri_param2),
            minRadius=int(min_r),
            maxRadius=int(max_r),
        )

        if circles is not None and len(circles) > 0:
            circles = np.round(circles[0]).astype(int)
            # Pick the largest circle as the dish
            cx, cy, r = max(circles, key=lambda c: c[2])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), int(r), 255, thickness=-1)
            img = img * (mask.astype(np.float32) / 255.0)
        else:
            raise RuntimeError("Petri dish circle not found; adjust Hough parameters.")

    # Auto contrast stretch to full dynamic range for visibility (optional)
    if do_stretch:
        mode = str(stretch_mode).lower().strip()
        if mode == "stddev":
            mean_val = float(np.mean(img))
            std_val = float(np.std(img))
            p_lo = max(0.0, mean_val - std_lo_sigma * std_val)
            p_hi = mean_val + std_hi_sigma * std_val
        elif mode == "manual":
            if manual_high is None:
                raise ValueError("manual_high is required for stretch_mode='manual'")
            p_lo = 0.0
            p_hi = float(manual_high)
        else:
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


def _build_parser():
    parser = argparse.ArgumentParser(description="Adjust and stretch ch01 images.")
    parser.add_argument("ch01_path", help="Input ch01 image path.")
    parser.add_argument("save_path", help="Output image path.")
    parser.add_argument("--brightness", type=float, default=0.0, help="Additive offset.")
    parser.add_argument("--exposure", type=float, default=3.0, help="Multiplicative gain.")
    parser.add_argument("--contrast-gain", type=float, default=1.0, help="Contrast gain.")
    parser.add_argument("--stretch-low", type=float, default=0.175, help="Percentile low.")
    parser.add_argument("--stretch-high", type=float, default=99.825, help="Percentile high.")
    parser.add_argument("--no-stretch", action="store_true", help="Disable stretch.")
    parser.add_argument(
        "--stretch-mode",
        choices=["percentile", "stddev", "manual"],
        default="stddev",
        help="Stretch method.",
    )
    parser.add_argument(
        "--std-lo-sigma",
        type=float,
        default=1.0,
        help="Stddev lower sigma bound.",
    )
    parser.add_argument(
        "--std-hi-sigma",
        type=float,
        default=5.0,
        help="Stddev upper sigma bound.",
    )
    parser.add_argument(
        "--manual-high",
        type=float,
        default=None,
        help="Manual high bound for stretch_mode=manual.",
    )
    parser.add_argument(
        "--no-median-blur",
        action="store_true",
        help="Disable median blur.",
    )
    parser.add_argument(
        "--median-ksize",
        type=int,
        default=3,
        help="Median blur kernel size (odd).",
    )
    parser.add_argument(
        "--use-bg-subtract",
        action="store_true",
        help="Enable large-scale background subtraction.",
    )
    parser.add_argument(
        "--bg-sigma",
        type=float,
        default=35.0,
        help="Gaussian sigma for background subtraction.",
    )
    parser.add_argument(
        "--use-petri-mask",
        action="store_true",
        help="Detect petri dish circle and zero-out pixels outside it.",
    )
    parser.add_argument(
        "--petri-dp",
        type=float,
        default=1.2,
        help="HoughCircles dp parameter.",
    )
    parser.add_argument(
        "--petri-param1",
        type=float,
        default=100.0,
        help="HoughCircles param1 (Canny high threshold).",
    )
    parser.add_argument(
        "--petri-param2",
        type=float,
        default=30.0,
        help="HoughCircles param2 (accumulator threshold).",
    )
    parser.add_argument(
        "--petri-min-radius",
        type=int,
        default=None,
        help="Minimum circle radius (pixels).",
    )
    parser.add_argument(
        "--petri-max-radius",
        type=int,
        default=None,
        help="Maximum circle radius (pixels).",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    adjust_ch01_image(
        args.ch01_path,
        args.save_path,
        brightness=args.brightness,
        exposure=args.exposure,
        contrast_gain=args.contrast_gain,
        stretch_low=args.stretch_low,
        stretch_high=args.stretch_high,
        do_stretch=not args.no_stretch,
        stretch_mode=args.stretch_mode,
        std_lo_sigma=args.std_lo_sigma,
        std_hi_sigma=args.std_hi_sigma,
        manual_high=args.manual_high,
        use_median_blur=not args.no_median_blur,
        median_ksize=args.median_ksize,
        use_bg_subtract=args.use_bg_subtract,
        bg_sigma=args.bg_sigma,
        use_petri_mask=args.use_petri_mask,
        petri_dp=args.petri_dp,
        petri_param1=args.petri_param1,
        petri_param2=args.petri_param2,
        petri_min_radius=args.petri_min_radius,
        petri_max_radius=args.petri_max_radius,
    )


if __name__ == "__main__":
    main()
