import argparse
from pathlib import Path

import cv2
import numpy as np


def overlay_mask(
    mask_path: str,
    base_path: str,
    save_path: str,
    thresh: float | None = 0.0,
    invert: bool = False,
) -> None:
    """Make white in mask transparent, overlay on base, save composited PNG."""
    base = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if base is None:
        raise FileNotFoundError(f"Cannot read base image: {base_path}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask image: {mask_path}")
    if base.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Size mismatch: base {base.shape[:2]} vs mask {mask.shape[:2]}")

    # Base -> grayscale 1-channel
    if base.ndim == 3 and base.shape[2] == 3:
        base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    elif base.ndim == 2:
        base_gray = base
    else:
        raise ValueError(f"Unsupported base shape: {base.shape}")

    # Mask -> BGR (keep colors), fallback to grayscale
    if mask.ndim == 2:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif mask.ndim == 3 and mask.shape[2] == 3:
        mask_bgr = mask
    elif mask.ndim == 3 and mask.shape[2] == 4:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    # Determine max value for base dtype
    if not np.issubdtype(base_gray.dtype, np.integer):
        raise ValueError(f"Base dtype not integer: {base_gray.dtype}")
    
    # --- Memory Optimization: Avoid float32 conversion ---
    target_dtype = base_gray.dtype
    base_max = np.iinfo(target_dtype).max
    
    mask_dtype = mask_bgr.dtype
    mask_max = np.iinfo(mask_dtype).max

    # Detect white in mask: pixels close to max in all channels
    thresh_val = int(mask_max * 0.95)
    white = (mask_bgr[:, :, 0] >= thresh_val) & \
            (mask_bgr[:, :, 1] >= thresh_val) & \
            (mask_bgr[:, :, 2] >= thresh_val)

    # Prepare output image (start with base converted to BGR)
    if base_gray.ndim == 2:
        output = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    else:
        output = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

    # Prepare mask in target dtype
    if mask_dtype == target_dtype:
        mask_scaled = mask_bgr
    else:
        # Scale mask to match base dtype
        scale = float(base_max) / float(mask_max)
        if target_dtype == np.uint16 and mask_dtype == np.uint8:
            mask_scaled = mask_bgr.astype(np.uint16) * 257
        elif target_dtype == np.uint8 and mask_dtype == np.uint16:
            mask_scaled = (mask_bgr // 257).astype(np.uint8)
        else:
            mask_scaled = (mask_bgr.astype(np.float32) * scale).astype(target_dtype)

    # Composite: mask over base
    mask_indices = white if invert else ~white
    np.copyto(output, mask_scaled, where=mask_indices[:, :, None])

    # Save as 3-channel image (no alpha in output since white was made transparent in composite)
    save_path = str(save_path)
    cv2.imwrite(save_path, output)
    print(f"Saved overlaid image to: {save_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Make mask white transparent over base image.")
    ap.add_argument("--mask", required=True, help="Path to mask image (white=transparent).")
    ap.add_argument("--base", required=True, help="Path to base image.")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    ap.add_argument(
        "--thresh",
        type=float,
        default=0.0,
        help="Threshold for mask white detection; default 0 means >0 is white. Use None for auto 50%% of max.",
    )
    ap.add_argument(
        "--auto-thresh",
        action="store_true",
        help="Use 50%% of mask max as threshold.",
    )
    ap.add_argument(
        "--invert",
        action="store_true",
        help="Invert mask logic (white becomes opaque instead of transparent).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    thresh = None if args.auto_thresh else args.thresh
    overlay_mask(
        mask_path=args.mask,
        base_path=args.base,
        save_path=args.out,
        thresh=thresh,
        invert=args.invert,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
