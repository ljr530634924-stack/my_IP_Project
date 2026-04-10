import cv2
import numpy as np
import os
import glob
from particle_analysis import run_refined_particle_extraction

# --- Configuration ---
INPUT_FOLDER = r"D:\Ingenieurpraixs\test_crop_WN"

# Output subfolder name (created inside INPUT_FOLDER)
OUTPUT_SUBFOLDER = "cropped_particles"

# Padding around each cropped bounding box (pixels)
CROP_PADDING = 10

# Parameters (same as batch_run_NN_MC.py)
CIRCLE_RADIUS_SCALE = 1
USE_WATERSHED = True
WATERSHED_MIN_DIST = 10
MIN_CIRCULARITY = 0.75
KEEP_AREA = 3000

BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3

STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5

# Area filter applied to mask contours
MIN_OBJECT_AREA = 10000
MAX_OBJECT_AREA = 12000


def process_image(ch00_path, output_dir):
    print(f"Processing: {os.path.basename(ch00_path)}")

    base_name = os.path.basename(ch00_path)
    name_no_ext = os.path.splitext(base_name)[0]
    prefix = name_no_ext.replace("ch00", "").replace("__", "_").strip(" _") or "output"

    try:
        # === 1. Extract mask (same as batch_run_NN_MC.py) ===
        print("  1. Extracting structure mask...")
        mask1 = run_refined_particle_extraction(
            ch00_path,
            save_prefix=f"temp_{prefix}_crop",
            circle_radius_scale=CIRCLE_RADIUS_SCALE,
            use_watershed=USE_WATERSHED,
            watershed_min_dist=WATERSHED_MIN_DIST,
            min_circularity=MIN_CIRCULARITY,
            keep_area=KEEP_AREA,
            large_sigma=BANDPASS_LARGE_SIGMA,
            noise_sigma=BANDPASS_SMALL_SIGMA,
            stretch_low=STRETCH_LOW_PERCENTILE,
            stretch_high=STRETCH_HIGH_PERCENTILE,
            simple_mode=True,
            save_intermediates=False
        )

        # === 1b. Filter mask contours by area (same as batch_run_NN_MC.py) ===
        if mask1.dtype == bool:
            mask1 = (mask1 * 255).astype(np.uint8)

        filtered_mask = np.zeros_like(mask1)
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept_contours = []
        removed = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_OBJECT_AREA <= area <= MAX_OBJECT_AREA:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
                kept_contours.append(cnt)
            else:
                removed += 1
        print(f"  [Area filter] kept={len(kept_contours)}, removed={removed} "
              f"(range {MIN_OBJECT_AREA}–{MAX_OBJECT_AREA} px²)")

        if not kept_contours:
            print("  [WARN] No contours passed the area filter. Skipping.")
            return 0

        # === 2. Load original image and crop each particle ===
        print("  2. Cropping particles from original image...")
        img = cv2.imread(ch00_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [ERROR] Cannot read {ch00_path}")
            return 0

        img_h, img_w = img.shape[:2]
        crop_count = 0

        for idx, cnt in enumerate(kept_contours):
            x, y, w, h = cv2.boundingRect(cnt)

            # Apply padding, clamped to image bounds
            x1 = max(0, x - CROP_PADDING)
            y1 = max(0, y - CROP_PADDING)
            x2 = min(img_w, x + w + CROP_PADDING)
            y2 = min(img_h, y + h + CROP_PADDING)

            crop = img[y1:y2, x1:x2]

            out_filename = f"{prefix}_particle_{idx + 1:03d}.tif"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, crop)
            crop_count += 1

        print(f"  -> Saved {crop_count} crops to: {output_dir}")
        return crop_count

    except Exception as e:
        print(f"  [ERROR] Failed processing {base_name}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    print(f"=== Starting Batch Crop NN in '{os.path.abspath(INPUT_FOLDER)}' ===")

    output_dir = os.path.join(INPUT_FOLDER, OUTPUT_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder: {output_dir}")

    search_pattern = os.path.join(INPUT_FOLDER, "*ch00*.tif")
    ch00_files = glob.glob(search_pattern)

    if not ch00_files:
        print(f"No files found matching {search_pattern}")
        return

    print(f"Found {len(ch00_files)} ch00 files.\n")

    total_crops = 0
    for ch00 in ch00_files:
        total_crops += process_image(ch00, output_dir)

    print(f"\n=== Done! Total crops saved: {total_crops} ===")


if __name__ == "__main__":
    main()
