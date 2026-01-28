import cv2
import os
import glob
import numpy as np
from PIL import Image
from particle_analysis import (
    run_refined_particle_extraction,
    find_notches_and_axes,
)
from measure_intensity import compute_quadrant_intensity

# Increase PIL image size limit to handle large scientific images
Image.MAX_IMAGE_PIXELS = None

# --- Configuration ---
# Set this to the folder containing your images.
INPUT_FOLDER = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Jinrui\CRP\Project002_CRP_ELISA_DirectBinding_qCAP_VaryingBiotin0,0.1,1,10mg (3images again)"

# Refined extraction parameters (Structure)
CIRCLE_RADIUS_SCALE = 1
USE_WATERSHED = True
WATERSHED_MIN_DIST = 10
TILED_WATERSHED = True
MIN_CIRCULARITY = 0.75
KEEP_AREA = 3000
BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3
STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5


def process_pair(ch00_path, ch01_path):
    print(f"Processing pair:\n  CH00: {os.path.basename(ch00_path)}\n  CH01: {os.path.basename(ch01_path)}")

    directory = os.path.dirname(ch00_path)
    base_name = os.path.basename(ch00_path)
    name_no_ext = os.path.splitext(base_name)[0]

    # Create a cleaner prefix
    prefix = name_no_ext.replace("ch00", "")
    prefix = prefix.replace("__", "_").strip(" _")
    if not prefix:
        prefix = "output"

    csv_output = os.path.join(directory, f"{prefix}_WN_MC_results.xlsx")
    id_map_output = os.path.join(directory, f"{prefix}_WN_MC_id_map.png")

    try:
        # 1. Extract Structure (Simple Mode)
        print("  1. Extracting structure...")
        structure_mask = run_refined_particle_extraction(
            ch00_path,
            save_prefix=f"temp_{prefix}",
            circle_radius_scale=CIRCLE_RADIUS_SCALE,
            use_watershed=USE_WATERSHED,
            watershed_min_dist=WATERSHED_MIN_DIST,
            tiled_watershed=TILED_WATERSHED,
            min_circularity=MIN_CIRCULARITY,
            keep_area=KEEP_AREA,
            large_sigma=BANDPASS_LARGE_SIGMA,
            noise_sigma=BANDPASS_SMALL_SIGMA,
            stretch_low=STRETCH_LOW_PERCENTILE,
            stretch_high=STRETCH_HIGH_PERCENTILE,
            save_intermediates=False,
            simple_mode=True,
            restrict_to_largest_circle=False,
        )

        # 2. Find Notches and Axes
        print("  2. Finding notches and axes...")
        axes_info = find_notches_and_axes(
            structure_mask,
            save_path=None,
        )

        # 3. Measure on raw ch01 using measurement circles
        print("  3. Measuring intensities...")
        raw_ch01 = cv2.imread(ch01_path, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            print(f"  [ERROR] Cannot read {ch01_path}")
            return

        num_labels, labels = cv2.connectedComponents(structure_mask)
        from skimage import measure
        regions = measure.regionprops(labels)

        compute_quadrant_intensity(
            brightness_image=raw_ch01,
            labels=labels,
            regions=regions,
            axes_info=axes_info,
            csv_path=csv_output,
            id_map_path=id_map_output,
        )

        print(f"  -> Done. Results: {csv_output}")

    except Exception as e:
        print(f"  [ERROR] Failed processing {base_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    print(f"=== Starting Batch WN MC Analysis in '{os.path.abspath(INPUT_FOLDER)}' ===")

    search_pattern = os.path.join(INPUT_FOLDER, "*ch00*.tif")
    ch00_files = glob.glob(search_pattern)

    if not ch00_files:
        print(f"No files found matching {search_pattern}")
        return

    print(f"Found {len(ch00_files)} candidate ch00 files.")

    count = 0
    for ch00 in ch00_files:
        directory, filename = os.path.split(ch00)
        filename_ch01 = filename.replace("ch00", "ch01")
        ch01 = os.path.join(directory, filename_ch01)

        if not os.path.exists(ch01):
            print(f"[WARN] Corresponding ch01 file not found for {filename}. Skipping.")
            continue

        process_pair(ch00, ch01)
        count += 1

    print(f"\n=== Batch Processing Complete! Processed {count} pairs. ===")


if __name__ == "__main__":
    main()
