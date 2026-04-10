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
# Refined extraction parameters (Structure)
CIRCLE_RADIUS_SCALE = 1
USE_WATERSHED = True
WATERSHED_MIN_DIST = 10
TILED_WATERSHED = True
MIN_CIRCULARITY = 0.6
KEEP_AREA = 3000
MIN_OBJECT_AREA = 9000
MAX_OBJECT_AREA = 11000
BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3
STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5
SAVE_MASK = False  # Set to True to save the binary structure mask as {prefix}_mask.png
SAVE_MASK_WITH_AXES = False  # Set to True to save the axes/notch overlay mask as {prefix}_mask_with_axes.png

# Measurement Parameters
MEASURE_RADIUS = 21        # Fixed measurement outer circle radius (pixels)
CENTER_OFFSET = 24         # Distance from particle centroid to measurement circle center along each local axis (pixels)
PERCENTAGE = 0.3           # Inner circle radius as a fraction of outer radius (0.0 - <1.0)


def make_annulus_mask(shape, cx, cy, r_outer, r_inner):
    """Build an annular mask between r_inner and r_outer centered at (cx, cy)."""
    mask_outer = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask_outer, (cx, cy), r_outer, 255, -1)
    if r_inner > 0:
        mask_inner = np.zeros(shape[:2], dtype=np.uint8)
        cv2.circle(mask_inner, (cx, cy), r_inner, 255, -1)
        return cv2.subtract(mask_outer, mask_inner)
    return mask_outer


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

    csv_output = os.path.join(directory, f"{prefix}_results.xlsx")
    id_map_output = os.path.join(directory, f"{prefix}_map.png")

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

        # Filter out particles larger than MAX_OBJECT_AREA
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(structure_mask, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > MAX_OBJECT_AREA or area < MIN_OBJECT_AREA:
                structure_mask[labels == i] = 0

        # Save mask if requested
        if SAVE_MASK:
            mask_output = os.path.join(directory, f"{prefix}_mask.png")
            cv2.imwrite(mask_output, structure_mask)
            print(f"  -> Saved mask: {mask_output}")

        # 2. Find Notches and Axes
        print("  2. Finding notches and axes...")
        temp_axes_path = os.path.join(directory, f"temp_axes_{prefix}.png")
        axes_info = find_notches_and_axes(
            structure_mask,
            save_path=temp_axes_path,
        )

        # 2.1 Overlay axes on ch00 (Option 3: Black BG, Ch00 inside particles, Axes on top)
        if os.path.exists(temp_axes_path):
            print("  2.1 Generating overlay (Option 3)...")
            overlay_output = os.path.join(directory, f"{prefix}_visualization.png")
            try:
                axes_img = cv2.imread(temp_axes_path)
                base_img = cv2.imread(ch00_path, cv2.IMREAD_UNCHANGED)

                if axes_img is not None and base_img is not None:
                    # Normalize base to 8-bit for visualization
                    if base_img.dtype == np.uint16:
                        p_lo, p_hi = np.percentile(base_img, (1, 99))
                        if p_hi > p_lo:
                            base_vis = (base_img.astype(np.float32) - p_lo) * (255.0 / (p_hi - p_lo))
                            base_vis = np.clip(base_vis, 0, 255).astype(np.uint8)
                        else:
                            base_vis = (base_img / 256).astype(np.uint8)
                    elif base_img.dtype == np.uint8:
                        base_vis = base_img.copy()
                    else:
                        base_vis = cv2.normalize(base_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    if len(base_vis.shape) == 2:
                        base_vis = cv2.cvtColor(base_vis, cv2.COLOR_GRAY2BGR)

                    if axes_img.shape[:2] != base_vis.shape[:2]:
                        axes_img = cv2.resize(axes_img, (base_vis.shape[1], base_vis.shape[0]))

                    # Detect white particle interior (255,255,255)
                    gray_axes = cv2.cvtColor(axes_img, cv2.COLOR_BGR2GRAY)
                    _, mask_white = cv2.threshold(gray_axes, 250, 255, cv2.THRESH_BINARY)

                    # Composite: Start with axes_img (Black BG + Axes), replace white with base
                    final_comp = axes_img.copy()
                    final_comp[mask_white == 255] = base_vis[mask_white == 255]

                    cv2.imwrite(overlay_output, final_comp)
                    print(f"  -> Saved overlay: {overlay_output}")

                if SAVE_MASK_WITH_AXES:
                    mask_axes_output = os.path.join(directory, f"{prefix}_mask_with_axes.png")
                    os.rename(temp_axes_path, mask_axes_output)
                    print(f"  -> Saved mask with axes: {mask_axes_output}")
                else:
                    os.remove(temp_axes_path)
            except Exception as e:
                print(f"  [WARN] Failed to generate overlay: {e}")
                if os.path.exists(temp_axes_path):
                    os.remove(temp_axes_path)

        # 3. Measure on raw ch01 using annulus measurement circles
        print("  3. Measuring intensities (with ring)...")
        raw_ch01 = cv2.imread(ch01_path, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            print(f"  [ERROR] Cannot read {ch01_path}")
            return

        num_labels, labels = cv2.connectedComponents(structure_mask)
        from skimage import measure
        regions = measure.regionprops(labels)

        circles_to_draw = compute_quadrant_intensity(
            brightness_image=raw_ch01,
            labels=labels,
            regions=regions,
            axes_info=axes_info,
            csv_path=csv_output,
            id_map_path=id_map_output,
            inner_ratio=PERCENTAGE,
            measure_radius=MEASURE_RADIUS,
            center_offset=CENTER_OFFSET,
        )

        # 4. Overlay measurement circles onto visualization
        overlay_output = os.path.join(directory, f"{prefix}_visualization.png")
        if circles_to_draw and os.path.exists(overlay_output):
            print("  4. Drawing measurement circles on visualization...")
            vis_img = cv2.imread(overlay_output)
            if vis_img is not None:
                for circ in circles_to_draw:
                    cx, cy = circ["center"]
                    r_outer = circ["radius"]
                    color = circ["color"]
                    r_inner = max(1, int(round(r_outer * PERCENTAGE)))
                    # Outer arc circle
                    cv2.circle(vis_img, (cx, cy), r_outer, color, 1, cv2.LINE_AA)
                    # Inner arc circle
                    cv2.circle(vis_img, (cx, cy), r_inner, color, 1, cv2.LINE_AA)
                cv2.imwrite(overlay_output, vis_img)
                print(f"  -> Updated visualization with circles: {overlay_output}")

        print(f"  -> Done. Results: {csv_output}")

    except Exception as e:
        print(f"  [ERROR] Failed processing {base_name}: {e}")
        import traceback
        traceback.print_exc()

def run_batch_wn_mc_wr(input_folder):
    """
    Main entry point for running the batch WN_MC (With Ring) analysis.
    :param input_folder: The path to the folder containing image pairs.
    """
    print(f"=== Starting Batch WN MC (With Ring) Analysis in '{os.path.abspath(input_folder)}' ===")

    search_pattern = os.path.join(input_folder, "*ch00*.tif")
    ch00_files = glob.glob(search_pattern)

    if not ch00_files:
        print(f"No files found matching {search_pattern}")
        return

    print(f"Found {len(ch00_files)} candidate ch00 files to process.")

    processed_count = 0
    for ch00 in ch00_files:
        directory, filename = os.path.split(ch00)
        filename_ch01 = filename.replace("ch00", "ch01")
        ch01 = os.path.join(directory, filename_ch01)

        if not os.path.exists(ch01):
            print(f"[WARN] Corresponding ch01 file not found for {filename}. Skipping.")
            continue

        process_pair(ch00, ch01)
        processed_count += 1

    print(f"\n=== Batch Processing Complete! Processed {processed_count} pairs. ===")


if __name__ == "__main__":
    # This block allows the script to be run standalone for testing.
    # The GUI will call the `run_batch_wn_mc_wr` function directly.
    DEFAULT_INPUT_FOLDER = r"D:\Ingenieurpraixs\test_WN_WR"
    if not os.path.isdir(DEFAULT_INPUT_FOLDER):
        print(f"[ERROR] Default test folder not found: {DEFAULT_INPUT_FOLDER}")
        print("Please update the DEFAULT_INPUT_FOLDER path in the script.")
    else:
        run_batch_wn_mc_wr(DEFAULT_INPUT_FOLDER)
