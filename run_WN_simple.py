import cv2
import os
import numpy as np
from particle_analysis import (
    run_refined_particle_extraction,
    find_notches_and_axes,
    extract_inner_boundaries,
)
from handle_ch01 import adjust_ch01_image
from measure_intensity import compute_masked_quadrant_intensity


# --- Configuration ---
CH00_PATH = "Project002_CRP_ELISA_DirectBinding_qCAP_VaryingBiotin0,0.1,1,10mg_TileScan 3_A_1_Merged_CRP_0ng_ch00.tif"
CH01_PATH = "Project002_CRP_ELISA_DirectBinding_qCAP_VaryingBiotin0,0.1,1,10mg_TileScan 3_A_1_Merged_CRP_0ng_ch01.tif"

# Temporary intermediate files (will be deleted automatically)
TEMP_ADJUSTED_CH01 = "temp_adjusted_ch01.png"
TEMP_SIGNAL_MASK_VIS = "temp_signal_mask_vis.png"
TEMP_AXES_VIS = "temp_axes_vis.png"

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

# Signal Extraction Parameters
SIGNAL_OPENING_RADIUS = 1

# Output files
CSV_OUTPUT = "1merged_NC_WN_results.xlsx"
VIS_OUTPUT = "1merged_NC_WN_visualization.png"

def main():
    print("=== Starting WN Simple Analysis (With Notches, Minimal Output) ===")

    try:
        # 1. Adjust ch01 (Signal Preparation)
        print(f"1. Adjusting ch01 -> {TEMP_ADJUSTED_CH01} (Temporary)...")
        adjust_ch01_image(
            CH01_PATH, 
            TEMP_ADJUSTED_CH01,
            exposure=8.0,
            brightness=10,
            contrast_gain=1.0,
            stretch_low=2.0,
            stretch_high=98.5,
            do_stretch=True
        )

        # 2. Extract Signal Mask
        print(f"2. Extracting signal mask (Temporary)...")
        # extract_inner_boundaries returns the mask array
        signal_mask = extract_inner_boundaries(
            TEMP_ADJUSTED_CH01, 
            save_path=TEMP_SIGNAL_MASK_VIS, # Temporary visualization
            thin_opening_radius=SIGNAL_OPENING_RADIUS,
            min_size=80,
            keep_area=80,
            canny_sigma=2.0,
            clahe_clip_limit=0.01,
            brightness_threshold=0.2
        )

        # 3. Extract Structure (Simple Mode)
        print(f"3. Extracting structure from {CH00_PATH}...")
        structure_mask = run_refined_particle_extraction(
            CH00_PATH, 
            save_prefix="temp_structure", 
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
            save_intermediates=False, # No intermediate images
           simple_mode=True,         # Simple mode (Canny/Watershed only)
            restrict_to_largest_circle=False
        )

        # 4. Find Notches and Axes
        print(f"4. Finding notches and axes...")
        # find_notches_and_axes requires a save_path for the visualization
        axes_info = find_notches_and_axes(
            structure_mask, 
            save_path=TEMP_AXES_VIS # Temporary visualization
        )

        # 5. Measure and Visualize
        print(f"5. Measuring intensities and generating final visualization...")
        raw_ch01 = cv2.imread(CH01_PATH, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            raise FileNotFoundError(f"Cannot read {CH01_PATH}")

        compute_masked_quadrant_intensity(
            brightness_image=raw_ch01,
            structure_mask=structure_mask,
            signal_mask=signal_mask,
            axes_info=axes_info,
            csv_path=CSV_OUTPUT,
            overlay_path=VIS_OUTPUT
        )

        print("\n=== Pipeline Complete! ===")
        print(f"Results: {CSV_OUTPUT}")
        print(f"Visualization: {VIS_OUTPUT}")

    finally:
        # Cleanup temporary files
        print("Cleaning up temporary files...")
        temp_files = [TEMP_ADJUSTED_CH01, TEMP_SIGNAL_MASK_VIS, TEMP_AXES_VIS]
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {f}: {e}")

if __name__ == "__main__":
    main() 