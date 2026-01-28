import cv2
import os
import numpy as np
from particle_analysis import (
    run_refined_particle_extraction,
)
from measure_intensity import compute_global_signal_intensity

# --- Configuration ---
CH00_PATH = "1_Merged_ch00_0ngmL.tif"
CH01_PATH = "1_Merged_ch01_0ngmL.tif"

# Refined extraction parameters
CIRCLE_RADIUS_SCALE = 1  # Factor to scale ideal circles in Mask B.
USE_WATERSHED = True      # Separate touching particles
WATERSHED_MIN_DIST = 20
MIN_CIRCULARITY = 0.75
KEEP_AREA = 3000

# Bandpass Filter Parameters
BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3

# Saturation Stretch Parameters
STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5

# Output files
CSV_OUTPUT = "1final_global_results.csv"
VIS_OUTPUT = "1final_global_visualization.png"


def main():
    print("=== Starting Global Analysis Pipeline (No Axes) ===")

    # 1. Extract structure from ch00 (No Axes)
    print(f"1. Extracting structure from {CH00_PATH}...")
    structure_mask = run_refined_particle_extraction(
        CH00_PATH, 
        save_prefix="structure_refined",
        circle_radius_scale=CIRCLE_RADIUS_SCALE,
        use_watershed=USE_WATERSHED,
        watershed_min_dist=WATERSHED_MIN_DIST,
        min_circularity=MIN_CIRCULARITY,
        keep_area=KEEP_AREA,
        large_sigma=BANDPASS_LARGE_SIGMA,
        noise_sigma=BANDPASS_SMALL_SIGMA,
        stretch_low=STRETCH_LOW_PERCENTILE,
        stretch_high=STRETCH_HIGH_PERCENTILE
    )
    
    # Save the structure mask explicitly
    cv2.imwrite("global_structure_mask.png", structure_mask)
    print(" -> Saved global_structure_mask.png")

    # Note: Skipped find_notches_and_axes and overlay generation as requested.

    # 2. Measure Global Intensity
    # Using the new compute_global_signal_intensity function
    print(f"2. Measuring global signal intensities...")
    raw_ch01 = cv2.imread(CH01_PATH, cv2.IMREAD_UNCHANGED)
    if raw_ch01 is None:
        raise FileNotFoundError(f"Cannot read {CH01_PATH}")

    # Since we removed the specific signal mask generation, we treat the entire particle as the signal area.
    compute_global_signal_intensity(
        brightness_image=raw_ch01,
        structure_mask=structure_mask,
        signal_mask=structure_mask, # Use structure mask as signal mask (measure whole particle)
        csv_path=CSV_OUTPUT,
        overlay_path=VIS_OUTPUT
    )

    print("\n=== Pipeline Complete! ===")
    print(f"Results: {CSV_OUTPUT}")
    print(f"Visualization: {VIS_OUTPUT}")

if __name__ == "__main__":
    main()