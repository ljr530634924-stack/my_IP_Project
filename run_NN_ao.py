import cv2
import os
import numpy as np
from particle_analysis import (
    run_refined_particle_extraction,
)
from measure_intensity import compute_global_signal_intensity

# --- Configuration ---
CH00_PATH = "1_Merged_NC_t00_ch00.tif"
CH01_PATH = "1_Merged_NC_t00_ch01.tif"

# Refined extraction parameters
CIRCLE_RADIUS_SCALE = 1  # Factor to scale ideal circles in Mask B.
USE_WATERSHED = True      # Separate touching particles
WATERSHED_MIN_DIST = 10   # 调小此值 (原20) 以分离靠得更近的粒子
TILED_WATERSHED = True   # 建议您在这里设为 False 来测试全局效果
MIN_CIRCULARITY = 0.75
KEEP_AREA = 3000

# Bandpass Filter Parameters
BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3

# Saturation Stretch Parameters
STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5

# Output files (Only these will be generated)
CSV_OUTPUT = "1merged_NC_results.xlsx" # Explicitly use .xlsx
VIS_OUTPUT = "1merged_NC_visualization.png"


def main():
    print("=== Starting Optimized Global Analysis (Minimal Output) ===")

    # 1. Extract structure from ch00
    # save_intermediates=False prevents generating A, B, C, D images
    print(f"1. Extracting structure from {CH00_PATH}...")
    structure_mask = run_refined_particle_extraction(
        CH00_PATH, 
        save_prefix="structure_refined",
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
        save_intermediates=True,  # <--- Optimization here
        simple_mode=True,  # 开启简易模式：只计算 C 图，速度快且省内存
        restrict_to_largest_circle=False # 关闭 ROI 优化：防止误检导致图像被错误裁剪
    )
    
    # Note: We do NOT save structure_mask explicitly here.

    # 2. Measure Global Intensity
    print(f"2. Measuring global signal intensities...")
    raw_ch01 = cv2.imread(CH01_PATH, cv2.IMREAD_UNCHANGED)
    if raw_ch01 is None:
        raise FileNotFoundError(f"Cannot read {CH01_PATH}")

    compute_global_signal_intensity(
        brightness_image=raw_ch01,
        structure_mask=structure_mask,
        signal_mask=structure_mask, 
        csv_path=CSV_OUTPUT,
        overlay_path=VIS_OUTPUT
    )

    print("\n=== Pipeline Complete! ===")
    print(f"Results: {CSV_OUTPUT}")
    print(f"Visualization: {VIS_OUTPUT}")

if __name__ == "__main__":
    main()