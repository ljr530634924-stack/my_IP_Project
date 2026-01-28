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
from overlay_mask import overlay_mask

# --- Configuration ---
CH00_PATH = "Project001_qCAPs_ACPEGN3_4conc_0,0.1,1,4mgmL_A1 (2)_Merged_ch00.tif"
CH01_PATH = "Project001_qCAPs_ACPEGN3_4conc_0,0.1,1,4mgmL_A1 (2)_Merged_ch01.tif"

# Intermediate files
SHOW_CH01_PATH = "1show_ch01.png"
SIGNAL_MASK_PATH = "1mask_signal_from_ch01.png"

# Refined extraction parameters
CIRCLE_RADIUS_SCALE = 1  # Factor to scale ideal circles in Mask B. <1.0 to shrink, >1.0 to expand.
SIGNAL_OPENING_RADIUS = 1 # 既然不管粘连，就调小此值 (如 1 或 0) 以保留较暗/较小的信号点
USE_WATERSHED = True      # 是否启用分水岭算法分割粘连粒子
WATERSHED_MIN_DIST = 20   # 分水岭算法的最小峰值距离 (根据粒子大小调整)
MIN_CIRCULARITY = 0.75   # 圆度过滤阈值 (1.0为完美圆)
KEEP_AREA = 3000         # 新增：单个粒子的最小面积。

# Bandpass Filter Parameters (Fiji-like style)
BANDPASS_LARGE_SIGMA = 40  # 背景平滑半径 (对应 Fiji "Filter large structures down to")
BANDPASS_SMALL_SIGMA = 3   # 噪点平滑半径 (对应 Fiji "Filter small structures up to")

# Saturation Stretch Parameters (after Bandpass)
STRETCH_LOW_PERCENTILE = 0.5  # Saturate bottom 0.5% of pixels to black
STRETCH_HIGH_PERCENTILE = 99.5 # Saturate top 0.5% of pixels to white

# Output files
CSV_OUTPUT = "1project2Test_masked_results2.csv"
VIS_OUTPUT = "1project2Test_result_visualization.png"


def main():
    print("=== Starting Final Analysis Pipeline ===")

    # 1. Adjust ch01 -> show_ch01.png (for signal mask generation)
    print(f"1. Adjusting ch01 -> {SHOW_CH01_PATH}")
    adjust_ch01_image(
        CH01_PATH, 
        SHOW_CH01_PATH,
        exposure=3.0,
        contrast_gain=1.0,
        stretch_low=0.175,
        stretch_high=99.825,
        do_stretch=True,
    )



    # 2. Extract signal mask from show_ch01.png
    # Using thin_opening_radius=0 to keep all details as requested
    print(f"2. Extracting preliminary signal mask from {SHOW_CH01_PATH}...")
    prelim_signal_mask = extract_inner_boundaries(
        SHOW_CH01_PATH, 
        save_path=SIGNAL_MASK_PATH,
        thin_opening_radius=SIGNAL_OPENING_RADIUS,
        min_size=80,       # 稍微调大 (60 -> 80) 以抵消提高灵敏度带来的噪点
        keep_area=80,      # 同上
        canny_sigma=2.0,   # 新增：设为 1.0 (原 2.0) 以检测更细微/模糊的边缘
        clahe_clip_limit=0.01, # 新增：设为 0.03 (原 0.01) 以增强微弱信号的对比度
        brightness_threshold=0.2 # 新增：直接提取亮度大于30%的区域，防止高亮信号丢失
    )

    # 3. Extract structure & axes from ch00
    # Using refined method (A/B/C/D) for best particle definition
    print(f"3. Extracting structure & axes from {CH00_PATH}...")
    structure_mask = run_refined_particle_extraction(
        CH00_PATH, 
        save_prefix="structure_refined",
        circle_radius_scale=CIRCLE_RADIUS_SCALE,
        use_watershed=USE_WATERSHED,
        watershed_min_dist=WATERSHED_MIN_DIST,
        min_circularity=MIN_CIRCULARITY,
        keep_area=KEEP_AREA,
        large_sigma=BANDPASS_LARGE_SIGMA,   # 传入 Bandpass 参数 (对应 Large Sigma)
        noise_sigma=BANDPASS_SMALL_SIGMA,   # 传入 Bandpass 参数
        stretch_low=STRETCH_LOW_PERCENTILE,
        stretch_high=STRETCH_HIGH_PERCENTILE
    )
    axes_info = find_notches_and_axes(
        structure_mask, 
        save_path="structure_axes.png"
    )

    # --- Generate Overlays for Inspection ---
    print("Generating structure axes overlays...")
    # Overlay 1: Axes + CH00
    overlay_mask(
        mask_path="structure_axes.png",
        base_path=CH00_PATH,
        save_path="structure_axes_overlay.png"
    )
    print(" -> Saved structure_axes_overlay.png")

    # Overlay 2: Axes + Show01
    overlay_mask(
        mask_path="structure_axes.png",
        base_path=SHOW_CH01_PATH,
        save_path="structure_axes_show01_overlay.png"
    )
    print(" -> Saved structure_axes_show01_overlay.png")


    # 4. (Skipped) Refine Signal Mask
    # We now use the preliminary signal mask directly with the new area-based circle logic.

    # 5. Measure and Visualize
    print(f"5. Measuring intensities and generating overlay...")
    raw_ch01 = cv2.imread(CH01_PATH, cv2.IMREAD_UNCHANGED)
    if raw_ch01 is None:
        raise FileNotFoundError(f"Cannot read {CH01_PATH}")

    compute_masked_quadrant_intensity(
        brightness_image=raw_ch01,
        structure_mask=structure_mask,
        signal_mask=prelim_signal_mask, # Use the mask from Step 2 directly
        axes_info=axes_info,
        csv_path=CSV_OUTPUT,
        overlay_path=VIS_OUTPUT
    )

    print("\n=== Pipeline Complete! ===")
    print(f"Results: {CSV_OUTPUT}")
    print(f"Visualization: {VIS_OUTPUT}")

if __name__ == "__main__":
    main()