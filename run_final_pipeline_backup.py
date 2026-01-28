import cv2
import os
import numpy as np
from particle_analysis import (
    run_refined_particle_extraction,
    find_notches_and_axes,
    extract_inner_boundaries,
    cut_mask_with_axes,
    refine_mask_with_ellipses,
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
CUT_MASK_PATH = "1mask_signal_cut_by_axes.png"
REFINED_MASK_PATH = "1mask_signal_refined_ellipses.png"

# Refined extraction parameters
CIRCLE_RADIUS_SCALE = 1  # Factor to scale ideal circles in Mask B. <1.0 to shrink, >1.0 to expand.
SIGNAL_OPENING_RADIUS = 1 # 既然不管粘连，就调小此值 (如 1 或 0) 以保留较暗/较小的信号点

# Output files
CSV_OUTPUT = "1final_masked_results2.csv"
VIS_OUTPUT = "1final_result_visualization.png"


def main():
    print("=== Starting Final Analysis Pipeline ===")

    # 1. Adjust ch01 -> show_ch01.png (for signal mask generation)
    print(f"1. Adjusting ch01 -> {SHOW_CH01_PATH}")
    adjust_ch01_image(
        CH01_PATH, 
        SHOW_CH01_PATH,
        exposure=8.0,      # 大幅增加曝光 (建议 10.0 到 20.0)
        brightness=10,      # 降低亮度底数 (原 50 -> 10)，避免把背景抬得太高
        contrast_gain=1.0,  # 降低对比度增益，防止暗部被压黑 (默认 1.0)
        stretch_low=2.0,    # 修改：大幅降低此值 (10.0 -> 2.0)，防止微弱信号被当成背景切除
        stretch_high=98.5,  # 关键：让前10%的亮点过曝，从而强行拉伸并提亮剩下的90%暗部细节
        do_stretch=True    # 启用自动拉伸
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
        circle_radius_scale=CIRCLE_RADIUS_SCALE
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

    # Overlay 3: Axes + Raw CH01
    overlay_mask(
        mask_path="structure_axes.png",
        base_path=CH01_PATH,
        save_path="structure_axes_raw_ch01_overlay.png"
    )
    print(" -> Saved structure_axes_raw_ch01_overlay.png")

    # 4. Refine Signal Mask (Cut with axes -> Fit Ellipses)
    print(f"4. Refining signal mask...")
    
    # 4a. Cut with axes
    cut_mask = cut_mask_with_axes(prelim_signal_mask, axes_info, line_thickness=3)
    cv2.imwrite(CUT_MASK_PATH, cut_mask)
    print(f" -> Saved cut mask: {CUT_MASK_PATH}")
    
    # 4b. Fit ellipses
    final_signal_mask = refine_mask_with_ellipses(cut_mask, min_area=10)
    cv2.imwrite(REFINED_MASK_PATH, final_signal_mask)
    print(f" -> Saved refined mask: {REFINED_MASK_PATH}")

    # 5. Measure and Visualize
    print(f"5. Measuring intensities and generating overlay...")
    raw_ch01 = cv2.imread(CH01_PATH, cv2.IMREAD_UNCHANGED)
    if raw_ch01 is None:
        raise FileNotFoundError(f"Cannot read {CH01_PATH}")

    compute_masked_quadrant_intensity(
        brightness_image=raw_ch01,
        structure_mask=structure_mask,
        signal_mask=final_signal_mask,
        axes_info=axes_info,
        csv_path=CSV_OUTPUT,
        overlay_path=VIS_OUTPUT
    )

    print("\n=== Pipeline Complete! ===")
    print(f"Results: {CSV_OUTPUT}")
    print(f"Visualization: {VIS_OUTPUT}")

if __name__ == "__main__":
    main()