import cv2

from particle_analysis import (
    run_refined_particle_extraction,
    find_notches_and_axes,
)
from measure_intensity import compute_quadrant_intensity

# --- Configuration ---
CH00_PATH = "Project002_CRP_ELISA_DirectBinding_qCAP_VaryingBiotin0,0.1,1,10mg_TileScan 3_A_1_Merged_CRP_0ng_ch00.tif"
CH01_PATH = "Project002_CRP_ELISA_DirectBinding_qCAP_VaryingBiotin0,0.1,1,10mg_TileScan 3_A_1_Merged_CRP_0ng_ch01.tif"

# Refined extraction parameters
CIRCLE_RADIUS_SCALE = 1  # Factor to scale ideal circles in Mask B. <1.0 to shrink, >1.0 to expand.
SIGNAL_OPENING_RADIUS = 1  # retained for parity, unused in this pipeline
USE_WATERSHED = True      # 是否启用分水岭算法分割粘连颗粒?
WATERSHED_MIN_DIST = 20   # 分水岭算法的最小峰值距离(根据粒子大小调整)
MIN_CIRCULARITY = 0.75   # 圆度过滤阈值(1.0为完美圆)
KEEP_AREA = 3000         # 单个粒子的最小面积

# Bandpass Filter Parameters (Fiji-like style)
BANDPASS_LARGE_SIGMA = 40  # 背景平滑半径 (对应 Fiji "Filter large structures down to")
BANDPASS_SMALL_SIGMA = 3   # 噪点平滑半径 (对应 Fiji "Filter small structures up to")

# Saturation Stretch Parameters (after Bandpass)
STRETCH_LOW_PERCENTILE = 0.5  # Saturate bottom 0.5% of pixels to black
STRETCH_HIGH_PERCENTILE = 99.5  # Saturate top 0.5% of pixels to white

# Output files
CSV_OUTPUT = "1project2final_masked_results2.csv"
ID_MAP_OUTPUT = "particle_id_map.png"


def main():
    print("=== Starting WN MC Pipeline ===")

    # 1. Extract structure & axes from ch00
    print(f"1. Extracting structure & axes from {CH00_PATH}...")
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
        stretch_high=STRETCH_HIGH_PERCENTILE,
        simple_mode=True,
    )
    axes_info = find_notches_and_axes(
        structure_mask,
        save_path="structure_axes.png",
    )

    # 2. Measure on raw ch01 using measurement circles
    print("2. Measuring intensities...")
    raw_ch01 = cv2.imread(CH01_PATH, cv2.IMREAD_UNCHANGED)
    if raw_ch01 is None:
        raise FileNotFoundError(f"Cannot read {CH01_PATH}")

    # compute_quadrant_intensity expects regionprops regions and labels.
    # Reuse structure_mask labels here.
    num_labels, labels = cv2.connectedComponents(structure_mask)
    from skimage import measure
    regions = measure.regionprops(labels)

    compute_quadrant_intensity(
        brightness_image=raw_ch01,
        labels=labels,
        regions=regions,
        axes_info=axes_info,
        csv_path=CSV_OUTPUT,
        id_map_path=ID_MAP_OUTPUT,
    )

    print("\n=== Pipeline Complete! ===")
    print(f"Results: {CSV_OUTPUT}")
    print(f"Visualization: {ID_MAP_OUTPUT}")


if __name__ == "__main__":
    main()
