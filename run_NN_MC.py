import cv2
import numpy as np
import os
from particle_analysis import run_refined_particle_extraction, find_inner_holes_contours
from measure_intensity import compute_circles_intensity

# --- Configuration ---
CH00_PATH = "1_Merged_ch00_0ngmL.tif"
CH01_PATH = "1_Merged_ch01_0ngmL.tif" # 确保这里有对应的 ch01 文件

# Output Files
CSV_OUTPUT = "NN_MC_results.xlsx"
VIS_OUTPUT = "NN_MC_01visualization.png"
FINAL_RESULT_PATH = "NN_MC_00visualization.png"

# Parameters (Same as GIC_4C)
CIRCLE_RADIUS_SCALE = 1
USE_WATERSHED = True
WATERSHED_MIN_DIST = 20
MIN_CIRCULARITY = 0.75
KEEP_AREA = 3000

BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3

STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5

def main():
    if not os.path.exists(CH00_PATH):
        print(f"[Error] Input file not found: {CH00_PATH}")
        return
    if not os.path.exists(CH01_PATH):
        print(f"[Error] Input file not found: {CH01_PATH}")
        return

    print("=== 1. Extract Outer Boundaries (Structure from ch00) ===")
    mask1 = run_refined_particle_extraction(
        CH00_PATH, 
        save_prefix="structure_refined_NN_MC",
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

    print("=== 2. Preprocess ch00 for Hole Detection ===")
    # Replicating the GIC preprocessing steps to ensure consistent hole detection
    img = cv2.imread(CH00_PATH, cv2.IMREAD_UNCHANGED)
    
    if mask1.dtype == bool:
        mask1 = (mask1 * 255).astype(np.uint8)
    
    masked_img = cv2.bitwise_and(img, img, mask=mask1)
    
    if masked_img.ndim == 3:
        masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    else:
        masked_img_gray = masked_img

    if mask1.dtype == bool:
        mask_for_norm = mask1
    else:
        mask_for_norm = mask1 > 0

    masked_vals = masked_img_gray[mask_for_norm]
    if masked_vals.size == 0:
        print("[Error] Mask is empty.")
        return

    v_min, v_max = np.percentile(masked_vals, (1, 99))

    if v_max > v_min:
        scaled = (masked_img_gray.astype(np.float32) - v_min) * (255.0 / (v_max - v_min))
        masked_img_gray = np.clip(scaled, 0, 255).astype(np.uint8)
    else:
        masked_img_gray = np.zeros_like(masked_img_gray, dtype=np.uint8)

    masked_img_flattened = cv2.GaussianBlur(masked_img_gray, (5, 5), 0)

    print("=== 3. Find Inner Holes & Predict 4 Circles (on ch00) ===")
    # We use return_data=True to get the coordinates
    _, circles_data = find_inner_holes_contours(
        image_gray=masked_img_flattened,
        structure_mask=mask1,
        save_path=FINAL_RESULT_PATH, 
        min_area=20,
        max_area=5000,
        min_circularity=0.5,
        block_size=13,
        c_value=1,
        erosion_size=15, 
        debug=False,
        detect_dark=False, 
        predict_4c=True,    # Enable 4C prediction
        return_data=True    # Request data return
    )
    print(f"=== Done! Final result: {FINAL_RESULT_PATH} ===")

    print("=== 4. Measure Intensity on ch01 ===")
    raw_ch01 = cv2.imread(CH01_PATH, cv2.IMREAD_UNCHANGED)
    
    compute_circles_intensity(
        brightness_image=raw_ch01,
        circles_data=circles_data,
        csv_path=CSV_OUTPUT,
        overlay_path=VIS_OUTPUT
    )

if __name__ == "__main__":
    main()