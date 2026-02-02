import cv2
import numpy as np
import os
from particle_analysis import extract_outer_boundaries, find_inner_holes_contours, run_refined_particle_extraction
from overlay_mask import overlay_mask

ch00_path = "1_Merged_ch00_0ngmL.tif"
INPUTPATH = "1_Merged_ch00_0ngmL.tif"

CIRCLE_RADIUS_SCALE = 1
SIGNAL_OPENING_RADIUS = 1
USE_WATERSHED = True
WATERSHED_MIN_DIST = 20
MIN_CIRCULARITY = 0.75
KEEP_AREA = 3000

BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3

STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5

STEP2_OVERLAY_PATH = "GIC_4C_step2_overlay.png"
STEP3_MASK_PATH = "GIC_4C_step3_refined_mask.png"
FINAL_RESULT_PATH = "GIC_step4_holes_result.png" # 保持您要求的文件名

def main():
    if not os.path.exists(ch00_path):
        print(f"[Error] Input file not found: {ch00_path}")
        return

    print("=== 1. Extract Outer Boundaries (Initial) ===")
    mask1=run_refined_particle_extraction(
        ch00_path, 
        save_prefix="structure_refined_4C",
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

    print("=== 2. Overlay on ch00 (Masking) ===")
    
    img = cv2.imread(ch00_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read input image.")
    
    if mask1.dtype == bool:
        mask1 = (mask1 * 255).astype(np.uint8)
    
    masked_img = cv2.bitwise_and(img, img, mask=mask1)
    
    cv2.imwrite(STEP2_OVERLAY_PATH, masked_img)
    print(f" -> Saved masked image to: {STEP2_OVERLAY_PATH}")

    print("flatten")
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
        raise ValueError("Mask is empty; cannot normalize.")

    v_min, v_max = np.percentile(masked_vals, (1, 99))

    if v_max > v_min:
        scaled = (masked_img_gray.astype(np.float32) - v_min) * (255.0 / (v_max - v_min))
        masked_img_gray = np.clip(scaled, 0, 255).astype(np.uint8)
    else:
        masked_img_gray = np.zeros_like(masked_img_gray, dtype=np.uint8)

    masked_img_flattened = cv2.GaussianBlur(masked_img_gray, (5, 5), 0)
    cv2.imwrite("masked_img_flattened_4C.png", masked_img_flattened)    

    print("=== 4. Find Inner Holes (4C Mode) ===")

    result_img = find_inner_holes_contours(
        image_gray=masked_img_flattened,
        structure_mask=mask1,
        save_path=FINAL_RESULT_PATH,
        min_area=20,
        max_area=5000,
        min_circularity=0.5,
        block_size=13,
        c_value=1,
        erosion_size=15, 
        debug=True,
        detect_dark=False, # 检测亮斑
        predict_4c=True    # [New] 开启4C预测模式
    )
    
    print(f"=== Done! Final result: {FINAL_RESULT_PATH} ===")

if __name__ == "__main__":
    main()