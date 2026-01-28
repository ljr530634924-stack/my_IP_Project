import cv2
import numpy as np
import os
from particle_analysis import extract_outer_boundaries, find_inner_holes_contours, run_refined_particle_extraction
from overlay_mask import overlay_mask

# --- Configuration ---
# 杈撳叆鍥剧墖璺緞
ch00_path = "1_Merged_ch00_0ngmL.tif"
INPUTPATH = "1_Merged_ch00_0ngmL.tif"

CIRCLE_RADIUS_SCALE = 1  # Factor to scale ideal circles in Mask B. <1.0 to shrink, >1.0 to expand.
SIGNAL_OPENING_RADIUS = 1 # 鏃㈢劧涓嶇绮樿繛锛屽氨璋冨皬姝ゅ€?(濡?1 鎴?0) 浠ヤ繚鐣欒緝鏆?杈冨皬鐨勪俊鍙风偣
USE_WATERSHED = True      # 鏄惁鍚敤鍒嗘按宀畻娉曞垎鍓茬矘杩炵矑瀛?
WATERSHED_MIN_DIST = 20   # 鍒嗘按宀畻娉曠殑鏈€灏忓嘲鍊艰窛绂?(鏍规嵁绮掑瓙澶у皬璋冩暣)
MIN_CIRCULARITY = 0.75   # 鍦嗗害杩囨护闃堝€?(1.0涓哄畬缇庡渾)
KEEP_AREA = 3000         # 鏂板锛氬崟涓矑瀛愮殑鏈€灏忛潰绉€?

# Bandpass Filter Parameters (Fiji-like style)
BANDPASS_LARGE_SIGMA = 40  # 鑳屾櫙骞虫粦鍗婂緞 (瀵瑰簲 Fiji "Filter large structures down to")
BANDPASS_SMALL_SIGMA = 3   # 鍣偣骞虫粦鍗婂緞 (瀵瑰簲 Fiji "Filter small structures up to")

# Saturation Stretch Parameters (after Bandpass)
STRETCH_LOW_PERCENTILE = 0.5  # Saturate bottom 0.5% of pixels to black
STRETCH_HIGH_PERCENTILE = 99.5 # Saturate top 0.5% of pixels to white

# 涓棿缁撴灉鍜屾渶缁堣緭鍑鸿矾寰?
STEP2_OVERLAY_PATH = "GIC_step2_overlay.png"
STEP3_MASK_PATH = "GIC_step3_refined_mask.png"
FINAL_RESULT_PATH = "GIC_step4_holes_result.png"

def main():
    if not os.path.exists(ch00_path):
        print(f"[Error] Input file not found: {ch00_path}")
        return

    print("=== 1. Extract Outer Boundaries (Initial) ===")
    # 绗竴姝ワ細鎻愬彇鍒濆杞粨
    # 鐩殑锛氫粠澶嶆潅鐨勫師濮嬭儗鏅腑鍒嗙鍑虹矑瀛?
    mask1=run_refined_particle_extraction(
        ch00_path, 
        save_prefix="structure_refined",
        circle_radius_scale=CIRCLE_RADIUS_SCALE,
        use_watershed=USE_WATERSHED,
        watershed_min_dist=WATERSHED_MIN_DIST,
        min_circularity=MIN_CIRCULARITY,
        keep_area=KEEP_AREA,
        large_sigma=BANDPASS_LARGE_SIGMA,   # 浼犲叆 Bandpass 鍙傛暟 (瀵瑰簲 Large Sigma)
        noise_sigma=BANDPASS_SMALL_SIGMA,   # 浼犲叆 Bandpass 鍙傛暟
        stretch_low=STRETCH_LOW_PERCENTILE,
        stretch_high=STRETCH_HIGH_PERCENTILE,
        simple_mode=True,  # 寮€鍚畝鏄撴ā寮忥細鍙绠?C 鍥撅紝閫熷害蹇瓨
    )

    print("=== 2. Overlay on ch00 (Masking) ===")
    # 绗簩姝ワ細Overlay (Masking)
    # 鍙傜収 run_overlay.py 鐨勯€昏緫锛歁ask鐧借壊閮ㄥ垎淇濈暀鍘熷浘锛岄粦鑹查儴鍒嗗彉榛戙€?
    
    # 璇诲彇鍘熷浘
    img = cv2.imread(ch00_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read input image.")
    
    # 纭繚 mask 鏄?uint8 鏍煎紡 (0, 255)
    if mask1.dtype == bool:
        mask1 = (mask1 * 255).astype(np.uint8)
    
    # 浣跨敤 bitwise_and 瀹炵幇 Overlay 鏁堟灉锛?
    # 鍙湁 mask1 涓?255 鐨勫湴鏂逛繚鐣?img 鐨勫儚绱狅紝鍏朵綑涓?0 (榛戣壊)
    masked_img = cv2.bitwise_and(img, img, mask=mask1)
    
    # 淇濆瓨 Step 2 缁撴灉锛屼緵 Step 3 璇诲彇锛屼篃浣滀负涓棿缁撴灉妫€鏌?
    cv2.imwrite(STEP2_OVERLAY_PATH, masked_img)
    print(f" -> Saved masked image to: {STEP2_OVERLAY_PATH}")

    print("flatten")
    # 纭繚杞负鐏板害鍥?
    if masked_img.ndim == 3:
        masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    else:
        masked_img_gray = masked_img

    # [Fix] cv2.adaptiveThreshold 蹇呴』浣跨敤 8-bit 鍥惧儚
    # 鍙湪 mask 鍐呴儴鍋氬綊涓€鍖栵紝閬垮厤榛戣壊鑳屾櫙鎷変綆瀵规瘮搴?
    if mask1.dtype == bool:
        mask_for_norm = mask1
    else:
        mask_for_norm = mask1 > 0

    masked_vals = masked_img_gray[mask_for_norm]
    if masked_vals.size == 0:
        raise ValueError("Mask is empty; cannot normalize.")

    v_min = float(masked_vals.min())
    v_max = float(masked_vals.max())

    if v_max > v_min:
        scaled = (masked_img_gray.astype(np.float32) - v_min) * (255.0 / (v_max - v_min))
        masked_img_gray = np.clip(scaled, 0, 255).astype(np.uint8)
    else:
        masked_img_gray = np.zeros_like(masked_img_gray, dtype=np.uint8)

    # Step 3.5: 杞诲害 Gaussian 鍘诲櫔锛屽緱鍒扮敤浜庢壘瀛旂殑杈撳叆鍥?
    masked_img_flattened = cv2.GaussianBlur(masked_img_gray, (5, 5), 0)
    cv2.imwrite("masked_img_flattened.png", masked_img_flattened)    

    print("=== 4. Find Inner Holes ===")
    # 绗洓姝ワ細鎵惧瓟娲?
    # 浣跨敤 Step 2 鐨?masked_img (鐏板害) 鍜?Step 3 鐨?mask2 (缁撴瀯鎺╄啘)
    


    result_img = find_inner_holes_contours(
        image_gray=masked_img_flattened,
        structure_mask=mask1,
        save_path=FINAL_RESULT_PATH,
        min_area=5,       # 最小区域面积
        max_area=5000,     # 最大区域面积   
        min_circularity=0.2,
        block_size=3,     # 自适应阈值块大小
        opening_kernel_size=3,  # 形态学开运算核大小（奇数）
        debug=True         #是否生成debug图片
    )
    
    print(f"=== Done! Final result: {FINAL_RESULT_PATH} ===")

if __name__ == "__main__":
    main()

