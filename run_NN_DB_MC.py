import cv2
import numpy as np
import os
from scipy import ndimage
from skimage import morphology, measure
from particle_analysis import find_inner_holes_contours, separate_particles_watershed
from measure_intensity import compute_circles_intensity

# --- Configuration ---
CH00_PATH = "Project002_05122025_nTProBNP_TileScan 2_A_1_Merged_NC_t00_ch00.tif"
CH01_PATH = "Project002_05122025_nTProBNP_TileScan 2_A_1_Merged_NC_t00_ch01.tif"

# Output Files
CSV_OUTPUT = "P2_NN_DB_MC_results.xlsx"
VIS_OUTPUT = "P2_NN_DB_MC_01visualization.png"
FINAL_RESULT_PATH = "P2_NN_DB_MC_00visualization.png"

# --- Parameters for Dirty Background (DB) Extraction ---
# 1. Median Blur: Suppresses salt-and-pepper/foggy noise while preserving edges.
MEDIAN_BLUR_KSIZE = 5       # Kernel size (3, 5, 7). Larger = more smoothing.

# 2. Adaptive Threshold: Handles non-uniform background (fog).
ADAPTIVE_BLOCK_SIZE = 51    # Size of the pixel neighborhood. Must be odd.
                            # Should be roughly the size of the features you want to isolate.
ADAPTIVE_C = -2             # Constant subtracted from the mean. 
                            # Threshold = Mean - C.
                            # Negative C (e.g. -2) -> Threshold = Mean + 2. Stricter (reduces fog).
                            # Positive C (e.g. 2) -> Threshold = Mean - 2. More lenient (fills holes but adds noise).

# 3. Morphological Reconstruction: Connects fragmented parts.
CLOSING_RADIUS = 5          # Radius of the disk element for closing (Dilation -> Erosion).
                            # Helps bridge gaps in the particle structure.

# 4. Filtering
PRE_CLOSING_MIN_AREA = 3000  # [New] 在闭操作前移除细小噪点（关键！防止噪点被膨胀连接）
MIN_OBJECT_AREA = 80000   # Minimum area to keep a particle.
FILL_HOLES = True           # Fill internal holes to make solid masks.
MIN_CIRCULARITY = 0.7       # [New] 最小圆度 (0-1)，低于此值将被剔除
MAX_ASPECT_RATIO = 1.3      # [New] 最大长宽比，高于此值将被剔除
WATERSHED_MIN_DIST = 10     # [Modified] 分水岭最小距离 (原20)，调小有助于分开粘连粒子

# --- Parameters for Hole Detection (Same as run_NN_MC) ---
# These are passed to find_inner_holes_contours
HOLE_MIN_AREA = 20
HOLE_MAX_AREA = 5000
HOLE_MIN_CIRCULARITY = 0.5

def extract_structure_adaptive(image_path, save_prefix="debug_db"):
    """
    Custom extraction logic for dirty/foggy backgrounds.
    Pipeline: Median Blur -> Adaptive Threshold -> Morphological Closing -> Fill Holes.
    """
    print(f"Loading {image_path}...")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
        
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read image.")

    # --- Step 1: Median Blur ---
    print(f"  [DB] Applying Median Blur (ksize={MEDIAN_BLUR_KSIZE})...")
    img_blur = cv2.medianBlur(img, MEDIAN_BLUR_KSIZE)
    cv2.imwrite(f"{save_prefix}_1_median.png", img_blur)

    # --- Step 2: Adaptive Thresholding ---
    print(f"  [DB] Applying Adaptive Threshold (Block={ADAPTIVE_BLOCK_SIZE}, C={ADAPTIVE_C})...")
    # ADAPTIVE_THRESH_GAUSSIAN_C is generally better for natural lighting changes
    thresh = cv2.adaptiveThreshold(
        img_blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        ADAPTIVE_BLOCK_SIZE, 
        ADAPTIVE_C
    )
    cv2.imwrite(f"{save_prefix}_2_adaptive_thresh.png", thresh)

    # --- Step 2.5: Pre-Closing Filter (关键步骤) ---
    if PRE_CLOSING_MIN_AREA > 0:
        print(f"  [DB] Pre-filtering noise (Min Area={PRE_CLOSING_MIN_AREA})...")
        thresh_bool = morphology.remove_small_objects(thresh > 0, min_size=PRE_CLOSING_MIN_AREA)
        thresh = (thresh_bool.astype(np.uint8) * 255)

    # --- Step 3: Morphological Reconstruction (Closing) ---
    print(f"  [DB] Morphological Closing (Radius={CLOSING_RADIUS})...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_RADIUS*2+1, CLOSING_RADIUS*2+1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{save_prefix}_3_closed.png", closed)

    # --- Step 4: Fill Holes ---
    mask = closed > 0
    if FILL_HOLES:
        print("  [DB] Filling holes...")
        mask = ndimage.binary_fill_holes(mask)

    # --- Step 5: Filter Small Objects ---
    print(f"  [DB] Removing small objects (Min Area={MIN_OBJECT_AREA})...")
    mask = morphology.remove_small_objects(mask, min_size=MIN_OBJECT_AREA)
    
    # --- Step 6: Watershed Separation (Optional but recommended) ---
    print(f"  [DB] Applying Watershed separation (Min Dist={WATERSHED_MIN_DIST})...")
    # Using the function from particle_analysis to split touching particles
    labels = separate_particles_watershed(mask, min_distance=WATERSHED_MIN_DIST)
    
    # --- Step 7: Advanced Filtering (Circularity & Aspect Ratio) ---
    print(f"  [DB] Filtering particles (Circularity>={MIN_CIRCULARITY}, Aspect Ratio<={MAX_ASPECT_RATIO})...")
    regions = measure.regionprops(labels)
    final_mask = np.zeros(labels.shape, dtype=np.uint8)
    
    kept_count = 0
    for r in regions:
        # 1. Aspect Ratio (长宽比)
        minr, minc, maxr, maxc = r.bbox
        h = maxr - minr
        w = maxc - minc
        aspect_ratio = max(h, w) / max(1, min(h, w))
        
        # 2. Circularity (圆度)
        perimeter = max(r.perimeter, 1e-6)
        circularity = 4 * np.pi * r.area / (perimeter ** 2)
        
        if aspect_ratio <= MAX_ASPECT_RATIO and circularity >= MIN_CIRCULARITY:
            final_mask[r.coords[:, 0], r.coords[:, 1]] = 255
            kept_count += 1
            
    print(f"  [DB] Kept {kept_count} particles after filtering.")
    cv2.imwrite(f"{save_prefix}_4_final_mask.png", final_mask)
    
    return final_mask

def main():
    print("=== Starting NN_DB_MC Analysis (Dirty Background Mode) ===")
    
    # 1. Extract Structure using new Adaptive method
    print("=== 1. Extracting structure (Adaptive Method) ===")
    try:
        mask1 = extract_structure_adaptive(CH00_PATH, save_prefix="debug_db")
    except Exception as e:
        print(f"[Error] Extraction failed: {e}")
        return

    # 2. Preprocess ch00 for Hole Detection (Masking & Flattening)
    print("=== 2. Preprocessing ch00 for Hole Detection ===")
    img = cv2.imread(CH00_PATH, cv2.IMREAD_UNCHANGED)
    
    # Ensure mask is uint8
    if mask1.dtype == bool:
        mask1 = (mask1 * 255).astype(np.uint8)
    
    # Mask the original image
    masked_img = cv2.bitwise_and(img, img, mask=mask1)
    
    # Convert to grayscale if needed
    if masked_img.ndim == 3:
        masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    else:
        masked_img_gray = masked_img

    # Normalize brightness within the mask (Contrast Stretch)
    mask_for_norm = mask1 > 0
    masked_vals = masked_img_gray[mask_for_norm]
    
    if masked_vals.size == 0:
        print("[Error] Mask is empty after extraction.")
        return

    v_min, v_max = np.percentile(masked_vals, (1, 99))
    if v_max > v_min:
        scaled = (masked_img_gray.astype(np.float32) - v_min) * (255.0 / (v_max - v_min))
        masked_img_gray = np.clip(scaled, 0, 255).astype(np.uint8)
    else:
        masked_img_gray = np.zeros_like(masked_img_gray, dtype=np.uint8)

    # Slight blur to reduce noise for hole detection
    masked_img_flattened = cv2.GaussianBlur(masked_img_gray, (5, 5), 0)

    # 3. Find Inner Holes & Predict 4 Circles
    print("=== 3. Find Inner Holes & Predict 4 Circles ===")
    _, circles_data = find_inner_holes_contours(
        image_gray=masked_img_flattened,
        structure_mask=mask1,
        save_path=FINAL_RESULT_PATH, 
        min_area=HOLE_MIN_AREA,
        max_area=HOLE_MAX_AREA,
        min_circularity=HOLE_MIN_CIRCULARITY,
        block_size=13,
        c_value=1,
        erosion_size=15, 
        debug=False,
        detect_dark=False, 
        predict_4c=True,    # Enable 4C prediction
        return_data=True
    )
    print(f"  -> Saved visualization 00: {FINAL_RESULT_PATH}")

    # 4. Measure Intensity on ch01
    print("=== 4. Measure Intensity on ch01 ===")
    if not os.path.exists(CH01_PATH):
        print(f"[Error] CH01 file not found: {CH01_PATH}")
        return

    raw_ch01 = cv2.imread(CH01_PATH, cv2.IMREAD_UNCHANGED)
    
    compute_circles_intensity(
        brightness_image=raw_ch01,
        circles_data=circles_data,
        csv_path=CSV_OUTPUT,
        overlay_path=VIS_OUTPUT
    )
    print(f"  -> Done. Results: {CSV_OUTPUT}")
    print(f"  -> Saved visualization 01: {VIS_OUTPUT}")

if __name__ == "__main__":
    main()