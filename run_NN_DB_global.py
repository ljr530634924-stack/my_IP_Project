import cv2
import numpy as np
import os
import sys
import multiprocessing
from scipy import ndimage
from skimage import morphology, measure
from particle_analysis import separate_particles_watershed
from measure_intensity import compute_global_signal_intensity
from overlay_mask import overlay_mask

# --- Configuration ---
CH00_PATH = "Project002_05122025_nTProBNP_TileScan 2_A_1_Merged_NC_t00_ch00.tif"
CH01_PATH = "Project002_05122025_nTProBNP_TileScan 2_A_1_Merged_NC_t00_ch01.tif"

# Output Files
CSV_OUTPUT = "P2_NN_DB_global_results.xlsx"
VIS_OUTPUT = "P2_NN_DB_global_01visualization.png"
VIS_00_OUTPUT = "P2_NN_DB_global_00visualization.png"

# --- Parameters for Dirty Background (DB) Extraction ---
# 1. Median Blur: Suppresses salt-and-pepper/foggy noise while preserving edges.
MEDIAN_BLUR_KSIZE = 5       # Kernel size (3, 5, 7). Larger = more smoothing.

# 2. Adaptive Threshold: Handles non-uniform background (fog).
ADAPTIVE_BLOCK_SIZE = 51    # Size of the pixel neighborhood. Must be odd.
ADAPTIVE_C = -7        # Negative C -> Stricter threshold (reduces fog).

# 3. Morphological Reconstruction: Connects fragmented parts.
CLOSING_RADIUS = 5          # Radius for closing (Dilation -> Erosion).

# 4. Filtering (User requested defaults for small image testing)
PRE_CLOSING_MIN_AREA = 3000  # [Modified] 闭操作前移除小噪点 (原 30000 -> 100)
MIN_OBJECT_AREA = 100000      # [Modified] 最终保留的最小粒子面积 (原 30000 -> 3000)
FILL_HOLES = True           # Fill internal holes.
MIN_CIRCULARITY = 0.7       # 最小圆度
MAX_ASPECT_RATIO = 1.3      # 最大长宽比
WATERSHED_MIN_DIST = 4    # 分水岭最小距离

def extract_structure_adaptive(image_path, save_prefix="debug_db_global"):
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
    if save_prefix:
        cv2.imwrite(f"{save_prefix}_1_median.png", img_blur)

    # --- Step 2: Adaptive Thresholding ---
    print(f"  [DB] Applying Adaptive Threshold (Block={ADAPTIVE_BLOCK_SIZE}, C={ADAPTIVE_C})...")
    thresh = cv2.adaptiveThreshold(
        img_blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        ADAPTIVE_BLOCK_SIZE, 
        ADAPTIVE_C
    )
    if save_prefix:
        cv2.imwrite(f"{save_prefix}_2_adaptive_thresh.png", thresh)

    # --- Step 2.5: Pre-Closing Filter ---
    if PRE_CLOSING_MIN_AREA > 0:
        print(f"  [DB] Pre-filtering noise (Min Area={PRE_CLOSING_MIN_AREA})...")
        thresh_bool = morphology.remove_small_objects(thresh > 0, min_size=PRE_CLOSING_MIN_AREA)
        thresh = (thresh_bool.astype(np.uint8) * 255)

    # --- Step 3: Morphological Reconstruction (Closing) ---
    print(f"  [DB] Morphological Closing (Radius={CLOSING_RADIUS})...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_RADIUS*2+1, CLOSING_RADIUS*2+1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    if save_prefix:
        cv2.imwrite(f"{save_prefix}_3_closed.png", closed)

    # --- Step 4: Fill Holes ---
    mask = closed > 0
    if FILL_HOLES:
        print("  [DB] Filling holes...")
        mask = ndimage.binary_fill_holes(mask)

    # --- Step 5: Filter Small Objects ---
    print(f"  [DB] Removing small objects (Min Area={MIN_OBJECT_AREA})...")
    mask = morphology.remove_small_objects(mask, min_size=MIN_OBJECT_AREA)
    
    # --- Step 6: Watershed Separation ---
    print(f"  [DB] Applying Watershed separation (Min Dist={WATERSHED_MIN_DIST})...")
    labels = separate_particles_watershed(mask, min_distance=WATERSHED_MIN_DIST)
    
    # --- Step 7: Advanced Filtering (Circularity & Aspect Ratio) ---
    print(f"  [DB] Filtering particles (Circularity>={MIN_CIRCULARITY}, Aspect Ratio<={MAX_ASPECT_RATIO}, Area>={MIN_OBJECT_AREA})...")
    regions = measure.regionprops(labels)
    final_mask = np.zeros(labels.shape, dtype=np.uint8)
    
    kept_count = 0
    for r in regions:
        # 1. Aspect Ratio
        minr, minc, maxr, maxc = r.bbox
        h = maxr - minr
        w = maxc - minc
        aspect_ratio = max(h, w) / max(1, min(h, w))
        
        # 2. Circularity
        perimeter = max(r.perimeter, 1e-6)
        circularity = 4 * np.pi * r.area / (perimeter ** 2)
        
        if aspect_ratio <= MAX_ASPECT_RATIO and circularity >= MIN_CIRCULARITY and r.area >= MIN_OBJECT_AREA:
            final_mask[r.coords[:, 0], r.coords[:, 1]] = 255
            kept_count += 1
            
    print(f"  [DB] Kept {kept_count} particles after filtering.")
    if save_prefix:
        cv2.imwrite(f"{save_prefix}_4_final_mask.png", final_mask)
    
    return final_mask

def main():
    # Windows 下多进程保护 (防止 scikit-image/joblib 导致的无限重启)
    multiprocessing.freeze_support()

    print("=== Starting NN_DB_Global Analysis (Dirty Background + Global Measure) ===")
    
    # 1. Extract Structure
    print("=== 1. Extracting structure (Adaptive Method) ===")
    try:
        mask1 = extract_structure_adaptive(CH00_PATH, save_prefix="debug_db_global")
    except Exception as e:
        print(f"[Error] Extraction failed: {e}")
        return

    # 1.5 Visualize Structure on CH00 (New Step)
    print(f"=== 1.5 Visualizing structure on ch00 ===")
    
    # Use overlay_mask logic (similar to run_overlay.py)
    temp_mask_path = "temp_vis_mask.png"
    cv2.imwrite(temp_mask_path, mask1)
    try:
        overlay_mask(
            mask_path=temp_mask_path,
            base_path=CH00_PATH,
            save_path=VIS_00_OUTPUT,
            thresh=0.0,
            invert=False
        )
        
        # --- NEW: Add sorted IDs to VIS_00_OUTPUT ---
        # Read the generated overlay to draw IDs on it
        vis_00 = cv2.imread(VIS_00_OUTPUT)
        if vis_00 is not None:
            num_labels, labels = cv2.connectedComponents(mask1)
            regions = measure.regionprops(labels)
            # Sort by x-coordinate (col) to match the logic in compute_global_signal_intensity
            regions = sorted(regions, key=lambda r: r.centroid[1])
            
            for idx, r in enumerate(regions):
                pid = idx + 1
                cy, cx = r.centroid
                # Draw ID in Green
                cv2.putText(vis_00, str(pid), (int(cx), int(cy)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
            
            cv2.imwrite(VIS_00_OUTPUT, vis_00)
            print(f"  -> Added sorted IDs to: {VIS_00_OUTPUT}")
    finally:
        if os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)

    # 2. Measure Global Intensity on ch01
    print("=== 2. Measuring Global Intensity on ch01 ===")
    if not os.path.exists(CH01_PATH):
        print(f"[Error] CH01 file not found: {CH01_PATH}")
        return

    raw_ch01 = cv2.imread(CH01_PATH, cv2.IMREAD_UNCHANGED)
    
    # Use structure_mask as both structure and signal mask (measure whole particle)
    compute_global_signal_intensity(
        brightness_image=raw_ch01,
        structure_mask=mask1,
        signal_mask=mask1, 
        csv_path=CSV_OUTPUT,
        overlay_path=VIS_OUTPUT
    )
    print(f"  -> Done. Results: {CSV_OUTPUT}")
    print(f"  -> Saved visualization: {VIS_OUTPUT}")

if __name__ == "__main__":
    main()
    sys.exit(0) # 强制退出，防止异常挂起