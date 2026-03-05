import cv2
import numpy as np
import os
import glob
import sys
import multiprocessing
from scipy import ndimage
from skimage import morphology, measure
from particle_analysis import separate_particles_watershed, find_inner_holes_contours
from measure_intensity import compute_circles_intensity
from overlay_mask import overlay_mask

# --- Configuration ---
# Set this to the folder containing your images.
INPUT_FOLDER = r"D:\Ingenieurpraixs\test_NN_DB_4c"
SAVE_DEBUG_INPUTS = True # Set to True to save inputs for debug_holes.py

# --- Parameters for Dirty Background (DB) Extraction (From run_NN_DB_global.py) ---
MEDIAN_BLUR_KSIZE = 5      # Kernel size (3, 5, 7). Larger = more smoothing.
ADAPTIVE_BLOCK_SIZE = 101   # Should be larger than the particle diameter.
ADAPTIVE_C = -1          # Negative C is stricter (less noise). Positive C is lenient. Start with a negative value for cleaner results.
CLOSING_RADIUS = 3          # Radius for closing.
PRE_CLOSING_MIN_AREA = 100  # Lower this drastically to avoid deleting fragments before closing.
MIN_OBJECT_AREA = 50000     # Lower this drastically for debugging. 50k is too large for most cases.
MAX_OBJECT_AREA = 240000    # Maximum area to filter out overlapping particles.
FILL_HOLES = True           # Fill internal holes.
MIN_CIRCULARITY = 0.5       # Minimum circularity
MAX_ASPECT_RATIO = 1.4      # Maximum aspect ratio
WATERSHED_MIN_DIST = 15   # A value of 2 is very small and might over-segment. Try 10-20.

def extract_structure_adaptive(image_path, save_prefix=None):
    """
    Custom extraction logic for dirty/foggy backgrounds.
    Pipeline: Median Blur -> Adaptive Threshold -> Morphological Closing -> Fill Holes.
    """
    # print(f"  [DB] Loading {os.path.basename(image_path)}...")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
        
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read image.")

    # --- Step 1: Median Blur ---
    # print(f"  [DB] Applying Median Blur (ksize={MEDIAN_BLUR_KSIZE})...")
    img_blur = cv2.medianBlur(img, MEDIAN_BLUR_KSIZE)
    # if save_prefix:
    #     cv2.imwrite(f"{save_prefix}_1_median.png", img_blur)

    # --- Step 2: Adaptive Thresholding ---
    # print(f"  [DB] Applying Adaptive Threshold (Block={ADAPTIVE_BLOCK_SIZE}, C={ADAPTIVE_C})...")
    thresh = cv2.adaptiveThreshold(
        img_blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        ADAPTIVE_BLOCK_SIZE, 
        ADAPTIVE_C
    )
    # if save_prefix:
    #     cv2.imwrite(f"{save_prefix}_2_adaptive_thresh.png", thresh)

    # --- Step 2.5: Pre-Closing Filter ---
    if PRE_CLOSING_MIN_AREA > 0:
        # print(f"  [DB] Pre-filtering noise (Min Area={PRE_CLOSING_MIN_AREA})...")
        thresh_bool = morphology.remove_small_objects(thresh > 0, min_size=PRE_CLOSING_MIN_AREA)
        thresh = (thresh_bool.astype(np.uint8) * 255)

    # --- Step 3: Morphological Reconstruction (Closing) ---
    # print(f"  [DB] Morphological Closing (Radius={CLOSING_RADIUS})...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_RADIUS*2+1, CLOSING_RADIUS*2+1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # if save_prefix:
    #     cv2.imwrite(f"{save_prefix}_3_closed.png", closed)

    # --- Step 4: Fill Holes ---
    mask = closed > 0
    if FILL_HOLES:
        # print("  [DB] Filling holes...")
        mask = ndimage.binary_fill_holes(mask)

    # --- Step 5: Filter Small Objects ---
    # print(f"  [DB] Removing small objects (Min Area={MIN_OBJECT_AREA})...")
    mask = morphology.remove_small_objects(mask, min_size=MIN_OBJECT_AREA)
    
    # --- Step 6: Watershed Separation ---
    # print(f"  [DB] Applying Watershed separation (Min Dist={WATERSHED_MIN_DIST})...")
    labels = separate_particles_watershed(mask, min_distance=WATERSHED_MIN_DIST)
    
    # --- Step 7: Advanced Filtering (Circularity & Aspect Ratio) ---
    # print(f"  [DB] Filtering particles...")
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
        
        if aspect_ratio <= MAX_ASPECT_RATIO and circularity >= MIN_CIRCULARITY and MIN_OBJECT_AREA <= r.area <= MAX_OBJECT_AREA:
            # --- Hough Repair for Broken Particles (< 100k) ---
            if r.area < 100000:
                # 1. Extract local mask for this particle
                minr, minc, maxr, maxc = r.bbox
                local_mask = (labels[minr:maxr, minc:maxc] == r.label).astype(np.uint8) * 255
                
                # 2. Pad the mask (Crucial: center might be outside the bbox for partial arcs)
                pad = 100
                local_mask_padded = cv2.copyMakeBorder(local_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                
                # [Optimization] Blur the binary mask to improve gradient calculation for Hough
                local_mask_padded = cv2.GaussianBlur(local_mask_padded, (9, 9), 2)

                # 3. Run Hough Circle Transform
                # Radius ~178 for 100k area. Range [50, 300] covers variations.
                circles = cv2.HoughCircles(
                    local_mask_padded, 
                    cv2.HOUGH_GRADIENT, 
                    dp=1, 
                    minDist=max(local_mask.shape), # Expect only 1 circle
                    param1=50, 
                    param2=10, # Lower threshold to detect weak/partial circles (was 12)
                    minRadius=50, # Lowered from 100 to catch smaller fragments
                    maxRadius=300
                )
                
                if circles is not None:
                    # Found a circle -> Draw perfect circle on final_mask
                    circles = np.round(circles[0, :]).astype("int")
                    cx_pad, cy_pad, radius = circles[0]
                    
                    # print(f"    [Hough] Fixed particle (Area={r.area}): Found circle r={radius}")
                    
                    # Map coordinates back to global space
                    # local_padded (cx, cy) -> local (cx-pad, cy-pad) -> global (cx-pad+minc, cy-pad+minr)
                    global_cx = int(cx_pad - pad + minc)
                    global_cy = int(cy_pad - pad + minr)
                    
                    cv2.circle(final_mask, (global_cx, global_cy), int(radius), 255, -1)
                else:
                    # print(f"    [Hough] Failed to fix particle (Area={r.area}): No circle found.")
                    # No circle found -> Keep original shape
                    final_mask[r.coords[:, 0], r.coords[:, 1]] = 255
            else:
                # Large/Complete particle -> Keep original shape
                final_mask[r.coords[:, 0], r.coords[:, 1]] = 255
                
            kept_count += 1
            
    print(f"  [DB] Kept {kept_count} particles after filtering.")
    if save_prefix:
        cv2.imwrite(f"{save_prefix}_4_final_mask.png", final_mask)
    
    return final_mask

def process_pair(ch00_path, ch01_path):
    print(f"Processing pair:\n  CH00: {os.path.basename(ch00_path)}\n  CH01: {os.path.basename(ch01_path)}")
    
    directory = os.path.dirname(ch00_path)
    base_name = os.path.basename(ch00_path)
    name_no_ext = os.path.splitext(base_name)[0]
    
    # Create a cleaner prefix
    prefix = name_no_ext.replace("ch00", "")
    prefix = prefix.replace("__", "_").strip(" _")
    if not prefix:
        prefix = "output"
        
    # Define dynamic output paths
    csv_output = os.path.join(directory, f"{prefix}_NN_DB_global_results.xlsx")
    vis_01_output = os.path.join(directory, f"{prefix}_NN_DB_global_01visualization.png")
    vis_00_output = os.path.join(directory, f"{prefix}_NN_DB_global_00visualization.png")
    
    try:
        # 1. Extract Structure
        print("  1. Extracting structure (Adaptive Method)...")
        # --- DEBUGGING: Enable saving intermediate files ---
        # Create a prefix for debug images inside the loop to see each step's result
        # debug_prefix = os.path.join(directory, f"{prefix}_debug")
        mask1 = extract_structure_adaptive(ch00_path, save_prefix=None)

        # 2. Preprocess ch00 for Spot Detection (Masking & Flattening)
        print("  2. Preprocessing ch00 for spot detection...")
        img = cv2.imread(ch00_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [ERROR] Cannot read {ch00_path}")
            return

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
        if mask1.dtype == bool:
            mask_for_norm = mask1
        else:
            mask_for_norm = mask1 > 0

        masked_vals = masked_img_gray[mask_for_norm]
        if masked_vals.size == 0:
            print("  [Error] Mask is empty.")
            return

        v_min, v_max = np.percentile(masked_vals, (1, 99))

        if v_max > v_min:
            scaled = (masked_img_gray.astype(np.float32) - v_min) * (255.0 / (v_max - v_min))
            masked_img_gray = np.clip(scaled, 0, 255).astype(np.uint8)
        else:
            masked_img_gray = np.zeros_like(masked_img_gray, dtype=np.uint8)

        # Slight blur
        masked_img_flattened = cv2.GaussianBlur(masked_img_gray, (5, 5), 0)

        # --- Save inputs for debugging find_inner_holes_contours ---
        if SAVE_DEBUG_INPUTS:
            debug_gray_path = os.path.join(directory, f"{prefix}_debug_input_gray.png")
            debug_mask_path = os.path.join(directory, f"{prefix}_debug_input_mask.png")
            cv2.imwrite(debug_gray_path, masked_img_flattened)
            cv2.imwrite(debug_mask_path, mask1)
            print(f"  [DEBUG] Saved inputs for hole detection:\n    - {os.path.basename(debug_gray_path)}\n    - {os.path.basename(debug_mask_path)}")

        # 3. Find 4C spots on ch00
        print("  3. Finding 4C spots on ch00...")
        _, circles_data = find_inner_holes_contours(
            image_gray=masked_img_flattened,
            structure_mask=mask1,
            save_path=vis_00_output, 
            min_area=20,
            max_area=50000,
            min_circularity=0.6,
            block_size=13,
            c_value=1,
            erosion_size=25, 
            debug=False,
            detect_dark=True, 
            predict_4c=True,    # Enable 4C prediction
            return_data=True,   # Request data return
            use_median_blur=True, # [New] 开启中值滤波以去除噪点
            median_ksize=5,      # [New] 设置滤波核大小
            opening_ksize=9      # [New] 增大开运算核以去除背景噪点 (3 -> 5)
        )
        print(f"  -> Saved visualization 00: {vis_00_output}")

        # 4. Measure Intensity on ch01 (4 Circles)
        print("  4. Measuring Intensity on ch01...")
        raw_ch01 = cv2.imread(ch01_path, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            print(f"  [ERROR] Cannot read {ch01_path}")
            return
        
        compute_circles_intensity(
            brightness_image=raw_ch01,
            circles_data=circles_data,
            csv_path=csv_output,
            overlay_path=vis_01_output
        )
        print(f"  -> Done. Results: {csv_output}")
        print(f"  -> Saved visualization 01: {vis_01_output}")

    except Exception as e:
        print(f"  [ERROR] Failed processing {base_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    multiprocessing.freeze_support()
    print(f"=== Starting Batch NN_DB_Global Analysis in '{os.path.abspath(INPUT_FOLDER)}' ===")
    
    search_pattern = os.path.join(INPUT_FOLDER, "*ch00*.tif")
    ch00_files = glob.glob(search_pattern)
    
    if not ch00_files:
        print(f"No files found matching {search_pattern}")
        return

    print(f"Found {len(ch00_files)} candidate ch00 files.")

    count = 0
    for ch00 in ch00_files:
        directory, filename = os.path.split(ch00)
        filename_ch01 = filename.replace("ch00", "ch01")
        ch01 = os.path.join(directory, filename_ch01)
        
        if not os.path.exists(ch01):
            print(f"[WARN] Corresponding ch01 file not found for {filename}. Skipping.")
            continue
            
        process_pair(ch00, ch01)
        count += 1

    print(f"\n=== Batch Processing Complete! Processed {count} pairs. ===")

if __name__ == "__main__":
    main()
    sys.exit(0)