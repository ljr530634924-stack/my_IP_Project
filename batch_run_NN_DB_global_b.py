import cv2
import numpy as np
import os
import glob
import sys
import multiprocessing
from scipy import ndimage
from skimage import morphology, measure
from particle_analysis import separate_particles_watershed
from measure_intensity import compute_global_signal_intensity
from overlay_mask import overlay_mask

# --- Configuration ---
# Set this to the folder containing your images.
INPUT_FOLDER = r"D:\Ingenieurpraixs\test3_05122025"

# --- Parameters for Dirty Background (DB) Extraction (From run_NN_DB_global.py) ---
MEDIAN_BLUR_KSIZE = 5       # Kernel size (3, 5, 7). Larger = more smoothing.
ADAPTIVE_BLOCK_SIZE = 101   # Should be larger than the particle diameter.
ADAPTIVE_C = -1        # Negative C is stricter (less noise). Positive C is lenient. Start with a negative value for cleaner results.
CLOSING_RADIUS = 3          # Radius for closing.
PRE_CLOSING_MIN_AREA = 100  # Lower this drastically to avoid deleting fragments before closing.
MIN_OBJECT_AREA = 50000      # Lower this drastically for debugging. 50k is too large for most cases.
FILL_HOLES = True           # Fill internal holes.
MIN_CIRCULARITY = 0.5       # Minimum circularity
MAX_ASPECT_RATIO = 1.4      # Maximum aspect ratio
WATERSHED_MIN_DIST = 7   # A value of 2 is very small and might over-segment. Try 10-20.

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
        
        if aspect_ratio <= MAX_ASPECT_RATIO and circularity >= MIN_CIRCULARITY and r.area >= MIN_OBJECT_AREA:
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
        debug_prefix = os.path.join(directory, f"{prefix}_debug")
        mask1 = extract_structure_adaptive(ch00_path, save_prefix=debug_prefix)

        # 1.5 Visualize Structure on CH00 (Overlay + IDs)
        print("  1.5 Visualizing structure on ch00...")
        temp_mask_path = os.path.join(directory, f"temp_vis_mask_{prefix}.png")
        cv2.imwrite(temp_mask_path, mask1)
        
        try:
            # Create overlay
            overlay_mask(
                mask_path=temp_mask_path,
                base_path=ch00_path,
                save_path=vis_00_output,
                thresh=0.0,
                invert=False
            )
            
            # Add sorted IDs
            vis_00 = cv2.imread(vis_00_output)
            if vis_00 is not None:
                num_labels, labels = cv2.connectedComponents(mask1)
                regions = measure.regionprops(labels)
                # Sort by x-coordinate (col)
                regions = sorted(regions, key=lambda r: r.centroid[1])
                
                for idx, r in enumerate(regions):
                    pid = idx + 1
                    cy, cx = r.centroid
                    # Draw ID in Green
                    cv2.putText(vis_00, str(pid), (int(cx), int(cy)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
                
                cv2.imwrite(vis_00_output, vis_00)
                print(f"  -> Saved visualization 00: {vis_00_output}")
        finally:
            if os.path.exists(temp_mask_path):
                os.remove(temp_mask_path)

        # 2. Measure Global Intensity on ch01
        print("  2. Measuring Global Intensity on ch01...")
        raw_ch01 = cv2.imread(ch01_path, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            print(f"  [ERROR] Cannot read {ch01_path}")
            return
        
        compute_global_signal_intensity(
            brightness_image=raw_ch01,
            structure_mask=mask1,
            signal_mask=mask1, 
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