import cv2
import numpy as np
import os
import glob
import sys
import multiprocessing
from scipy import ndimage
from skimage import morphology, measure
from skimage.segmentation import find_boundaries
from particle_analysis import separate_particles_watershed, find_notches_and_axes
from measure_intensity import compute_quadrant_intensity

# --- Configuration ---
# Set this to the folder containing your images.
INPUT_FOLDER = r"F:\Jinrui\qCAP_QuantaRed_750um\Biotin_4Conc\t45min60min" # 请修改为您的图片文件夹路径
SAVE_DEBUG_IMAGES = False  # Set to True to save intermediate debug images

# --- Parameters for Dirty Background (DB) Extraction (From run_NN_DB_global.py) ---
MEDIAN_BLUR_KSIZE = 5       # Kernel size (3, 5, 7). Larger = more smoothing.
ADAPTIVE_BLOCK_SIZE = 101    # Size of the pixel neighborhood. Must be odd.
ADAPTIVE_C = -1   # Negative C -> Stricter threshold (reduces fog). Try -2 or -1 if missing particles.
CLOSING_RADIUS = 3        # Radius for closing.
PRE_CLOSING_MIN_AREA = 100 # Remove small noise before closing
MIN_OBJECT_AREA = 130000    # Minimum area to keep a particle. Lowered from 100k to 10k.
MAX_OBJECT_AREA = 250000    # Maximum area to keep a particle.
MAX_OBJECT_AREA_FINAL = 245000 # Final check to remove merged particles
FILL_HOLES = True           # Fill internal holes.
MIN_CIRCULARITY = 0.5       # Minimum circularity. Lowered to catch irregular shapes.
MAX_ASPECT_RATIO = 1.4      # Maximum aspect ratio. Increased to allow slightly elongated particles.
WATERSHED_MIN_DIST = 15    # Watershed min distance

def extract_structure_adaptive(image_path, save_prefix=None):
    """
    Custom extraction logic for dirty/foggy backgrounds.
    Pipeline: Median Blur -> Adaptive Threshold -> Morphological Closing -> Fill Holes.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
        
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read image.")

    # --- Step 1: Median Blur ---
    img_blur = cv2.medianBlur(img, MEDIAN_BLUR_KSIZE)
    if save_prefix:
        cv2.imwrite(f"{save_prefix}_1_median.png", img_blur)

    # --- Step 2: Adaptive Thresholding ---
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
        thresh_bool = morphology.remove_small_objects(thresh > 0, min_size=PRE_CLOSING_MIN_AREA)
        thresh = (thresh_bool.astype(np.uint8) * 255)

    # --- Step 3: Morphological Reconstruction (Closing) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_RADIUS*2+1, CLOSING_RADIUS*2+1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    if save_prefix:
        cv2.imwrite(f"{save_prefix}_3_closed.png", closed)

    # --- Step 4: Fill Holes ---
    mask = closed > 0
    if FILL_HOLES:
        mask = ndimage.binary_fill_holes(mask)

    # --- Step 5: Filter Small Objects ---
    mask = morphology.remove_small_objects(mask, min_size=MIN_OBJECT_AREA)
    
    # --- Step 6: Watershed Separation ---
    labels = separate_particles_watershed(mask, min_distance=WATERSHED_MIN_DIST)
    
    # --- Step 7: Advanced Filtering (Circularity & Aspect Ratio) ---
    regions = measure.regionprops(labels)
    final_mask = np.zeros(labels.shape, dtype=np.uint8)
    kept_labels = np.zeros(labels.shape, dtype=np.int32)
    
    kept_count = 0
    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        h = maxr - minr
        w = maxc - minc
        aspect_ratio = max(h, w) / max(1, min(h, w))
        
        perimeter = max(r.perimeter, 1e-6)
        circularity = 4 * np.pi * r.area / (perimeter ** 2)
        
        if aspect_ratio <= MAX_ASPECT_RATIO and circularity >= MIN_CIRCULARITY and MIN_OBJECT_AREA <= r.area <= MAX_OBJECT_AREA:
            final_mask[r.coords[:, 0], r.coords[:, 1]] = 255
            kept_labels[r.coords[:, 0], r.coords[:, 1]] = r.label
            kept_count += 1

    # --- Boundary Burning: Separate touching particles (划黑线方案) ---
    if kept_count > 0:
        # 找出不同粒子之间的接触边界（排除粒子与背景的边界）
        b_all = find_boundaries(kept_labels, mode='inner', background=0)
        b_bin = find_boundaries(kept_labels > 0, mode='inner', background=0)
        touching_boundaries = b_all & (~b_bin)
        
        if np.any(touching_boundaries):
            # 稍微膨胀边界线以确保彻底切断 (使用 disk(1) 产生约 3px 宽的线)
            thick_boundaries = morphology.binary_dilation(touching_boundaries, morphology.disk(1))
            final_mask[thick_boundaries] = 0
            print(f"  [DB] Burned boundaries to separate touching particles.")

    # --- Final Check: Remove particles that merged and became too large ---
    if MAX_OBJECT_AREA_FINAL > 0:
        num_labels_final, labels_final = cv2.connectedComponents(final_mask, connectivity=8)
        regions_final = measure.regionprops(labels_final)
        removed_merged = 0
        for rf in regions_final:
            if rf.area > MAX_OBJECT_AREA_FINAL:
                final_mask[labels_final == rf.label] = 0
                removed_merged += 1
        if removed_merged > 0:
            print(f"  [DB] Removed {removed_merged} merged particles > {MAX_OBJECT_AREA_FINAL} px.")
            
    print(f"  [DB] Kept {kept_count} particles after filtering.")
    if save_prefix:
        cv2.imwrite(f"{save_prefix}_4_final_mask.png", final_mask)
    
    return final_mask

def process_pair(ch00_path, ch01_path):
    print(f"Processing pair:\n  CH00: {os.path.basename(ch00_path)}\n  CH01: {os.path.basename(ch01_path)}")
    
    directory = os.path.dirname(ch00_path)
    base_name = os.path.basename(ch00_path)
    name_no_ext = os.path.splitext(base_name)[0]
    
    prefix = name_no_ext.replace("ch00", "").replace("__", "_").strip(" _")
    if not prefix:
        prefix = "output"

    csv_output = os.path.join(directory, f"{prefix}_WN_DB_MC_results.xlsx")
    id_map_output = os.path.join(directory, f"{prefix}_WN_DB_MC_id_map.png")
    overlay_output = os.path.join(directory, f"{prefix}_WN_DB_MC_overlay_axes_ch00.png")

    try:
        # 1. Extract Structure using adaptive method
        print("  1. Extracting structure (Adaptive Method)...")
        if SAVE_DEBUG_IMAGES:
            debug_prefix = os.path.join(directory, f"{prefix}_debug")
        else:
            debug_prefix = None
        structure_mask = extract_structure_adaptive(ch00_path, save_prefix=debug_prefix)

        # 2. Find Notches and Axes
        print("  2. Finding notches and axes...")
        temp_axes_path = os.path.join(directory, f"temp_axes_{prefix}.png")
        axes_info = find_notches_and_axes(
            structure_mask,
            save_path=temp_axes_path,
        )

        # 2.1 Overlay axes on ch00 (Option 3: Black BG, Ch00 inside particles, Axes on top)
        if os.path.exists(temp_axes_path):
            print("  2.1 Generating overlay...")
            try:
                axes_img = cv2.imread(temp_axes_path)
                base_img = cv2.imread(ch00_path, cv2.IMREAD_UNCHANGED)

                if axes_img is not None and base_img is not None:
                    # Normalize base to 8-bit for visualization
                    if base_img.dtype == np.uint16:
                        p_lo, p_hi = np.percentile(base_img, (1, 99))
                        if p_hi > p_lo:
                            base_vis = (base_img.astype(np.float32) - p_lo) * (255.0 / (p_hi - p_lo))
                            base_vis = np.clip(base_vis, 0, 255).astype(np.uint8)
                        else:
                            base_vis = (base_img / 256).astype(np.uint8)
                    elif base_img.dtype == np.uint8:
                        base_vis = base_img.copy()
                    else:
                        base_vis = cv2.normalize(base_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    if len(base_vis.shape) == 2:
                        base_vis = cv2.cvtColor(base_vis, cv2.COLOR_GRAY2BGR)

                    if axes_img.shape[:2] != base_vis.shape[:2]:
                        axes_img = cv2.resize(axes_img, (base_vis.shape[1], base_vis.shape[0]))

                    # Detect white particle interior (255,255,255)
                    gray_axes = cv2.cvtColor(axes_img, cv2.COLOR_BGR2GRAY)
                    _, mask_white = cv2.threshold(gray_axes, 250, 255, cv2.THRESH_BINARY)

                    # Composite: Start with axes_img (Black BG + Axes), replace white with base
                    final_comp = axes_img.copy()
                    final_comp[mask_white == 255] = base_vis[mask_white == 255]

                    cv2.imwrite(overlay_output, final_comp)
                    print(f"  -> Saved overlay: {overlay_output}")

                os.remove(temp_axes_path)
            except Exception as e:
                print(f"  [WARN] Failed to generate overlay: {e}")
                if os.path.exists(temp_axes_path):
                    os.remove(temp_axes_path)

        # 3. Measure on raw ch01 using measurement circles
        print("  3. Measuring intensities...")
        raw_ch01 = cv2.imread(ch01_path, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            print(f"  [ERROR] Cannot read {ch01_path}")
            return

        num_labels, labels = cv2.connectedComponents(structure_mask)
        from skimage import measure
        regions = measure.regionprops(labels)

        circles_to_draw = compute_quadrant_intensity(
            brightness_image=raw_ch01,
            labels=labels,
            regions=regions,
            axes_info=axes_info,
            csv_path=csv_output,
            id_map_path=id_map_output,
        )

        print(f"  -> Done. Results: {csv_output}")

    except Exception as e:
        print(f"  [ERROR] Failed processing {base_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    multiprocessing.freeze_support()
    print(f"=== Starting Batch WN_DB_MC Analysis in '{os.path.abspath(INPUT_FOLDER)}' ===")
    
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