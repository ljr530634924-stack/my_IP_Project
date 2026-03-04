import cv2
import numpy as np
import os
import glob
import sys
import multiprocessing
from scipy import ndimage
from skimage import morphology, measure

# 假设这些模块在您的Python环境中可用
from handle_ch01 import adjust_ch01_image
from particle_analysis import separate_particles_watershed, find_MC
from measure_intensity import compute_circles_intensity

# --- Configuration ---
# 设置包含您图像的文件夹
INPUT_FOLDER = r"D:\Ingenieurpraixs\test_p2"
SAVE_DEBUG_IMAGES = True  # Set to True to save intermediate debug images

# --- Parameters for Structure Extraction (Copied from batch_run_NN_DB_4c.py) ---
MEDIAN_BLUR_KSIZE = 5
ADAPTIVE_BLOCK_SIZE = 101
ADAPTIVE_C = -1
CLOSING_RADIUS = 3
PRE_CLOSING_MIN_AREA = 100
MIN_OBJECT_AREA = 50000
MAX_OBJECT_AREA = 180000
MAX_OBJECT_AREA_FINAL = 140000 # [New] 最终Mask的面积上限，用于剔除粘连后超标的粒子
FILL_HOLES = True
MIN_CIRCULARITY = 0.5
MAX_ASPECT_RATIO = 1.4
WATERSHED_MIN_DIST = 15
HOUGH_FIT_CIRCLE = False # Set to True to enable Hough circle repair for broken particles

# --- Parameters for Signal Extraction ---
SIGNAL_MIN_AREA = 10 # [Modified] 降低最小面积限制 (50->10)，防止漏掉小信号点

# --- Structure extraction function (Copied from batch_run_NN_DB_4c.py) ---
def extract_structure_adaptive(image_path, save_prefix=None, hough_fit_circle=True):
    """
    Extraction logic for dirty backgrounds with Hough circle repair for broken particles.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read image.")

    img_blur = cv2.medianBlur(img, MEDIAN_BLUR_KSIZE)
    thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
    
    if PRE_CLOSING_MIN_AREA > 0:
        thresh_bool = morphology.remove_small_objects(thresh > 0, min_size=PRE_CLOSING_MIN_AREA)
        thresh = (thresh_bool.astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_RADIUS*2+1, CLOSING_RADIUS*2+1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    mask = closed > 0
    if FILL_HOLES:
        mask = ndimage.binary_fill_holes(mask)

    mask = morphology.remove_small_objects(mask, min_size=MIN_OBJECT_AREA)
    labels = separate_particles_watershed(mask, min_distance=WATERSHED_MIN_DIST)
    
    regions = measure.regionprops(labels)
    final_mask = np.zeros(labels.shape, dtype=np.uint8)
    
    kept_count = 0
    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        h, w = maxr - minr, maxc - minc
        aspect_ratio = max(h, w) / max(1, min(h, w))
        perimeter = max(r.perimeter, 1e-6)
        circularity = 4 * np.pi * r.area / (perimeter ** 2)
        
        if aspect_ratio <= MAX_ASPECT_RATIO and circularity >= MIN_CIRCULARITY and MIN_OBJECT_AREA <= r.area <= MAX_OBJECT_AREA:
            if hough_fit_circle and r.area < 100000: # Apply Hough repair only to smaller/broken particles
                minr, minc, maxr, maxc = r.bbox
                local_mask = (labels[minr:maxr, minc:maxc] == r.label).astype(np.uint8) * 255
                pad = 100
                local_mask_padded = cv2.copyMakeBorder(local_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                local_mask_padded = cv2.GaussianBlur(local_mask_padded, (9, 9), 2)
                circles = cv2.HoughCircles(local_mask_padded, cv2.HOUGH_GRADIENT, dp=1, minDist=max(local_mask.shape), param1=50, param2=10, minRadius=50, maxRadius=300)
                
                if circles is not None:
                    cx_pad, cy_pad, radius = np.round(circles[0, 0]).astype("int")
                    global_cx = int(cx_pad - pad + minc)
                    global_cy = int(cy_pad - pad + minr)
                    cv2.circle(final_mask, (global_cx, global_cy), int(radius), 255, -1)
                else:
                    final_mask[r.coords[:, 0], r.coords[:, 1]] = 255
            else:
                final_mask[r.coords[:, 0], r.coords[:, 1]] = 255
            kept_count += 1
            
    # [New] 最终质检：剔除因粘连而导致面积超标的粒子
    if MAX_OBJECT_AREA_FINAL > 0:
        # 对最终生成的 Mask 再次进行连通域分析
        num_labels_final, labels_final = cv2.connectedComponents(final_mask)
        regions_final = measure.regionprops(labels_final)
        
        removed_merged = 0
        for r in regions_final:
            if r.area > MAX_OBJECT_AREA_FINAL:
                final_mask[labels_final == r.label] = 0
                removed_merged += 1
        
        if removed_merged > 0:
            print(f"  [Structure] Removed {removed_merged} merged particles > {MAX_OBJECT_AREA_FINAL} px from final mask.")

    print(f"  [Structure] Kept {kept_count} particles after filtering.")
    if save_prefix and SAVE_DEBUG_IMAGES:
        cv2.imwrite(f"{save_prefix}_structure_mask.png", final_mask)
    
    return final_mask

# --- Main Processing Function for a Pair of Images ---
def process_pair(ch00_path, ch01_path):
    print(f"Processing pair:\n  CH00: {os.path.basename(ch00_path)}\n  CH01: {os.path.basename(ch01_path)}")
    
    directory = os.path.dirname(ch00_path)
    base_name = os.path.basename(ch00_path).replace("_ch00", "")
    name_no_ext = os.path.splitext(base_name)[0]
    
    prefix = name_no_ext.replace("__", "_").strip(" _")
    if not prefix:
        prefix = "output"
        
    # Define dynamic output paths
    csv_output = os.path.join(directory, f"{prefix}_NN_DB_SM_results.xlsx")
    vis_output = os.path.join(directory, f"{prefix}_NN_DB_SM_visualization.png")
    debug_prefix = os.path.join(directory, f"{prefix}_debug")

    try:
        # --- Step 1: Extract Particle Structure (from ch00) ---
        print("1. Extracting particle structure...")
        structure_mask = extract_structure_adaptive(ch00_path, save_prefix=debug_prefix, hough_fit_circle=HOUGH_FIT_CIRCLE)

        # --- Step 2: Adjust ch01 for Signal Detection ---
        print("2. Adjusting ch01 for robust signal detection...")
        temp_adj_ch01_path = os.path.join(directory, f"temp_adj_{prefix}.png")
        adjust_ch01_image(
            ch01_path, 
            temp_adj_ch01_path,
            exposure=1.5,
            brightness=10,
            contrast_gain=1.0,
            stretch_low=20.0,
            stretch_high=98.5,
            do_stretch=True
        )

        # --- Step 3: Extract Signal Mask (from adjusted ch01) ---
        print("3. Extracting signal mask...")
        # Replaced call to extract_inner_boundaries to use cv2.imread, avoiding Pillow's size limit.
        # This strategy is consistent with other scripts like batch_run_NN_DB_4c.py.
        
        # 1. Read the adjusted image using cv2
        adj_ch01_img = cv2.imread(temp_adj_ch01_path, cv2.IMREAD_GRAYSCALE)
        if adj_ch01_img is None:
            raise IOError(f"Failed to read adjusted ch01 image: {temp_adj_ch01_path}")

        # 2. Apply a simple binary threshold to find bright spots (signals)
        brightness_threshold_value = int(0.5 * 255) # [Modified] 降低亮度阈值 (0.2 -> 0.1) 以检测较暗信号
        _, signal_mask_thresholded = cv2.threshold(adj_ch01_img, brightness_threshold_value, 255, cv2.THRESH_BINARY)

        # 3. Filter out small noise artifacts
        signal_mask_bool = morphology.remove_small_objects(signal_mask_thresholded > 0, min_size=SIGNAL_MIN_AREA)
        signal_mask_raw = (signal_mask_bool.astype(np.uint8) * 255)
        if SAVE_DEBUG_IMAGES:
            cv2.imwrite(f"{debug_prefix}_signal_mask_raw.png", signal_mask_raw)

        # **User Request: Fill holes in the signal mask**
        print("   Filling holes in signal mask...")
        signal_mask_filled = (ndimage.binary_fill_holes(signal_mask_raw > 0) * 255).astype(np.uint8)
        if SAVE_DEBUG_IMAGES:
            cv2.imwrite(f"{debug_prefix}_signal_mask_filled.png", signal_mask_filled)

        # --- Step 4: Intersect Masks ---
        print("4. Keeping only signals inside particles...")
        # This ensures we only measure signals that are within a detected particle
        final_signal_mask = cv2.bitwise_and(signal_mask_filled, signal_mask_filled, mask=structure_mask)
        if SAVE_DEBUG_IMAGES:
            # [Modified] 保存信号掩膜，并更名为 signal_mask_in_structure
            cv2.imwrite(os.path.join(directory, f"{prefix}_signal_mask_in_structure.png"), final_signal_mask)

        # --- Step 5: 4C Prediction & Measurement ---
        print("5. Predicting 4C spots and measuring on Ch01...")
        
        # [Modified] 将可视化图的名字改回 00visualization
        vis_00_output = os.path.join(directory, f"{prefix}_NN_DB_SM_00visualization.png")
        
        # [New] 定义 debug_v 图片路径
        debug_vis_output = os.path.join(directory, f"{prefix}_debug_v.png") if SAVE_DEBUG_IMAGES else None
        
        # [New] 读取 Ch00 原图用于可视化背景
        ch00_img = cv2.imread(ch00_path, cv2.IMREAD_UNCHANGED)
        
        # [Modified] 使用新的 find_MC 函数替代 find_inner_holes_contours
        _, circles_data, particle_areas = find_MC(
            structure_mask=structure_mask,
            signal_mask=final_signal_mask,
            save_path=vis_00_output,
            min_area=1000,
            max_area=50000, 
            min_circularity=0.1, # 保持较低的圆度要求，确保能找到基准点
            return_data=True,
            ch00_image=ch00_img, # [New] 传入原图
            debug_save_path=debug_vis_output # [Modified] 传入调试图片保存路径
        )

        # Measure on raw Ch01
        raw_ch01 = cv2.imread(ch01_path, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            raise IOError(f"Cannot read raw ch01 image: {ch01_path}")

        compute_circles_intensity(
            brightness_image=raw_ch01,
            circles_data=circles_data,
            particle_areas=particle_areas,
            csv_path=csv_output,
            overlay_path=vis_output
        )

        # Clean up temporary adjusted ch01 file
        if os.path.exists(temp_adj_ch01_path):
            os.remove(temp_adj_ch01_path)

    except Exception as e:
        print(f"  [ERROR] Failed processing {prefix}: {e}")
        import traceback
        traceback.print_exc()


# --- Main Entry Point ---
def main():
    """
    Main function to find and process all ch00/ch01 image pairs in the INPUT_FOLDER.
    """
    multiprocessing.freeze_support()
    print(f"=== Starting Batch NN_DB_SM Analysis in '{os.path.abspath(INPUT_FOLDER)}' ===")
    
    search_pattern = os.path.join(INPUT_FOLDER, "*ch00*.tif")
    ch00_files = glob.glob(search_pattern)
    
    if not ch00_files:
        print(f"No files found matching {search_pattern}")
        return

    print(f"Found {len(ch00_files)} candidate ch00 files.")

    processed_count = 0
    for ch00_path in ch00_files:
        ch01_path = ch00_path.replace("ch00", "ch01")
        
        if not os.path.exists(ch01_path):
            print(f"[WARN] Corresponding ch01 file not found for {os.path.basename(ch00_path)}. Skipping.")
            continue
            
        process_pair(ch00_path, ch01_path)
        processed_count += 1

    print(f"\n=== Batch Processing Complete! Processed {processed_count} pairs. ===")

if __name__ == "__main__":
    main()
    sys.exit(0)