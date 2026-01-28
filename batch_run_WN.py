import cv2
import os
import glob
import numpy as np
from PIL import Image
from particle_analysis import (
    run_refined_particle_extraction,
    find_notches_and_axes,
    extract_inner_boundaries,
)
from handle_ch01 import adjust_ch01_image
from measure_intensity import compute_masked_quadrant_intensity

# Increase PIL image size limit to handle large scientific images
Image.MAX_IMAGE_PIXELS = None

# --- Configuration ---
# Set this to the folder containing your images.
INPUT_FOLDER = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Jinrui\CRP\Project003_CRP_DirectBinding_qCAP_4Biotin_0-10mg" # <-- 1. 修改这里为你的图片文件夹路径


# Refined extraction parameters (Structure)
CIRCLE_RADIUS_SCALE = 1
USE_WATERSHED = True
WATERSHED_MIN_DIST = 10
TILED_WATERSHED = True
MIN_CIRCULARITY = 0.75
KEEP_AREA = 3000
BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3
STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5

# Signal Extraction Parameters
SIGNAL_OPENING_RADIUS = 1

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
        
    csv_output = os.path.join(directory, f"{prefix}_WN_results.xlsx")
    vis_output = os.path.join(directory, f"{prefix}_WN_visualization.png")
    
    # Temporary files (using prefix to avoid collisions in batch run)
    temp_adjusted_ch01 = os.path.join(directory, f"temp_adjusted_{prefix}.png")
    
    try:
        # 1. Adjust ch01 (Signal Preparation)
        print(f"  1. Adjusting ch01...")
        adjust_ch01_image(
            ch01_path, 
            temp_adjusted_ch01,
            exposure=8.0,
            brightness=10,
            contrast_gain=1.0,
            stretch_low=2.0,
            stretch_high=98.5,
            do_stretch=True
        )

        # 2. Extract Signal Mask
        print(f"  2. Extracting signal mask...")
        # extract_inner_boundaries returns the mask array
        signal_mask = extract_inner_boundaries(
            temp_adjusted_ch01, 
            save_path=None, # Optimization: Skip saving visualization
            thin_opening_radius=SIGNAL_OPENING_RADIUS,
            min_size=80,
            keep_area=80,
            canny_sigma=2.0,
            clahe_clip_limit=0.01,
            brightness_threshold=0.2
        )

        # 3. Extract Structure (Simple Mode)
        print(f"  3. Extracting structure...")
        structure_mask = run_refined_particle_extraction(
            ch00_path, 
            save_prefix=f"temp_{prefix}", 
            circle_radius_scale=CIRCLE_RADIUS_SCALE,
            use_watershed=USE_WATERSHED,
            watershed_min_dist=WATERSHED_MIN_DIST,
            tiled_watershed=TILED_WATERSHED,
            min_circularity=MIN_CIRCULARITY,
            keep_area=KEEP_AREA,
            large_sigma=BANDPASS_LARGE_SIGMA,
            noise_sigma=BANDPASS_SMALL_SIGMA,
            stretch_low=STRETCH_LOW_PERCENTILE,
            stretch_high=STRETCH_HIGH_PERCENTILE,
            save_intermediates=False, # No intermediate images
            simple_mode=True,         # Simple mode (Canny/Watershed only)
            restrict_to_largest_circle=False
        )

        # 4. Find Notches and Axes
        print(f"  4. Finding notches and axes...")
        axes_info = find_notches_and_axes(
            structure_mask, 
            save_path=None # Optimization: Skip saving visualization
        )

        # 5. Measure and Visualize
        print(f"  5. Measuring intensities...")
        raw_ch01 = cv2.imread(ch01_path, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            print(f"  [ERROR] Cannot read {ch01_path}")
            return

        compute_masked_quadrant_intensity(
            brightness_image=raw_ch01,
            structure_mask=structure_mask,
            signal_mask=signal_mask,
            axes_info=axes_info,
            csv_path=csv_output,
            overlay_path=vis_output
        )
        print(f"  -> Done. Results: {csv_output}")

    except Exception as e:
        print(f"  [ERROR] Failed processing {base_name}: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_adjusted_ch01):
            try:
                os.remove(temp_adjusted_ch01)
            except Exception as e:
                print(f"  [WARN] Could not remove temp file: {e}")

def main():
    print(f"=== Starting Batch WN Simple Analysis in '{os.path.abspath(INPUT_FOLDER)}' ===")
    
    # 2. 修改这里，使用更具体的文件名模式来筛选文件
    search_pattern = os.path.join(INPUT_FOLDER, "*Project003_CRP_Fullwell_TileScan 3_CRP_Alexa555_directBinding_B_4Biotin_0-10mg*ch00*.tif")
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