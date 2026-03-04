import cv2
import os
import glob
import numpy as np
from particle_analysis import (
    run_refined_particle_extraction,
)
from measure_intensity import compute_global_signal_intensity

# --- Configuration ---
# Set this to the folder containing your images. "." means current directory.
INPUT_FOLDER = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Jinrui\Direct Binding Ab488\SVA\qCAP_SVA_30mgmL_Ab488_19122025"
# Refined extraction parameters
CIRCLE_RADIUS_SCALE = 1
USE_WATERSHED = True
WATERSHED_MIN_DIST = 10   # 修正：从 20 改为 10，防止粒子合并后因长宽比超标被过滤
TILED_WATERSHED = False   # 修正：改为 False 以使用全局分水岭，效果更稳健
MIN_CIRCULARITY = 0.75
KEEP_AREA = 3000
# Bandpass Filter Parameters
BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3

# Saturation Stretch Parameters
STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5

def process_pair(ch00_path, ch01_path):
    print(f"Processing pair:\n  CH00: {os.path.basename(ch00_path)}\n  CH01: {os.path.basename(ch01_path)}")
    
    # Generate output filenames based on input name
    # Example: "1_Merged_ch00_0ngmL.tif" -> "1_Merged_0ngmL_global_results.xlsx"
    directory = os.path.dirname(ch00_path)
    base_name = os.path.basename(ch00_path)
    name_no_ext = os.path.splitext(base_name)[0]
    
    # Create a cleaner prefix by removing 'ch00'
    prefix = name_no_ext.replace("ch00", "")
    # Clean up double underscores or trailing/leading separators
    prefix = prefix.replace("__", "_").strip(" _")
    if not prefix: # Fallback if filename was just "ch00"
        prefix = "output"
        
    csv_output = os.path.join(directory, f"{prefix}_global_results.xlsx")
    vis_output = os.path.join(directory, f"{prefix}_global_visualization.png")
    
    # 1. Extract structure
    # save_intermediates=False prevents generating A, B, C, D images
    structure_mask = run_refined_particle_extraction(
        ch00_path, 
        save_prefix=f"temp_{prefix}", # Prefix won't be used for files if save_intermediates=False
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
        save_intermediates=False,
        simple_mode=True,  # 修正：关闭简易模式，启用 A/B/C/D 完整优化流程以保证质量
        restrict_to_largest_circle=False # 关闭 ROI 优化：防止误检导致图像被错误裁剪
    )

    # 2. Measure Global Intensity
    raw_ch01 = cv2.imread(ch01_path, cv2.IMREAD_UNCHANGED)
    if raw_ch01 is None:
        print(f"  [ERROR] Cannot read {ch01_path}")
        return

    compute_global_signal_intensity(
        brightness_image=raw_ch01,
        structure_mask=structure_mask,
        signal_mask=structure_mask, 
        csv_path=csv_output,
        overlay_path=vis_output
    )
    print(f"  -> Done. Results: {csv_output}")

def main():
    print(f"=== Starting Batch Global Analysis in '{os.path.abspath(INPUT_FOLDER)}' ===")
    
    # Find all ch00 files (case insensitive usually depends on OS, assuming lowercase here)
    search_pattern = os.path.join(INPUT_FOLDER, "*ch00*.tif")
    ch00_files = glob.glob(search_pattern)
    
    if not ch00_files:
        print(f"No files found matching {search_pattern}")
        return

    print(f"Found {len(ch00_files)} candidate ch00 files.")

    count = 0
    for ch00 in ch00_files:
        directory, filename = os.path.split(ch00)
        
        # Simple replacement of 'ch00' with 'ch01' to find the pair
        filename_ch01 = filename.replace("ch00", "ch01")
        ch01 = os.path.join(directory, filename_ch01)
        
        if not os.path.exists(ch01):
            print(f"[WARN] Corresponding ch01 file not found for {filename}. Skipping.")
            continue
            
        try:
            process_pair(ch00, ch01)
            count += 1
        except Exception as e:
            print(f"[ERROR] Failed processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Batch Processing Complete! Processed {count} pairs. ===")

if __name__ == "__main__":
    main()