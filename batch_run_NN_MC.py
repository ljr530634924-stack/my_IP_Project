import cv2
import numpy as np
import os
import glob
from particle_analysis import run_refined_particle_extraction, find_inner_holes_contours
from measure_intensity import compute_circles_intensity

# --- Configuration ---
# Set this to the folder containing your images.
INPUT_FOLDER = r"F:\Jinrui\qCAP-CRP_11122025"

# Parameters (Same as run_NN_MC.py)
CIRCLE_RADIUS_SCALE = 1
USE_WATERSHED = True
WATERSHED_MIN_DIST = 20
MIN_CIRCULARITY = 0.75
KEEP_AREA = 3000

BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3

STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5

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
    csv_output = os.path.join(directory, f"{prefix}_NN_MC_results.xlsx")
    vis_01_output = os.path.join(directory, f"{prefix}_NN_MC_01visualization.png") # Circles on Ch01
    vis_00_output = os.path.join(directory, f"{prefix}_NN_MC_00visualization.png") # Holes/4C on Ch00
    
    try:
        # === 1. Extract Outer Boundaries (Structure from ch00) ===
        print("  1. Extracting structure...")
        mask1 = run_refined_particle_extraction(
            ch00_path, 
            save_prefix=f"temp_{prefix}_NN_MC", # Temp prefix, intermediates usually skipped or overwritten
            circle_radius_scale=CIRCLE_RADIUS_SCALE,
            use_watershed=USE_WATERSHED,
            watershed_min_dist=WATERSHED_MIN_DIST,
            min_circularity=MIN_CIRCULARITY,
            keep_area=KEEP_AREA,
            large_sigma=BANDPASS_LARGE_SIGMA,
            noise_sigma=BANDPASS_SMALL_SIGMA,
            stretch_low=STRETCH_LOW_PERCENTILE,
            stretch_high=STRETCH_HIGH_PERCENTILE,
            simple_mode=True, # Keep simple mode for speed as in run_NN_MC.py
            save_intermediates=False 
        )

        # === 2. Preprocess ch00 for Hole Detection ===
        print("  2. Preprocessing ch00...")
        # Replicating the logic from run_NN_MC.py
        img = cv2.imread(ch00_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [ERROR] Cannot read {ch00_path}")
            return

        if mask1.dtype == bool:
            mask1 = (mask1 * 255).astype(np.uint8)
        
        masked_img = cv2.bitwise_and(img, img, mask=mask1)
        
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
            print("  [Error] Mask is empty.")
            return

        v_min, v_max = np.percentile(masked_vals, (1, 99))

        if v_max > v_min:
            scaled = (masked_img_gray.astype(np.float32) - v_min) * (255.0 / (v_max - v_min))
            masked_img_gray = np.clip(scaled, 0, 255).astype(np.uint8)
        else:
            masked_img_gray = np.zeros_like(masked_img_gray, dtype=np.uint8)

        masked_img_flattened = cv2.GaussianBlur(masked_img_gray, (5, 5), 0)

        # === 3. Find Inner Holes & Predict 4 Circles (on ch00) ===
        print("  3. Finding holes & predicting 4C...")
        # We use return_data=True to get the coordinates
        # save_path corresponds to FINAL_RESULT_PATH in run_NN_MC.py
        _, circles_data = find_inner_holes_contours(
            image_gray=masked_img_flattened,
            structure_mask=mask1,
            save_path=vis_00_output, 
            min_area=20,
            max_area=5000,
            min_circularity=0.5,
            block_size=13,
            c_value=1,
            erosion_size=15, 
            debug=False,
            detect_dark=False, 
            predict_4c=True,    # Enable 4C prediction
            return_data=True    # Request data return
        )
        print(f"  -> Saved visualization 00: {vis_00_output}")

        # === 4. Measure Intensity on ch01 ===
        print("  4. Measuring intensity on ch01...")
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
    print(f"=== Starting Batch NN MC Analysis in '{os.path.abspath(INPUT_FOLDER)}' ===")
    
    # Search for ch00 files
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