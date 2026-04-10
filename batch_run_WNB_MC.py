import cv2
import os
import glob
import numpy as np
from PIL import Image
from skimage import measure

# Import existing functions
from particle_analysis import run_refined_particle_extraction
from measure_intensity import compute_quadrant_intensity

# Increase PIL image size limit to handle large scientific images
Image.MAX_IMAGE_PIXELS = None

# --- Configuration ---
# Refined extraction parameters (Structure)
CIRCLE_RADIUS_SCALE = 1
USE_WATERSHED = True
WATERSHED_MIN_DIST = 10
TILED_WATERSHED = True
MIN_CIRCULARITY = 0.75
KEEP_AREA = 3000
MIN_OBJECT_AREA = 9000
MAX_OBJECT_AREA = 11000
BANDPASS_LARGE_SIGMA = 40
BANDPASS_SMALL_SIGMA = 3
STRETCH_LOW_PERCENTILE = 0.5
STRETCH_HIGH_PERCENTILE = 99.5


def find_notches_and_axes_bump(binary_mask, save_path=None):
    """
    Finds the big notch (Y-axis) and the small bump (X-axis).
    
    Logic:
    1. Y-axis: Defined by the point closest to the centroid (Big Notch).
    2. X-axis: Defined by analyzing the boundary variance (Standard Deviation) 
       in the regions orthogonal (75-105 degrees) to the Y-axis.
       The side with higher variance (due to the bump shape) is the positive X direction.
    """
    axes_info = {}
    num_labels, labels = cv2.connectedComponents(binary_mask)

    h, w = binary_mask.shape
    # Visualization canvas
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[binary_mask == 255] = (255, 255, 255)

    # Sort regions by x-coordinate for consistent ID assignment
    regions = measure.regionprops(labels)
    regions = sorted(regions, key=lambda r: r.centroid[1])
    region_id_map = {r.label: idx + 1 for idx, r in enumerate(regions)}

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue
        contour = contours[0][:, 0, :]  # Nx2 array of (x, y) points

        # Calculate Centroid
        M = cv2.moments(component)
        if M["m00"] == 0: continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid = np.array([cx, cy])

        # Calculate distances from centroid to all boundary points
        pts = contour.astype(float)
        vecs = pts - centroid
        d = np.linalg.norm(vecs, axis=1)

        # --- 1. Find Big Notch (Y-axis positive) ---
        # Strategy: Global minimum distance
        big_idx = np.argmin(d)
        big_notch = pts[big_idx]

        # Define Y-axis vector v_y (Centroid -> Big Notch)
        v_y = big_notch - centroid
        norm_v_y = np.linalg.norm(v_y)
        if norm_v_y == 0: continue
        v_y = v_y / norm_v_y

        # Draw Y-axis (Red)
        diameter = np.max(d) * 2.2
        p_center = (int(cx), int(cy))
        p_y_end = (int(cx + v_y[0] * diameter / 2), int(cy + v_y[1] * diameter / 2))
        cv2.line(canvas, p_center, p_y_end, (0, 0, 255), 2) # Red line for Y
        cv2.circle(canvas, (int(big_notch[0]), int(big_notch[1])), 4, (0, 0, 255), -1) # Red dot

        # --- 2. Find Small Bump (X-axis positive) ---
        # Strategy: Compare Standard Deviation of distances in Left vs Right sectors
        
        # Define orthogonal vectors
        # v_ortho_1 = Rotate v_y by 90 degrees
        v_ortho_1 = np.array([v_y[1], -v_y[0]]) 
        # v_ortho_2 = Rotate v_y by -90 degrees
        v_ortho_2 = -v_ortho_1                  

        # Normalize all point vectors for angle calculation
        d_safe = d.copy()
        d_safe[d_safe == 0] = 1.0
        vecs_norm = vecs / d_safe[:, np.newaxis]

        # Dot product with v_y to find points in the orthogonal band
        # We want points where angle with Y is between 75 and 105 degrees.
        # cos(75) approx 0.2588. So we want abs(dot_product) < 0.2588
        dot_products = np.sum(vecs_norm * v_y, axis=1)
        angle_threshold = np.cos(np.deg2rad(75)) # ~0.2588
        sector_mask = np.abs(dot_products) < angle_threshold

        # Split these sector points into Side 1 (v_ortho_1) and Side 2 (v_ortho_2)
        # Use dot product with v_ortho_1 to determine sign
        dot_ortho = np.sum(vecs_norm * v_ortho_1, axis=1)
        
        mask_side_1 = sector_mask & (dot_ortho > 0)
        mask_side_2 = sector_mask & (dot_ortho < 0)

        # Get distances for both sides
        d_side_1 = d[mask_side_1]
        d_side_2 = d[mask_side_2]

        # Calculate Standard Deviation (Measure of "waviness" or "bumpiness")
        std_1 = np.std(d_side_1) if len(d_side_1) > 2 else 0
        std_2 = np.std(d_side_2) if len(d_side_2) > 2 else 0

        # Determine X-axis direction (Side with higher SD is the Bump)
        if std_1 > std_2:
            v_x = v_ortho_1
            bump_side_pts = pts[mask_side_1]
        else:
            v_x = v_ortho_2
            bump_side_pts = pts[mask_side_2]

        # Find a representative point for the bump to draw the blue dot
        # We use the point with max distance in that sector
        if len(bump_side_pts) > 0:
            d_bump = np.linalg.norm(bump_side_pts - centroid, axis=1)
            bump_idx = np.argmax(d_bump)
            bump_pt = bump_side_pts[bump_idx]
        else:
            bump_pt = centroid + v_x * (np.max(d) if len(d)>0 else 10)

        # Draw X-axis (Blue)
        p_x_end = (int(cx + v_x[0] * diameter / 2), int(cy + v_x[1] * diameter / 2))
        cv2.line(canvas, p_center, p_x_end, (255, 0, 0), 2) # Blue line for X
        cv2.circle(canvas, (int(bump_pt[0]), int(bump_pt[1])), 4, (255, 0, 0), -1) # Blue dot

        # Store info
        axes_info[label] = {
            "ex": v_x,
            "ey": v_y,
            "centroid": centroid,
            "radius": np.max(d)
        }

        # Draw ID
        pid = region_id_map.get(label, label)
        cv2.putText(canvas, str(pid), (int(cx)+10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save visualization
    if save_path:
        cv2.imwrite(save_path, canvas)

    return axes_info, canvas


def process_pair(ch00_path, ch01_path):
    print(f"Processing pair:\n  CH00: {os.path.basename(ch00_path)}\n  CH01: {os.path.basename(ch01_path)}")

    directory = os.path.dirname(ch00_path)
    base_name = os.path.basename(ch00_path)
    name_no_ext = os.path.splitext(base_name)[0]

    prefix = name_no_ext.replace("ch00", "").replace("__", "_").strip(" _")
    if not prefix:
        prefix = "output"

    csv_output = os.path.join(directory, f"{prefix}_WNB_MC_results.xlsx")
    id_map_output = os.path.join(directory, f"{prefix}_WNB_MC_map.png")

    try:
        # 1. Extract Structure
        print("  1. Extracting structure...")
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
            save_intermediates=False,
            simple_mode=False,
            restrict_to_largest_circle=False,
        )

        # Filter objects by area
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(structure_mask, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > MAX_OBJECT_AREA or area < MIN_OBJECT_AREA:
                structure_mask[labels == i] = 0

        # 2. Find Notches (Y) and Bumps (X)
        print("  2. Finding notches (Y) and bumps (X)...")
        
        # Use the NEW function defined in this file
        axes_info, axes_img = find_notches_and_axes_bump(
            structure_mask,
            save_path=None,
        )

        # 2.1 Overlay axes on ch00 (Option 3: Black BG, Ch00 inside particles, Axes on top)
        print("  2.1 Generating overlay...")
        overlay_output = os.path.join(directory, f"{prefix}_visualization.png")
        try:
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

        except Exception as e:
            print(f"  [WARN] Failed to generate overlay: {e}")

        # 3. Measure
        print("  3. Measuring intensities...")
        raw_ch01 = cv2.imread(ch01_path, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            print(f"  [ERROR] Cannot read {ch01_path}")
            return

        # Re-label for measurement function
        num_labels, labels = cv2.connectedComponents(structure_mask)
        regions = measure.regionprops(labels)

        compute_quadrant_intensity(
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

def run_batch_wnb_mc(input_folder):
    print(f"=== Starting Batch WNB_MC Analysis in '{os.path.abspath(input_folder)}' ===")
    search_pattern = os.path.join(input_folder, "*ch00*.tif")
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
            print(f"[WARN] Missing ch01 for {filename}. Skipping.")
            continue
            
        process_pair(ch00, ch01)
        count += 1
    print(f"\n=== Batch Processing Complete! Processed {count} pairs. ===")

if __name__ == "__main__":
    # Default test path
    DEFAULT_INPUT_FOLDER = r"D:\Ingenieurpraixs\test_18032026_new"
    if os.path.isdir(DEFAULT_INPUT_FOLDER):
        run_batch_wnb_mc(DEFAULT_INPUT_FOLDER)
    else:
        print("Please set a valid input folder.")