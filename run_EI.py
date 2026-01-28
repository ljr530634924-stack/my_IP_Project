import os
from particle_analysis import extract_inner_boundaries

# --- Configuration ---
# Input image (signal channel or whichever you want to extract inner boundaries from)
INPUT_PATH = "1_Merged_ch00_0ngmL.tif"

# Output files
MASK_OUTPUT = "1_Merged_ch00_0ngmL_inner_boundaries.png"

# Inner extraction parameters (more sensitive by default)
CANNY_SIGMA = 1.2
CLAHE_CLIP_LIMIT = 3.0
SMALL_HOLE_AREA = 1000
MIN_SIZE = 80
KEEP_AREA = 80
THIN_OPENING_RADIUS = 1
MIN_CIRCULARITY = None


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    print("=== Running Extract Inner Boundaries (EI) ===")
    mask = extract_inner_boundaries(
        INPUT_PATH,
        save_path=MASK_OUTPUT,
        thin_opening_radius=THIN_OPENING_RADIUS,
        min_size=MIN_SIZE,
        keep_area=KEEP_AREA,
        canny_sigma=CANNY_SIGMA,
        clahe_clip_limit=CLAHE_CLIP_LIMIT,
        min_circularity=MIN_CIRCULARITY,
        small_hole_area=SMALL_HOLE_AREA,
    )

    print("Done.")
    print(f"Mask saved to: {MASK_OUTPUT}")
    print(f"Mask dtype: {mask.dtype}, shape: {mask.shape}")


if __name__ == "__main__":
    main()
