from particle_analysis import run_refined_particle_extraction, find_notches_and_axes

# --- Configuration ---
# Use the same configuration as run_final_pipeline.py for consistency
CH00_PATH = "Project001_qCAPs_ACPEGN3_4conc_0,0.1,1,4mgmL_A1 (2)_Merged_ch00.tif"
CIRCLE_RADIUS_SCALE = 1

BANDPASS_LARGE_SIGMA = 40  # 背景平滑半径 (对应 Fiji "Filter large structures down to")
BANDPASS_SMALL_SIGMA = 3   # 噪点平滑半径 (对应 Fiji "Filter small structures up to")

STRETCH_LOW_PERCENTILE = 0.5  # Saturate bottom 0.5% of pixels to black
STRETCH_HIGH_PERCENTILE = 99.5 # Saturate top 0.5% of pixels to white

def main():
    print("=== Testing Step 3: Structure & Axes Extraction ===")
    # --- 1. Run the new refined extraction process ---
    # This will generate A, B, C, and D images with the specified prefix.
    final_mask = run_refined_particle_extraction(
        ch00_path=CH00_PATH,
        save_prefix="structure_refined",
        circle_radius_scale=CIRCLE_RADIUS_SCALE
    )

    # --- 2. (Optional) Use the final mask for subsequent analysis ---
    # For example, find notches and axes on the refined mask
    find_notches_and_axes(
        binary_mask=final_mask,
        save_path="structure_axes.png"
    )

    print(f"\nStep 3 test complete. Axes image saved to structure_axes.png")


if __name__ == "__main__":
    main()