import cv2
from skimage import measure

from particle_analysis import run_refined_particle_extraction, find_notches_and_axes
from measure_intensity import compute_quadrant_intensity


# The original brightness image corresponding to the ch00 file. It must be a .tif file.
BRIGHTNESS_PATH = "Project001_Image 3_100ug_Ab_TRIS_ch01.tif"
# ch00 source used to derive mask/axes/labels/regions
CH00_PATH = "Project001_Image 3_100ug_Ab_TRIS_ch00.tif"


def main():
    # 1) derive mask from ch00 using refined extraction (A/B/C/D method)
    mask_img = run_refined_particle_extraction(
        CH00_PATH,
        save_prefix="measure_refined",
    )

    # 2) labels / regions
    num_labels, labels = cv2.connectedComponents(mask_img)
    regions = measure.regionprops(labels)

    # 3) axes per particle
    axes_info = find_notches_and_axes(mask_img, save_path="axes_output.png")

    # 4) read the clean brightness image for measurement
    # Use IMREAD_UNCHANGED to preserve the original bit-depth (e.g., 16-bit for TIFF files).
    # This is critical for accurate intensity measurements.
    brightness_image = cv2.imread(BRIGHTNESS_PATH, cv2.IMREAD_UNCHANGED)
    if brightness_image is None:
        raise FileNotFoundError(f"Cannot read brightness image: {BRIGHTNESS_PATH}")

    brightness_image = brightness_image.astype(float)

    # 5) quadrant sampling
    compute_quadrant_intensity(
        brightness_image=brightness_image,
        labels=labels,
        regions=regions,
        axes_info=axes_info,
        csv_path="quadrant_results5.csv",
        id_map_path="particle_id_map_real.png",
    )


if __name__ == "__main__":
    main()
