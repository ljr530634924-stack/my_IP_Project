from overlay_mask import overlay_mask

# --- configure your inputs/outputs here ---
MASK_PATH = "global_structure_mask.png"  # mask image; white部分视为透明

# overlay 1: real base
BASE_REAL = "1_Merged_ch00_0ngmL.tif"
OUTPUT_REAL = "1_Merged_ch00_0ngmL_overlay_real.png"

# overlay 2: show base
BASE_SHOW = "1_Merged_ch00_0ngmL.tif"
OUTPUT_SHOW = "1_Merged_ch00_0ngmL_overlay_show.png"

# Optional settings
THRESH = 0.0       # 0 表示 mask > 0 视为白；None 表示用 50% max 自动阈值（需 overlay_mask 支持）
INVERT = False     # True 时反转逻辑：白保持，非白透明


def main() -> int:
    thresh_val = None if THRESH is None else THRESH

    # 1) mask + real base
    overlay_mask(
        mask_path=MASK_PATH,
        base_path=BASE_REAL,
        save_path=OUTPUT_REAL,
        thresh=thresh_val,
        invert=INVERT,
    )

    # 2) mask + show base
    overlay_mask(
        mask_path=MASK_PATH,
        base_path=BASE_SHOW,
        save_path=OUTPUT_SHOW,
        thresh=thresh_val,
        invert=INVERT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
