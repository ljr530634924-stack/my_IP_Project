from particle_analysis import extract_outer_boundaries, find_notches_and_axes

# --- 1. 生成 mask ---
mask = extract_outer_boundaries(
     "Project001_Image 3_100ug_Ab_TRIS_ch00.tif",
    save_path="3ch00_boundary_only.png",
    thin_opening_radius=5
)

# --- 2. 用 mask 做坐标系绘制 ---o
find_notches_and_axes(
    binary_mask=mask,
    save_path="3MASK.png"
)
