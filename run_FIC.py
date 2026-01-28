import cv2
import numpy as np
from particle_analysis import find_inner_holes_contours


ch00_path = "1_Merged_ch00_0ngmL.tif"
structure_mask_path = "structure_refined_C_detailed_mask.png"  # 假设这是之前保存的结构掩码文件路径

# 1. 读取 ch00 (注意：如果是 16-bit，需要转 8-bit)
ch00_img = cv2.imread(ch00_path, cv2.IMREAD_UNCHANGED)

# 如果是 16-bit，转换为 8-bit
if ch00_img.dtype == np.uint16:
    ch00_img = (ch00_img / 256).astype(np.uint8)

# 读取结构掩码
structure_mask = cv2.imread(structure_mask_path, cv2.IMREAD_GRAYSCALE)
if structure_mask is None:
    raise FileNotFoundError(f"结构掩码文件未找到: {structure_mask_path}")

# 2. 调用函数
result_img = find_inner_holes_contours(
    image_gray=ch00_img,
    structure_mask=structure_mask,
    save_path="1_holes_visualization.png",
    min_area=20,       # 稍微调大，过滤极小噪点
    max_area=5000,     # 大幅调大！防止大孔洞被过滤 (500太小了)
    block_size=41,     # 调大！必须大于孔洞直径 (建议 31-61)
    c_value=3,         # 稍微降低对比度要求
    debug=True         # 开启调试，会生成 debug_particle_X_... 图片
)

print("孔洞检测完成，结果图片已返回并保存。")
