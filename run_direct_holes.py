import cv2
import numpy as np
import os
from particle_analysis import find_inner_holes_contours

# --- Configuration ---
# 输入：您提供的干净的、背景全黑的粒子图
INPUT_IMAGE_PATH = "1_Merged_ch00_0ngmL_overlay_realNew.png"
# 输出：标注了红点的结果图
OUTPUT_IMAGE_PATH = "1_Merged_ch00_0ngmL_holes_result.png"

def process_clean_image(image_path, save_path):
    """
    读取一张背景全黑的粒子图像，自动生成 Mask，并检测内部孔洞。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input file not found: {image_path}")

    print(f"Processing: {image_path}")

    # 1. 读取图像
    # 使用 IMREAD_UNCHANGED 读取，以防是 16-bit 或有 Alpha 通道
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Failed to read image.")

    # 确保转为灰度图 (如果是 RGB/BGR)
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # 2. 生成 Structure Mask
    # 因为背景是干净的黑色 (0)，我们直接通过阈值生成 Mask
    # 阈值设为 5 是为了容忍可能的 JPG 压缩噪点或边缘抗锯齿
    print("Generating mask from non-black pixels...")
    _, structure_mask = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)
    structure_mask = structure_mask.astype(np.uint8)

    # 3. 调用核心找洞函数
    print("Detecting inner holes per particle...")
    result_img = find_inner_holes_contours(
        image_gray=img_gray,
        structure_mask=structure_mask,
        save_path=save_path,
        min_area=20,       # 最小面积 (根据之前测试调整)
        max_area=5000,     # 最大面积
        min_circularity=0.5, # 圆度要求
        block_size=41,     # 自适应阈值邻域大小
        c_value=3,         # 自适应阈值对比度要求
        debug=False        # 如果需要查看中间过程，改为 True
    )
    
    return result_img

if __name__ == "__main__":
    try:
        process_clean_image(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)
        print(f"Done! Result saved to: {OUTPUT_IMAGE_PATH}")
    except Exception as e:
        print(f"Error: {e}")