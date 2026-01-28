import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, exposure, feature, measure
import cv2
from scipy.signal import argrelextrema


def extract_outer_boundaries(ch00_path,
                             save_path="closed_particles.png",
                             min_circularity=None,
                             thin_opening_radius=2):

    # --- 1. 读取原图 ---
    img = io.imread(ch00_path)

    # --- 2. 归一化到 0-1 ---
    img = img.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min())

    # --- 3. CLAHE增强 ---
    img_eq = exposure.equalize_adapthist(img, clip_limit=0.01)

    # --- 4. Canny边缘 ---
    edges = feature.canny(img_eq, sigma=2)

    # --- 5. 闭操作，使边界闭合 ---
    edges_closed = morphology.binary_closing(edges, morphology.disk(5))

    # --- 6. 填充洞 ---
    filled = morphology.remove_small_holes(edges_closed, area_threshold=5000)

    # --- 7. 去除极小噪点 ---
    prelim = morphology.remove_small_objects(filled, min_size=2000)

    # 可选：用小结构元素做开运算，再轻微膨胀回去，剃掉头发丝等细长附加物
    if thin_opening_radius and thin_opening_radius > 0:
        selem = morphology.disk(thin_opening_radius)
        prelim = morphology.binary_opening(prelim, selem)
        prelim = morphology.binary_dilation(prelim, selem)

    # --- 8. 连通域分析 ---
    lbl = measure.label(prelim)
    regions = measure.regionprops(lbl)

    # --- 9. 过滤细长区域 ---
    filtered_mask = np.zeros_like(prelim, dtype=bool)
    kept_regions = []

    for r in regions:
        area = r.area
        minr, minc, maxr, maxc = r.bbox
        h = maxr - minr
        w = maxc - minc
        aspect_ratio = max(h, w) / max(1, min(h, w))
        # 如果提供 min_circularity，则用圆度过滤；否则仅用面积与长宽比
        keep = area > 3000 and aspect_ratio < 1.8
        if min_circularity is not None:
            perimeter = max(r.perimeter, 1e-6)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            keep = keep and (circularity >= min_circularity)

        if keep:
            filtered_mask[r.coords[:, 0], r.coords[:, 1]] = True
            kept_regions.append(r)

    print(f"Final kept particles: {len(kept_regions)}")

    # --- 10. 画黑底白粒子 + 黄边界（保持原图尺寸） ---
    h, w = img.shape

    # 黑底白粒子 mask
    mask_img = np.zeros((h, w), dtype=np.uint8)
    for r in kept_regions:
        mask_img[r.coords[:, 0], r.coords[:, 1]] = 255

    # 创建尺寸与原图一致的 figure
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])  # 无边框

    ax.imshow(mask_img, cmap="gray")

    # 画轮廓
    for r in kept_regions:
        contour = measure.find_contours(r.filled_image, 0.5)[0]
        minr, minc, _, _ = r.bbox
        contour[:, 0] += minr
        contour[:, 1] += minc
        

    ax.axis("off")

    # 保存为原图尺寸
    fig.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    return mask_img


# === 新增：从黑白图找到豁口 + 画坐标系 ===
def find_notches_and_axes(binary_mask, save_path="axes_output1.png"):
    """
    binary_mask: 黑白 mask 图（255=粒子）
    """
    axes_info={}
    # 1. 连通域
    num_labels, labels = cv2.connectedComponents(binary_mask)

    h, w = binary_mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[binary_mask == 255] = (255, 255, 255)

    # 按质心 x 坐标排序，生成稳定编号
    regions = measure.regionprops(labels)
    regions = sorted(regions, key=lambda r: r.centroid[1])  # centroid=(row, col); col=x
    region_id_map = {r.label: idx + 1 for idx, r in enumerate(regions)}

    # 遍历每个粒子
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue
        contour = contours[0][:, 0, :]  # Nx2

        # 计算质心
        M = cv2.moments(component)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid = np.array([cx, cy])

        # 到质心距离
        pts = contour
        d = np.linalg.norm(pts - centroid, axis=1)

        # 大缺口
        big_idx = np.argmin(d)
        big_notch = pts[big_idx]

        # 小缺口
        minima_idx = argrelextrema(d, np.less)[0]
        minima_idx = [i for i in minima_idx if i != big_idx]

        if len(minima_idx) == 0:
            continue

        small_idx = minima_idx[np.argmin(d[minima_idx])]
        small_notch = pts[small_idx]

        # 坐标系方向
        v = big_notch - centroid
        v = v / np.linalg.norm(v)
        perp = np.array([v[1], -v[0]])
        diameter = np.max(d) * 2

        axes_info[label] = {
        "ex": perp,
        "ey": v,
        "centroid": centroid}
         
 

        # Y轴
        p1 = (int(centroid[0] - v[0] * diameter / 2), int(centroid[1] - v[1] * diameter / 2))
        p2 = (int(centroid[0] + v[0] * diameter / 2), int(centroid[1] + v[1] * diameter / 2))
        cv2.line(canvas, p1, p2, (255, 0, 0), 2)

        # X轴
        p3 = (int(centroid[0] - perp[0] * diameter / 2), int(centroid[1] - perp[1] * diameter / 2))
        p4 = (int(centroid[0] + perp[0] * diameter / 2), int(centroid[1] + perp[1] * diameter / 2))
        cv2.line(canvas, p3, p4, (0, 0, 255), 2)

        # 标记缺口
        cv2.circle(canvas, tuple(big_notch), 6, (255, 0, 0), -1)
        cv2.circle(canvas, tuple(small_notch), 6, (0, 0, 255), -1)

        # 在圆心右侧绘制编号：沿水平线找到最右侧点 A，再向右偏移
        pid = region_id_map.get(label, label)
        text = str(pid)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))  # cx=x, cy=y
        rows = [cy, cy - 1, cy + 1]  # 若质心行无命中，尝试邻行
        ax = None
        for r in rows:
            if r < 0 or r >= h:
                continue
            cols = np.where(component[r, :] > 0)[0]
            if cols.size:
                ax = cols.max()
                break
        if ax is None:
            ax = cx  # fallback
        d = max(0, ax - cx)
        margin = 10
        center_x = cx + d + margin
        center_y = cy
        org = (int(center_x - tw / 2), int(center_y + th / 2))
        # 边界保护：如果超出右侧，回退到左侧
        if org[0] + tw > w:
            org = (max(0, cx - d - margin - int(tw / 2)), org[1])
        cv2.putText(canvas, text, org, font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    # 保存
    # 保存，与原图尺寸一致
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = fig.add_axes([0,0,1,1])  # 占满画布
    ax.imshow(canvas[:, :, ::-1])
    ax.axis("off")

    fig.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return axes_info


def run_refined_particle_extraction(ch00_path, save_prefix="refined"):
    """
    Implements a refined particle extraction strategy based on intersecting
    an ideal circular mask with a detailed raw mask.

    1. Generates a detailed mask C (no opening).
    2. Generates a smoothed mask A (opening_radius=5).
    3. Fits circles to mask A to create an ideal mask B.
    4. Intersects B and C to get the final mask D.

    Saves intermediate images A, B, C and final D for review.
    """
    # --- Step 1: Generate Image C (Noisy but complete mask) ---
    print("Step 1/4: Generating detailed mask C (no opening)...")
    # extract_outer_boundaries returns the raw mask array.
    image_C = extract_outer_boundaries(
        ch00_path,
        thin_opening_radius=0 # Disable opening to get max detail
    )
    cv2.imwrite(f"{save_prefix}_C_detailed_mask.png", image_C)
    print(f" -> Saved {save_prefix}_C_detailed_mask.png")

    # --- Step 2: Generate Image A (Smoothed mask) ---
    print("Step 2/4: Generating smoothed mask A (opening radius 5)...")
    image_A = extract_outer_boundaries(
        ch00_path,
        thin_opening_radius=5
    )
    cv2.imwrite(f"{save_prefix}_A_smoothed_mask.png", image_A)
    print(f" -> Saved {save_prefix}_A_smoothed_mask.png")

    # --- Step 3: Generate Image B (Ideal circles from A) ---
    print("Step 3/4: Generating ideal circle mask B from smoothed mask A...")
    # Find contours in the smoothed mask A
    contours, _ = cv2.findContours(image_A, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image for the ideal circles
    image_B = np.zeros_like(image_A)

    # Fit a circle to each contour and draw it on image_B
    for cnt in contours:
        # Get the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        # Draw the filled circle
        cv2.circle(image_B, center, radius, 255, -1)

    cv2.imwrite(f"{save_prefix}_B_ideal_circles.png", image_B)
    print(f" -> Saved {save_prefix}_B_ideal_circles.png")

    # --- Step 4: Generate Image D (Intersection) ---
    print("Step 4/4: Generating final mask D by intersecting B and C...")
    image_D = cv2.bitwise_and(image_B, image_C)
    cv2.imwrite(f"{save_prefix}_D_final_mask.png", image_D)
    print(f" -> Saved {save_prefix}_D_final_mask.png")

    print("\nRefined extraction complete!")

    return image_D
