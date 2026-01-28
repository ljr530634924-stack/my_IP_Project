import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, exposure, feature, measure
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max
from scipy import ndimage
import cv2
from scipy.signal import argrelextrema
import gc

def tiled_canny(image, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, chunk_size=4000):
    """
    内存优化版的 Canny 边缘检测。
    通过将图像切分为重叠的块来运行 skimage.feature.canny，从而大幅降低峰值内存占用。
    """
    rows, cols = image.shape
    output = np.zeros((rows, cols), dtype=bool)
    
    # 边缘余量：4*sigma 足以覆盖高斯核的影响范围，
    # 额外增加一点余量以保证边缘连接性（虽然 Hysteresis 是全局的，但在大块处理下通常影响可忽略）
    margin = int(max(50, 4 * sigma))
    
    for r in range(0, rows, chunk_size):
        for c in range(0, cols, chunk_size):
            # 定义包含余量的 Tile 坐标
            r0 = max(0, r - margin)
            r1 = min(rows, r + chunk_size + margin)
            c0 = max(0, c - margin)
            c1 = min(cols, c + chunk_size + margin)
            
            tile = image[r0:r1, c0:c1]
            
            # 处理 Mask 切片
            tile_mask = None
            if mask is not None:
                tile_mask = mask[r0:r1, c0:c1]
            
            # 在 Tile 上运行 skimage Canny
            edges_tile = feature.canny(tile, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, mask=tile_mask)
            
            # 计算输出图像中的有效区域（去除余量）
            out_r_start = r
            out_r_end = min(r + chunk_size, rows)
            out_c_start = c
            out_c_end = min(c + chunk_size, cols)
            
            # 计算 Tile 中的对应区域
            tile_r_start = out_r_start - r0
            tile_r_end = tile_r_start + (out_r_end - out_r_start)
            tile_c_start = out_c_start - c0
            tile_c_end = tile_c_start + (out_c_end - out_c_start)
            
            # 填入结果
            output[out_r_start:out_r_end, out_c_start:out_c_end] = \
                edges_tile[tile_r_start:tile_r_end, tile_c_start:tile_c_end]
    
    return output


def extract_inner_boundaries(
    ch00_path,
    save_path=None,
    thin_opening_radius=0,
    min_size=50,
    keep_area=None,
    canny_sigma=1.0,
    clahe_clip_limit=2.0,
    brightness_threshold=None, # 兼容性参数，本逻辑主要基于Canny，暂不使用亮度阈值
    min_circularity=None,
    small_hole_area=None,
    closing_radius=5,
    detect_dark=True  # 新增参数：True=提取暗洞(Case A), False=提取亮斑(Case B)
):
    """
    基于 "实心Mask + 亮度阈值" 的逻辑提取内部特征。
    
    核心逻辑：
    1. 生成实心的粒子 Mask (Container)。
    2. 在 Container 内部应用亮度阈值筛选。
       - detect_dark=True:  提取 < threshold 的区域 (暗洞)
       - detect_dark=False: 提取 > threshold 的区域 (亮信号)
    3. 过滤噪点。
    """
    
    # 兼容 keep_area 参数 (通常用于指代最小保留面积)
    if keep_area is not None:
        min_size = keep_area

    # --- 1. 预处理 (与外轮廓保持一致) ---
    img_eq = preprocess_structure_image(
        ch00_path,
        flatten_background=True,
        clahe_clip_limit=clahe_clip_limit
    )
    
    # 关键修复：将 uint8 (0-255) 转换为 float (0-1) 以便与 brightness_threshold (如 0.5) 进行比较
    if img_eq.dtype == np.uint8:
        img_float = img_eq.astype(np.float32) / 255.0
    else:
        img_float = img_eq

    # --- 2. 生成实心容器 Mask ---
    # 调用 extract_outer_boundaries 获取粒子主体
    # 使用较小的 min_size (如500) 确保能捕获到粒子，避免漏掉
    container_mask_img = extract_outer_boundaries(
        ch00_path,
        save_path=None,
        preprocessed_img=img_eq, # 复用预处理结果
        min_size=500, 
        thin_opening_radius=0,   # 不需要开运算，保留最大轮廓
        canny_sigma=canny_sigma
    )
    container_mask = container_mask_img > 0
    
    # 确保容器是实心的 (填充所有内部孔洞，作为筛选范围)
    container_mask = ndimage.binary_fill_holes(container_mask)

    # --- 3. 应用亮度阈值 ---
    # 如果未指定阈值，给一个默认值 (0.5)
    thresh_val = float(brightness_threshold) if brightness_threshold is not None else 0.5
    
    if detect_dark:
        # Case A: 粒子内部的暗洞 (Pixel < Threshold)
        threshold_mask = img_float < thresh_val
    else:
        # Case B: 粒子内部的亮信号 (Pixel > Threshold)
        threshold_mask = img_float > thresh_val

    # --- 4. 取交集 (Container AND Threshold) ---
    result_mask = container_mask & threshold_mask

    # --- 5. 过滤结果 ---
    # 移除过小的噪点
    final_holes = morphology.remove_small_objects(result_mask, min_size=min_size)

    # 可选：圆度过滤
    if min_circularity is not None:
        labels = measure.label(final_holes)
        regions = measure.regionprops(labels)
        final_holes = np.zeros_like(final_holes, dtype=bool)
        for r in regions:
            perimeter = max(r.perimeter, 1e-6)
            circularity = 4 * np.pi * r.area / (perimeter ** 2)
            if circularity >= min_circularity:
                final_holes[r.coords[:, 0], r.coords[:, 1]] = True

    # 可选：形态学开运算整理形状
    if thin_opening_radius > 0:
        selem = morphology.disk(thin_opening_radius)
        final_holes = morphology.binary_opening(final_holes, selem)

    # --- 6. 输出 ---
    result_img = (final_holes * 255).astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, result_img)

    return result_img


def separate_particles_watershed(binary_mask, min_distance=20):
    """
    Separates connected particles using the Watershed algorithm.
    Returns a labeled image where each particle has a unique ID.
    """
    # 1. Distance transform: calculate distance from background
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # 2. Find peaks: local maxima in the distance map are particle centers
    # labels=binary_mask ensures we only look within foreground.
    # IMPORTANT: exclude_border=False is crucial when processing small crops (ROIs).
    # Otherwise, peaks within min_distance of the crop edge are dropped.
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_mask, exclude_border=False)
    
    # 3. Create markers
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    
    if markers.max() == 0:
        # Fallback: if no peaks found, treat as single particle to avoid losing it
        return binary_mask.astype(np.int32)
    
    # 4. Watershed segmentation
    # We use -distance because watershed works on basins (minima)
    labels = watershed(-distance, markers, mask=binary_mask)
    
    return labels


def preprocess_structure_image(
    ch00_path, 
    flatten_background=True, 
    clahe_clip_limit=2.0, 
    large_sigma=40,    # Renamed from flatten_sigma for clarity (Bandpass Large)
    noise_sigma=3.0,    # Renamed from noise_sigma for clarity (Bandpass Small)
    debug_prefix=None,
    stretch_low=0.5,   # New: Percentile for saturation stretch (low end)
    stretch_high=99.5, # New: Percentile for saturation stretch (high end)
    restrict_to_largest_circle=False # New: Auto-detect well/dish and mask outside
):
    """
    Helper function to perform heavy preprocessing (IO, Background Flatten, CLAHE) once.
    Returns the preprocessed image (img_eq) ready for Canny edge detection.
    """
    # --- 1. 读取原图 ---
    print("Step 0/4: Preprocessing image (Background flattening)...")
    img = io.imread(ch00_path)

    # --- 1.5 霍夫圆检测 ROI (可选) ---
    if restrict_to_largest_circle:
        # 为了速度，先降采样做检测
        h_raw, w_raw = img.shape
        scale = 512.0 / w_raw
        small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        
        # 归一化到 0-255 uint8 供 Hough 使用
        if small_img.dtype != np.uint8:
            small_img = ((small_img - small_img.min()) / (small_img.max() - small_img.min()) * 255).astype(np.uint8)
        
        # 霍夫圆检测
        # param2: 累加器阈值，越小越容易检测到圆（但也容易误检）
        # minRadius: 假设圆至少占据宽度的 30%
        circles = cv2.HoughCircles(small_img, cv2.HOUGH_GRADIENT, dp=1, minDist=small_img.shape[0]/2,
                                   param1=50, param2=30, 
                                   minRadius=int(small_img.shape[0]*0.3), 
                                   maxRadius=int(small_img.shape[0]*0.6))
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 找最大的圆
            largest_circle = max(circles[0, :], key=lambda c: c[2])
            cx, cy, r = largest_circle
            
            # 映射回原图坐标
            cx = int(cx / scale)
            cy = int(cy / scale)
            r = int(r / scale)
            
            # --- 安全检查：如果圆心偏离图像中心太远，可能是误检 ---
            img_cx, img_cy = w_raw // 2, h_raw // 2
            offset = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
            max_offset = min(w_raw, h_raw) * 0.15  # 允许 15% 的偏离
            
            if offset > max_offset:
                print(f" -> [WARN] Detected ROI circle center ({cx}, {cy}) is too far from image center ({img_cx}, {img_cy}). Offset={int(offset)}. Ignoring ROI to avoid cropping data.")
            else:
                # 创建掩膜并应用
                mask = np.zeros((h_raw, w_raw), dtype=img.dtype)
                cv2.circle(mask, (cx, cy), r, 1, -1) # 1 inside
                img = img * mask # 圆外变黑
                print(f" -> ROI applied: Center=({cx},{cy}), Radius={r}")
        else:
            print(" -> [WARN] No circle detected. Skipping ROI masking.")

    # --- 2. 归一化到 0-1 ---
    img = img.astype(np.float32)
    # 优化：使用百分位数归一化，防止极亮噪点(Hot Pixels)或标尺导致整体对比度被压缩
    p_min, p_max = np.percentile(img, (0.1, 99.9))
    if p_max > p_min:
        img = (img - p_min) / (p_max - p_min)
    img = np.clip(img, 0, 1)

    # --- 2.5 背景校正 (针对中间亮四周暗的情况) ---
    if flatten_background:
        # 使用 Difference of Gaussians (DoG) 带通滤波替代单纯的背景减除
        # 1. 小核高斯模糊：去除高频噪点 (保留信号)
        blur_small = cv2.GaussianBlur(img, (0, 0), noise_sigma)

        # 2. 大核高斯模糊：估算背景 (去除低频)
        # 优化：先降采样再模糊，大幅减少大核模糊的计算量
        h, w = img.shape
        down_factor = 0.25
        small_h, small_w = int(h * down_factor), int(w * down_factor)
        img_down = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # 在小图上做模糊，sigma 也要相应缩小 (100 * 0.25 = 25)
        bg_small = cv2.GaussianBlur(img_down, (0, 0), large_sigma * down_factor)
        
        # 放大回原尺寸
        blur_large = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 3. DoG = Blur_small - Blur_large
        # 优化：使用原地减法节省内存 (blur_small 被复用为 img)
        cv2.subtract(blur_small, blur_large, dst=blur_small)
        img = blur_small
        del blur_large # 立即释放大数组
        gc.collect()   # 强制垃圾回收
        
        # 4. Fiji-style percentile stretching (saturation stretch)
        # This avoids hard clipping and handles outliers gracefully.
        p_lo, p_hi = np.percentile(img, (stretch_low, stretch_high))
        if p_hi > p_lo:
            # Stretch the percentile range to [0, 1]
            # 优化：原地运算
            np.subtract(img, p_lo, out=img)
            np.multiply(img, 1.0 / (p_hi - p_lo), out=img)
        
        # Clip to [0, 1] range, effectively saturating the ends
        np.clip(img, 0, 1, out=img)

        if debug_prefix:
            debug_path = f"{debug_prefix}_flattened_dog.png"
            cv2.imwrite(debug_path, (img * 255).astype(np.uint8))
            print(f" -> [Debug] Saved flattened image (DoG): {debug_path}")

    # --- 3. CLAHE增强 ---
    # 优化：使用 OpenCV CLAHE (SIMD加速)
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_uint8)
    
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_preprocessed_final.png", img_eq)
        print(f" -> [Debug] Saved final preprocessed image: {debug_prefix}_preprocessed_final.png")

    return img_eq


def extract_outer_boundaries(ch00_path,
                             save_path="closed_particles.png",
                             min_circularity=None,
                             thin_opening_radius=0,
                             min_size=2000,      # 默认值保持 2000 (用于结构提取)
                             keep_area=3000,     # 默认值保持 3000 (用于结构提取)
                             canny_sigma=2.0,    # 默认值保持 2.0
                             clahe_clip_limit=2.0,
                             flatten_background=True, # 新增：是否启用背景校正
                             preprocessed_img=None, # 新增：接收预处理后的图像以跳过重复计算
                             use_watershed=False,   # 新增：是否使用分水岭算法分割粘连
                             watershed_min_dist=20, # 新增：分水岭算法的最小峰值距离
                             tiled_watershed=True,  # 新增：是否使用分块方式运行分水岭(省内存但可能影响效果)
                             ): 
    """
    Clean version of boundary extraction for Structure (Step 3).
    Removes brightness_threshold logic (Path B) to avoid capturing internal bright spots.
    """
    
    # 如果提供了预处理后的图像，直接使用，跳过 IO/Hough/Flatten/CLAHE
    if preprocessed_img is not None:
        img_eq = preprocessed_img
        # 获取形状用于后续创建 mask
        img_h, img_w = img_eq.shape
    else:
        # 否则执行完整流程
        img_eq = preprocess_structure_image(ch00_path, flatten_background, clahe_clip_limit, debug_prefix="debug_outer")
        img_h, img_w = img_eq.shape

    # --- 4. Canny边缘 ---
    edges = tiled_canny(img_eq, sigma=canny_sigma)

    # --- 5. 闭操作，使边界闭合 ---
    edges_closed = morphology.binary_closing(edges, morphology.disk(5))

    # --- 6. 填充洞 ---
    filled = morphology.remove_small_holes(edges_closed, area_threshold=5000)

    # (Path B: Brightness Threshold logic removed here)

    # --- 7. 去除极小噪点 ---
    prelim = morphology.remove_small_objects(filled, min_size=min_size)

    # 可选：用小结构元素做开运算，再轻微膨胀回去，剃掉头发丝等细长附加物
    if thin_opening_radius and thin_opening_radius > 0:
        selem = morphology.disk(thin_opening_radius)
        prelim = morphology.binary_opening(prelim, selem)
        prelim = morphology.binary_dilation(prelim, selem)

    # --- 7.5 分水岭分割 (处理粘连) ---
    if use_watershed:
        if tiled_watershed:
            # 内存优化：逐个对连通域应用分水岭，而不是对整张大图
            print(" -> Applying watershed per-component (Tiled) to save memory...")
            
            # 1. 先找到所有粘连的团块
            initial_labels = measure.label(prelim)
            
            # 2. 创建一个空的输出图像，用于收集分割后的结果
            final_labels = np.zeros_like(initial_labels)
            label_offset = 0  # 用于确保最终的 label ID 是唯一的
            
            # 3. 遍历每个团块
            for region in measure.regionprops(initial_labels):
                # 提取该团块的 bounding box，以减小处理范围
                minr, minc, maxr, maxc = region.bbox
                pad = 2  # 增加一点 padding，防止边界效应
                r0, r1 = max(0, minr - pad), min(prelim.shape[0], maxr + pad)
                c0, c1 = max(0, minc - pad), min(prelim.shape[1], maxc + pad)
                
                # 在这个小块上运行分水岭
                component_mask = (initial_labels[r0:r1, c0:c1] == region.label)
                sub_labels = separate_particles_watershed(component_mask, min_distance=watershed_min_dist)
                
                # 将分割后的结果放回最终的 label 图中，并确保 label ID 不冲突
                if sub_labels.max() > 0:
                    valid_sub_labels = sub_labels > 0
                    final_labels[r0:r1, c0:c1][valid_sub_labels] = sub_labels[valid_sub_labels] + label_offset
                    label_offset += sub_labels.max()
            
            prelim = final_labels > 0
            lbl = final_labels  # 直接复用 lbl，避免再次调用 measure.label
        else:
            # 全局模式：对整张图运行分水岭 (内存消耗大，但效果可能更好)
            print(" -> Applying Global watershed (High Memory Usage)...")
            lbl = separate_particles_watershed(prelim, min_distance=watershed_min_dist)
            prelim = lbl > 0
    else:
        lbl = measure.label(prelim)

    # --- 8. 连通域分析 ---
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
        
        # 结构提取通常不需要 min_circularity，保持原有逻辑
        keep = area > keep_area and aspect_ratio < 1.3
        if min_circularity is not None:
            perimeter = max(r.perimeter, 1e-6)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            keep = keep and (circularity >= min_circularity)

        if keep:
            filtered_mask[r.coords[:, 0], r.coords[:, 1]] = True
            kept_regions.append(r)

    # --- 10. 画图并保存 (保持原有逻辑) ---
    mask_img = np.zeros((img_h, img_w), dtype=np.uint8)
    for r in kept_regions:
        mask_img[r.coords[:, 0], r.coords[:, 1]] = 255

    # NEW: 如果使用了分水岭，必须在二值图中“烧”出边界线，防止粒子再次粘连
    if use_watershed and len(kept_regions) > 0:
        # 1. 重建带有不同 ID 的 label 图
        kept_labels = np.zeros((img_h, img_w), dtype=np.int32)
        for i, r in enumerate(kept_regions):
            kept_labels[r.coords[:, 0], r.coords[:, 1]] = i + 1
        
        # 2. 找出“内部边界”：即不同 label 接触的地方 (b_all)，排除掉 label 与背景接触的地方 (b_bin)
        b_all = find_boundaries(kept_labels, mode='inner', background=0)
        b_bin = find_boundaries(kept_labels > 0, mode='inner', background=0)
        touching_boundaries = b_all & (~b_bin)
        
        # 3. 加粗接触线并置为黑色 (0)
        # 使用膨胀操作将 1 像素的线变粗，确保 8-连通性被彻底切断
        # morphology.disk(1) 会产生 3x3 的结构，足以切断对角线连接
        thick_boundaries = morphology.binary_dilation(touching_boundaries, morphology.disk(1))
        mask_img[thick_boundaries] = 0

    # (省略了画图代码以保持简洁，但核心返回的是 mask_img)
    # 注意：如果需要保存 closed_particles.png，这里应该保留画图代码。
    # 鉴于 run_refined_particle_extraction 主要使用返回值，这里简化处理。
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

        # --- 1. 确定大缺口 (红点): 全局距离质心最近的点 ---
        # --- 1. 寻找所有候选极小值点 (包括全局最小值) ---
        minima_idx = argrelextrema(d, np.less)[0]
        global_min_idx = np.argmin(d)
        # 确保全局最小值也在候选列表中
        if global_min_idx not in minima_idx:
            minima_idx = np.append(minima_idx, global_min_idx)

        # (可选) 统一过滤：禁止落在分水岭分界线上 (邻居检测)
        check_radius = 5
        natural_candidates = []
        
        for idx in minima_idx:
            pt = pts[idx]
            px, py = int(pt[0]), int(pt[1])
            
            # 提取 ROI (注意边界检查)
            y_min, y_max = max(0, py - check_radius), min(h, py + check_radius + 1)
            x_min, x_max = max(0, px - check_radius), min(w, px + check_radius + 1)
            roi = labels[y_min:y_max, x_min:x_max]
            
            # 检查是否有邻居：ROI中存在非0且非当前label的像素
            has_neighbor = np.any((roi > 0) & (roi != label))
            
            if not has_neighbor:
                natural_candidates.append(idx)

        if len(natural_candidates) > 0:
            best_idx = natural_candidates[np.argmin(d[natural_candidates])]
            big_idx = best_idx
            big_notch = pts[big_idx]
        else:
            # 回退：如果没有自然边界点，使用全局最小值
            big_idx = global_min_idx
            big_notch = pts[big_idx]

        # --- 2. 定义 Y 轴向量 (从圆心指向红点) ---
        v = big_notch - centroid
        v_norm = np.linalg.norm(v)
        if v_norm == 0: continue # 质心和豁口重合，无法定义方向
        v = v / v_norm

        # 只要有大豁口，就绘制Y轴和蓝点
        diameter = np.max(d) * 2
        p1 = (int(centroid[0] - v[0] * diameter / 2), int(centroid[1] - v[1] * diameter / 2))
        p2 = (int(centroid[0] + v[0] * diameter / 2), int(centroid[1] + v[1] * diameter / 2))
        cv2.line(canvas, p1, p2, (0, 0, 255), 2) # Red Y-axis (Swapped)
        cv2.circle(canvas, tuple(big_notch), 6, (0, 0, 255), -1) # Red dot for big notch (Swapped)

        # --- 3. 确定小缺口 (蓝点): 在垂直于 Y 轴的区域内寻找最近点 ---
        # 策略：寻找与 Y 轴夹角接近 90 度 (60~120度) 的轮廓点
        
        # 计算所有轮廓点相对于 Y 轴的夹角余弦值 (Dot Product)
        # cos(theta) = (v . p) / (|v|*|p|)
        # 我们想要 theta 接近 90度，即 cos(theta) 接近 0
        # 设定阈值：cos(80)≈0.174, cos(100)≈-0.174。所以绝对值 < 0.174 即可
        
        ortho_candidates = []
        
        # 为了效率，我们直接向量化计算
        # 向量 P = pts - centroid
        vecs = pts - centroid
        # 归一化 P (避免除以0)
        norms = np.linalg.norm(vecs, axis=1)
        valid_mask = norms > 0
        
        # 计算点积 (v 已经是单位向量)
        # dot_products = (vecs . v) / norms
        dot_products = np.sum(vecs[valid_mask] * v, axis=1) / norms[valid_mask]
        
        # 筛选出 cos(theta) 在 [-0.174, 0.174] 之间的点 (即夹角在 80~100 度之间)
        # 对应的索引需要映射回原始 pts
        threshold = np.cos(np.deg2rad(80))
        candidate_indices = np.where(np.abs(dot_products) < threshold)[0]
        
        # 映射回 valid_mask 为 True 的原始索引 (如果 norms 有 0 的情况)
        # 这里简单起见，假设 norms 都 > 0 (因为是轮廓点，离质心肯定有距离)
        
        if len(candidate_indices) > 0:
            # 在这些“侧面”的点中，找到距离质心最近的那个
            # 注意：这里我们直接在几何筛选出的区域里找最小值，不再依赖 argrelextrema
            # 这样可以防止因为局部抖动而漏掉真正的几何最近点
            
            local_d = d[candidate_indices]
            best_local_idx = np.argmin(local_d)
            small_idx = candidate_indices[best_local_idx]
            small_notch = pts[small_idx]

            # X 轴: 与 Y 轴垂直，且正方向面向小豁口
            perp = np.array([v[1], -v[0]])  # 候选 X 轴 (Y 轴旋转+90度)
            v_small = small_notch - centroid
            if np.dot(v_small, perp) < 0:
                perp = -perp

            # 存储完整的坐标系信息
            axes_info[label] = {
                "ex": perp,
                "ey": v,
                "centroid": centroid,
                "radius": np.max(d)
            }

            # 绘制X轴和红点
            p3 = (int(centroid[0] - perp[0] * diameter / 2), int(centroid[1] - perp[1] * diameter / 2))
            p4 = (int(centroid[0] + perp[0] * diameter / 2), int(centroid[1] + perp[1] * diameter / 2))
            cv2.line(canvas, p3, p4, (255, 0, 0), 2) # Blue X-axis (Swapped)
            cv2.circle(canvas, tuple(small_notch), 6, (255, 0, 0), -1) # Blue dot for small notch (Swapped)
        else:
            # 如果没有小豁口，只存储Y轴信息
            axes_info[label] = {
                "ex": np.array([np.nan, np.nan]), # X-axis is undefined
                "ey": v,
                "centroid": centroid,
                "radius": np.max(d)
            }

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
    # 优化：如果 save_path 为 None，则跳过耗内存的画图步骤
    if save_path is not None:
        # 保存，与原图尺寸一致
        fig = plt.figure(figsize=(w/100, h/100), dpi=100)
        ax = fig.add_axes([0,0,1,1])  # 占满画布
        ax.imshow(canvas[:, :, ::-1])
        ax.axis("off")

        fig.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0)
        plt.close(fig)
    return axes_info


def run_refined_particle_extraction(
    ch00_path, 
    save_prefix="refined", 
    circle_radius_scale=1.0, 
    use_watershed=False, 
    watershed_min_dist=20, 
    min_circularity=None, 
    keep_area=3000,
    large_sigma=25,    # 新增：对应 Fiji "Filter large structures down to"
    noise_sigma=1.0,   # 新增：对应 Fiji "Filter small structures up to"
    stretch_low=0.5,   # 新增：饱和度拉伸低位
    stretch_high=99.5, # 新增：饱和度拉伸高位
    save_intermediates=True, # 新增：控制是否保存中间过程图片
    simple_mode=False,  # 新增：如果为True，直接返回Mask C，跳过后续步骤
    restrict_to_largest_circle=False, # 新增：传递给预处理
    tiled_watershed=True # 新增：控制分水岭策略
):
    """
    Implements a refined particle extraction strategy based on intersecting
    an ideal circular mask with a detailed raw mask.

    1. Generates a detailed mask C (no opening).
    2. Generates a smoothed mask A (opening_radius=5).
    3. Fits circles to mask A to create an ideal mask B.
    4. Intersects B and C to get the final mask D.

    Saves intermediate images A, B, C and final D for review.

    Args:
        circle_radius_scale (float): Factor to scale the radius of the ideal circles in Mask B.
        use_watershed (bool): Whether to use watershed segmentation to separate touching particles.
        watershed_min_dist (int): Minimum distance between peaks for watershed.
        min_circularity (float): Minimum circularity threshold (0-1).
        keep_area (int): Minimum area for a single particle.
        large_sigma (float): Sigma for background estimation (Large structures).
        noise_sigma (float): Sigma for noise suppression (Small structures).
        stretch_low (float): Low percentile for saturation stretch.
        stretch_high (float): High percentile for saturation stretch.
    """
    # --- Step 0: Preprocess image once (Heavy lifting), save debug images with "structure_refined" prefix ---
    img_preprocessed = preprocess_structure_image(
        ch00_path, 
        flatten_background=True, 
        debug_prefix=save_prefix if save_intermediates else None,
        large_sigma=large_sigma,     # 传入参数
        noise_sigma=noise_sigma,     # 传入参数
        stretch_low=stretch_low,     # 传入参数
        stretch_high=stretch_high,   # 传入参数
        restrict_to_largest_circle=restrict_to_largest_circle # 传入参数
    )

    # --- Step 1: Generate Image C (Noisy but complete mask) ---
    print("Step 1/4: Generating detailed mask C (no opening)...")
    # extract_outer_boundaries returns the raw mask array.
    image_C = extract_outer_boundaries(
        ch00_path,
        thin_opening_radius=0, # Disable opening to get max detail
        save_path=None,        # Ensure no internal saving
        preprocessed_img=img_preprocessed,
        use_watershed=use_watershed,
        watershed_min_dist=watershed_min_dist,
        tiled_watershed=tiled_watershed,
        min_circularity=min_circularity,
        keep_area=keep_area
    )
    if save_intermediates:
        cv2.imwrite(f"{save_prefix}_C_detailed_mask.png", image_C)
        print(f" -> Saved {save_prefix}_C_detailed_mask.png")

    if simple_mode:
        print("Simple mode enabled: Returning detailed mask C directly.")
        return image_C

    # --- Step 2: Generate Image A (Smoothed mask) ---
    print("Step 2/4: Generating smoothed mask A (opening radius 5)...")
    image_A = extract_outer_boundaries(
        ch00_path,
        thin_opening_radius=5,  # 修改这里：控制结构提取时的平滑程度
        save_path=None,
        preprocessed_img=img_preprocessed,
        use_watershed=use_watershed,
        watershed_min_dist=watershed_min_dist,
        tiled_watershed=tiled_watershed,
        min_circularity=min_circularity,
        keep_area=keep_area
    )
    if save_intermediates:
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
        radius = int(radius * circle_radius_scale)
        # Draw the filled circle
        cv2.circle(image_B, center, radius, 255, -1)

    if save_intermediates:
        cv2.imwrite(f"{save_prefix}_B_ideal_circles.png", image_B)
        print(f" -> Saved {save_prefix}_B_ideal_circles.png")

    # --- Step 4: Generate Image D (Intersection) ---
    print("Step 4/4: Generating final mask D by intersecting B and C...")
    image_D = cv2.bitwise_and(image_B, image_C)
    if save_intermediates:
        cv2.imwrite(f"{save_prefix}_D_final_mask.png", image_D)
        print(f" -> Saved {save_prefix}_D_final_mask.png")

    print("\nRefined extraction complete!")

    return image_D


def cut_mask_with_axes(binary_mask, axes_info, line_thickness=3):
    """
    Cuts the binary mask by drawing black lines along the axes of each particle.
    This helps separate connected signal spots that cross quadrant boundaries.
    """
    h, w = binary_mask.shape
    cut_mask = binary_mask.copy()
    
    for label, ax in axes_info.items():
        centroid = ax["centroid"]
        ex = ax["ex"]
        ey = ax["ey"]
        
        # Use radius from axes_info to limit cut length
        # Fallback to large value if not present (compatibility)
        radius = ax.get("radius", np.sqrt(h**2 + w**2))
        cut_len = radius * 1.2  # Slightly larger to ensure full cut through the boundary

        cx, cy = centroid
        
        # Draw X axis cut (Red axis direction)
        p1_x = int(cx - ex[0] * cut_len)
        p1_y = int(cy - ex[1] * cut_len)
        p2_x = int(cx + ex[0] * cut_len)
        p2_y = int(cy + ex[1] * cut_len)
        cv2.line(cut_mask, (p1_x, p1_y), (p2_x, p2_y), 0, line_thickness)
        
        # Draw Y axis cut (Blue axis direction)
        p3_x = int(cx - ey[0] * cut_len)
        p3_y = int(cy - ey[1] * cut_len)
        p4_x = int(cx + ey[0] * cut_len)
        p4_y = int(cy + ey[1] * cut_len)
        cv2.line(cut_mask, (p3_x, p3_y), (p4_x, p4_y), 0, line_thickness)
        
    return cut_mask


def refine_mask_with_ellipses(binary_mask, min_area=10):
    """
    Fits ellipses to the connected components of the mask to regularize shapes.
    """
    h, w = binary_mask.shape
    refined_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Find contours on the cut mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
            
        # Fit ellipse requires at least 5 points
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(refined_mask, ellipse, 255, -1)
            except:
                # Fallback if fitEllipse fails (e.g. collinear points)
                cv2.drawContours(refined_mask, [cnt], -1, 255, -1)
        else:
            # Fallback for small shapes
            cv2.drawContours(refined_mask, [cnt], -1, 255, -1)
            
    return refined_mask


def find_inner_holes_contours(
    image_gray,
    structure_mask,
    save_path=None,
    min_area=50,
    max_area=2000,
    min_circularity=0.5,
    block_size=41,
    c_value=5,
    debug=False  # 新增调试参数
):
    """
    针对每个粒子区域，分别寻找内部的4个孔洞。
    
    流程：
    1. 遍历 structure_mask 中的每个连通域 (粒子)。
    2. 对每个粒子 ROI 应用硬遮罩 (Hard Mask)。
    3. 使用自适应阈值 + 轮廓分析寻找孔洞。
    4. 返回所有孔洞的全局坐标列表。
    """
    # 确保 mask 是 uint8
    if structure_mask.dtype == bool:
        structure_mask = (structure_mask * 255).astype(np.uint8)
    
    # 自动修正 block_size 为奇数 (OpenCV 要求)
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3: block_size = 3

    # 1. 连通域分析
    labels = measure.label(structure_mask)
    regions = measure.regionprops(labels)
    
    all_holes = []
    
    # 2. 遍历每个粒子 (Per-Region Search)
    for r in regions:
        # 获取 ROI
        minr, minc, maxr, maxc = r.bbox
        
        # 提取局部图像
        roi_img = image_gray[minr:maxr, minc:maxc]
        # 提取局部 mask (只包含当前粒子，排除 ROI 内的其他粒子)
        roi_mask = (labels[minr:maxr, minc:maxc] == r.label).astype(np.uint8) * 255
        
        # Step 2: 硬遮罩 (Overlay/Masking) - 只保留粒子内部
        masked_roi = cv2.bitwise_and(roi_img, roi_img, mask=roi_mask)
        
        # Step 3: 寻找小圆
        # a. 高斯模糊
        blurred = cv2.GaussianBlur(masked_roi, (5, 5), 0)
        
        # b. 自适应阈值
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_value
        )
        
        # 再次应用 mask 清理边界
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)
        
        # c. 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # d. 轮廓查找
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- DEBUG: 保存前几个粒子的中间结果 ---
        if debug and r.label <= 3:
            debug_vis = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_vis, contours, -1, (0, 255, 0), 1)
            cv2.imwrite(f"debug_particle_{r.label}_1_thresh.png", opened)
            cv2.imwrite(f"debug_particle_{r.label}_2_contours.png", debug_vis)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > min_circularity:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        # 转换回全局坐标
                        candidates.append({
                            'center': (cx + minc, cy + minr),
                            'circularity': circularity
                        })
        
        # e. 排序并取前4
        candidates.sort(key=lambda x: x['circularity'], reverse=True)
        all_holes.extend([c['center'] for c in candidates[:4]])

    # 4. 生成结果图片 (无论是否保存，都生成以便返回)
    if image_gray.ndim == 2:
        vis_img = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image_gray.copy()

    # 按照您的要求修改可视化效果：
    # 1. structure_mask 内部透明 (显示原图 ch00)
    # 2. 其余部分黑色 (遮挡背景)
    
    # 创建全黑背景
    final_vis = np.zeros_like(vis_img)
    
    # 只复制 Mask 区域内的像素 (透出原图)
    mask_bool = structure_mask > 0
    final_vis[mask_bool] = vis_img[mask_bool]

    # 画红点 (遍历到的所有孔洞)
    for pt in all_holes:
        cv2.circle(final_vis, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

    # 如果需要保存
    if save_path:
        cv2.imwrite(save_path, final_vis)
        print(f"Saved hole visualization to: {save_path}")

    # 返回标注好的图片
    return final_vis
