import csv
import cv2
import numpy as np
import os


def compute_quadrant_intensity(
    brightness_image,
    labels,
    regions,
    axes_info,
    csv_path="quadrant_intensity.csv",
    id_map_path="particle_id_map.png",
):
    """
    Per particle: place one sampling circle in each quadrant (radius = minor_axis_length / 2,
    center offset along ex/ey by the same radius), compute mean intensity inside each circle,
    and draw IDs + circles onto id_map for visual validation.
    """
    h, w = brightness_image.shape

    # 1) stable IDs: sort by x (col) from left to right
    region_sorted = sorted(regions, key=lambda r: r.centroid[1])
    region_id_map = {r.label: idx + 1 for idx, r in enumerate(region_sorted)}

    # 2) base id map
    # Create a displayable 8-bit background for the ID map by normalizing the input brightness image.
    # This handles both 8-bit and 16-bit inputs correctly for visualization purposes,
    # without affecting the original `brightness_image` used for measurement.
    min_val, max_val = brightness_image.min(), brightness_image.max()
    if max_val > min_val:
        # Normalize to 0-255 range
        display_base = 255.0 * (brightness_image - min_val) / (max_val - min_val)
    else:
        # Handle flat image
        display_base = np.zeros_like(brightness_image)
    id_map = cv2.cvtColor(display_base.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for r in region_sorted:
        pid = region_id_map[r.label]
        cx, cy = r.centroid[1], r.centroid[0]
        cv2.putText(
            id_map,
            str(pid),
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # 3) circles per quadrant
    #Q1 is blue, Q2 is green, Q3 is purple, Q4 is yellow
    results = []
    color_map = {"Q1": (255, 0, 0), "Q2": (0, 255, 0), "Q3": (255, 0, 255), "Q4": (0, 255, 255)}
    offsets = {
        "Q1": (+1, +1),
        "Q2": (-1, +1),
        "Q3": (-1, -1),
        "Q4": (+1, -1),
    }

    for r in region_sorted:
        label = r.label
        if label not in axes_info:
            continue
        pid = region_id_map[label]
        mask = labels == label

        ax = axes_info[label]  # (x, y)
        C_xy = np.array(ax["centroid"], dtype=np.float32)
        ex_xy = np.array(ax["ex"], dtype=np.float32)
        ey_xy = np.array(ax["ey"], dtype=np.float32)

        # convert to (row, col)
        C = np.array([C_xy[1], C_xy[0]], dtype=np.float32)
        ex_rc = np.array([ex_xy[1], ex_xy[0]], dtype=np.float32)
        ey_rc = np.array([ey_xy[1], ey_xy[0]], dtype=np.float32)

        def _norm(v):
            n = np.linalg.norm(v)
            return v if n == 0 else v / n

        ex_rc = _norm(ex_rc)
        ey_rc = _norm(ey_rc)

        radius = float(r.minor_axis_length) / 6
        if radius <= 0:
            results.append([pid, np.nan, np.nan, np.nan, np.nan])
            continue

        ys, xs = np.where(mask)
        coords = np.stack([ys, xs], axis=1).astype(np.float32)
        vals = brightness_image[mask]

        quadrant_means = []
        quadrant_areas = []

        for q_name, (sx, sy) in offsets.items():
            center = C + ex_rc * (sx * radius) + ey_rc * (sy * radius)

            dy = coords[:, 0] - center[0]
            dx = coords[:, 1] - center[1]
            in_circle = (dx * dx + dy * dy) <= (radius * radius)

            q_vals = vals[in_circle]
            if q_vals.size > 0:
                mean_val = round(float(np.mean(q_vals)), 2)
                area_val = q_vals.size
            else:
                mean_val = np.nan
                area_val = 0

            quadrant_means.append(mean_val)
            quadrant_areas.append(area_val)

            cv2.circle(
                id_map,
                (int(round(center[1])), int(round(center[0]))),  # (x, y)
                int(round(radius)),
                color_map[q_name],
                1,
                cv2.LINE_AA,
            )

        results.append([pid, *quadrant_means, *quadrant_areas])

    # 4) write CSV
    mean_row = []
    sd_row = []
    sem_row = []
    if results:
        mean_row = ["Mean"]
        sd_row = ["SD"]
        sem_row = ["Std. Error"]
        # Columns 1 to 8 (Q1_Mean...Q4_Mean, Q1_Area...Q4_Area)
        for col_idx in range(1, 9):
            col_vals = [row[col_idx] for row in results]
            valid_vals = [v for v in col_vals if not np.isnan(v)]
            if valid_vals:
                mean_row.append(round(np.mean(valid_vals), 2))
                n = len(valid_vals)
                if n > 1:
                    sd = np.std(valid_vals, ddof=1)
                    sd_row.append(round(sd, 2))
                    sem_row.append(round(sd / np.sqrt(n), 2))
                else:
                    sd_row.append(0.0)
                    sem_row.append(0.0)
            else:
                mean_row.append(0.0)
                sd_row.append(0.0)
                sem_row.append(0.0)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["particle_id", 
                  "Q1_Mean", "Q2_Mean", "Q3_Mean", "Q4_Mean", 
                  "Q1_Area", "Q2_Area", "Q3_Area", "Q4_Area"]
        writer.writerow(header)
        writer.writerows(results)
        if mean_row:
            writer.writerow(mean_row)
        if sd_row:
            writer.writerow(sd_row)
        if sem_row:
            writer.writerow(sem_row)

    # 5) Save population summary (Inter-particle stats)
    base, ext = os.path.splitext(csv_path)
    _save_population_stats(results, f"{base}_summary{ext}")

    cv2.imwrite(id_map_path, id_map)

    print(f"[OK] quadrant table written: {csv_path}")
    print(f"[OK] id map saved: {id_map_path}")


def compute_masked_quadrant_intensity(
    brightness_image,
    structure_mask,
    signal_mask,
    axes_info,
    csv_path="masked_quadrant_results.csv",
    overlay_path="final_overlay_visualization.png"
):
    """
    Compute intensity based on the intersection of structure_mask (particle)
    and signal_mask (bright spots).
    """
    h, w = brightness_image.shape

    # 1. Prepare structure labels
    num_labels, labels = cv2.connectedComponents(structure_mask)
    from skimage import measure
    regions = measure.regionprops(labels)
    regions = sorted(regions, key=lambda r: r.centroid[1])
    region_id_map = {r.label: idx + 1 for idx, r in enumerate(regions)}

    # 2. Prepare visualization base (8-bit)
    min_val, max_val = brightness_image.min(), brightness_image.max()
    if max_val > min_val:
        display_base = 255.0 * (brightness_image - min_val) / (max_val - min_val)
    else:
        display_base = np.zeros_like(brightness_image)
    vis_img = cv2.cvtColor(display_base.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Quadrant colors for visualization
    # Q1: Blue, Q2: Green, Q3: Purple, Q4: Yellow
    q_colors = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (255, 0, 255),
        3: (0, 255, 255)
    }

    results = []

    for r in regions:
        label = r.label
        if label not in axes_info:
            continue

        pid = region_id_map[label]
        ax = axes_info[label]
        centroid = ax["centroid"]
        ex = ax["ex"]
        ey = ax["ey"]

        minr, minc, maxr, maxc = r.bbox
        q_pixels = [[], [], [], []]  # Store pixel values for each quadrant

        for row in range(minr, maxr):
            for col in range(minc, maxc):
                # Must be inside the particle structure
                if labels[row, col] != label:
                    continue

                # Must be inside the signal mask (bright spot)
                if signal_mask[row, col] == 0:
                    # Dim the background pixels for visualization
                    vis_img[row, col] = (vis_img[row, col] * 0.3).astype(np.uint8)
                    continue

                # --- Pixel is valid signal ---
                vec = np.array([col - centroid[0], row - centroid[1]])
                proj_x = np.dot(vec, ex)
                proj_y = np.dot(vec, ey)

                # Determine quadrant
                # Q1(+,+), Q2(-,+), Q3(-,-), Q4(+,-)
                if proj_x >= 0 and proj_y >= 0:
                    q_idx = 0
                elif proj_x < 0 and proj_y >= 0:
                    q_idx = 1
                elif proj_x < 0 and proj_y < 0:
                    q_idx = 2
                else:
                    q_idx = 3

                val = brightness_image[row, col]
                q_pixels[q_idx].append(val)

                # Colorize pixel
                color = q_colors[q_idx]
                current_bgr = vis_img[row, col].astype(float)
                new_bgr = current_bgr * 0.5 + np.array(color) * 0.5
                vis_img[row, col] = np.clip(new_bgr, 0, 255).astype(np.uint8)

        # Calculate stats from collected pixels
        means, areas = [], []
        for q_idx in range(4):
            vals = np.array(q_pixels[q_idx])
            if vals.size > 0:
                means.append(round(float(np.mean(vals)), 2))
                areas.append(vals.size)
            else:
                means.append(0.0)
                areas.append(0)

        row_data = [pid] + means + areas
        results.append(row_data)

        # Draw axes
        p_cen = (int(centroid[0]), int(centroid[1]))
        p_y = (int(centroid[0] + ey[0] * 20), int(centroid[1] + ey[1] * 20))
        p_x = (int(centroid[0] + ex[0] * 20), int(centroid[1] + ex[1] * 20))
        cv2.line(vis_img, p_cen, p_y, (255, 0, 0), 2) # Blue axis (Y -> Big Notch)
        cv2.line(vis_img, p_cen, p_x, (0, 0, 255), 2) # Red axis (X -> Small Notch)
        cv2.putText(vis_img, str(pid), p_cen, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Calculate column means for the footer row
    mean_row = []
    sd_row = []
    sem_row = []
    if results:
        mean_row = ["Mean"]
        sd_row = ["SD"]
        sem_row = ["Std. Error"]
        # Columns 1 to 8 (Q1_Mean...Q4_Mean, Q1_Area...Q4_Area)
        for col_idx in range(1, 9):
            col_vals = [row[col_idx] for row in results]
            valid_vals = [v for v in col_vals if not np.isnan(v)]
            if valid_vals:
                mean_row.append(round(np.mean(valid_vals), 2))
                n = len(valid_vals)
                if n > 1:
                    sd = np.std(valid_vals, ddof=1)
                    sd_row.append(round(sd, 2))
                    sem_row.append(round(sd / np.sqrt(n), 2))
                else:
                    sd_row.append(0.0)
                    sem_row.append(0.0)
            else:
                mean_row.append(0.0)
                sd_row.append(0.0)
                sem_row.append(0.0)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["particle_id", 
                  "Q1_Mean", "Q2_Mean", "Q3_Mean", "Q4_Mean", 
                  "Q1_Area", "Q2_Area", "Q3_Area", "Q4_Area"]
        writer.writerow(header)
        writer.writerows(results)
        if mean_row:
            writer.writerow(mean_row)
        if sd_row:
            writer.writerow(sd_row)
        if sem_row:
            writer.writerow(sem_row)

    # Save population summary
    base, ext = os.path.splitext(csv_path)
    _save_population_stats(results, f"{base}_summary{ext}")

    cv2.imwrite(overlay_path, vis_img)
    print(f"[OK] Masked measurement saved: {csv_path}")
    print(f"[OK] Visualization saved: {overlay_path}")


def _save_population_stats(results, summary_path):
    """
    Calculate Mean, Std, and SEM across all particles (Inter-particle statistics)
    and save to a separate CSV.
    """
    # results structure: [pid, Q1_mean, Q2_mean, Q3_mean, Q4_mean, ...]
    # We extract indices 1, 2, 3, 4
    q_data = [[], [], [], []]

    for row in results:
        # row[0] is pid
        # row[1]..row[4] are Q1..Q4 means
        for i in range(4):
            val = row[1 + i]
            if not np.isnan(val):
                q_data[i].append(val)

    summary_rows = []
    # Headers
    headers = ["Statistic", "Q1", "Q2", "Q3", "Q4"]

    # Calculate stats
    means = ["Population_Mean"]
    stds = ["Population_Std"]
    sems = ["Population_SEM"]

    for i in range(4):
        arr = np.array(q_data[i])
        if arr.size > 0:
            m = np.mean(arr)
            s = np.std(arr, ddof=1) if arr.size > 1 else 0.0
            sem = s / np.sqrt(arr.size)

            means.append(round(m, 2))
            stds.append(round(s, 2))
            sems.append(round(sem, 2))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            sems.append(np.nan)

    summary_rows = [means, stds, sems]

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(summary_rows)

    print(f"[OK] Population statistics (Inter-particle) saved: {summary_path}")
