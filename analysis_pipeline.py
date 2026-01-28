"""
Unified image analysis and compositing utilities.
All functions are ASCII only to avoid encoding issues.
"""
from pathlib import Path
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, exposure, feature, morphology, measure
from scipy.signal import argrelextrema
from PIL import Image


# ---------- Basic helpers ----------
def _ensure_path(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


def adjust_ch01_image(ch01_path, save_path,
                      brightness=10.0,
                      exposure=0.3,
                      contrast_gain=0.2):
    """
    Load a channel-1 image, apply exposure/brightness/contrast tweaks, and save.
    """
    ch01_path = _ensure_path(ch01_path)
    img = cv2.imread(str(ch01_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {ch01_path}")

    img = img.astype(np.float32)
    img = img * exposure
    img = img + brightness
    img = (img - 128.0) * contrast_gain + 128.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    cv2.imwrite(str(save_path), img)
    return Path(save_path)


def extract_outer_boundaries(ch00_path,
                             save_path=None,
                             clahe_clip=0.01,
                             canny_sigma=2,
                             closing_radius=5,
                             hole_area=5000,
                             min_size=2000,
                             keep_area=3000,
                             keep_aspect_max=1.8):
    """
    Generate a binary mask of particles and optionally save a boundary visualization.
    """
    ch00_path = _ensure_path(ch00_path)
    img = io.imread(str(ch00_path))
    img = img.astype(np.float64)
    img = (img - img.min()) / max(1e-8, (img.max() - img.min()))

    img_eq = exposure.equalize_adapthist(img, clip_limit=clahe_clip)
    edges = feature.canny(img_eq, sigma=canny_sigma)
    edges_closed = morphology.binary_closing(edges, morphology.disk(closing_radius))
    filled = morphology.remove_small_holes(edges_closed, area_threshold=hole_area)
    prelim = morphology.remove_small_objects(filled, min_size=min_size)

    lbl = measure.label(prelim)
    regions = measure.regionprops(lbl)

    filtered_mask = np.zeros_like(prelim, dtype=bool)
    kept_regions = []
    for r in regions:
        area = r.area
        minr, minc, maxr, maxc = r.bbox
        h = maxr - minr
        w = maxc - minc
        aspect_ratio = max(h, w) / max(1, min(h, w))
        if area > keep_area and aspect_ratio < keep_aspect_max:
            filtered_mask[r.coords[:, 0], r.coords[:, 1]] = True
            kept_regions.append(r)

    mask_img = (filtered_mask.astype(np.uint8)) * 255

    if save_path:
        h, w = mask_img.shape
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(mask_img, cmap="gray")

        for r in kept_regions:
            contour = measure.find_contours(r.filled_image, 0.5)
            if not contour:
                continue
            contour = contour[0]
            minr, minc, _, _ = r.bbox
            contour[:, 0] += minr
            contour[:, 1] += minc
            ax.plot(contour[:, 1], contour[:, 0], color="yellow", linewidth=2)

        ax.axis("off")
        fig.savefig(str(save_path), dpi=100, bbox_inches=None, pad_inches=0)
        plt.close(fig)

    return mask_img


def find_notches_and_axes(binary_mask, save_path=None):
    """
    Find major/minor notches per particle and draw axes.
    Returns a dict: label -> {ex, ey, centroid}.
    """
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be 2D")

    h, w = binary_mask.shape
    num_labels, labels = cv2.connectedComponents(binary_mask)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[binary_mask == 255] = (255, 255, 255)

    axes_info = {}
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        if cv2.countNonZero(component) == 0:
            continue

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue
        contour = contours[0][:, 0, :]

        M = cv2.moments(component)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid = np.array([cx, cy])

        d = np.linalg.norm(contour - centroid, axis=1)
        if d.size == 0:
            continue

        big_idx = np.argmin(d)
        big_notch = contour[big_idx]

        minima_idx = argrelextrema(d, np.less)[0]
        minima_idx = [i for i in minima_idx if i != big_idx]
        if len(minima_idx) == 0:
            continue

        small_idx = minima_idx[np.argmin(d[minima_idx])]
        small_notch = contour[small_idx]

        v = big_notch - centroid
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            continue
        v = v / norm_v
        perp = np.array([v[1], -v[0]])

        diameter = np.max(d) * 2
        axes_info[label] = {
            "ex": perp,
            "ey": v,
            "centroid": centroid,
        }

        p1 = (int(centroid[0] - v[0] * diameter / 2), int(centroid[1] - v[1] * diameter / 2))
        p2 = (int(centroid[0] + v[0] * diameter / 2), int(centroid[1] + v[1] * diameter / 2))
        cv2.line(canvas, p1, p2, (255, 0, 0), 2)

        p3 = (int(centroid[0] - perp[0] * diameter / 2), int(centroid[1] - perp[1] * diameter / 2))
        p4 = (int(centroid[0] + perp[0] * diameter / 2), int(centroid[1] + perp[1] * diameter / 2))
        cv2.line(canvas, p3, p4, (0, 0, 255), 2)

        cv2.circle(canvas, tuple(big_notch), 6, (255, 0, 0), -1)
        cv2.circle(canvas, tuple(small_notch), 6, (0, 0, 255), -1)

    if save_path:
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(canvas[:, :, ::-1])
        ax.axis("off")
        fig.savefig(str(save_path), dpi=100, bbox_inches=None, pad_inches=0)
        plt.close(fig)

    return axes_info


def compute_quadrant_intensity(brightness_image,
                               labels,
                               regions,
                               axes_info,
                               csv_path="quadrant_intensity.csv",
                               id_map_path="particle_id_map.png"):
    """
    Compute mean intensity per quadrant for each particle using
    a sampling circle per quadrant (radius = minor_axis/2, offset = radius).
    """
    h, w = brightness_image.shape
    region_sorted = sorted(regions, key=lambda r: r.centroid[1])
    region_id_map = {r.label: idx + 1 for idx, r in enumerate(region_sorted)}

    base = np.clip(brightness_image, 0, 255)
    id_map = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for r in region_sorted:
        pid = region_id_map[r.label]
        cx, cy = r.centroid[1], r.centroid[0]
        cv2.putText(id_map, str(pid), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    results = []
    warnings = []
    offsets = {"Q1": (+1, +1), "Q2": (-1, +1), "Q3": (-1, -1), "Q4": (+1, -1)}
    color_map = {"Q1": (0, 255, 0), "Q2": (255, 0, 0), "Q3": (0, 255, 255), "Q4": (255, 0, 255)}

    for r in region_sorted:
        label = r.label
        if label not in axes_info:
            warnings.append(f"skip label {label}: missing axes info")
            continue

        pid = region_id_map[label]
        mask = (labels == label)

        ax = axes_info[label]  # stored as (x, y)
        C_xy = np.array(ax["centroid"], dtype=np.float32)
        ex_xy = np.array(ax["ex"], dtype=np.float32)
        ey_xy = np.array(ax["ey"], dtype=np.float32)

        C = np.array([C_xy[1], C_xy[0]], dtype=np.float32)  # (row, col)
        ex_rc = np.array([ex_xy[1], ex_xy[0]], dtype=np.float32)
        ey_rc = np.array([ey_xy[1], ey_xy[0]], dtype=np.float32)

        def _norm(v):
            n = np.linalg.norm(v)
            return v if n == 0 else v / n

        ex_rc = _norm(ex_rc)
        ey_rc = _norm(ey_rc)

        radius = float(r.minor_axis_length) / 2.0
        if radius <= 0:
            warnings.append(f"label {label}: minor_axis_length <= 0")
            results.append([pid, float("nan"), float("nan"), float("nan"), float("nan")])
            continue

        ys, xs = np.where(mask)
        coords = np.stack([ys, xs], axis=1).astype(np.float32)
        vals = brightness_image[mask]

        quadrant_means = []
        for q_name, (sx, sy) in offsets.items():
            center = C + ex_rc * (sx * radius) + ey_rc * (sy * radius)

            dy = coords[:, 0] - center[0]
            dx = coords[:, 1] - center[1]
            in_circle = (dx * dx + dy * dy) <= (radius * radius)

            q_vals = vals[in_circle]
            mean_val = float(np.mean(q_vals)) if q_vals.size > 0 else float("nan")
            quadrant_means.append(mean_val)

            cv2.circle(
                id_map,
                (int(round(center[1])), int(round(center[0]))),
                int(round(radius)),
                color_map[q_name],
                1,
                cv2.LINE_AA
            )

        results.append([pid, *quadrant_means])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["particle_id", "Q1_mean", "Q2_mean", "Q3_mean", "Q4_mean"])
        writer.writerows(results)

    cv2.imwrite(str(id_map_path), id_map)
    return {"warnings": warnings, "csv_path": csv_path, "id_map_path": id_map_path}


def make_A_transparent(A_path, A1_path="A1.png"):
    """
    Make all white pixels transparent.
    """
    A_path = _ensure_path(A_path)
    img = Image.open(A_path).convert("RGB")
    arr = np.array(img)

    alpha = np.ones((arr.shape[0], arr.shape[1]), dtype=np.uint8) * 255
    white_mask = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 255) & (arr[:, :, 2] == 255)
    alpha[white_mask] = 0

    rgba = np.dstack([arr, alpha])
    Image.fromarray(rgba).save(A1_path)
    return A1_path


def overlap_A1_B(A1_path, B_path, C1_path="C1.png"):
    """
    Alpha-composite A1 over B.
    """
    A1_path = _ensure_path(A1_path)
    B_path = _ensure_path(B_path)

    A1 = Image.open(A1_path).convert("RGBA")
    B = Image.open(B_path).convert("RGBA")
    if A1.size != B.size:
        raise ValueError(f"Size mismatch: A1 {A1.size} vs B {B.size}")

    C1 = Image.alpha_composite(B, A1)
    C1.save(C1_path)
    return C1_path


def generate_final_image(A_path, B_path, A1_path="A1.png", C1_path="C1.png"):
    """
    Pipeline: make A transparent, then overlay on B.
    """
    make_A_transparent(A_path, A1_path)
    overlap_A1_B(A1_path, B_path, C1_path)
    return C1_path


def run_full_analysis(ch00_path,
                      brightness_image_path,
                      mask_out="mask.png",
                      axes_out="axes_output.png",
                      csv_out="quadrant_results.csv",
                      id_map_out="particle_id_map.png"):
    """
    Convenience pipeline: mask -> axes -> quadrant intensities.
    """
    mask_img = extract_outer_boundaries(ch00_path, save_path=mask_out)
    num_labels, labels = cv2.connectedComponents(mask_img)
    regions = measure.regionprops(labels)

    axes_info = find_notches_and_axes(mask_img, save_path=axes_out)

    brightness_image_path = _ensure_path(brightness_image_path)
    brightness_image = cv2.imread(str(brightness_image_path), cv2.IMREAD_GRAYSCALE)
    if brightness_image is None:
        raise ValueError(f"Failed to read brightness image: {brightness_image_path}")
    brightness_image = brightness_image.astype(float)

    return compute_quadrant_intensity(
        brightness_image=brightness_image,
        labels=labels,
        regions=regions,
        axes_info=axes_info,
        csv_path=csv_out,
        id_map_path=id_map_out,
    )
