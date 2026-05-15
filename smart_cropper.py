#!/usr/bin/env python3
"""Center-out smart cropper using fixed Canny settings.

Pipeline:
1) Grayscale + Gaussian blur (blur slider value 1 -> kernel 3)
2) Canny with threshold=255 (same logic as edge_threshold_tuner)
3) Scan from image center outward in 4 directions (up/down/left/right)
4) Ignore first 10% of each scan path to reduce false positives
5) Trim crop exactly at first stable edge in each direction
6) Visualize:
   - edge map
   - compared scan lines
   - detected crop lines/points
   - cropped result
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart crop from center-out edge scanning.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--ignore-ratio", type=float, default=0.10, help="Fraction of initial scan path to ignore.")
    parser.add_argument(
        "--save-crop",
        type=Path,
        default=None,
        help="Optional output path for cropped image.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="Smart Cropper",
        help="Display window title.",
    )
    return parser.parse_args()


def odd_kernel_from_slider(v: int) -> int:
    return max(1, (2 * int(v)) + 1)


def compute_edges(gray: np.ndarray) -> np.ndarray:
    """Use same edge logic as edge_threshold_tuner with fixed values.

    Fixed values requested by user:
    - threshold=255
    - blur slider=1
    """
    threshold = 255
    blur_slider = 1

    k = odd_kernel_from_slider(blur_slider)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    low = max(0, int(threshold))
    high = min(255, int(max(low + 1, low * 3)))
    return cv2.Canny(blurred, low, high, L2gradient=True)


def stable_edge_vertical(edges: np.ndarray, y: int, x: int, band: int = 3, min_hits: int = 2) -> bool:
    h, w = edges.shape[:2]
    x1 = max(0, x - band)
    x2 = min(w, x + band + 1)
    y1 = max(0, y - 1)
    y2 = min(h, y + 2)
    roi = edges[y1:y2, x1:x2]
    return int(np.count_nonzero(roi)) >= min_hits


def stable_edge_horizontal(edges: np.ndarray, y: int, x: int, band: int = 3, min_hits: int = 2) -> bool:
    h, w = edges.shape[:2]
    y1 = max(0, y - band)
    y2 = min(h, y + band + 1)
    x1 = max(0, x - 1)
    x2 = min(w, x + 2)
    roi = edges[y1:y2, x1:x2]
    return int(np.count_nonzero(roi)) >= min_hits


def scan_up(edges: np.ndarray, cx: int, cy: int, ignore_ratio: float) -> int:
    path_len = cy + 1
    ignore = int(path_len * ignore_ratio)
    for step in range(ignore, path_len):
        y = cy - step
        if y < 0:
            break
        if stable_edge_vertical(edges, y, cx):
            return y
    return 0


def scan_down(edges: np.ndarray, cx: int, cy: int, ignore_ratio: float) -> int:
    h = edges.shape[0]
    path_len = h - cy
    ignore = int(path_len * ignore_ratio)
    for step in range(ignore, path_len):
        y = cy + step
        if y >= h:
            break
        if stable_edge_vertical(edges, y, cx):
            return y
    return h - 1


def scan_left(edges: np.ndarray, cx: int, cy: int, ignore_ratio: float) -> int:
    path_len = cx + 1
    ignore = int(path_len * ignore_ratio)
    for step in range(ignore, path_len):
        x = cx - step
        if x < 0:
            break
        if stable_edge_horizontal(edges, cy, x):
            return x
    return 0


def scan_right(edges: np.ndarray, cx: int, cy: int, ignore_ratio: float) -> int:
    w = edges.shape[1]
    path_len = w - cx
    ignore = int(path_len * ignore_ratio)
    for step in range(ignore, path_len):
        x = cx + step
        if x >= w:
            break
        if stable_edge_horizontal(edges, cy, x):
            return x
    return w - 1


def detect_crop_bounds(edges: np.ndarray, ignore_ratio: float) -> Dict[str, int]:
    h, w = edges.shape[:2]
    cx = w // 2
    cy = h // 2

    top = scan_up(edges, cx, cy, ignore_ratio)
    bottom = scan_down(edges, cx, cy, ignore_ratio)
    left = scan_left(edges, cx, cy, ignore_ratio)
    right = scan_right(edges, cx, cy, ignore_ratio)

    if top >= bottom:
        top, bottom = 0, h - 1
    if left >= right:
        left, right = 0, w - 1

    return {
        "cx": cx,
        "cy": cy,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def draw_scan_lines(image: np.ndarray, bounds: Dict[str, int], ignore_ratio: float) -> np.ndarray:
    out = image.copy()
    h, w = image.shape[:2]
    cx = bounds["cx"]
    cy = bounds["cy"]

    # Center cross used for scans.
    cv2.line(out, (cx, 0), (cx, h - 1), (255, 255, 0), 1)
    cv2.line(out, (0, cy), (w - 1, cy), (255, 255, 0), 1)

    # Ignored central zones (10% by default), drawn in gray.
    up_ig = int((cy + 1) * ignore_ratio)
    dn_ig = int((h - cy) * ignore_ratio)
    lf_ig = int((cx + 1) * ignore_ratio)
    rt_ig = int((w - cx) * ignore_ratio)

    cv2.line(out, (cx, cy), (cx, max(0, cy - up_ig)), (100, 100, 100), 2)
    cv2.line(out, (cx, cy), (cx, min(h - 1, cy + dn_ig)), (100, 100, 100), 2)
    cv2.line(out, (cx, cy), (max(0, cx - lf_ig), cy), (100, 100, 100), 2)
    cv2.line(out, (cx, cy), (min(w - 1, cx + rt_ig), cy), (100, 100, 100), 2)

    # Active scan segments (outside ignored zone).
    cv2.line(out, (cx, max(0, cy - up_ig)), (cx, 0), (0, 255, 255), 2)
    cv2.line(out, (cx, min(h - 1, cy + dn_ig)), (cx, h - 1), (0, 255, 255), 2)
    cv2.line(out, (max(0, cx - lf_ig), cy), (0, cy), (0, 255, 255), 2)
    cv2.line(out, (min(w - 1, cx + rt_ig), cy), (w - 1, cy), (0, 255, 255), 2)

    cv2.putText(out, "Scan lines (gray=ignored zone)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_crop_detection(image: np.ndarray, bounds: Dict[str, int]) -> np.ndarray:
    out = image.copy()
    top = bounds["top"]
    bottom = bounds["bottom"]
    left = bounds["left"]
    right = bounds["right"]
    cx = bounds["cx"]
    cy = bounds["cy"]

    # Crop rectangle and boundary lines.
    cv2.rectangle(out, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.line(out, (0, top), (out.shape[1] - 1, top), (0, 200, 0), 1)
    cv2.line(out, (0, bottom), (out.shape[1] - 1, bottom), (0, 200, 0), 1)
    cv2.line(out, (left, 0), (left, out.shape[0] - 1), (0, 200, 0), 1)
    cv2.line(out, (right, 0), (right, out.shape[0] - 1), (0, 200, 0), 1)

    # Cropping points on scan axes.
    cv2.circle(out, (cx, top), 5, (0, 0, 255), -1)
    cv2.circle(out, (cx, bottom), 5, (0, 0, 255), -1)
    cv2.circle(out, (left, cy), 5, (0, 0, 255), -1)
    cv2.circle(out, (right, cy), 5, (0, 0, 255), -1)
    cv2.circle(out, (cx, cy), 4, (255, 255, 0), -1)

    cv2.putText(out, "Detected crop points + rectangle", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_to(img: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    tw, th = target_wh
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


def build_panel(edges: np.ndarray, scan_vis: np.ndarray, detect_vis: np.ndarray, crop: np.ndarray) -> np.ndarray:
    h, w = scan_vis.shape[:2]
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_bgr = resize_to(edges_bgr, (w, h))

    crop_bgr = crop
    if crop_bgr.size == 0:
        crop_bgr = np.zeros_like(scan_vis)
    else:
        crop_bgr = resize_to(crop_bgr, (w, h))
    cv2.putText(crop_bgr, "Cropped image", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(edges_bgr, "Edge map", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    top_row = cv2.hconcat([edges_bgr, scan_vis])
    bot_row = cv2.hconcat([detect_vis, crop_bgr])
    return cv2.vconcat([top_row, bot_row])


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = compute_edges(gray)

    bounds = detect_crop_bounds(edges, ignore_ratio=float(np.clip(args.ignore_ratio, 0.0, 0.49)))
    x1, x2 = bounds["left"], bounds["right"] + 1
    y1, y2 = bounds["top"], bounds["bottom"] + 1
    crop = image[y1:y2, x1:x2].copy()

    print(
        f"Crop bounds: left={bounds['left']}, right={bounds['right']}, "
        f"top={bounds['top']}, bottom={bounds['bottom']}"
    )
    print(f"Cropped size: {crop.shape[1]}x{crop.shape[0]}")

    if args.save_crop is not None:
        args.save_crop.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_crop), crop)
        print(f"Saved cropped image to {args.save_crop}")

    scan_vis = draw_scan_lines(image, bounds, ignore_ratio=float(np.clip(args.ignore_ratio, 0.0, 0.49)))
    detect_vis = draw_crop_detection(image, bounds)
    panel = build_panel(edges, scan_vis, detect_vis, crop)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window, 1400, 900)
    cv2.imshow(args.window, panel)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
