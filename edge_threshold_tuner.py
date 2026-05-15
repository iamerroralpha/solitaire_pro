#!/usr/bin/env python3
"""Interactive edge-detection tuner for simple high-contrast images.

Usage:
    python edge_threshold_tuner.py path/to/image.png

Controls:
    - Slider "Threshold": tune edge sensitivity.
    - Slider "Blur": Gaussian blur kernel size (odd values only).
    - Press 's' to save current edge map.
    - Press 'q' or ESC to quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive edge threshold tuner.")
    parser.add_argument("image", type=Path, help="Path to input image.")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("edge_output.png"),
        help="Output path when pressing 's'.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="Edge Threshold Tuner",
        help="Window title.",
    )
    return parser.parse_args()


def odd_kernel_from_slider(v: int) -> int:
    # Map slider values to odd kernel sizes: 1, 3, 5, ...
    return max(1, (2 * int(v)) + 1)


def compute_edges(gray: np.ndarray, threshold: int, blur_slider: int) -> np.ndarray:
    k = odd_kernel_from_slider(blur_slider)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    # Single-threshold tuning: upper is derived for stable Canny behavior.
    low = max(0, int(threshold))
    high = min(255, int(max(low + 1, low * 3)))
    return cv2.Canny(blurred, low, high, L2gradient=True)


def build_preview(original_bgr: np.ndarray, edges: np.ndarray, threshold: int, blur_slider: int) -> np.ndarray:
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = original_bgr.copy()
    overlay[edges > 0] = (0, 255, 0)

    top = cv2.hconcat([original_bgr, edges_bgr, overlay])
    text = f"Threshold={threshold}  BlurKernel={odd_kernel_from_slider(blur_slider)}"
    cv2.putText(top, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        top,
        "q/ESC: quit    s: save edges",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return top


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window, 1400, 700)

    # Threshold starts in a useful range for high-contrast glyph-like images.
    cv2.createTrackbar("Threshold", args.window, 60, 255, lambda _v: None)
    cv2.createTrackbar("Blur", args.window, 1, 10, lambda _v: None)

    while True:
        threshold = cv2.getTrackbarPos("Threshold", args.window)
        blur_slider = cv2.getTrackbarPos("Blur", args.window)

        edges = compute_edges(gray, threshold=threshold, blur_slider=blur_slider)
        preview = build_preview(image, edges, threshold=threshold, blur_slider=blur_slider)

        cv2.imshow(args.window, preview)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord("s"):
            args.save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save), edges)
            print(f"Saved edge map to {args.save}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
