#!/usr/bin/env python3
"""Run full slot processing once and show final original-vs-processed grid.

Pipeline per calibrated slot:
  1) crop from screenshot using board_config.json
  2) edge detection (Canny + Gaussian blur)
  3) centroid-distance filtering on edge pixels
  4) optional tiny-component cleanup

The script uses parameters from a JSON file (default: edge_centroid_compare_params.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from calibrate import _ensure_exapunks_fullscreen
from edge_detection_slots_tuner import (
    LABEL_H,
    edge_map,
    load_config,
    capture_all_crops,
    upscale_to_min,
    odd_kernel_from_slider,
)
from edge_detection_centroid_compare import filter_edges_by_centroid_distance


DEFAULT_CONFIG = Path("board_config.json")
DEFAULT_PARAMS = Path("edge_centroid_compare_params.json")
DEFAULT_OUT = Path("final_processed_slots_grid.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full edge+centroid filtering over calibrated slots and show final grid."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Calibration config JSON.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS, help="Processing params JSON.")
    parser.add_argument("--save-image", type=Path, default=DEFAULT_OUT, help="Where to save final grid image.")
    parser.add_argument("--no-save", action="store_true", help="Do not save image to disk.")
    parser.add_argument("--no-display", action="store_true", help="Do not open preview window.")
    parser.add_argument("--tile-gap", type=int, default=6, help="Gap between tiles.")
    return parser.parse_args()


def load_params(path: Path) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Params not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either edge-only params or edge+centroid params.
    low = int(data.get("low_threshold", 60))
    high = int(data.get("high_threshold", 180))
    blur_slider = int(data.get("blur_slider", 1))
    min_radius_pct = int(data.get("min_radius_pct", 0))
    max_radius_pct = int(data.get("max_radius_pct", 100))
    min_component_area = int(data.get("min_component_area", 0))

    high = min(255, max(low + 1, high))
    min_radius_pct = min(100, max(0, min_radius_pct))
    max_radius_pct = min(100, max(min_radius_pct, max_radius_pct))
    min_component_area = max(0, min_component_area)

    return {
        "low_threshold": low,
        "high_threshold": high,
        "blur_slider": blur_slider,
        "min_radius_pct": min_radius_pct,
        "max_radius_pct": max_radius_pct,
        "min_component_area": min_component_area,
    }


def annotate_tile(original: np.ndarray, processed: np.ndarray, label: str) -> np.ndarray:
    orig_up = upscale_to_min(original)
    h, w = orig_up.shape[:2]

    proc_up = cv2.resize(processed, (w, h), interpolation=cv2.INTER_NEAREST)
    proc_bgr = cv2.cvtColor(proc_up, cv2.COLOR_GRAY2BGR)
    proc_bgr[proc_up > 0] = (0, 255, 0)

    side_by_side = cv2.hconcat([orig_up, proc_bgr])

    strip = np.zeros((LABEL_H, side_by_side.shape[1], 3), dtype=np.uint8)
    cv2.putText(strip, label, (4, LABEL_H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return cv2.vconcat([side_by_side, strip])


def build_final_grid(crops: List[List[np.ndarray]], params: Dict[str, int], tile_gap: int) -> np.ndarray:
    cols = len(crops)
    rows = len(crops[0]) if cols > 0 else 0
    if cols == 0 or rows == 0:
        return np.zeros((220, 420, 3), dtype=np.uint8)

    sample = upscale_to_min(crops[0][0])
    disp_h, disp_w = sample.shape[:2]
    tile_w = disp_w * 2
    tile_h = disp_h + LABEL_H
    header_h = 34

    canvas_h = header_h + rows * tile_h + (rows + 1) * tile_gap
    canvas_w = cols * tile_w + (cols + 1) * tile_gap
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)

    for c in range(cols):
        for r in range(rows):
            crop = crops[c][r]
            edges = edge_map(
                crop,
                low=params["low_threshold"],
                high=params["high_threshold"],
                blur_slider=params["blur_slider"],
            )
            filtered = filter_edges_by_centroid_distance(
                edges,
                min_radius_pct=params["min_radius_pct"],
                max_radius_pct=params["max_radius_pct"],
                min_component_area=params["min_component_area"],
            )
            tile = annotate_tile(crop, filtered, label=f"c{c} r{r}")

            y1 = header_h + tile_gap + r * (tile_h + tile_gap)
            x1 = tile_gap + c * (tile_w + tile_gap)
            canvas[y1:y1 + tile_h, x1:x1 + tile_w] = tile

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], header_h), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "Left: original  |  Right: final processed",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    slot_boxes: List[List[Dict[str, int]]] = cfg["slot_boxes"]
    if not slot_boxes or not slot_boxes[0]:
        raise ValueError("Config slot_boxes is empty.")

    params = load_params(args.params)
    print(
        "Using params: "
        f"low={params['low_threshold']} high={params['high_threshold']} "
        f"blur_k={odd_kernel_from_slider(params['blur_slider'])} "
        f"min_r%={params['min_radius_pct']} max_r%={params['max_radius_pct']} "
        f"min_area={params['min_component_area']}"
    )

    _ensure_exapunks_fullscreen()
    print("Capturing screenshot and cropping calibrated slots...")
    crops = capture_all_crops(slot_boxes)

    grid = build_final_grid(crops, params=params, tile_gap=max(0, args.tile_gap))

    if not args.no_save:
        args.save_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_image), grid)
        print(f"Saved final grid to {args.save_image}")

    if not args.no_display:
        cv2.namedWindow("Final Processed Slots", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Final Processed Slots", 1700, 950)
        cv2.imshow("Final Processed Slots", grid)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
