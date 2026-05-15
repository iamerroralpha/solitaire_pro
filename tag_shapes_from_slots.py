#!/usr/bin/env python3
"""Interactive shape tagging from calibrated EXAPUNKS slots.

Workflow:
1) Ensure EXAPUNKS is fullscreen and capture current screen.
2) Crop all slots from board_config.json.
3) Process each slot with the same pipeline used in final processed view:
   - edge detection
   - centroid-distance filtering
   - zero-border trimming
4) Ask user to label each processed shape as one of:
   6, 7, 8, 9, 10, fh, fd, fs, fc
5) Save processed shape mask image only.

Naming:
  label_1, label_2, label_3, label_4
Example:
  6_1.png, 6_2.png, ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from calibrate import _ensure_exapunks_fullscreen
from edge_detection_slots_tuner import (
    LABEL_H,
    capture_all_crops,
    edge_map,
    load_config,
    odd_kernel_from_slider,
    upscale_to_min,
)
from edge_detection_centroid_compare import filter_edges_by_centroid_distance


ALLOWED_LABELS: List[str] = ["6", "7", "8", "9", "10", "fh", "fd", "fs", "fc"]
DEFAULT_CONFIG = Path("board_config.json")
DEFAULT_PARAMS = Path("edge_centroid_compare_params.json")
DEFAULT_OUT_ROOT = Path("tagged_shapes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag processed slot shapes for classifier dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Calibration config JSON.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS, help="Processing params JSON.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output dataset root folder.")
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=4,
        help="Maximum number of examples per label (default: 4).",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Also prompt for slots whose processed mask is empty.",
    )
    return parser.parse_args()


def load_params(path: Path) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

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


def trim_zero_borders(edge_img: np.ndarray) -> np.ndarray:
    ys, xs = np.where(edge_img > 0)
    if xs.size == 0:
        return edge_img
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    return edge_img[y1:y2, x1:x2]


def build_processed_mask(crop: np.ndarray, params: Dict[str, int]) -> np.ndarray:
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
    return trim_zero_borders(filtered)


def build_example_image(crop: np.ndarray, processed_mask: np.ndarray) -> np.ndarray:
    orig_up = upscale_to_min(crop)
    disp_h, disp_w = orig_up.shape[:2]

    proc_up = cv2.resize(processed_mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    proc_bgr = cv2.cvtColor(proc_up, cv2.COLOR_GRAY2BGR)
    proc_bgr[proc_up > 0] = (0, 255, 0)

    side = cv2.hconcat([orig_up, proc_bgr])
    strip = np.zeros((LABEL_H, side.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        strip,
        "Left: original  |  Right: processed",
        (4, LABEL_H - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([side, strip])


def next_index_for_label(out_dir: Path, label: str, max_per_label: int) -> Optional[int]:
    used = set()
    for p in out_dir.glob(f"{label}_*.png"):
        stem = p.stem
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        if parts[0] != label:
            continue
        try:
            used.add(int(parts[1]))
        except ValueError:
            continue

    for i in range(1, max_per_label + 1):
        if i not in used:
            return i
    return None


def prompt_label() -> str:
    msg = "Label [6/7/8/9/10/fh/fd/fs/fc], s=skip, q=quit: "
    return input(msg).strip().lower()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    params = load_params(args.params)
    slot_boxes: List[List[Dict[str, int]]] = cfg["slot_boxes"]

    if not slot_boxes or not slot_boxes[0]:
        raise ValueError("Config slot_boxes is empty.")

    shapes_dir = args.out_root / "processed_shapes"
    shapes_dir.mkdir(parents=True, exist_ok=True)

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

    # Flatten slots in deterministic order by column then row.
    flat_slots: List[Tuple[int, int, np.ndarray]] = []
    for c, col in enumerate(crops):
        for r, crop in enumerate(col):
            flat_slots.append((c, r, crop))

    win = "Shape Tagging Preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 500)

    saved_count = 0
    for idx, (col, row, crop) in enumerate(flat_slots, start=1):
        processed = build_processed_mask(crop, params)
        nonzero = int(np.count_nonzero(processed))

        if nonzero == 0 and not args.include_empty:
            print(f"[{idx}/{len(flat_slots)}] c{col} r{row}: empty processed mask -> auto-skip")
            continue

        preview = build_example_image(crop, processed)
        header = np.zeros((30, preview.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header,
            f"Slot c{col} r{row}   processed_pixels={nonzero}",
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        preview_show = cv2.vconcat([header, preview])
        cv2.imshow(win, preview_show)
        cv2.waitKey(1)

        while True:
            label = prompt_label()
            if label == "q":
                cv2.destroyAllWindows()
                print("Stopped by user.")
                print(f"Saved {saved_count} sample(s).")
                return
            if label in ("", "s", "skip"):
                break
            if label not in ALLOWED_LABELS:
                print(f"Invalid label '{label}'. Allowed: {', '.join(ALLOWED_LABELS)}")
                continue

            sample_idx = next_index_for_label(shapes_dir, label, args.max_per_label)
            if sample_idx is None:
                print(f"Label {label} already has {args.max_per_label} samples. Skipping.")
                break

            base_name = f"{label}_{sample_idx}.png"
            shape_path = shapes_dir / base_name

            cv2.imwrite(str(shape_path), processed)

            saved_count += 1
            print(f"Saved {base_name}")
            break

    cv2.destroyAllWindows()
    print(f"Done. Saved {saved_count} sample(s).")


if __name__ == "__main__":
    main()
