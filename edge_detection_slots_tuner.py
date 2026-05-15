#!/usr/bin/env python3
"""Tune edge-detection parameters over all calibrated slot boxes.

This script uses `board_config.json` from calibration, captures the current screen,
crops every configured slot, and displays edge results for all slots while you tune
Canny + blur parameters with sliders.

Controls:
  - Sliders in "Controls" window:
      * low_threshold
      * high_threshold
      * blur (kernel = 2*blur + 1)
      * view_scale_pct
  - r: recapture screenshot and re-crop all slots
  - s: print current parameters (and optionally save them)
  - q / ESC: quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import ImageGrab

# Re-use the EXAPUNKS fullscreen helper from calibrate.py
from calibrate import _ensure_exapunks_fullscreen


CONFIG_DEFAULT = Path("board_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive edge tuner for all calibrated slot boxes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DEFAULT,
        help="Path to calibration config JSON.",
    )
    parser.add_argument(
        "--save-params",
        type=Path,
        default=None,
        help="Optional JSON path to save tuned parameters when pressing 's'.",
    )
    parser.add_argument(
        "--tile-gap",
        type=int,
        default=6,
        help="Gap (pixels) between tiles.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "slot_boxes" not in data:
        raise ValueError("Invalid config: missing 'slot_boxes'.")
    return data


def capture_screen_bgr() -> np.ndarray:
    screenshot = ImageGrab.grab()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


def crop_slot(img: np.ndarray, slot: Dict[str, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, int(slot["x"]))
    y1 = max(0, int(slot["y"]))
    x2 = min(w, x1 + int(slot["w"]))
    y2 = min(h, y1 + int(slot["h"]))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2].copy()


DISPLAY_MIN_W = 80   # minimum tile display width in pixels
DISPLAY_MIN_H = 80   # minimum tile display height in pixels
LABEL_H = 18         # pixels reserved below each tile for the label


def upscale_to_min(img: np.ndarray) -> np.ndarray:
    """Upscale image to at least DISPLAY_MIN_W x DISPLAY_MIN_H, preserving aspect ratio."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((DISPLAY_MIN_H, DISPLAY_MIN_W, 3), dtype=np.uint8)
    scale = max(DISPLAY_MIN_W / max(w, 1), DISPLAY_MIN_H / max(h, 1))
    if scale > 1.0:
        new_w = max(DISPLAY_MIN_W, int(round(w * scale)))
        new_h = max(DISPLAY_MIN_H, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img


def odd_kernel_from_slider(v: int) -> int:
    return max(1, 2 * int(v) + 1)


def edge_map(crop_bgr: np.ndarray, low: int, high: int, blur_slider: int) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    k = odd_kernel_from_slider(blur_slider)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    low = max(0, int(low))
    high = max(low + 1, int(high))
    high = min(255, high)
    return cv2.Canny(blurred, low, high, L2gradient=True)


def annotate_tile(original: np.ndarray, edges: np.ndarray, label: str) -> np.ndarray:
    # Upscale both images to a useful viewing size before concatenating.
    orig_up = upscale_to_min(original)
    disp_h, disp_w = orig_up.shape[:2]

    edges_up = cv2.resize(edges, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    edges_bgr = cv2.cvtColor(edges_up, cv2.COLOR_GRAY2BGR)
    edges_bgr[edges_up > 0] = (0, 255, 0)

    side_by_side = cv2.hconcat([orig_up, edges_bgr])

    # Add label strip below the image.
    label_strip = np.zeros((LABEL_H, side_by_side.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        label_strip,
        label,
        (4, LABEL_H - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([side_by_side, label_strip])


def build_mosaic(
    crops: List[List[np.ndarray]],
    low: int,
    high: int,
    blur_slider: int,
    tile_gap: int,
) -> np.ndarray:
    cols = len(crops)
    rows = len(crops[0]) if cols > 0 else 0
    if cols == 0 or rows == 0:
        return np.zeros((200, 400, 3), dtype=np.uint8)

    # Derive tile size from a representative upscaled crop.
    sample = upscale_to_min(crops[0][0])
    disp_h, disp_w = sample.shape[:2]
    tile_w = disp_w * 2          # original + edges side by side
    tile_h = disp_h + LABEL_H   # image + label strip

    canvas_h = rows * tile_h + (rows + 1) * tile_gap
    canvas_w = cols * tile_w + (cols + 1) * tile_gap
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)

    for c in range(cols):
        for r in range(rows):
            crop = crops[c][r]
            edges = edge_map(crop, low=low, high=high, blur_slider=blur_slider)
            tile = annotate_tile(crop, edges, label=f"c{c} r{r}")

            y1 = tile_gap + r * (tile_h + tile_gap)
            x1 = tile_gap + c * (tile_w + tile_gap)
            canvas[y1:y1 + tile_h, x1:x1 + tile_w] = tile

    return canvas


def add_footer(img: np.ndarray, low: int, high: int, blur_slider: int) -> np.ndarray:
    out = img.copy()
    footer_h = 36
    canvas = np.full((out.shape[0] + footer_h, out.shape[1], 3), 20, dtype=np.uint8)
    canvas[: out.shape[0], :] = out

    text = (
        f"low={low}  high={high}  blur_kernel={odd_kernel_from_slider(blur_slider)}    "
        "r=recapture  s=save/print  q/esc=quit"
    )
    cv2.putText(
        canvas,
        text,
        (10, out.shape[0] + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def capture_all_crops(slot_boxes: List[List[Dict[str, int]]]) -> List[List[np.ndarray]]:
    screen = capture_screen_bgr()
    out: List[List[np.ndarray]] = []
    for col_slots in slot_boxes:
        col_crops: List[np.ndarray] = []
        for slot in col_slots:
            col_crops.append(crop_slot(screen, slot))
        out.append(col_crops)
    return out


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    slot_boxes: List[List[Dict[str, int]]] = config["slot_boxes"]

    if not slot_boxes or not slot_boxes[0]:
        raise ValueError("Config slot_boxes is empty.")

    _ensure_exapunks_fullscreen()
    print("Capturing initial screenshot and crops from calibrated slot boxes...")
    crops = capture_all_crops(slot_boxes)

    controls_win = "Controls"
    view_win = "Slot Edge Detection Tuner"

    cv2.namedWindow(controls_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(view_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(view_win, 1600, 900)

    cv2.createTrackbar("low_threshold", controls_win, 60, 255, lambda _v: None)
    cv2.createTrackbar("high_threshold", controls_win, 180, 255, lambda _v: None)
    cv2.createTrackbar("blur", controls_win, 1, 12, lambda _v: None)
    cv2.createTrackbar("view_scale_pct", controls_win, 100, 200, lambda _v: None)

    while True:
        low = cv2.getTrackbarPos("low_threshold", controls_win)
        high = cv2.getTrackbarPos("high_threshold", controls_win)
        blur_slider = cv2.getTrackbarPos("blur", controls_win)
        scale_pct = max(20, cv2.getTrackbarPos("view_scale_pct", controls_win))

        mosaic = build_mosaic(
            crops=crops,
            low=low,
            high=high,
            blur_slider=blur_slider,
            tile_gap=max(0, args.tile_gap),
        )
        frame = add_footer(mosaic, low=low, high=high, blur_slider=blur_slider)

        if scale_pct != 100:
            w = int(frame.shape[1] * (scale_pct / 100.0))
            h = int(frame.shape[0] * (scale_pct / 100.0))
            frame = cv2.resize(frame, (max(1, w), max(1, h)), interpolation=cv2.INTER_AREA)

        cv2.imshow(view_win, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord("r"):
            print("Recapturing screenshot and slot crops...")
            _ensure_exapunks_fullscreen()
            crops = capture_all_crops(slot_boxes)
        if key == ord("s"):
            payload = {
                "low_threshold": int(low),
                "high_threshold": int(max(low + 1, high)),
                "blur_slider": int(blur_slider),
                "blur_kernel": int(odd_kernel_from_slider(blur_slider)),
            }
            print(f"Current parameters: {payload}")
            if args.save_params is not None:
                args.save_params.parent.mkdir(parents=True, exist_ok=True)
                with args.save_params.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                print(f"Saved parameters to {args.save_params}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
