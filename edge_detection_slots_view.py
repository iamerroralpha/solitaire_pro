#!/usr/bin/env python3
"""Capture screen, apply saved edge-detection parameters to all calibrated slots, show result.

Usage:
    python edge_detection_slots_view.py
    python edge_detection_slots_view.py --config board_config.json --params edge_params.json
    python edge_detection_slots_view.py --save-image edge_result.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import ImageGrab

from calibrate import _ensure_exapunks_fullscreen
from edge_detection_slots_tuner import (
    DISPLAY_MIN_W,
    DISPLAY_MIN_H,
    LABEL_H,
    upscale_to_min,
    odd_kernel_from_slider,
    capture_all_crops,
    build_mosaic,
    add_footer,
)


DEFAULT_CONFIG = Path("board_config.json")
DEFAULT_PARAMS = Path("edge_params.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show edge detection on all calibrated slots using saved parameters."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument(
        "--save-image",
        type=Path,
        default=None,
        help="Optional path to save the result image.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the display window (useful with --save-image).",
    )
    return parser.parse_args()


def load_json(path: Path, name: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    config = load_json(args.config, "Board config")
    params = load_json(args.params, "Edge params")

    slot_boxes: List[List[Dict]] = config["slot_boxes"]
    low = int(params["low_threshold"])
    high = int(params["high_threshold"])
    blur_slider = int(params["blur_slider"])

    print(f"Parameters: low={low}  high={high}  blur_kernel={odd_kernel_from_slider(blur_slider)}")

    _ensure_exapunks_fullscreen()
    print("Capturing screenshot and cropping slots...")
    crops = capture_all_crops(slot_boxes)

    mosaic = build_mosaic(crops=crops, low=low, high=high, blur_slider=blur_slider, tile_gap=6)
    frame = add_footer(mosaic, low=low, high=high, blur_slider=blur_slider)

    if args.save_image is not None:
        args.save_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_image), frame)
        print(f"Saved result image to {args.save_image}")

    if not args.no_display:
        cv2.namedWindow("Edge Detection — Slots", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Edge Detection — Slots", 1600, 900)
        cv2.imshow("Edge Detection — Slots", frame)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
