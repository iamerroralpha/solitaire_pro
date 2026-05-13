#!/usr/bin/env python3
"""Calibrate EXAPUNKS solitaire board geometry for peek-strip card detection.

5-click calibration sequence
─────────────────────────────
  1. Board top-left corner       (overall boundary)
  2. Board bottom-right corner   (overall boundary)
  3. Top-left pixel of card at col 0, row 0  (anchor)
  4. Top-left pixel of card at col 1, row 0  → measures column spacing
  5. Top-left pixel of card at col 0, row 1  → measures stack/row spacing

From those five points the script derives:
  col_spacing = p4.x − p3.x
  row_spacing = p5.y − p3.y
  peek_w      = col_spacing − peek_w_margin  (pixels)
  peek_h      = row_spacing − peek_h_margin  (pixels)
  slot boxes  = one rectangle per (col, row) anchored from p3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import ImageGrab

CONFIG_PATH = Path("board_config.json")

# Labels shown in the overlay for each of the 5 clicks
_CLICK_PROMPTS: List[str] = [
    "1/5  Board TOP-LEFT corner",
    "2/5  Board BOTTOM-RIGHT corner",
    "3/5  TOP-LEFT of card  col=0, row=0  (top-left card)",
    "4/5  TOP-LEFT of card  col=1, row=0  (first card, second column)",
    "5/5  TOP-LEFT of card  col=0, row=1  (second card, first column)",
]


class _MultiClickSelector:
    """Captures N labeled clicks on a screenshot."""

    def __init__(self, prompts: List[str]) -> None:
        self.prompts = prompts
        self.points: List[Tuple[int, int]] = []
        self.image: Optional[np.ndarray] = None
        self._display: Optional[np.ndarray] = None
        self._win = "Calibration — click to place points"

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        idx = len(self.points)
        if idx >= len(self.prompts):
            return
        self.points.append((x, y))
        cv2.circle(self._display, (x, y), 7, (0, 255, 0), -1)
        cv2.circle(self._display, (x, y), 9, (255, 255, 255), 1)
        self._show_prompt()

    def _show_prompt(self) -> None:
        idx = len(self.points)
        out = self._display.copy()
        if idx < len(self.prompts):
            text = self.prompts[idx]
        else:
            text = "All points collected — press Y to save, ESC to cancel"
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(self._win, out)

    def run(self) -> Optional[List[Tuple[int, int]]]:
        print("\nCapturing screen…")
        screenshot = ImageGrab.grab()
        self.image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        self._display = self.image.copy()

        cv2.namedWindow(self._win)
        cv2.setMouseCallback(self._win, self._mouse_cb)
        self._show_prompt()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                print("Cancelled.")
                cv2.destroyWindow(self._win)
                return None
            if len(self.points) == len(self.prompts):
                # Re-render final state (all dots visible) and wait for Y/ESC
                self._show_prompt()
                while True:
                    k2 = cv2.waitKey(0) & 0xFF
                    if k2 in (ord("y"), ord("Y")):
                        cv2.destroyWindow(self._win)
                        return self.points
                    if k2 == 27:
                        print("Cancelled.")
                        cv2.destroyWindow(self._win)
                        return None


def _build_grid(
    board_p1: Tuple[int, int],
    board_p2: Tuple[int, int],
    anchor: Tuple[int, int],
    col1_anchor: Tuple[int, int],
    row1_anchor: Tuple[int, int],
    peek_w_margin: int,
    peek_h_margin: int,
    cols: int = 9,
    rows: int = 4,
) -> Dict[str, object]:
    x1 = min(board_p1[0], board_p2[0])
    y1 = min(board_p1[1], board_p2[1])
    x2 = max(board_p1[0], board_p2[0])
    y2 = max(board_p1[1], board_p2[1])

    col_spacing = col1_anchor[0] - anchor[0]
    row_spacing = row1_anchor[1] - anchor[1]

    if col_spacing <= 0 or row_spacing <= 0:
        raise ValueError(
            f"Bad reference clicks: col_spacing={col_spacing}, row_spacing={row_spacing}. "
            "Make sure col=1 is to the right of col=0 and row=1 is below row=0."
        )

    peek_w = max(4, col_spacing - peek_w_margin)
    peek_h = max(4, row_spacing - peek_h_margin)

    slots: List[List[Dict[str, int]]] = []
    col_lefts: List[int] = []
    row_tops: List[int] = []

    for col in range(cols):
        col_lefts.append(anchor[0] + col_spacing * col)

    for row in range(rows):
        row_tops.append(anchor[1] + row_spacing * row)

    for col in range(cols):
        column_slots: List[Dict[str, int]] = []
        for row in range(rows):
            # Center the peek strip horizontally within the column slot
            sx = col_lefts[col] + (col_spacing - peek_w) // 2
            sy = row_tops[row]
            column_slots.append({"x": int(sx), "y": int(sy), "w": int(peek_w), "h": int(peek_h)})
        slots.append(column_slots)

    return {
        "board_rect": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "rows": rows,
        "cols": cols,
        "col_spacing": col_spacing,
        "row_spacing": row_spacing,
        "anchor": {"x": anchor[0], "y": anchor[1]},
        "col_lefts": col_lefts,
        "row_tops": row_tops,
        "peek_w": peek_w,
        "peek_h": peek_h,
        "slot_boxes": slots,
    }


def _draw_overlay(image: np.ndarray, config: Dict[str, object]) -> np.ndarray:
    out = image.copy()
    rect = config["board_rect"]

    cv2.rectangle(
        out,
        (rect["x1"], rect["y1"]),
        (rect["x2"], rect["y2"]),
        (0, 255, 255),
        2,
    )

    for col_idx, col_slots in enumerate(config["slot_boxes"]):
        for row_idx, slot in enumerate(col_slots):
            tl = (slot["x"], slot["y"])
            br = (slot["x"] + slot["w"], slot["y"] + slot["h"])
            cv2.rectangle(out, tl, br, (0, 255, 0), 1)
            cv2.putText(
                out,
                f"{col_idx},{row_idx}",
                (slot["x"], slot["y"] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(
        out,
        "Preview — press Y to save, ESC to cancel",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def run_calibration(args: argparse.Namespace) -> bool:
    selector = _MultiClickSelector(_CLICK_PROMPTS)
    points = selector.run()
    if not points:
        return False

    board_p1, board_p2, anchor, col1_anchor, row1_anchor = (
        points[0], points[1], points[2], points[3], points[4],
    )

    try:
        config = _build_grid(
            board_p1, board_p2, anchor, col1_anchor, row1_anchor,
            peek_w_margin=args.peek_w_margin,
            peek_h_margin=args.peek_h_margin,
        )
    except ValueError as exc:
        print(f"Calibration error: {exc}")
        return False

    print(f"  col_spacing = {config['col_spacing']} px")
    print(f"  row_spacing = {config['row_spacing']} px")
    print(f"  peek box    = {config['peek_w']} × {config['peek_h']} px")

    overlay = _draw_overlay(selector.image, config)
    win = "Calibration Preview"
    cv2.namedWindow(win)
    cv2.imshow(win, overlay)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("y"), ord("Y")):
            with CONFIG_PATH.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            print(f"Saved calibration to {CONFIG_PATH}")
            cv2.destroyWindow(win)
            return True
        if key == 27:
            print("Cancelled. No config written.")
            cv2.destroyWindow(win)
            return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate EXAPUNKS solitaire board geometry (5-click).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--peek-w-margin", type=int, default=6,
        help="Pixels to subtract from col_spacing to get peek strip width.",
    )
    parser.add_argument(
        "--peek-h-margin", type=int, default=4,
        help="Pixels to subtract from row_spacing to get peek strip height.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()
