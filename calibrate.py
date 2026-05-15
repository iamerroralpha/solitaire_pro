#!/usr/bin/env python3
"""Calibrate EXAPUNKS solitaire board geometry for peek-strip card detection.

5-click calibration sequence
─────────────────────────────
  1. Board top-left corner       (overall boundary)
  2. Board bottom-right corner   (overall boundary)
  3. Top-left pixel of card at col 0, row 0  (anchor)
  4. Top-left pixel of card at col 1, row 0  → measures column spacing
  5. Top-left pixel of card at col 0, row 1  → measures stack/row spacing

From those five points the script derives an initial guess, then lets you
fine-tune with 6 sliders:
    1) start_x, 2) start_y, 3) dist_x, 4) dist_y, 5) box_w, 6) box_h
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
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


def _is_exapunks_running() -> bool:
    """Return True if an EXAPUNKS process is currently running."""
    try:
        proc = subprocess.run(
            ["pgrep", "-ifl", "EXAPUNKS"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"Warning: could not check EXAPUNKS process state: {exc}")
        return False
    return bool(proc.stdout.strip())


def _osascript(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )


def _is_exapunks_fullscreen() -> Optional[bool]:
    """Best-effort check of EXAPUNKS fullscreen state.

    Returns:
        True/False if readable, None if state cannot be determined.
    """
    script = (
        'tell application "System Events"\n'
        '  if not (exists process "EXAPUNKS") then return "not_running"\n'
        '  tell process "EXAPUNKS"\n'
        '    if not (exists window 1) then return "no_window"\n'
        '    try\n'
        '      return (value of attribute "AXFullScreen" of window 1) as string\n'
        '    on error\n'
        '      return "unknown"\n'
        '    end try\n'
        '  end tell\n'
        'end tell'
    )
    proc = _osascript(script)
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip().lower()
    if out == "true":
        return True
    if out == "false":
        return False
    return None


def _ensure_exapunks_fullscreen() -> None:
    """If EXAPUNKS is running, bring it to front and ensure fullscreen."""
    if not _is_exapunks_running():
        print("EXAPUNKS not detected. Continuing without app focus/fullscreen automation.")
        return

    print("EXAPUNKS detected. Activating app window…")
    activate = _osascript('tell application "EXAPUNKS" to activate')
    if activate.returncode != 0:
        err = activate.stderr.strip() or activate.stdout.strip()
        print(f"Warning: could not activate EXAPUNKS automatically: {err}")
        return

    time.sleep(0.35)

    fs = _is_exapunks_fullscreen()
    if fs is True:
        print("EXAPUNKS already fullscreen.")
        time.sleep(0.25)
        return

    print("Switching EXAPUNKS to fullscreen (Ctrl+Cmd+F)…")
    toggle = _osascript(
        'tell application "System Events" to keystroke "f" using {control down, command down}'
    )
    if toggle.returncode != 0:
        err = toggle.stderr.strip() or toggle.stdout.strip()
        print(f"Warning: fullscreen toggle failed: {err}")
        print("You can manually fullscreen EXAPUNKS and rerun calibration.")
        return

    time.sleep(0.8)


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
    start_x: int,
    start_y: int,
    dist_x: int,
    dist_y: int,
    box_w: int,
    box_h: int,
    cols: int = 9,
    rows: int = 4,
) -> Dict[str, object]:
    x1 = min(board_p1[0], board_p2[0])
    y1 = min(board_p1[1], board_p2[1])
    x2 = max(board_p1[0], board_p2[0])
    y2 = max(board_p1[1], board_p2[1])

    if dist_x <= 0 or dist_y <= 0:
        raise ValueError(
            f"Invalid spacing: dist_x={dist_x}, dist_y={dist_y}. Both must be > 0."
        )
    if box_w <= 0 or box_h <= 0:
        raise ValueError(
            f"Invalid box size: box_w={box_w}, box_h={box_h}. Both must be > 0."
        )

    slots: List[List[Dict[str, int]]] = []
    col_lefts: List[int] = []
    row_tops: List[int] = []

    for col in range(cols):
        col_lefts.append(start_x + dist_x * col)

    for row in range(rows):
        row_tops.append(start_y + dist_y * row)

    for col in range(cols):
        column_slots: List[Dict[str, int]] = []
        for row in range(rows):
            sx = col_lefts[col]
            sy = row_tops[row]
            column_slots.append({"x": int(sx), "y": int(sy), "w": int(box_w), "h": int(box_h)})
        slots.append(column_slots)

    return {
        "board_rect": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "rows": rows,
        "cols": cols,
        "col_spacing": dist_x,
        "row_spacing": dist_y,
        "anchor": {"x": start_x, "y": start_y},
        "col_lefts": col_lefts,
        "row_tops": row_tops,
        "peek_w": box_w,
        "peek_h": box_h,
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
    _ensure_exapunks_fullscreen()

    selector = _MultiClickSelector(_CLICK_PROMPTS)
    points = selector.run()
    if not points:
        return False

    board_p1, board_p2, anchor, col1_anchor, row1_anchor = (
        points[0], points[1], points[2], points[3], points[4],
    )

    base_start_x = anchor[0]
    base_start_y = anchor[1]
    base_dist_x = col1_anchor[0] - anchor[0]
    base_dist_y = row1_anchor[1] - anchor[1]

    if base_dist_x <= 0 or base_dist_y <= 0:
        print(
            "Calibration error: bad reference clicks. "
            "Need col 1 anchor right of col 0 and row 1 anchor below row 0."
        )
        return False

    base_box_w = max(4, base_dist_x - args.peek_w_margin)
    base_box_h = max(4, base_dist_y - args.peek_h_margin)

    print(f"  initial start = ({base_start_x}, {base_start_y}) px")
    print(f"  initial dist  = ({base_dist_x}, {base_dist_y}) px")
    print(f"  initial box   = {base_box_w} × {base_box_h} px")

    _SLIDER_CENTRE = 200
    _SLIDER_MAX = 400

    win = "Calibration Preview  |  Y = save   ESC = cancel"
    cv2.namedWindow(win)
    cv2.createTrackbar("1) offset_x", win, _SLIDER_CENTRE, _SLIDER_MAX, lambda _: None)
    cv2.createTrackbar("2) offset_y", win, _SLIDER_CENTRE, _SLIDER_MAX, lambda _: None)
    cv2.createTrackbar("3) distance_x", win, _SLIDER_CENTRE, _SLIDER_MAX, lambda _: None)
    cv2.createTrackbar("4) distance_y", win, _SLIDER_CENTRE, _SLIDER_MAX, lambda _: None)
    cv2.createTrackbar("5) width", win, _SLIDER_CENTRE, _SLIDER_MAX, lambda _: None)
    cv2.createTrackbar("6) height", win, _SLIDER_CENTRE, _SLIDER_MAX, lambda _: None)

    def _read_values() -> Tuple[int, int, int, int, int, int]:
        off_x = cv2.getTrackbarPos("1) offset_x", win) - _SLIDER_CENTRE
        off_y = cv2.getTrackbarPos("2) offset_y", win) - _SLIDER_CENTRE
        delta_dist_x = cv2.getTrackbarPos("3) distance_x", win) - _SLIDER_CENTRE
        delta_dist_y = cv2.getTrackbarPos("4) distance_y", win) - _SLIDER_CENTRE
        delta_w = cv2.getTrackbarPos("5) width", win) - _SLIDER_CENTRE
        delta_h = cv2.getTrackbarPos("6) height", win) - _SLIDER_CENTRE

        start_x = base_start_x + off_x
        start_y = base_start_y + off_y
        dist_x = max(1, base_dist_x + delta_dist_x)
        dist_y = max(1, base_dist_y + delta_dist_y)
        box_w = max(1, base_box_w + delta_w)
        box_h = max(1, base_box_h + delta_h)
        return start_x, start_y, dist_x, dist_y, box_w, box_h

    def _redraw() -> Optional[Dict[str, object]]:
        start_x, start_y, dist_x, dist_y, box_w, box_h = _read_values()
        try:
            cfg = _build_grid(
                board_p1,
                board_p2,
                start_x,
                start_y,
                dist_x,
                dist_y,
                box_w,
                box_h,
            )
        except ValueError:
            return None

        overlay = _draw_overlay(selector.image, cfg)
        cv2.rectangle(overlay, (0, overlay.shape[0] - 62), (overlay.shape[1], overlay.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"start=({start_x},{start_y})  dist=({dist_x},{dist_y})  box=({box_w},{box_h})",
            (10, overlay.shape[0] - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "Y = save    ESC = cancel",
            (10, overlay.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, overlay)
        return cfg

    latest_cfg = _redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        latest_cfg = _redraw()

        if key in (ord("y"), ord("Y")):
            if latest_cfg is None:
                print("Cannot save: current slider combination is invalid.")
                continue
            with CONFIG_PATH.open("w", encoding="utf-8") as f:
                json.dump(latest_cfg, f, indent=2)
            vals = _read_values()
            print(
                f"Saved calibration to {CONFIG_PATH}  "
                f"(start=({vals[0]},{vals[1]}), dist=({vals[2]},{vals[3]}), box=({vals[4]},{vals[5]}))"
            )
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
