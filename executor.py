#!/usr/bin/env python3
"""Automated EXAPUNKS solitaire executor.

Full pipeline:
  1. Run solver.py  (which itself runs shape_comparer.py first)
  2. Load planned_moves.json
  3. Translate each move into screen coordinates using board_config.json
  4. Execute clicks on the EXAPUNKS window

Coordinate model:
  - Board columns  0..8  →  col_lefts[col] + peek_w//2
  - Rows beyond calibration  →  row_tops[0] + row_index * row_spacing + peek_h//2
  - Spare slot (stack 9)  →  holder_box center  (calibrated or prompted on first run)

Click model (click-to-select, then click-to-place):
  1. Click top card of source stack
  2. Click destination (top card of dest stack, or empty column anchor)

Usage:
    python executor.py                   # solve + execute
    python executor.py --dry-run         # solve only, print clicks without executing
    python executor.py --delay 1.2       # seconds between moves (default 0.01)
    python executor.py --recalibrate-holder   # re-ask for holder position
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import threading

import cv2
import numpy as np

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0   # we manage all delays manually
except ImportError:
    print("ERROR: pyautogui is not installed.  Run:  pip install pyautogui")
    raise SystemExit(1)

try:
    from pynput import keyboard as _pynput_keyboard
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False

# Global stop flag set by the ESC listener.
_STOP = threading.Event()


def _start_esc_listener() -> None:
    """Start a background thread that sets _STOP when ESC is pressed."""
    if not _PYNPUT_AVAILABLE:
        print("[INFO] pynput not available – ESC abort disabled.")
        return

    def on_press(key):
        if key == _pynput_keyboard.Key.esc:
            _STOP.set()
            print("\n[ESC] Abort requested – stopping after current move.")
            return False  # stop the listener

    listener = _pynput_keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

from calibrate import _ensure_exapunks_fullscreen

BOARD_STACK_COUNT = 9
SPARE_STACK_INDEX = 9
TOTAL_STACKS = 10

DEFAULT_CONFIG = Path("board_config.json")
DEFAULT_PLAN = Path("planned_moves.json")
DEFAULT_SOLVER = Path("solver.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run solver then execute moves on the EXAPUNKS window.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--solver", type=Path, default=DEFAULT_SOLVER)
    p.add_argument("--delay", type=float, default=0.01,
                   help="Seconds to wait between moves (default 0.01).")
    p.add_argument("--drag-duration", type=float, default=0.01,
                   help="Seconds the mouse takes to travel during a drag (default 0.01).")
    p.add_argument("--pre-hold", type=float, default=0.01,
                   help="Seconds to hold mouseDown before starting drag motion (default 0.01).")
    p.add_argument("--post-hold", type=float, default=0.01,
                   help="Seconds to hold after reaching destination before mouseUp (default 0.01).")
    p.add_argument("--approach-delay", type=float, default=0.01,
                   help="Seconds to wait after moving to source before mouseDown (default 0.01).")
    p.add_argument("--focus-delay", type=float, default=0.01,
                   help="Seconds to wait after focusing EXAPUNKS before first move (default 0.01).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned clicks without moving the mouse.")
    p.add_argument("--recalibrate-holder", action="store_true",
                   help="Force re-calibration of the spare/holder slot position.")
    p.add_argument("--skip-solve", action="store_true",
                   help="Skip running the solver and use existing planned_moves.json.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"board_config.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Spare/holder calibration
# ---------------------------------------------------------------------------

def calibrate_holder(cfg: Dict, config_path: Path) -> Dict:
    """Interactive click-to-set calibration for the spare/holder square.

    Shows a live screenshot with crosshair overlay.  The user clicks on the
    centre of the spare slot.  Result is saved into board_config.json.
    """
    print("\n=== Holder / Spare slot calibration ===")
    print("The spare square position is not yet configured.")
    print("A screenshot will open.  Click on the CENTRE of the spare/holder square.")
    print("Press  q  to abort without saving.\n")
    time.sleep(1.0)

    from PIL import ImageGrab
    screenshot = ImageGrab.grab()
    screen_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Keep a reference so the callback can write into it.
    result: Dict = {"x": None, "y": None}

    win = "Holder calibration – click the spare square, q to quit"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            result["x"] = x
            result["y"] = y
            overlay = screen_bgr.copy()
            cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(overlay, (x, y), 20, (0, 255, 0), 2)
            cv2.putText(overlay, f"({x}, {y})", (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, overlay)
        elif event == cv2.EVENT_MOUSEMOVE:
            overlay = screen_bgr.copy()
            # Draw crosshair
            h, w = overlay.shape[:2]
            cv2.line(overlay, (x, 0), (x, h), (0, 200, 255), 1)
            cv2.line(overlay, (0, y), (w, y), (0, 200, 255), 1)
            if result["x"] is not None:
                cv2.circle(overlay, (result["x"], result["y"]), 8, (0, 255, 0), -1)
            cv2.imshow(win, overlay)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    cv2.imshow(win, screen_bgr)
    cv2.setMouseCallback(win, on_mouse)

    print("Waiting for click...  (Press Enter/s to confirm, q to cancel)")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise RuntimeError("Holder calibration cancelled by user.")
        if key in (13, ord("s"), ord("\r")):   # Enter or s
            if result["x"] is not None:
                break
            print("No point selected yet – click the holder square first.")

    cv2.destroyAllWindows()

    # Store as a box centred on the clicked point.
    peek_w = int(cfg.get("peek_w", 29))
    peek_h = int(cfg.get("peek_h", 24))
    hx = result["x"] - peek_w // 2
    hy = result["y"] - peek_h // 2
    cfg["holder_box"] = {"x": hx, "y": hy, "w": peek_w, "h": peek_h}
    save_config(cfg, config_path)
    print(f"Holder position saved: centre ({result['x']}, {result['y']})")
    return cfg


# ---------------------------------------------------------------------------
# Coordinate calculation
# ---------------------------------------------------------------------------

def col_center_x(cfg: Dict, col_index: int) -> int:
    col_lefts = cfg["col_lefts"]
    peek_w = int(cfg.get("peek_w", 29))
    return int(col_lefts[col_index]) + peek_w // 2


def card_center_y(cfg: Dict, row_index: int) -> int:
    row_tops = cfg["row_tops"]
    row_spacing = int(cfg.get("row_spacing", 29))
    peek_h = int(cfg.get("peek_h", 24))
    base_y = int(row_tops[0])
    return base_y + row_index * row_spacing + peek_h // 2


def board_card_screen_pos(cfg: Dict, col_index: int, row_index: int) -> Tuple[int, int]:
    """Centre pixel of the card at (col_index, row_index) on the board."""
    return col_center_x(cfg, col_index), card_center_y(cfg, row_index)


def spare_screen_pos(cfg: Dict) -> Tuple[int, int]:
    box = cfg["holder_box"]
    return int(box["x"]) + int(box["w"]) // 2, int(box["y"]) + int(box["h"]) // 2


def source_pos(cfg: Dict, stack_index: int, stack_size: int, count: int = 1) -> Tuple[int, int]:
    """Screen position of the BOTTOM card of the sequence being moved.

    When moving `count` cards the grab point is the lowest card in the
    movable suffix: row_index = stack_size - count.
    """
    if stack_index == SPARE_STACK_INDEX:
        return spare_screen_pos(cfg)
    # Clamp to row 0 in case count > stack_size (shouldn't happen with a valid plan).
    grip_row = max(0, stack_size - count)
    return board_card_screen_pos(cfg, stack_index, grip_row)


def dest_pos(cfg: Dict, stack_index: int, stack_size: int) -> Tuple[int, int]:
    """Screen position to drop onto for the given stack.

    We target the NEXT empty row (stack_size), which sits just below the
    current top card.  For an empty column this is row 0.
    """
    if stack_index == SPARE_STACK_INDEX:
        return spare_screen_pos(cfg)
    drop_row = stack_size   # 0 when empty, top+1 when non-empty
    return board_card_screen_pos(cfg, stack_index, drop_row)


# ---------------------------------------------------------------------------
# Solver invocation
# ---------------------------------------------------------------------------

def run_solver(solver_path: Path, plan_path: Path, no_refresh: bool = False) -> None:
    if not solver_path.exists():
        raise FileNotFoundError(f"solver.py not found: {solver_path}")

    mode_msg = "no refresh" if no_refresh else "with shape_comparer refresh"
    print(f"Running solver ({mode_msg})...")
    cmd = [sys.executable, str(solver_path), "--plan-out", str(plan_path)]
    if no_refresh:
        cmd.append("--no-refresh")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Solver failed with exit code {exc.returncode}.") from exc


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------

def load_plan(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    moves = data.get("planned_moves")
    if not isinstance(moves, list):
        raise ValueError("planned_moves.json missing 'planned_moves' list.")
    return moves


def load_initial_stack_sizes(state_path: Path) -> List[int]:
    """Return list of 10 stack sizes from normalized_board_state.json."""
    sizes = [0] * TOTAL_STACKS
    if not state_path.exists():
        return sizes
    with state_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    columns = raw.get("columns", [])
    for i, col in enumerate(columns):
        if i < BOARD_STACK_COUNT and isinstance(col, list):
            sizes[i] = len(col)
    holder = raw.get("holder", {})
    if isinstance(holder, dict) and holder.get("card") is not None:
        sizes[SPARE_STACK_INDEX] = 1
    return sizes


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _fmt_sizes(sizes: List[int]) -> str:
    """One-line display of all stack sizes.  spare shown separately."""
    board = " ".join(f"c{i+1}:{s}" for i, s in enumerate(sizes[:BOARD_STACK_COUNT]))
    spare = f"spare:{sizes[SPARE_STACK_INDEX]}"
    return f"[{board}  {spare}]"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

# Drag duration controls how long the mouse takes to travel from src to dst.
# Long enough for the game to register the grab, short enough to feel snappy.
DRAG_DURATION = 0.01
PRE_DRAG_HOLD = 0.01   # seconds to hold mouseDown before starting motion
POST_DROP_HOLD = 0.01  # seconds to hold after reaching destination


def perform_drag(sx: int, sy: int, dx: int, dy: int, dry_run: bool,
                 src_label: str, dst_label: str,
                 drag_duration: float = DRAG_DURATION,
                 pre_hold: float = PRE_DRAG_HOLD,
                 post_hold: float = POST_DROP_HOLD,
                 approach_delay: float = 0.05) -> None:
    """Drag from (sx,sy) to (dx,dy): mouseDown → move → mouseUp."""
    if dry_run:
        print(f"    [DRY] drag {src_label}({sx},{sy}) -> {dst_label}({dx},{dy})")
        return

    pyautogui.moveTo(sx, sy, duration=0.05)
    time.sleep(approach_delay)
    pyautogui.mouseDown(button="left")
    time.sleep(pre_hold)
    pyautogui.moveTo(dx, dy, duration=drag_duration)
    time.sleep(post_hold)
    pyautogui.mouseUp(button="left")


def execute_moves(
    moves: List[Dict],
    cfg: Dict,
    stack_sizes: List[int],
    delay: float,
    drag_duration: float,
    pre_hold: float,
    post_hold: float,
    approach_delay: float,
    dry_run: bool,
) -> int:
    """Execute all moves. Returns number of moves completed before stop."""
    sizes = list(stack_sizes)   # mutable copy

    for idx, move in enumerate(moves, start=1):
        if _STOP.is_set():
            print(f"[ABORT] Stopped before move {idx}.")
            return idx - 1

        src = int(move["source_stack"])
        dst = int(move["target_stack"])
        count = int(move["move_count"])
        reason = move.get("reason", "")

        src_label = "spare" if src == SPARE_STACK_INDEX else f"col {src + 1}"
        dst_label = "spare" if dst == SPARE_STACK_INDEX else f"col {dst + 1}"

        sx, sy = source_pos(cfg, src, sizes[src], count)
        dx, dy = dest_pos(cfg, dst, sizes[dst])

        print(f"{idx:02d}/{len(moves)}. Drag {count} card(s) from {src_label} to {dst_label}  [{reason}]")
        print(f"       src=({sx},{sy}) row={max(0, sizes[src]-count)}  dst=({dx},{dy}) row={sizes[dst]}")
        print(f"       sizes before: {_fmt_sizes(sizes)}")

        perform_drag(sx, sy, dx, dy, dry_run, src_label, dst_label,
                     drag_duration=drag_duration, pre_hold=pre_hold,
                     post_hold=post_hold, approach_delay=approach_delay)

        # Update tracked sizes.
        sizes[src] = max(0, sizes[src] - count)
        sizes[dst] = sizes[dst] + count

        print(f"       sizes after:  {_fmt_sizes(sizes)}")

        if not dry_run:
            time.sleep(delay)

    return len(moves)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    # Ensure holder/spare slot is calibrated.
    needs_calibration = "holder_box" not in cfg or args.recalibrate_holder
    if needs_calibration:
        _ensure_exapunks_fullscreen()
        cfg = calibrate_holder(cfg, args.config)

    # Run solver (also runs shape_comparer internally) unless skipped.
    if not args.skip_solve:
        try:
            run_solver(args.solver, args.plan)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Solver error: {exc}")
            raise SystemExit(2) from exc

    # Load moves + initial state.
    try:
        moves = load_plan(args.plan)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Plan load error: {exc}")
        raise SystemExit(2) from exc

    state_path = Path("normalized_board_state.json")
    stack_sizes = load_initial_stack_sizes(state_path)

    if not moves:
        print("No moves in plan.  Nothing to execute.")
        raise SystemExit(0)

    if not args.dry_run:
        _start_esc_listener()
        print("ESC to abort at any time.")

    print(
        f"\n{len(moves)} move(s) to execute. between={args.delay}s "
        f"drag={args.drag_duration}s pre={args.pre_hold}s post={args.post_hold}s "
        f"approach={args.approach_delay}s"
    )
    if args.dry_run:
        print("DRY RUN – no actual drags.\n")
    else:
        print("Activating EXAPUNKS window...\n")
        _ensure_exapunks_fullscreen()
        time.sleep(args.focus_delay)

    completed = execute_moves(
        moves=moves,
        cfg=cfg,
        stack_sizes=stack_sizes,
        delay=args.delay,
        drag_duration=args.drag_duration,
        pre_hold=args.pre_hold,
        post_hold=args.post_hold,
        approach_delay=args.approach_delay,
        dry_run=args.dry_run,
    )

    if _STOP.is_set():
        print(f"\nAborted after {completed}/{len(moves)} moves.")
    else:
        print(f"\nDone. {completed} move(s) executed.")


if __name__ == "__main__":
    main()
