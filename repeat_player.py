#!/usr/bin/env python3
"""Looped EXAPUNKS solitaire player.

Flow per round:
1) Click New Game button (calibrate if missing)
2) Repeatedly run shape_comparer until produced state looks coherent
3) Run solver in no-refresh mode using that coherent state
4) Execute planned moves
5) Repeat

ESC can be used at any time to stop after the current action.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0
except ImportError:
    print("ERROR: pyautogui is not installed.  Run: pip install pyautogui")
    raise SystemExit(1)

try:
    from PIL import ImageGrab
except ImportError:
    print("ERROR: Pillow is not installed.  Run: pip install pillow")
    raise SystemExit(1)

from calibrate import _ensure_exapunks_fullscreen
from card_encoding import FACE_LABELS, NUMBERED_LABELS
from executor import (
    _STOP,
    _start_esc_listener,
    calibrate_holder,
    execute_moves,
    load_config,
    load_initial_stack_sizes,
    load_plan,
    run_solver,
    save_config,
)

DEFAULT_CONFIG = Path("board_config.json")
DEFAULT_PLAN = Path("planned_moves.json")
DEFAULT_SOLVER = Path("solver.py")
DEFAULT_STATE = Path("normalized_board_state.json")
DEFAULT_SHAPE_COMPARER = Path("shape_comparer.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play EXAPUNKS solitaire in a loop.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--solver", type=Path, default=DEFAULT_SOLVER)
    p.add_argument("--shape-comparer", type=Path, default=DEFAULT_SHAPE_COMPARER)
    p.add_argument("--state", type=Path, default=DEFAULT_STATE)

    # Execution timing (mirrors executor defaults).
    p.add_argument("--delay", type=float, default=0.01)
    p.add_argument("--drag-duration", type=float, default=0.01)
    p.add_argument("--pre-hold", type=float, default=0.01)
    p.add_argument("--post-hold", type=float, default=0.01)
    p.add_argument("--approach-delay", type=float, default=0.01)
    p.add_argument("--focus-delay", type=float, default=0.01)

    # New game + vision polling timing.
    p.add_argument("--new-game-clicks", type=int, default=1,
                   help="How many clicks to send to New Game each round (default 1).")
    p.add_argument("--new-game-hold", type=float, default=0.03,
                   help="Seconds to hold mouse button during New Game click (default 0.03).")
    p.add_argument("--new-game-between", type=float, default=0.06,
                   help="Seconds between repeated New Game clicks (default 0.06).")
    p.add_argument("--animation-start-delay", type=float, default=1.0,
                   help="Seconds to wait before first shape_comparer attempt (default 1.0).")
    p.add_argument("--animation-max-wait", type=float, default=8.0,
                   help="Max seconds to wait for a coherent vision state (default 8.0).")
    p.add_argument("--sample-interval", type=float, default=0.05,
                   help="Seconds between shape_comparer retries while waiting (default 0.05).")
    p.add_argument("--stable-frames", type=int, default=4,
                   help="Consecutive low-diff frames required for settle (default 4).")
    p.add_argument("--diff-threshold", type=float, default=1.2,
                   help="Mean abs grayscale diff threshold for movement (default 1.2).")

    p.add_argument("--max-games", type=int, default=0,
                   help="0 means infinite loop; otherwise stop after N rounds.")
    p.add_argument("--log-file", type=Path, default=Path("repeat_player_log.jsonl"),
                   help="Path to JSONL stats log (default repeat_player_log.jsonl).")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not click/drag; print actions only.")
    p.add_argument("--recalibrate-holder", action="store_true")
    p.add_argument("--recalibrate-new-game", action="store_true")
    return p.parse_args()


def _grab_screen_bgr() -> np.ndarray:
    screenshot = ImageGrab.grab()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


def _board_crop(frame_bgr: np.ndarray, cfg: Dict) -> np.ndarray:
    rect = cfg.get("board_rect", {})
    x1 = int(rect.get("x1", 0))
    y1 = int(rect.get("y1", 0))
    x2 = int(rect.get("x2", frame_bgr.shape[1]))
    y2 = int(rect.get("y2", frame_bgr.shape[0]))

    x1 = max(0, min(x1, frame_bgr.shape[1] - 1))
    y1 = max(0, min(y1, frame_bgr.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, frame_bgr.shape[1]))
    y2 = max(y1 + 1, min(y2, frame_bgr.shape[0]))

    return frame_bgr[y1:y2, x1:x2]


def _new_game_center(cfg: Dict) -> Tuple[int, int]:
    box = cfg["new_game_button"]
    cx = int(box["x"]) + int(box["w"]) // 2
    cy = int(box["y"]) + int(box["h"]) // 2
    return cx, cy


def calibrate_new_game_button(cfg: Dict, config_path: Path) -> Dict:
    """Interactive click-to-set calibration for the New Game button center."""
    print("\n=== New Game button calibration ===")
    print("A screenshot will open. Click on the CENTER of the New Game button.")
    print("Press q to cancel. Press Enter or s to confirm after clicking.\n")

    screen_bgr = _grab_screen_bgr()
    result = {"x": None, "y": None}
    win = "New Game calibration - click center, Enter/s to save, q to quit"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            result["x"] = x
            result["y"] = y
            overlay = screen_bgr.copy()
            cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(overlay, (x, y), 20, (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"({x}, {y})",
                (x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, overlay)
        elif event == cv2.EVENT_MOUSEMOVE:
            overlay = screen_bgr.copy()
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

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise RuntimeError("New Game calibration canceled by user.")
        if key in (13, ord("s"), ord("\r")):
            if result["x"] is None:
                print("No point selected yet - click the New Game button first.")
                continue
            break

    cv2.destroyAllWindows()

    # Store as a small box around the clicked center for future flexibility.
    box_w = 24
    box_h = 24
    cfg["new_game_button"] = {
        "x": int(result["x"]) - box_w // 2,
        "y": int(result["y"]) - box_h // 2,
        "w": box_w,
        "h": box_h,
    }
    save_config(cfg, config_path)
    print(f"Saved New Game button center at ({result['x']}, {result['y']}).")
    return cfg


def click_new_game(
    cfg: Dict,
    clicks: int,
    dry_run: bool,
    hold_s: float,
    between_s: float,
) -> None:
    x, y = _new_game_center(cfg)
    if dry_run:
        print(f"[DRY] New Game click at ({x}, {y}) x{clicks}")
        return

    for i in range(max(1, int(clicks))):
        if _STOP.is_set():
            return
        pyautogui.moveTo(x, y, duration=0.03)
        time.sleep(0.01)
        pyautogui.mouseDown(x=x, y=y, button="left")
        time.sleep(max(0.0, hold_s))
        pyautogui.mouseUp(x=x, y=y, button="left")
        if i + 1 < clicks:
            time.sleep(max(0.0, between_s))


def _run_shape_comparer(shape_comparer: Path, state_path: Path) -> None:
    if not shape_comparer.exists():
        raise FileNotFoundError(f"shape_comparer.py not found: {shape_comparer}")
    subprocess.run(
        [
            sys.executable,
            str(shape_comparer),
            "--solver-mode",
            "--state-out",
            str(state_path),
        ],
        check=True,
    )


def _state_looks_sane(state_path: Path) -> Tuple[bool, str]:
    if not state_path.exists():
        return False, "state file not created"

    try:
        with state_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, "state file unreadable"

    columns = raw.get("columns")
    if not isinstance(columns, list) or len(columns) != 9:
        return False, "invalid columns layout"

    holder = raw.get("holder")
    holder_card = holder.get("card") if isinstance(holder, dict) else None
    if holder_card is not None:
        return False, "holder not empty"

    labels = []
    for col in columns:
        if not isinstance(col, list):
            return False, "column is not a list"
        for card in col:
            if not isinstance(card, dict):
                return False, "malformed card object"
            lbl = card.get("label")
            if not isinstance(lbl, str):
                return False, "missing card label"
            labels.append(lbl)

    if len(labels) != 36:
        return False, f"expected 36 cards, got {len(labels)}"

    counts = Counter(labels)
    for lbl in FACE_LABELS:
        if counts.get(lbl, 0) != 4:
            return False, f"bad count for {lbl}: {counts.get(lbl, 0)}"
    for lbl in NUMBERED_LABELS:
        if counts.get(lbl, 0) != 2:
            return False, f"bad count for {lbl}: {counts.get(lbl, 0)}"

    return True, "ok"


def wait_for_vision_ready(
    shape_comparer: Path,
    state_path: Path,
    start_delay: float,
    max_wait: float,
    retry_delay: float,
    dry_run: bool,
) -> None:
    """Poll shape_comparer until generated state passes sanity checks."""
    if dry_run:
        print(
            "[DRY] Vision-ready wait "
            f"start={start_delay}s max={max_wait}s retry={retry_delay}s"
        )
        return

    time.sleep(max(0.0, start_delay))
    deadline = time.time() + max(0.1, max_wait)
    attempts = 0
    last_reason = "no attempt"

    while time.time() < deadline and not _STOP.is_set():
        attempts += 1
        try:
            _run_shape_comparer(shape_comparer, state_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            last_reason = f"shape_comparer failed: {exc}"
        else:
            ok, reason = _state_looks_sane(state_path)
            if ok:
                print(f"Vision-ready state accepted after {attempts} attempt(s).")
                return
            last_reason = reason

        time.sleep(max(0.01, retry_delay))

    raise RuntimeError(
        f"Timed out waiting for vision-ready state after {attempts} attempts ({last_reason})."
    )


def ensure_calibration(cfg: Dict, args: argparse.Namespace) -> Dict:
    needs_holder = "holder_box" not in cfg or args.recalibrate_holder
    needs_new_game = "new_game_button" not in cfg or args.recalibrate_new_game

    if needs_holder or needs_new_game:
        _ensure_exapunks_fullscreen()

    if needs_holder:
        cfg = calibrate_holder(cfg, args.config)
    if needs_new_game:
        cfg = calibrate_new_game_button(cfg, args.config)

    return cfg


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_round_log(log_path: Path, row: Dict[str, Any]) -> None:
    """Append one round record as a JSON line."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def play_round(round_idx: int, cfg: Dict, args: argparse.Namespace) -> Dict[str, Any]:
    """Run one round and return structured stats."""
    started_at = time.perf_counter()

    def result(status: str, planned: int = 0, completed: int = 0, error: str = "") -> Dict[str, Any]:
        return {
            "round": round_idx,
            "status": status,
            "planned_moves": int(planned),
            "completed_moves": int(completed),
            "duration_s": round(time.perf_counter() - started_at, 4),
            "error": error,
        }

    if _STOP.is_set():
        return result("aborted")

    print(f"\n=== Round {round_idx} ===")
    print("Clicking New Game...")
    if not args.dry_run:
        _ensure_exapunks_fullscreen()
        time.sleep(max(0.0, args.focus_delay))
    click_new_game(
        cfg,
        clicks=args.new_game_clicks,
        dry_run=args.dry_run,
        hold_s=args.new_game_hold,
        between_s=args.new_game_between,
    )

    if _STOP.is_set():
        return result("aborted")

    print("Waiting for coherent vision state...")
    try:
        wait_for_vision_ready(
            shape_comparer=args.shape_comparer,
            state_path=args.state,
            start_delay=args.animation_start_delay,
            max_wait=args.animation_max_wait,
            retry_delay=args.sample_interval,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        print(f"Vision readiness failed this round: {exc}")
        return result("solver_fail", error=str(exc))

    if _STOP.is_set():
        return result("aborted")

    print("Running solver (no refresh)...")
    try:
        run_solver(args.solver, args.plan, no_refresh=True)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Solver failed this round: {exc}")
        return result("solver_fail", error=str(exc))

    try:
        moves = load_plan(args.plan)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Plan load failed: {exc}")
        return result("solver_fail", error=str(exc))

    if not moves:
        print("Solver returned no moves.")
        return result("no_moves")

    sizes = load_initial_stack_sizes(args.state)
    completed = execute_moves(
        moves=moves,
        cfg=cfg,
        stack_sizes=sizes,
        delay=args.delay,
        drag_duration=args.drag_duration,
        pre_hold=args.pre_hold,
        post_hold=args.post_hold,
        approach_delay=args.approach_delay,
        dry_run=args.dry_run,
    )

    if _STOP.is_set():
        print(f"Round aborted after {completed}/{len(moves)} moves.")
        return result("aborted", planned=len(moves), completed=completed)

    print(f"Round completed. Executed {completed}/{len(moves)} moves.")
    return result("solved", planned=len(moves), completed=completed)


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    cfg = ensure_calibration(cfg, args)

    if not args.dry_run:
        _ensure_exapunks_fullscreen()
        time.sleep(max(0.0, args.focus_delay))

    _start_esc_listener()
    print("ESC to abort.")
    print(f"Logging rounds to: {args.log_file}")

    round_idx = 1
    while not _STOP.is_set():
        if args.max_games > 0 and round_idx > args.max_games:
            break

        round_stats = play_round(round_idx, cfg, args)
        log_row = {
            "ts_utc": _utc_now_iso(),
            "round": round_stats["round"],
            "status": round_stats["status"],
            "duration_s": round_stats["duration_s"],
            "planned_moves": round_stats["planned_moves"],
            "completed_moves": round_stats["completed_moves"],
            "dry_run": bool(args.dry_run),
            "error": round_stats["error"],
        }
        append_round_log(args.log_file, log_row)
        print(
            "Round log: "
            f"status={log_row['status']} "
            f"duration={log_row['duration_s']:.3f}s "
            f"moves={log_row['completed_moves']}/{log_row['planned_moves']}"
        )

        if round_stats["status"] == "aborted":
            break

        round_idx += 1

    print("\nRepeat player stopped.")


if __name__ == "__main__":
    main()
