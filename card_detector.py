#!/usr/bin/env python3
"""Detect EXAPUNKS solitaire cards from visible peek strips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import ImageGrab

from calibrate import run_calibration
from card_encoding import CARD_TO_CODE, decode_code

CONFIG_PATH = Path("board_config.json")
TEMPLATES_DIR = Path("templates")


class CardDetector:
    """Template-matching detector over peek strips for all 9x4 board slots."""

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        templates_dir: Path = TEMPLATES_DIR,
        min_score: float = 0.65,
    ):
        self.config_path = config_path
        self.templates_dir = templates_dir
        self.min_score = min_score
        self.config = self._load_config(config_path)
        self.templates = self._load_templates(templates_dir)

    @staticmethod
    def _load_config(path: Path) -> Dict[str, object]:
        if not path.exists():
            raise FileNotFoundError(f"Missing config: {path}. Run calibrate.py first.")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _prep_gray(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def _load_templates(self, templates_dir: Path) -> Dict[str, np.ndarray]:
        if not templates_dir.exists():
            raise FileNotFoundError(f"Missing templates dir: {templates_dir}")

        templates: Dict[str, np.ndarray] = {}
        for file_path in sorted(templates_dir.glob("*.png")):
            label = file_path.stem
            if label not in CARD_TO_CODE:
                continue
            img = cv2.imread(str(file_path))
            if img is None:
                continue
            templates[label] = self._prep_gray(img)

        if not templates:
            raise RuntimeError("No valid templates found. Run capture_templates.py first.")

        return templates

    @staticmethod
    def capture_screen_bgr() -> np.ndarray:
        screenshot = ImageGrab.grab()
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _crop_slot(img: np.ndarray, slot: Dict[str, int]) -> np.ndarray:
        h, w = img.shape[:2]
        x1 = max(0, int(slot["x"]))
        y1 = max(0, int(slot["y"]))
        x2 = min(w, x1 + int(slot["w"]))
        y2 = min(h, y1 + int(slot["h"]))
        return img[y1:y2, x1:x2]

    def detect_slot(self, slot_img: np.ndarray) -> Tuple[int, float, str]:
        """Return (code, score, label) for a single slot image."""
        if slot_img.size == 0:
            return 0, 0.0, "empty"

        slot_gray = self._prep_gray(slot_img)

        best_label = "empty"
        best_score = -1.0

        for label, template in self.templates.items():
            th, tw = template.shape[:2]
            resized_slot = cv2.resize(slot_gray, (tw, th), interpolation=cv2.INTER_AREA)
            score = float(cv2.matchTemplate(resized_slot, template, cv2.TM_CCOEFF_NORMED)[0][0])
            if score > best_score:
                best_score = score
                best_label = label

        if best_score < self.min_score:
            return 0, best_score, "unknown"

        return CARD_TO_CODE[best_label], best_score, best_label

    def detect_board(self, image: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        """Detect all cards; returns (codes, scores, labels) shaped (9, 4)."""
        board_img = image if image is not None else self.capture_screen_bgr()
        slots = self.config["slot_boxes"]

        codes = np.zeros((9, 4), dtype=np.int8)
        scores = np.zeros((9, 4), dtype=np.float32)
        labels: List[List[str]] = [["empty" for _ in range(4)] for _ in range(9)]

        for col_idx, column_slots in enumerate(slots):
            for row_idx, slot in enumerate(column_slots):
                slot_img = self._crop_slot(board_img, slot)
                code, score, label = self.detect_slot(slot_img)
                codes[col_idx, row_idx] = code
                scores[col_idx, row_idx] = score
                labels[col_idx][row_idx] = label

        return codes, scores, labels



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect EXAPUNKS solitaire board state.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to board config JSON.")
    parser.add_argument("--templates-dir", type=Path, default=TEMPLATES_DIR, help="Template directory path.")
    parser.add_argument("--min-score", type=float, default=0.65, help="Minimum template score for known class.")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration before detection.")
    parser.add_argument("--show-labels", action="store_true", help="Print labels matrix in addition to codes.")
    parser.add_argument("--show-scores", action="store_true", help="Print match confidence matrix.")
    parser.add_argument("--peek-w-margin", type=int, default=6, help="Pixels subtracted from col_spacing for peek width.")
    parser.add_argument("--peek-h-margin", type=int, default=4, help="Pixels subtracted from row_spacing for peek height.")
    return parser.parse_args()



def _ensure_calibration(args: argparse.Namespace) -> None:
    if args.calibrate or not args.config.exists():
        ok = run_calibration(
            argparse.Namespace(
                peek_w_margin=args.peek_w_margin,
                peek_h_margin=args.peek_h_margin,
            )
        )
        if not ok:
            raise RuntimeError("Calibration cancelled or failed.")



def _print_matrix(title: str, matrix: np.ndarray) -> None:
    print(f"\n{title}")
    print(matrix)



def _print_labels(labels: List[List[str]]) -> None:
    print("\nLabels (col x row):")
    for col_idx, column_labels in enumerate(labels):
        print(f"col {col_idx}: {column_labels}")



def _print_decoded(codes: np.ndarray) -> None:
    print("\nDecoded (col x row):")
    for col_idx in range(codes.shape[0]):
        decoded = [decode_code(int(code)) for code in codes[col_idx]]
        print(f"col {col_idx}: {decoded}")



def main() -> None:
    args = _parse_args()
    _ensure_calibration(args)

    detector = CardDetector(
        config_path=args.config,
        templates_dir=args.templates_dir,
        min_score=args.min_score,
    )

    codes, scores, labels = detector.detect_board()

    _print_matrix("Codes matrix (shape 9x4)", codes)
    _print_decoded(codes)

    if args.show_labels:
        _print_labels(labels)
    if args.show_scores:
        _print_matrix("Scores matrix (shape 9x4)", scores)


if __name__ == "__main__":
    main()
