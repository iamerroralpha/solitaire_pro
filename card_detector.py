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
    """Basic template-matching detector over peek strips for all 9x4 board slots."""

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        templates_dir: Path = TEMPLATES_DIR,
        min_score: float = 0.65,
        red_ratio_threshold: float = 0.04,
        resolve_debug: bool = False,
    ):
        self.config_path = config_path
        self.templates_dir = templates_dir
        self.min_score = min_score
        self.red_ratio_threshold = red_ratio_threshold
        self.resolve_debug = resolve_debug
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

    @staticmethod
    def _base_template_label(raw_label: str) -> str | None:
        """Map template filename stem to canonical class label.

        Supports numbered variants like 9b_1 -> 9b.
        """
        if raw_label in CARD_TO_CODE:
            return raw_label
        if "_" in raw_label:
            prefix = raw_label.split("_", 1)[0]
            if prefix in CARD_TO_CODE:
                return prefix
        return None

    @staticmethod
    def _next_variant_path(templates_dir: Path, base_label: str) -> Path:
        idx = 1
        while True:
            candidate = templates_dir / f"{base_label}_{idx}.png"
            if not candidate.exists():
                return candidate
            idx += 1

    def _load_templates(self, templates_dir: Path) -> List[Tuple[str, str, np.ndarray]]:
        if not templates_dir.exists():
            raise FileNotFoundError(f"Missing templates dir: {templates_dir}")

        templates: List[Tuple[str, str, np.ndarray]] = []
        for file_path in sorted(templates_dir.glob("*.png")):
            raw_label = file_path.stem.lower()
            base_label = self._base_template_label(raw_label)
            if base_label is None:
                continue
            img = cv2.imread(str(file_path))
            if img is None:
                continue
            templates.append((raw_label, base_label, self._prep_gray(img)))

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

    @staticmethod
    def _is_numbered_label(label: str) -> bool:
        return len(label) >= 2 and label[-1] in {"r", "b"} and label[:-1].isdigit()

    @staticmethod
    def _numbered_rank(label: str) -> str:
        return label[:-1]

    @staticmethod
    def _is_face_label(label: str) -> bool:
        return len(label) == 2 and label[0] in {"j", "q", "k", "a"} and label[1] in {"h", "d", "c", "s"}

    @staticmethod
    def _face_label_color(label: str) -> str:
        # Hearts/diamonds are red; clubs/spades are black.
        if not (len(label) == 2 and label[1] in {"h", "d", "c", "s"}):
            return "b"
        return "r" if label[1] in {"h", "d"} else "b"

    @staticmethod
    def _match_score(slot_gray: np.ndarray, template: np.ndarray) -> float:
        resized_slot = cv2.resize(slot_gray, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
        score = cv2.matchTemplate(resized_slot, template, cv2.TM_CCOEFF_NORMED)[0][0]
        if np.isnan(score):
            return -1.0
        return float(score)

    def _infer_number_color(self, slot_img: np.ndarray) -> Tuple[str, float]:
        """Infer red/black from a corner ROI using HSV red-pixel ratio."""
        if slot_img.size == 0:
            return "b", 0.0

        h, w = slot_img.shape[:2]
        roi = slot_img[0:max(1, int(h * 0.90)), 0:max(1, int(w * 0.55))]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 70, 40], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, 70, 40], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_ratio = float(np.count_nonzero(red_mask)) / float(red_mask.size)
        inferred = "r" if red_ratio >= self.red_ratio_threshold else "b"
        return inferred, red_ratio

    @staticmethod
    def _opposite_color(color: str) -> str:
        return "b" if color == "r" else "r"

    @staticmethod
    def _max_label_count(label: str) -> int:
        # Exact face cards exist once each; numbered rank+color classes represent two suits.
        if len(label) == 2 and label[0] in {"j", "q", "k", "a"} and label[1] in {"h", "d", "c", "s"}:
            return 1
        if len(label) >= 2 and label[-1] in {"r", "b"} and label[:-1].isdigit():
            return 2
        return 9999

    def _slot_candidates(self, slot_img: np.ndarray, topk: int = 10) -> List[Tuple[str, float]]:
        """Return ranked candidate labels for one slot crop."""
        if slot_img.size == 0:
            return []

        slot_gray = self._prep_gray(slot_img)
        inferred_color, _ = self._infer_number_color(slot_img)
        best_by_label: Dict[str, float] = {}

        for _raw_label, label, template in self.templates:
            score = self._match_score(slot_gray, template)
            if score > best_by_label.get(label, -1.0):
                best_by_label[label] = score

        # Build rank-wise numbered scores so color can be inferred robustly.
        rank_scores: Dict[str, float] = {}
        for label, score in best_by_label.items():
            if self._is_numbered_label(label):
                rank = self._numbered_rank(label)
                if score > rank_scores.get(rank, -1.0):
                    rank_scores[rank] = score

        merged: Dict[str, float] = {}

        # Face candidates: strict HSV-based color gating.
        for label, score in best_by_label.items():
            if self._is_face_label(label):
                if self._face_label_color(label) != inferred_color:
                    continue
                merged[label] = max(merged.get(label, -1.0), score)

        # Numbered candidates: strict inferred-color gating.
        # If HSV says red, we only compare/use red numbered classes (and vice versa for black).
        for rank, score in rank_scores.items():
            preferred = f"{rank}{inferred_color}"
            merged[preferred] = max(merged.get(preferred, -1.0), score)

        ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)
        return ranked[:topk]

    def detect_slot(self, slot_img: np.ndarray) -> Tuple[int, float, str, List[Tuple[str, float]]]:
        """Return (code, score, label, candidates) for a single slot image."""
        if slot_img.size == 0:
            return 0, 0.0, "empty", []

        candidates = self._slot_candidates(slot_img)
        if not candidates:
            return 0, 0.0, "unknown", []

        best_label, best_score = candidates[0]
        if best_score < self.min_score:
            return 0, best_score, "unknown", candidates

        return CARD_TO_CODE[best_label], best_score, best_label, candidates

    def _resolve_duplicate_labels(
        self,
        labels: List[List[str]],
        scores: np.ndarray,
        candidates_grid: List[List[List[Tuple[str, float]]]],
    ) -> List[str]:
        """Reassign weakest duplicates to next-best valid candidates using deck-count constraints."""
        logs: List[str] = []
        usage: Dict[str, int] = {}
        for col_idx in range(len(labels)):
            for row_idx in range(len(labels[col_idx])):
                label = labels[col_idx][row_idx]
                if label in {"empty", "unknown"}:
                    continue
                usage[label] = usage.get(label, 0) + 1

        max_iters = 200
        for _ in range(max_iters):
            overfull = [
                label
                for label, count in usage.items()
                if count > self._max_label_count(label)
            ]
            if not overfull:
                break

            progress = False
            for label in overfull:
                slots_for_label: List[Tuple[int, int, float]] = []
                for col_idx in range(len(labels)):
                    for row_idx in range(len(labels[col_idx])):
                        if labels[col_idx][row_idx] == label:
                            slots_for_label.append((col_idx, row_idx, float(scores[col_idx, row_idx])))

                # Weakest predictions first for reassignment.
                slots_for_label.sort(key=lambda item: item[2])

                for col_idx, row_idx, _score in slots_for_label:
                    if usage.get(label, 0) <= self._max_label_count(label):
                        break

                    for alt_label, alt_score in candidates_grid[col_idx][row_idx]:
                        if alt_label == label:
                            continue
                        if alt_score < self.min_score:
                            continue
                        if usage.get(alt_label, 0) >= self._max_label_count(alt_label):
                            continue

                        old_label = labels[col_idx][row_idx]
                        old_score = float(scores[col_idx, row_idx])
                        labels[col_idx][row_idx] = alt_label
                        scores[col_idx, row_idx] = alt_score
                        usage[label] = max(0, usage.get(label, 0) - 1)
                        usage[alt_label] = usage.get(alt_label, 0) + 1
                        logs.append(
                            f"({col_idx},{row_idx}) {old_label} [{old_score:.3f}] -> "
                            f"{alt_label} [{alt_score:.3f}]"
                        )
                        progress = True
                        break

            if not progress:
                break

        return logs

    def detect_board(self, image: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        """Detect all cards; returns (codes, scores, labels) shaped (9, 4)."""
        board_img = image if image is not None else self.capture_screen_bgr()
        slots = self.config["slot_boxes"]

        codes = np.zeros((9, 4), dtype=np.int8)
        scores = np.zeros((9, 4), dtype=np.float32)
        labels: List[List[str]] = [["empty" for _ in range(4)] for _ in range(9)]
        candidates_grid: List[List[List[Tuple[str, float]]]] = [[[] for _ in range(4)] for _ in range(9)]

        for col_idx, column_slots in enumerate(slots):
            for row_idx, slot in enumerate(column_slots):
                slot_img = self._crop_slot(board_img, slot)
                code, score, label, candidates = self.detect_slot(slot_img)
                codes[col_idx, row_idx] = code
                scores[col_idx, row_idx] = score
                labels[col_idx][row_idx] = label
                candidates_grid[col_idx][row_idx] = candidates

        reassign_logs = self._resolve_duplicate_labels(labels, scores, candidates_grid)

        if self.resolve_debug:
            if reassign_logs:
                print("\nDuplicate-resolution reassignments:")
                for line in reassign_logs:
                    print(f"  - {line}")
            else:
                print("\nDuplicate-resolution reassignments: none")

        for col_idx in range(codes.shape[0]):
            for row_idx in range(codes.shape[1]):
                label = labels[col_idx][row_idx]
                codes[col_idx, row_idx] = CARD_TO_CODE[label] if label in CARD_TO_CODE else 0

        return codes, scores, labels

    def _save_feedback_exemplar(self, label: str, slot_img: np.ndarray) -> Path:
        """Save corrected crop as template exemplar and return file path."""
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        base_path = self.templates_dir / f"{label}.png"
        if not base_path.exists():
            out_path = base_path
        else:
            out_path = self._next_variant_path(self.templates_dir, label)
        cv2.imwrite(str(out_path), slot_img)
        return out_path

    def interactive_feedback_loop(
        self,
        board_img: np.ndarray,
        labels: List[List[str]],
        scores: np.ndarray,
        feedback_all: bool = False,
        feedback_threshold: float = 0.80,
        feedback_max: int = 0,
    ) -> int:
        """Interactive correction loop; returns number of newly saved exemplars."""
        slots = self.config["slot_boxes"]
        to_review: List[Tuple[int, int]] = []
        for col_idx, column_slots in enumerate(slots):
            for row_idx, _slot in enumerate(column_slots):
                if feedback_all or float(scores[col_idx, row_idx]) < feedback_threshold:
                    to_review.append((col_idx, row_idx))

        if feedback_max > 0:
            to_review = to_review[:feedback_max]

        if not to_review:
            print("\nFeedback loop: no slots selected for review.")
            return 0

        print("\nFeedback loop commands:")
        print("  [Enter]/a = accept prediction")
        print("  <label>   = correct to canonical label (e.g., 9b, qh)")
        print("  s         = skip slot")
        print("  q         = quit feedback loop")

        saved = 0
        win = "Feedback Slot Crop"
        cv2.namedWindow(win)

        for idx, (col_idx, row_idx) in enumerate(to_review, start=1):
            slot = slots[col_idx][row_idx]
            slot_img = self._crop_slot(board_img, slot)
            candidates = self._slot_candidates(slot_img, topk=5)
            pred = labels[col_idx][row_idx]
            score = float(scores[col_idx, row_idx])

            cv2.imshow(win, slot_img)
            cv2.waitKey(1)

            cand_text = ", ".join([f"{lab}:{sc:.3f}" for lab, sc in candidates]) if candidates else "none"
            print(
                f"\n[{idx}/{len(to_review)}] slot ({col_idx},{row_idx}) "
                f"pred={pred} score={score:.3f}"
            )
            print(f"  candidates: {cand_text}")

            while True:
                answer = input("  feedback> ").strip().lower()
                if answer in {"", "a"}:
                    break
                if answer == "s":
                    break
                if answer == "q":
                    cv2.destroyWindow(win)
                    if saved > 0:
                        self.templates = self._load_templates(self.templates_dir)
                    return saved
                if answer in CARD_TO_CODE:
                    out_path = self._save_feedback_exemplar(answer, slot_img)
                    print(f"  saved exemplar: {out_path.name}")
                    saved += 1
                    break
                print("  invalid label; use canonical label like 9b, 10r, qh, as")

        cv2.destroyWindow(win)
        if saved > 0:
            self.templates = self._load_templates(self.templates_dir)
        return saved

    def render_debug_overlay(self, image: np.ndarray, show_indices: bool = True) -> np.ndarray:
        """Draw all sampling slot boxes on top of the provided screenshot."""
        out = image.copy()
        slots = self.config["slot_boxes"]

        rect = self.config.get("board_rect")
        if rect:
            cv2.rectangle(
                out,
                (int(rect["x1"]), int(rect["y1"])),
                (int(rect["x2"]), int(rect["y2"])),
                (0, 255, 255),
                2,
            )

        for col_idx, column_slots in enumerate(slots):
            for row_idx, slot in enumerate(column_slots):
                x = int(slot["x"])
                y = int(slot["y"])
                w = int(slot["w"])
                h = int(slot["h"])
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
                if show_indices:
                    cv2.putText(
                        out,
                        f"{col_idx},{row_idx}",
                        (x, max(15, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            out,
            "Slot-box debug overlay (sampling regions)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect EXAPUNKS solitaire board state.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to board config JSON.")
    parser.add_argument("--templates-dir", type=Path, default=TEMPLATES_DIR, help="Template directory path.")
    parser.add_argument("--min-score", type=float, default=0.65, help="Minimum template score for known class.")
    parser.add_argument("--red-ratio-threshold", type=float, default=0.04, help="Red pixel ratio threshold for numbered-card color inference.")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration before detection.")
    parser.add_argument("--show-labels", action="store_true", help="Print labels matrix in addition to codes.")
    parser.add_argument("--show-scores", action="store_true", help="Print match confidence matrix.")
    parser.add_argument("--show-color-pass", action="store_true", help="Print first-pass r/b labels for each slot.")
    parser.add_argument("--show-red-ratios", action="store_true", help="Print per-slot red ratios used for r/b decision.")
    parser.add_argument("--color-pass-only", action="store_true", help="Run only the r/b first pass and exit.")
    parser.add_argument("--resolve-debug", action="store_true", help="Print duplicate-resolution reassignment logs.")
    parser.add_argument("--feedback-loop", action="store_true", help="Run interactive accept/correct loop and save new exemplars.")
    parser.add_argument("--feedback-all", action="store_true", help="Review all slots in feedback loop (default: only low-confidence).")
    parser.add_argument("--feedback-threshold", type=float, default=0.80, help="Review slots below this score in feedback loop.")
    parser.add_argument("--feedback-max", type=int, default=0, help="Max slots to review in feedback loop (0 = no limit).")
    parser.add_argument("--debug-overlay", action="store_true", help="Show slot_boxes overlay window before matching.")
    parser.add_argument("--debug-save", type=Path, default=None, help="Save slot_boxes overlay image to this path.")
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


def _print_color_labels(color_labels: List[List[str]]) -> None:
    print("\nColor Pass (col x row):")
    for col_idx, column in enumerate(color_labels):
        print(f"col {col_idx}: {column}")


def _print_red_ratios(red_ratios: np.ndarray) -> None:
    print("\nRed Ratios (col x row):")
    print(np.array2string(red_ratios, precision=3, suppress_small=False))



def main() -> None:
    args = _parse_args()
    _ensure_calibration(args)

    detector = CardDetector(
        config_path=args.config,
        templates_dir=args.templates_dir,
        min_score=args.min_score,
        red_ratio_threshold=args.red_ratio_threshold,
        resolve_debug=args.resolve_debug,
    )

    board_img = detector.capture_screen_bgr()

    if args.debug_overlay or args.debug_save is not None:
        overlay = detector.render_debug_overlay(board_img)
        if args.debug_save is not None:
            args.debug_save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.debug_save), overlay)
            print(f"Saved debug overlay to {args.debug_save}")
        if args.debug_overlay:
            win = "Card Detector Debug Overlay"
            cv2.namedWindow(win)
            cv2.imshow(win, overlay)
            cv2.waitKey(0)
            cv2.destroyWindow(win)

    # First pass: classify each slot only as red/black using HSV red ratio.
    color_labels: List[List[str]] = [["b" for _ in range(4)] for _ in range(9)]
    red_ratios = np.zeros((9, 4), dtype=np.float32)
    slots = detector.config["slot_boxes"]
    for col_idx, column_slots in enumerate(slots):
        for row_idx, slot in enumerate(column_slots):
            slot_img = detector._crop_slot(board_img, slot)
            color, ratio = detector._infer_number_color(slot_img)
            color_labels[col_idx][row_idx] = color
            red_ratios[col_idx, row_idx] = ratio

    if args.show_color_pass or args.color_pass_only:
        _print_color_labels(color_labels)
    if args.show_red_ratios or args.color_pass_only:
        _print_red_ratios(red_ratios)
    if args.color_pass_only:
        return

    codes, scores, labels = detector.detect_board(image=board_img)

    if args.feedback_loop:
        saved = detector.interactive_feedback_loop(
            board_img=board_img,
            labels=labels,
            scores=scores,
            feedback_all=args.feedback_all,
            feedback_threshold=args.feedback_threshold,
            feedback_max=args.feedback_max,
        )
        print(f"\nFeedback loop saved {saved} new exemplar(s).")

    _print_matrix("Codes matrix (shape 9x4)", codes)
    _print_decoded(codes)

    if args.show_labels:
        _print_labels(labels)
    if args.show_scores:
        _print_matrix("Scores matrix (shape 9x4)", scores)


if __name__ == "__main__":
    main()
