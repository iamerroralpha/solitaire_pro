#!/usr/bin/env python3
"""Interactive template capture for EXAPUNKS solitaire peek strips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import ImageGrab

from card_encoding import ALL_LABELS, is_valid_label

CONFIG_PATH = Path("board_config.json")
TEMPLATES_DIR = Path("templates")


def _is_numbered_base_label(label: str) -> bool:
    return len(label) >= 2 and label[-1] in {"r", "b"} and label[:-1].isdigit()


def _next_numbered_variant_path(templates_dir: Path, base_label: str) -> Path:
    """Return first available variant path: base_1.png, base_2.png, ..."""
    idx = 1
    while True:
        candidate = templates_dir / f"{base_label}_{idx}.png"
        if not candidate.exists():
            return candidate
        idx += 1



def _load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}. Run calibrate.py first.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def _capture_screen_bgr() -> np.ndarray:
    screenshot = ImageGrab.grab()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)



def _crop(img: np.ndarray, slot: Dict[str, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, int(slot["x"]))
    y1 = max(0, int(slot["y"]))
    x2 = min(w, x1 + int(slot["w"]))
    y2 = min(h, y1 + int(slot["h"]))
    return img[y1:y2, x1:x2].copy()



def _canonicalize_label(raw: str) -> Optional[str]:
    s = raw.strip().lower()
    if not s:
        return None
    if s in {"skip", "s"}:
        return "__skip__"
    if s in {"quit", "q", "exit"}:
        return "__quit__"

    if is_valid_label(s):
        return s

    suit_map = {"h": "r", "d": "r", "c": "b", "s": "b"}

    # Number aliases: 8h -> 8r, 10d -> 10r, 7c -> 7b
    for rank in ("6", "7", "8", "9", "10"):
        if s.startswith(rank) and len(s) == len(rank) + 1 and s[-1] in suit_map:
            return f"{rank}{suit_map[s[-1]]}"

    # Face labels are now exact rank+suit: jh/qh/kh/ah ... js/qs/ks/as
    if len(s) == 2 and s[0] in {"j", "q", "k", "a"} and s[1] in {"h", "d", "c", "s"}:
        return s

    return None



def _draw_slot_overlay(base: np.ndarray, slot: Dict[str, int], pos_text: str) -> np.ndarray:
    out = base.copy()
    x, y, w, h = int(slot["x"]), int(slot["y"]), int(slot["w"]), int(slot["h"])
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(
        out,
        pos_text,
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out



def capture_templates(config_path: Path, templates_dir: Path, overwrite: bool) -> None:
    config = _load_config(config_path)
    slots = config["slot_boxes"]

    templates_dir.mkdir(parents=True, exist_ok=True)

    board_img = _capture_screen_bgr()
    preview_window = "Slot Preview"
    crop_window = "Peek Crop"
    cv2.namedWindow(preview_window)
    cv2.namedWindow(crop_window)

    print("\nCapture labels for each visible slot.")
    print("Valid canonical labels:", ", ".join(ALL_LABELS))
    print("Aliases: 8h->8r, 8c->8b  |  face labels must be exact: jh qh kh ah jd qd ... as")
    print("Commands: skip/s, quit/q\n")

    saved = 0
    skipped_existing = 0

    for col_idx, column_slots in enumerate(slots):
        for row_idx, slot in enumerate(column_slots):
            crop = _crop(board_img, slot)
            if crop.size == 0:
                print(f"[{col_idx},{row_idx}] empty crop, skipping")
                continue

            pos_text = f"col {col_idx} row {row_idx}"
            overlay = _draw_slot_overlay(board_img, slot, pos_text)
            cv2.imshow(preview_window, overlay)
            cv2.imshow(crop_window, crop)
            cv2.waitKey(1)

            while True:
                raw = input(f"[{col_idx},{row_idx}] label: ")
                label = _canonicalize_label(raw)
                if label is None:
                    print("Invalid label. Try again.")
                    continue
                if label == "__skip__":
                    break
                if label == "__quit__":
                    cv2.destroyAllWindows()
                    print(f"Stopped. Saved={saved}, skipped_existing={skipped_existing}")
                    return

                out_path = templates_dir / f"{label}.png"
                if out_path.exists() and not overwrite:
                    if _is_numbered_base_label(label):
                        out_path = _next_numbered_variant_path(templates_dir, label)
                        print(f"Base template exists, saving numbered variant as {out_path.name}")
                    else:
                        skipped_existing += 1
                        print(f"{out_path} already exists, skipped")
                        break

                cv2.imwrite(str(out_path), crop)
                saved += 1
                print(f"Saved {out_path}")
                break

    cv2.destroyAllWindows()
    print(f"Done. Saved={saved}, skipped_existing={skipped_existing}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture peek-strip templates for EXAPUNKS solitaire.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to board config JSON.")
    parser.add_argument("--templates-dir", type=Path, default=TEMPLATES_DIR, help="Templates output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing templates.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    capture_templates(args.config, args.templates_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
