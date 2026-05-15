#!/usr/bin/env python3
"""Match live slot shapes against tagged reference examples using contour matching.

Pipeline:
  1. Fullscreen EXAPUNKS and capture screen.
  2. Crop all calibrated slots from board_config.json.
  3. Process each slot (same pipeline as final_processed_slots_view).
  4. Match against reference shapes in tagged_shapes/processed_shapes/ using
     cv2.matchShapes (Hu-moment contour distance).
  5. Print predictions and display a visual grid for accuracy review.

Usage:
    python shape_comparer.py
    python shape_comparer.py --shapes-dir tagged_shapes/processed_shapes
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


DEFAULT_CONFIG = Path("board_config.json")
DEFAULT_PARAMS = Path("edge_centroid_compare_params.json")
DEFAULT_SHAPES_DIR = Path("tagged_shapes/processed_shapes")
ALLOWED_LABELS = ["6", "7", "8", "9", "10", "fh", "fd", "fs", "fc"]
RADIAL_ANGLES = 180
SIG_CACHE_FILENAME = "signatures.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match slot shapes against tagged examples.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument(
        "--shapes-dir",
        type=Path,
        default=DEFAULT_SHAPES_DIR,
        help="Folder containing tagged reference shapes (label_index.png).",
    )
    parser.add_argument(
        "--method",
        type=int,
        default=cv2.CONTOURS_MATCH_I2,
        help="cv2.matchShapes method (1=I1, 2=I2, 3=I3). Default: 2.",
    )
    parser.add_argument("--tile-gap", type=int, default=6)
    parser.add_argument(
        "--sigs-cache",
        type=Path,
        default=None,
        help="Path to precomputed radial signatures JSON. Defaults to <shapes-dir>/signatures.json.",
    )
    parser.add_argument(
        "--precompute-sigs",
        action="store_true",
        help="Recompute and overwrite the radial signatures cache, then exit.",
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
    return {
        "low_threshold": low,
        "high_threshold": high,
        "blur_slider": blur_slider,
        "min_radius_pct": min(100, max(0, min_radius_pct)),
        "max_radius_pct": min(100, max(0, max_radius_pct)),
        "min_component_area": max(0, min_component_area),
    }


def trim_zero_borders(img: np.ndarray) -> np.ndarray:
    ys, xs = np.where(img > 0)
    if xs.size == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def build_processed_mask(crop: np.ndarray, params: Dict[str, int]) -> np.ndarray:
    edges = edge_map(crop, low=params["low_threshold"], high=params["high_threshold"], blur_slider=params["blur_slider"])
    filtered = filter_edges_by_centroid_distance(
        edges,
        min_radius_pct=params["min_radius_pct"],
        max_radius_pct=params["max_radius_pct"],
        min_component_area=params["min_component_area"],
    )
    return trim_zero_borders(filtered)


def load_reference_shapes(shapes_dir: Path) -> Dict[str, List[np.ndarray]]:
    """Load all label_index.png files and group contours by label."""
    if not shapes_dir.exists():
        raise FileNotFoundError(f"Shapes directory not found: {shapes_dir}")

    refs: Dict[str, List[np.ndarray]] = defaultdict(list)
    for p in sorted(shapes_dir.glob("*.png")):
        parts = p.stem.split("_")
        if len(parts) < 2:
            continue
        # Label may contain underscores for multi-char labels like "10", but we
        # use only the last part as index, everything before as label.
        label = "_".join(parts[:-1])
        if label not in ALLOWED_LABELS:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        refs[label].append(img)

    if not refs:
        raise ValueError(f"No valid reference shapes found in {shapes_dir}.")

    print(f"Loaded references: { {k: len(v) for k, v in refs.items()} }")
    return refs


# ---------------------------------------------------------------------------
# Radial signature helpers
# ---------------------------------------------------------------------------

def radial_signature(mask: np.ndarray, n_angles: int = RADIAL_ANGLES) -> Optional[np.ndarray]:
    """Compute a normalized radial support signature from the mask centroid.

    For each of *n_angles* directions evenly spaced over [0, π), project every
    white pixel onto that direction and record the span (max − min projection
    from centroid).  The result is normalised so the maximum value is 1.
    Using [0, π) instead of [0, 2π) is deliberate: the projection span is
    symmetric, so we only need half the circle — giving 180 independent
    measurements across the full 360°.

    Returns an ndarray of shape (n_angles,), or None for an empty mask.
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    cx, cy = xs.mean(), ys.mean()
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    sig = np.empty(n_angles, dtype=np.float64)
    for i, a in enumerate(angles):
        proj = dx * np.cos(a) + dy * np.sin(a)
        sig[i] = proj.max() - proj.min()
    max_val = sig.max()
    if max_val > 0:
        sig /= max_val
    return sig


def precompute_and_save_signatures(
    refs: Dict[str, List[np.ndarray]],
    cache_path: Path,
    n_angles: int = RADIAL_ANGLES,
) -> Dict[str, List[List[float]]]:
    """Compute radial signatures for every reference image and save to JSON.

    The JSON structure is  {label: [[sig_float, ...], ...], ...}  — one list
    of floats per reference image (multiple examples per label are all kept).
    """
    data: Dict[str, List[List[float]]] = {}
    for label, imgs in refs.items():
        sigs = []
        for img in imgs:
            s = radial_signature(img, n_angles)
            if s is not None:
                sigs.append(s.tolist())
        data[label] = sigs
        print(f"  {label}: {len(sigs)} signature(s) computed")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({"n_angles": n_angles, "signatures": data}, f)
    print(f"Saved signatures cache → {cache_path}")
    return data


def load_or_build_signatures(
    refs: Dict[str, List[np.ndarray]],
    cache_path: Path,
    n_angles: int = RADIAL_ANGLES,
) -> Dict[str, List[np.ndarray]]:
    """Load pre-saved signatures from JSON, or build them if cache is absent.

    Returns {label: [sig_array, ...]} where each sig_array has shape (n_angles,).
    """
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        cached_n = int(raw.get("n_angles", n_angles))
        if cached_n != n_angles:
            print(
                f"WARNING: cached n_angles={cached_n} != requested {n_angles}; recomputing."
            )
        else:
            print(f"Loaded radial signatures cache from {cache_path}")
            result: Dict[str, List[np.ndarray]] = {}
            for lbl, sigs in raw["signatures"].items():
                result[lbl] = [np.array(s, dtype=np.float64) for s in sigs]
            return result

    print("No signatures cache found; computing now...")
    raw_data = precompute_and_save_signatures(refs, cache_path, n_angles)
    return {lbl: [np.array(s, dtype=np.float64) for s in sigs] for lbl, sigs in raw_data.items()}


def match_radial(
    query_mask: np.ndarray,
    radial_refs: Dict[str, List[np.ndarray]],
    n_angles: int = RADIAL_ANGLES,
) -> Tuple[Optional[str], float]:
    """Match query mask to reference signatures using L2 distance. Lower = better."""
    q_sig = radial_signature(query_mask, n_angles)
    if q_sig is None:
        return None, float("inf")
    best_label: Optional[str] = None
    best_score = float("inf")
    for label, sigs in radial_refs.items():
        for ref_sig in sigs:
            score = float(np.linalg.norm(q_sig - ref_sig))
            if score < best_score:
                best_score = score
                best_label = label
    return best_label, best_score


def predict(
    query_mask: np.ndarray,
    refs: Dict[str, List[np.ndarray]],
    radial_refs: Dict[str, List[np.ndarray]],
    method: int,
) -> Tuple[Optional[str], float, str]:
    """Combined prediction: contour match first; fall back to radial if score != 0.

    Returns (label, score, method_name) where method_name is 'hu' or 'rad'.
    """
    label, score = match_shape(query_mask, refs, method)
    if score == 0.0:
        return label, score, "hu"
    rad_label, rad_score = match_radial(query_mask, radial_refs)
    return rad_label, rad_score, "rad"


# ---------------------------------------------------------------------------
# Contour helpers
# ---------------------------------------------------------------------------

def contour_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest contour from a binary mask, or None if empty."""
    if mask is None or np.count_nonzero(mask) == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def match_shape(
    query_mask: np.ndarray,
    refs: Dict[str, List[np.ndarray]],
    method: int,
) -> Tuple[Optional[str], float]:
    """Return (best_label, score) where lower score = better match."""
    query_cnt = contour_from_mask(query_mask)
    if query_cnt is None:
        return None, float("inf")

    best_label: Optional[str] = None
    best_score = float("inf")

    for label, ref_imgs in refs.items():
        for ref_img in ref_imgs:
            ref_cnt = contour_from_mask(ref_img)
            if ref_cnt is None:
                continue
            try:
                score = cv2.matchShapes(query_cnt, ref_cnt, method, 0.0)
            except cv2.error:
                continue
            if score < best_score:
                best_score = score
                best_label = label

    return best_label, best_score


def annotate_prediction_tile(
    crop: np.ndarray,
    processed: np.ndarray,
    prediction: Optional[str],
    score: float,
    label: str,
    method_name: str = "",
) -> np.ndarray:
    orig_up = upscale_to_min(crop)
    h, w = orig_up.shape[:2]

    if np.count_nonzero(processed) == 0:
        proc_up = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        p_resized = cv2.resize(processed, (w, h), interpolation=cv2.INTER_NEAREST)
        proc_up = cv2.cvtColor(p_resized, cv2.COLOR_GRAY2BGR)
        proc_up[p_resized > 0] = (0, 255, 0)

    side = cv2.hconcat([orig_up, proc_up])

    strip_h = LABEL_H + 4
    strip = np.zeros((strip_h, side.shape[1], 3), dtype=np.uint8)
    if prediction is None:
        pred_text = f"{label}: NO MATCH"
        color = (0, 100, 255)
    else:
        method_tag = f"[{method_name}]" if method_name else ""
        pred_text = f"{label}: {prediction} {method_tag} ({score:.3f})"
        color = (0, 255, 0) if method_name == "hu" else (0, 200, 255)
    cv2.putText(strip, pred_text, (4, strip_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return cv2.vconcat([side, strip])


def build_result_mosaic(
    crops: List[List[np.ndarray]],
    processed_grid: List[List[np.ndarray]],
    predictions: List[List[Tuple[Optional[str], float]]],
    tile_gap: int,
) -> np.ndarray:
    cols = len(crops)
    rows = len(crops[0]) if cols else 0
    if cols == 0 or rows == 0:
        return np.zeros((200, 400, 3), dtype=np.uint8)

    sample = upscale_to_min(crops[0][0])
    disp_h, disp_w = sample.shape[:2]
    tile_w = disp_w * 2
    tile_h = disp_h + LABEL_H + 4
    header_h = 34

    canvas_h = header_h + rows * tile_h + (rows + 1) * tile_gap
    canvas_w = cols * tile_w + (cols + 1) * tile_gap
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], header_h), (0, 0, 0), -1)
    cv2.putText(canvas, "Left: original  |  Right: processed  |  Label: prediction (score)",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    for c in range(cols):
        for r in range(rows):
            pred, score, mname = predictions[c][r]
            slot_label = f"c{c}r{r}"
            tile = annotate_prediction_tile(crops[c][r], processed_grid[c][r], pred, score, slot_label, mname)
            y1 = header_h + tile_gap + r * (tile_h + tile_gap)
            x1 = tile_gap + c * (tile_w + tile_gap)
            canvas[y1:y1 + tile_h, x1:x1 + tile_w] = tile

    return canvas


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    params = load_params(args.params)
    slot_boxes: List[List[Dict[str, int]]] = cfg["slot_boxes"]
    if not slot_boxes or not slot_boxes[0]:
        raise ValueError("Config slot_boxes is empty.")

    refs = load_reference_shapes(args.shapes_dir)

    sigs_cache = args.sigs_cache if args.sigs_cache else args.shapes_dir / SIG_CACHE_FILENAME

    if args.precompute_sigs:
        precompute_and_save_signatures(refs, sigs_cache)
        print("Done. Exiting.")
        return

    radial_refs = load_or_build_signatures(refs, sigs_cache)

    print(
        f"Params: low={params['low_threshold']} high={params['high_threshold']} "
        f"blur_k={odd_kernel_from_slider(params['blur_slider'])} "
        f"min_r%={params['min_radius_pct']} max_r%={params['max_radius_pct']} "
        f"min_area={params['min_component_area']}"
    )

    _ensure_exapunks_fullscreen()
    print("Capturing screenshot and processing slots...")
    crops = capture_all_crops(slot_boxes)

    # Build processed masks and run matching.
    processed_grid: List[List[np.ndarray]] = []
    predictions: List[List[Tuple[Optional[str], float]]] = []

    cols = len(crops)
    for c in range(cols):
        col_processed = []
        col_preds = []
        for r in range(len(crops[c])):
            mask = build_processed_mask(crops[c][r], params)
            pred, score, mname = predict(mask, refs, radial_refs, method=args.method)
            col_processed.append(mask)
            col_preds.append((pred, score, mname))

            print(f"  slot c{c} r{r}: predicted={pred}  method={mname}  score={score:.4f}")

        processed_grid.append(col_processed)
        predictions.append(col_preds)

    mosaic = build_result_mosaic(crops, processed_grid, predictions, tile_gap=args.tile_gap)

    cv2.namedWindow("Shape Comparer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Shape Comparer", 1800, 950)
    cv2.imshow("Shape Comparer", mosaic)
    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
