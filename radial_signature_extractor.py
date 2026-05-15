#!/usr/bin/env python3
"""Extract radial distance signatures from images in crops/original/.

Pipeline per image:
  1. Load image -> grayscale
  2. Gaussian blur  (kernel 3, i.e. blur-slider=1)
  3. Canny edge detection  (threshold=60, upper = 60*3 = 180)
  4. Compute radial distance signature from edge pixels
  5. Write all signatures to JSON

Usage:
    python radial_signature_extractor.py
    python radial_signature_extractor.py --input-dir crops/original --output radial_signatures.json --bins 72
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract radial signatures from images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("crops/original"),
        help="Folder containing input images (default: crops/original).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("radial_signatures.json"),
        help="Output JSON file (default: radial_signatures.json).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=72,
        help="Number of angular bins for each signature (default: 72).",
    )
    parser.add_argument(
        "--canny-threshold",
        type=int,
        default=60,
        help="Canny lower threshold (upper = threshold * 3, default: 60).",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=1,
        help="Blur slider value (kernel = 2*blur+1, default: 1 -> kernel 3).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip the interactive Gaussian preview grid.",
    )
    return parser.parse_args()


def apply_gaussian(gray: np.ndarray, blur_slider: int) -> np.ndarray:
    """Return Gaussian-blurred grayscale image."""
    kernel = max(1, 2 * int(blur_slider) + 1)
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def compute_edges(blurred: np.ndarray, threshold: int) -> np.ndarray:
    """Run Canny on an already-blurred grayscale image."""
    low = max(0, int(threshold))
    high = min(255, int(max(low + 1, low * 3)))
    return cv2.Canny(blurred, low, high, L2gradient=True)


def show_gaussian_grid(names: List[str], blurred_images: List[np.ndarray]) -> None:
    """Display a grid of Gaussian-filtered images and wait for the window to be closed."""
    n = len(blurred_images)
    ncols = min(8, n)
    nrows = max(1, int(np.ceil(n / ncols)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.2))
    fig.suptitle("Step 1 — Gaussian filtered images", fontsize=13)

    axes_flat = np.array(axes).flatten().tolist() if n > 1 else [axes]

    for ax, name, img in zip(axes_flat, names, blurred_images):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=7, pad=2)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def interpolate_circular_zeros(signature: np.ndarray) -> np.ndarray:
    """Fill zero bins by circular nearest-neighbor interpolation."""
    n = signature.size
    if n == 0:
        return signature
    nonzero = np.where(signature > 0)[0]
    if nonzero.size == 0:
        return signature
    # Wrap support points around the circle.
    x = np.concatenate([nonzero - n, nonzero, nonzero + n])
    y = np.concatenate([signature[nonzero]] * 3)
    out = signature.copy()
    zero_idx = np.where(out == 0)[0]
    if zero_idx.size > 0:
        out[zero_idx] = np.interp(zero_idx, x, y)
    return out


def radial_distance_signature(edges: np.ndarray, bins: int = 72) -> Dict[str, object]:
    """Compute radial distance signature from a binary edge image."""
    ys, xs = np.where(edges > 0)

    if xs.size == 0:
        h, w = edges.shape[:2]
        return {
            "centroid": {"x": float(w) / 2.0, "y": float(h) / 2.0},
            "edge_pixels": 0,
            "max_radius": 0.0,
            "signature": [0.0] * bins,
        }

    cx = float(xs.mean())
    cy = float(ys.mean())

    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    radii = np.sqrt(dx * dx + dy * dy)
    angles = np.arctan2(dy, dx)
    angles = np.where(angles < 0, angles + 2.0 * np.pi, angles)

    bin_idx = np.clip(
        np.floor((angles / (2.0 * np.pi)) * bins).astype(np.int32), 0, bins - 1
    )

    signature = np.zeros((bins,), dtype=np.float32)
    for i, r in zip(bin_idx, radii):
        if r > signature[i]:
            signature[i] = r

    signature = interpolate_circular_zeros(signature)
    max_radius = float(signature.max())

    if max_radius > 0:
        signature = signature / max_radius

    return {
        "centroid": {"x": cx, "y": cy},
        "edge_pixels": int(xs.size),
        "max_radius": max_radius,
        "signature": signature.tolist(),
    }


def collect_images(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No supported images found in {input_dir}")
    return files


def main() -> None:
    args = parse_args()

    images = collect_images(args.input_dir)
    print(f"Found {len(images)} image(s) in {args.input_dir}")

    # ── Pass 1: load images and apply Gaussian filter ──────────────────────
    loaded: List[Tuple[Path, np.ndarray, np.ndarray]] = []  # (path, gray, blurred)
    for img_path in images:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [SKIP] Could not read: {img_path.name}")
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = apply_gaussian(gray, blur_slider=args.blur)
        loaded.append((img_path, gray, blurred))

    if not args.no_preview and loaded:
        show_gaussian_grid(
            names=[p.name for p, _, _ in loaded],
            blurred_images=[b for _, _, b in loaded],
        )

    # ── Pass 2: edge detection + radial signature ────────────────────────────
    entries: List[Dict[str, object]] = []
    for img_path, _gray, blurred in loaded:
        edges = compute_edges(blurred, threshold=args.canny_threshold)
        sig = radial_distance_signature(edges, bins=args.bins)

        print(
            f"  {img_path.name}: edge_pixels={sig['edge_pixels']}, "
            f"max_radius={sig['max_radius']:.2f}"
        )

        entries.append({
            "file": img_path.name,
            "centroid": sig["centroid"],
            "edge_pixels": sig["edge_pixels"],
            "max_radius": sig["max_radius"],
            "signature": sig["signature"],
        })

    payload = {
        "bins": args.bins,
        "canny_threshold": args.canny_threshold,
        "blur_kernel": 2 * args.blur + 1,
        "input_dir": str(args.input_dir),
        "total": len(entries),
        "entries": entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved {len(entries)} signature(s) to {args.output}")


if __name__ == "__main__":
    main()
