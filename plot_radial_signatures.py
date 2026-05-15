#!/usr/bin/env python3
"""Plot radial distance signatures from a JSON file produced by radial_signature_extractor.py.

Each signature is shown as a polar curve. Files are laid out in a grid sorted by filename.

Usage:
    python plot_radial_signatures.py
    python plot_radial_signatures.py --input radial_signatures.json --show
    python plot_radial_signatures.py --output plot.png --no-save
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot radial signatures.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("radial_signatures.json"),
        help="Input JSON produced by radial_signature_extractor.py (default: radial_signatures.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("radial_signatures_plot.png"),
        help="Output image path (default: radial_signatures_plot.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Radial Distance Signatures",
        help="Plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the interactive plot window.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the plot to disk.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=0,
        help="Number of columns in the subplot grid (0 = auto).",
    )
    return parser.parse_args()


def load_signatures(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Signatures file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    if not entries:
        raise ValueError("No entries found in signatures file.")
    return entries


def make_theta(n: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, n, endpoint=False)


def close_curve(theta: np.ndarray, r: np.ndarray):
    """Append first point to close the polar curve."""
    return np.append(theta, theta[0]), np.append(r, r[0])


def plot_signatures(entries: List[Dict[str, Any]], title: str, ncols: int) -> plt.Figure:
    n = len(entries)
    if ncols <= 0:
        ncols = min(8, n)
    nrows = max(1, int(np.ceil(n / ncols)))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={"projection": "polar"},
        figsize=(ncols * 2.2, nrows * 2.2),
    )
    fig.suptitle(title, fontsize=14, y=1.01)

    # Normalize axes to a flat list for easy iteration.
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.array(axes).flatten().tolist()

    for idx, (ax, entry) in enumerate(zip(axes_flat, entries)):
        sig = np.array(entry["signature"], dtype=np.float32)
        theta = make_theta(len(sig))
        theta_c, r_c = close_curve(theta, sig)

        ax.plot(theta_c, r_c, linewidth=1.2, color="steelblue")
        ax.fill(theta_c, r_c, alpha=0.15, color="steelblue")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        label = entry.get("file", f"#{idx}")
        ax.set_title(label, fontsize=7, pad=2)

    # Hide unused axes.
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    entries = load_signatures(args.input)
    print(f"Loaded {len(entries)} signature(s) from {args.input}")

    fig = plot_signatures(entries, title=args.title, ncols=args.cols)

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(args.output), dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.output}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
