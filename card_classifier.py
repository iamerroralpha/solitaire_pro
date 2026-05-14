#!/usr/bin/env python3
"""Learned template-based classifier for EXAPUNKS solitaire cards.

This is a lightweight real classifier:
- feature extraction from slot crops
- augmentation of the captured template images
- per-class prototype learning via nearest centroid
- cosine-similarity prediction with confidence / margin rejection
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from card_encoding import CARD_TO_CODE


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    margin: float
    topk: List[Tuple[str, float]]


class CardClassifier:
    """Nearest-centroid classifier trained from augmented template crops."""

    def __init__(
        self,
        templates_dir: Path,
        confidence_threshold: float = 0.72,
        margin_threshold: float = 0.035,
        feature_size: Tuple[int, int] = (32, 32),
    ) -> None:
        self.templates_dir = templates_dir
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.feature_size = feature_size
        self.centroids = self._build_centroids(self._load_templates(templates_dir))

    @staticmethod
    def _load_templates(templates_dir: Path) -> Dict[str, np.ndarray]:
        if not templates_dir.exists():
            raise FileNotFoundError(f"Missing templates dir: {templates_dir}")

        templates: Dict[str, np.ndarray] = {}
        for file_path in sorted(templates_dir.glob("*.png")):
            label = file_path.stem.lower()
            if label not in CARD_TO_CODE:
                continue
            img = cv2.imread(str(file_path))
            if img is None:
                continue
            templates[label] = img

        if not templates:
            raise RuntimeError("No valid templates found. Run capture_templates.py first.")

        return templates

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, self.feature_size, interpolation=cv2.INTER_AREA)

    def _augmentations(self, img: np.ndarray) -> Iterable[np.ndarray]:
        """Generate mild synthetic variants of a template crop."""
        yield img

        rows, cols = img.shape[:2]
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in shifts:
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(img, matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT101)
            yield shifted

        for alpha, beta in [(0.90, 0), (1.10, 0), (1.00, -12), (1.00, 12)]:
            adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            yield adjusted

        for ksize in [(3, 3), (5, 5)]:
            blurred = cv2.GaussianBlur(img, ksize, 0)
            yield blurred

    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        resized = self._resize(img)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        edge = cv2.Canny((gray * 255.0).astype(np.uint8), 40, 140).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        hist_parts: List[np.ndarray] = []
        hist_specs = [
            (0, 12, [0, 180]),
            (1, 8, [0, 256]),
            (2, 8, [0, 256]),
        ]
        for channel, bins, value_range in hist_specs:
            hist = cv2.calcHist([hsv], [channel], None, [bins], value_range)
            hist = cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
            hist_parts.append(hist)

        feature = np.concatenate([
            gray.flatten(),
            edge.flatten(),
            *hist_parts,
            hsv.mean(axis=(0, 1)).astype(np.float32) / np.array([180.0, 255.0, 255.0], dtype=np.float32),
        ]).astype(np.float32)

        norm = float(np.linalg.norm(feature))
        if norm > 0:
            feature /= norm
        return feature

    def _build_centroids(self, templates: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        centroids: Dict[str, np.ndarray] = {}
        for label, template_img in templates.items():
            samples = [self._extract_features(augmented) for augmented in self._augmentations(template_img)]
            centroid = np.mean(np.stack(samples, axis=0), axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid /= norm
            centroids[label] = centroid.astype(np.float32)
        return centroids

    def _similarities(self, feature: np.ndarray) -> List[Tuple[str, float]]:
        sims: List[Tuple[str, float]] = []
        for label, centroid in self.centroids.items():
            score = float(np.dot(feature, centroid))
            sims.append((label, score))
        sims.sort(key=lambda item: item[1], reverse=True)
        return sims

    def predict(self, img: np.ndarray, topk: int = 3) -> Prediction:
        """Predict the best card label for a crop."""
        if img.size == 0:
            return Prediction(label="unknown", score=0.0, margin=0.0, topk=[])

        feature = self._extract_features(img)
        sims = self._similarities(feature)
        if not sims:
            return Prediction(label="unknown", score=0.0, margin=0.0, topk=[])

        best_label, best_score = sims[0]
        second_score = sims[1][1] if len(sims) > 1 else -1.0
        margin = best_score - second_score
        if best_score < self.confidence_threshold or margin < self.margin_threshold:
            return Prediction(label="unknown", score=best_score, margin=margin, topk=sims[:topk])

        return Prediction(label=best_label, score=best_score, margin=margin, topk=sims[:topk])
