#!/usr/bin/env python3
"""Shared card label and integer encoding utilities for EXAPUNKS solitaire CV."""

from __future__ import annotations

from typing import Dict, List

NUMBERED_LABELS: List[str] = [
    "6b",
    "6r",
    "7b",
    "7r",
    "8b",
    "8r",
    "9b",
    "9r",
    "10b",
    "10r",
]

FACE_LABELS: List[str] = ["face_h", "face_d", "face_c", "face_s"]

ALL_LABELS: List[str] = NUMBERED_LABELS + FACE_LABELS

CARD_TO_CODE: Dict[str, int] = {label: idx + 1 for idx, label in enumerate(ALL_LABELS)}
CODE_TO_CARD: Dict[int, str] = {code: label for label, code in CARD_TO_CODE.items()}


def is_valid_label(label: str) -> bool:
    """Return True if label belongs to the known 14 classes."""
    return label in CARD_TO_CODE


def encode_label(label: str) -> int:
    """Encode card label into compact integer code (1..14)."""
    try:
        return CARD_TO_CODE[label]
    except KeyError as exc:
        raise ValueError(f"Unknown card label: {label}") from exc


def decode_code(code: int) -> str:
    """Decode integer code (1..14) into card label."""
    if code == 0:
        return "empty"
    try:
        return CODE_TO_CARD[code]
    except KeyError as exc:
        raise ValueError(f"Unknown card code: {code}") from exc
