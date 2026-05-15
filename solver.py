#!/usr/bin/env python3
"""Best-first logical solver for EXAPUNKS solitaire.

This solver is intentionally pure logic and follows the provided pseudocode:
- Data model with 10 stacks (0..8 board, 9 spare)
- Move generation using largest movable suffix
- Finished-stack detection rules
- Best-first search on (unfinished_stacks, move_count)

On each run, it first refreshes board state by invoking shape_comparer.py in
solver mode to produce normalized_board_state.json.
"""

from __future__ import annotations

import argparse
import heapq
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Rank convention: 6..10 for numbered cards, 12 for face cards (king-like class).
KING_RANK = 12
BOARD_STACK_COUNT = 9
SPARE_STACK_INDEX = 9
TOTAL_STACKS = 10

# Suit convention: 0..3, with deterministic color mapping.
# 0 = clubs (black), 1 = spades (black), 2 = diamonds (red), 3 = hearts (red)
SUIT_COLOR = {
    0: 0,  # black
    1: 0,  # black
    2: 1,  # red
    3: 1,  # red
}

SUIT_FROM_CHAR = {
    "c": 0,
    "s": 1,
    "d": 2,
    "h": 3,
}

COLOR_NAME = {
    0: "black",
    1: "red",
}


@dataclass(frozen=True)
class Card:
    rank: int
    suit: int

    @property
    def color(self) -> int:
        return SUIT_COLOR[self.suit]

    def label(self) -> str:
        if self.rank == KING_RANK:
            suit_char = {0: "c", 1: "s", 2: "d", 3: "h"}[self.suit]
            return f"f{suit_char}"
        suffix = "r" if self.color == 1 else "b"
        return f"{self.rank}{suffix}"


@dataclass(frozen=True)
class Move:
    src_index: int
    dst_index: int
    count: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "source_stack": self.src_index,
            "target_stack": self.dst_index,
            "move_count": self.count,
        }


@dataclass
class State:
    stacks: List[List[Card]]
    moves: List[Move]


class SolverError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve EXAPUNKS solitaire board using best-first search.")
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("normalized_board_state.json"),
        help="Normalized board state path (refreshed automatically before solve).",
    )
    parser.add_argument(
        "--plan-out",
        type=Path,
        default=Path("planned_moves.json"),
        help="Output JSON file for planned move sequence.",
    )
    parser.add_argument(
        "--shape-comparer",
        type=Path,
        default=Path("shape_comparer.py"),
        help="Path to shape_comparer script used to refresh state before solving.",
    )
    parser.add_argument(
        "--max-expanded",
        type=int,
        default=200000,
        help="Safety cap on expanded states.",
    )
    return parser.parse_args()


def refresh_state_with_shape_comparer(shape_comparer: Path, state_path: Path) -> None:
    if not shape_comparer.exists():
        raise SolverError(f"shape_comparer script not found: {shape_comparer}")

    cmd = [
        sys.executable,
        str(shape_comparer),
        "--state-out",
        str(state_path),
        "--solver-mode",
    ]

    print("Refreshing board state via shape_comparer...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SolverError(
            "shape_comparer failed while refreshing board state "
            f"(exit code {exc.returncode})."
        ) from exc

    if not state_path.exists():
        raise SolverError(f"shape_comparer did not produce state file: {state_path}")


def parse_card_from_label(label: str) -> Card:
    """Parse card label from shape_comparer output into rank/suit model.

    Supported labels:
    - Numbered: 6b, 7r, ..., 10b
    - Face-like: fc, fs, fd, fh (mapped to rank=12)
    """
    label = label.strip().lower()

    if len(label) >= 2 and label[0] == "f" and label[1] in SUIT_FROM_CHAR:
        return Card(rank=KING_RANK, suit=SUIT_FROM_CHAR[label[1]])

    if label.endswith("b") or label.endswith("r"):
        rank_str = label[:-1]
        try:
            rank = int(rank_str)
        except ValueError as exc:
            raise SolverError(f"Invalid numbered label: {label}") from exc

        color = 0 if label.endswith("b") else 1
        # Number labels do not encode full suit, only color. Use deterministic
        # pseudo-suit representative per color to preserve color behavior.
        suit = 0 if color == 0 else 2
        return Card(rank=rank, suit=suit)

    raise SolverError(f"Unrecognized card label: {label}")


def load_initial_state(path: Path) -> State:
    if not path.exists():
        raise SolverError(f"State file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    raw_columns = raw.get("columns")
    if not isinstance(raw_columns, list):
        raise SolverError("State JSON missing 'columns' list")

    if len(raw_columns) != BOARD_STACK_COUNT:
        raise SolverError(f"Expected {BOARD_STACK_COUNT} board columns, got {len(raw_columns)}")

    stacks: List[List[Card]] = [[] for _ in range(TOTAL_STACKS)]

    for i in range(BOARD_STACK_COUNT):
        col = raw_columns[i]
        if not isinstance(col, list):
            raise SolverError(f"Column {i} is not a list")

        parsed_col: List[Card] = []
        for card_obj in col:
            if not isinstance(card_obj, dict):
                continue
            label = card_obj.get("label")
            if not isinstance(label, str):
                continue
            parsed_col.append(parse_card_from_label(label))

        stacks[i] = parsed_col

    # Spare stack at index 9: optional single card from holder.
    holder = raw.get("holder")
    holder_card: Optional[Card] = None
    if isinstance(holder, dict):
        holder_obj = holder.get("card")
        if isinstance(holder_obj, dict):
            holder_label = holder_obj.get("label")
            if isinstance(holder_label, str):
                holder_card = parse_card_from_label(holder_label)

    stacks[SPARE_STACK_INDEX] = [holder_card] if holder_card is not None else []

    return State(stacks=stacks, moves=[])


def is_valid_sequence(bottom: Card, top: Card) -> bool:
    # Normal descending alternating-color link.
    if bottom.rank == top.rank + 1 and bottom.color != top.color:
        return True

    # Special rule: King on King of same suit.
    if bottom.rank == KING_RANK and top.rank == KING_RANK and bottom.suit == top.suit:
        return True

    return False


def is_stack_finished(stack: List[Card]) -> bool:
    # Pattern A: 4 kings of same suit.
    if len(stack) == 4:
        all_kings = all(card.rank == KING_RANK for card in stack)
        same_suit = all(card.suit == stack[0].suit for card in stack)
        if all_kings and same_suit:
            return True

    # Pattern B: 10-9-8-7-6 alternating colors.
    if len(stack) == 5:
        ranks_ok = (
            stack[0].rank == 10
            and stack[1].rank == 9
            and stack[2].rank == 8
            and stack[3].rank == 7
            and stack[4].rank == 6
        )
        colors_ok = (
            stack[0].color != stack[1].color
            and stack[1].color != stack[2].color
            and stack[2].color != stack[3].color
            and stack[3].color != stack[4].color
        )
        if ranks_ok and colors_ok:
            return True

    return False


def count_unfinished(state: State) -> int:
    count = 0
    for i in range(BOARD_STACK_COUNT):
        if not is_stack_finished(state.stacks[i]):
            count += 1
    return count


def movable_suffix_start(stack: List[Card]) -> int:
    # Returns first index of largest movable tail segment.
    if not stack:
        return -1

    i = len(stack) - 1
    while i > 0 and is_valid_sequence(stack[i - 1], stack[i]):
        i -= 1
    return i


def generate_moves(state: State) -> List[Move]:
    moves: List[Move] = []

    unfinished = [i for i in range(BOARD_STACK_COUNT) if not is_stack_finished(state.stacks[i])]

    # 1) Any top card -> spare, only if spare empty.
    if not state.stacks[SPARE_STACK_INDEX]:
        for src in unfinished:
            if state.stacks[src]:
                moves.append(Move(src_index=src, dst_index=SPARE_STACK_INDEX, count=1))

    # 2) Spare -> any unfinished stack.
    if state.stacks[SPARE_STACK_INDEX]:
        spare_card = state.stacks[SPARE_STACK_INDEX][0]
        for dst in unfinished:
            if not state.stacks[dst]:
                moves.append(Move(src_index=SPARE_STACK_INDEX, dst_index=dst, count=1))
            else:
                dst_top = state.stacks[dst][-1]
                if is_valid_sequence(dst_top, spare_card):
                    moves.append(Move(src_index=SPARE_STACK_INDEX, dst_index=dst, count=1))

    # 3) Largest movable tail from src -> dst.
    for src in unfinished:
        src_stack = state.stacks[src]
        if not src_stack:
            continue

        start = movable_suffix_start(src_stack)
        if start < 0:
            continue

        count = len(src_stack) - start
        moving_bottom = src_stack[start]

        for dst in unfinished:
            if dst == src:
                continue

            dst_stack = state.stacks[dst]
            if not dst_stack:
                moves.append(Move(src_index=src, dst_index=dst, count=count))
            else:
                dst_top = dst_stack[-1]
                if is_valid_sequence(dst_top, moving_bottom):
                    moves.append(Move(src_index=src, dst_index=dst, count=count))

    return moves


def apply_move(state: State, move: Move) -> State:
    src, dst, count = move.src_index, move.dst_index, move.count

    if src < 0 or src >= TOTAL_STACKS or dst < 0 or dst >= TOTAL_STACKS:
        raise SolverError(f"Move stack index out of range: {move}")

    if count <= 0:
        raise SolverError(f"Move count must be positive: {move}")

    # deep copy stacks
    new_stacks = [list(stack) for stack in state.stacks]

    if len(new_stacks[src]) < count:
        raise SolverError(f"Cannot move {count} cards from stack {src}; only {len(new_stacks[src])}")

    cards = new_stacks[src][-count:]
    del new_stacks[src][-count:]
    new_stacks[dst].extend(cards)

    new_moves = list(state.moves)
    new_moves.append(move)
    return State(stacks=new_stacks, moves=new_moves)


def state_key(state: State) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    return tuple(
        tuple((card.rank, card.suit) for card in stack)
        for stack in state.stacks
    )


def priority(state: State) -> Tuple[int, int]:
    return count_unfinished(state), len(state.moves)


def find_winning_moves(initial: State, max_expanded: int) -> List[Move]:
    frontier: List[Tuple[int, int, int, State]] = []
    seen = set()
    counter = 0

    prio = priority(initial)
    heapq.heappush(frontier, (prio[0], prio[1], counter, initial))
    seen.add(state_key(initial))

    expanded = 0

    while frontier and expanded < max_expanded:
        _p_unfinished, _p_moves, _p_counter, s = heapq.heappop(frontier)

        for move in generate_moves(s):
            s2 = apply_move(s, move)
            key = state_key(s2)

            if key in seen:
                continue

            unfinished = count_unfinished(s2)
            if unfinished == 1:
                return s2.moves

            seen.add(key)
            counter += 1
            prio2 = priority(s2)
            heapq.heappush(frontier, (prio2[0], prio2[1], counter, s2))

        expanded += 1

    raise SolverError("No winning path found")


def summarize_state(state: State) -> None:
    total_cards = sum(len(state.stacks[i]) for i in range(BOARD_STACK_COUNT)) + len(state.stacks[SPARE_STACK_INDEX])
    spare_text = state.stacks[SPARE_STACK_INDEX][0].label() if state.stacks[SPARE_STACK_INDEX] else "empty"

    print("Board state summary:")
    print(f"  board_stacks={BOARD_STACK_COUNT} total_cards={total_cards} spare={spare_text}")

    parts = []
    for i in range(BOARD_STACK_COUNT):
        labels = ",".join(card.label() for card in state.stacks[i]) if state.stacks[i] else "_"
        done = "done" if is_stack_finished(state.stacks[i]) else "open"
        parts.append(f"c{i + 1}[{len(state.stacks[i])},{done}]={labels}")
    print(f"  {' | '.join(parts)}")


def describe_stack(index: int) -> str:
    if index == SPARE_STACK_INDEX:
        return "spare"
    return f"column {index + 1}"


def format_move_for_terminal(move: Move, idx: int, state_before: State) -> str:
    src_name = describe_stack(move.src_index)
    dst_name = describe_stack(move.dst_index)

    src_stack = state_before.stacks[move.src_index]
    moving_cards = src_stack[-move.count:]
    card_labels = ", ".join(card.label() for card in moving_cards)
    card_word = "card" if move.count == 1 else "cards"

    return f"{idx:02d}. Move {move.count} {card_word} ({card_labels}) from {src_name} to {dst_name}."


def move_reason(move: Move) -> str:
    if move.dst_index == SPARE_STACK_INDEX:
        return "holder_store"
    if move.src_index == SPARE_STACK_INDEX:
        return "holder_release"
    return "largest_suffix"


def state_to_output_dict(state: State) -> Dict[str, Any]:
    stacks_out: List[List[Dict[str, Any]]] = []
    for stack in state.stacks:
        stack_out = []
        for card in stack:
            stack_out.append(
                {
                    "rank": card.rank,
                    "suit": card.suit,
                    "color": COLOR_NAME[card.color],
                    "label": card.label(),
                }
            )
        stacks_out.append(stack_out)

    return {
        "stacks": stacks_out,
        "unfinished_count": sum(1 for i in range(BOARD_STACK_COUNT) if not is_stack_finished(state.stacks[i])),
    }


def simulate_to_state(initial: State, moves: List[Move]) -> State:
    s = State(stacks=[list(stack) for stack in initial.stacks], moves=[])
    for m in moves:
        s = apply_move(s, m)
    return s


def main() -> None:
    args = parse_args()

    try:
        refresh_state_with_shape_comparer(args.shape_comparer, args.state)
        initial = load_initial_state(args.state)
    except (FileNotFoundError, SolverError, json.JSONDecodeError) as exc:
        print(f"Solver input error: {exc}")
        raise SystemExit(2) from exc

    summarize_state(initial)

    try:
        moves = find_winning_moves(initial, max_expanded=max(1, int(args.max_expanded)))
    except SolverError as exc:
        print(f"Solver failed: {exc}")
        raise SystemExit(3) from exc

    final_state = simulate_to_state(initial, moves)

    planned_moves = [
        {
            "source_stack": m.src_index,
            "target_stack": m.dst_index,
            "move_count": m.count,
            "reason": move_reason(m),
        }
        for m in moves
    ]

    output = {
        "search_strategy": "best_first",
        "priority": "(unfinished_stacks, move_count)",
        "initial_unfinished": count_unfinished(initial),
        "final_unfinished": count_unfinished(final_state),
        "planned_moves": planned_moves,
        "move_confidence": 1.0 if count_unfinished(final_state) <= 1 else 0.0,
        "expected_resulting_state": state_to_output_dict(final_state),
    }

    with args.plan_out.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"State file: {args.state}")
    print(f"Plan file:  {args.plan_out}")
    print(f"Initial unfinished stacks: {count_unfinished(initial)}")
    print(f"Final unfinished stacks:   {count_unfinished(final_state)}")

    if moves:
        print("\nPlanned moves:")
        cur = State(stacks=[list(stack) for stack in initial.stacks], moves=[])
        for i, m in enumerate(moves, start=1):
            print(format_move_for_terminal(m, i, cur))
            cur = apply_move(cur, m)
    else:
        print("\nNo moves required.")


if __name__ == "__main__":
    main()
