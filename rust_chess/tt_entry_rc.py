"""Class that represents a transposition table entry."""

from dataclasses import dataclass
from typing import TypeAlias  # For flags

import numpy as np

import rust_chess as rc

# Transposition table entry flags
Flag: TypeAlias = np.int8
EXACT: Flag = np.int8(1)
LOWERBOUND: Flag = np.int8(2)  # Beta (fail-high)
UPPERBOUND: Flag = np.int8(3)  # Alpha (fail-low)


@dataclass(frozen=True)  # (Immutable)
class TTEntry:  # TODO: Pack into an integer to save space
    """Class to represent a transposition table entry.

    Stores the depth, value, flag, and best move for a position.
    """

    __slots__ = ["best_move", "depth", "flag", "value"]  # Optimization for faster lookups

    depth: np.int8
    value: np.int16
    flag: Flag
    best_move: rc.Move | None


if __name__ == "__main__":
    # Print the size of the TTEntry class
    entry = TTEntry(depth=np.int8(0), value=np.int16(0), flag=EXACT, best_move=None)
    print(f"Size of TTEntry: {entry.__sizeof__()} bytes")
