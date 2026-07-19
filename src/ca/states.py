"""Cell states for the forest-fire cellular automaton."""

from __future__ import annotations

EMPTY = 0
TREE = 1
BURNING = 2
BURNED = 3

STATE_NAMES = {
    EMPTY: "empty",
    TREE: "tree",
    BURNING: "burning",
    BURNED: "burned",
}
