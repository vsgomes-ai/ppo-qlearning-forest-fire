"""Backward-compatible CA import path."""

from .ca import *  # noqa: F401,F403
from .ca.states import BURNED, BURNING, EMPTY, STATE_NAMES, TREE  # noqa: F401
