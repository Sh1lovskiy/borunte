from __future__ import annotations

from typing import List, Tuple, Optional
from utils.logger import Logger

LOG = Logger.get_logger("tk.graph_traverse")


class EdgeNavigator:
    """Cyclic iterator over edges with index state."""

    def __init__(self, edges: List[Tuple[int, int]]) -> None:
        self.edges = edges
        self.idx = -1

    def _step(self, d: int) -> Optional[Tuple[int, int]]:
        if not self.edges:
            return None
        self.idx = (self.idx + d) % len(self.edges)
        return self.edges[self.idx]

    def next(self) -> Optional[Tuple[int, int]]:
        return self._step(+1)

    def prev(self) -> Optional[Tuple[int, int]]:
        return self._step(-1)
