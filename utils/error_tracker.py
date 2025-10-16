# utils/error_tracker.py
"""Centralised error tracking utility for pipelines and services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from utils.logger import get_logger


@dataclass(slots=True)
class ErrorTracker:
    """Collect exceptions and contextual information during execution."""

    context: str
    errors: Dict[str, List[str]] = field(default_factory=dict)

    def record(self, key: str, message: str) -> None:
        logger = get_logger(self.context)
        logger.error("{}: {}", key, message)
        self.errors.setdefault(key, []).append(message)

    def summary(self) -> Dict[str, List[str]]:
        if not self.errors:
            return {}
        logger = get_logger(self.context)
        for key, messages in self.errors.items():
            logger.warning("Encountered {} issues for {}", len(messages), key)
        return dict(self.errors)


__all__ = ["ErrorTracker"]
