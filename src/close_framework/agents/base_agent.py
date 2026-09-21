"""Base classes shared by CLOSE agents."""

from __future__ import annotations

from abc import ABC
import logging


class BaseAgent(ABC):
    """Common agent base with consistent naming and logging."""

    agent_name = "BaseAgent"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.agent_name}")
