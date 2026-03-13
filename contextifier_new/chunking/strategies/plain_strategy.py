# contextifier_new/chunking/strategies/plain_strategy.py
"""
PlainChunkingStrategy — Simple recursive text splitting.

Fallback strategy for text without structural markers.
Uses RecursiveCharacterTextSplitter with standard separators.

Priority: 100 (lowest — fallback)
"""

from __future__ import annotations

from typing import Any, List, Union

from contextifier_new.config import ProcessingConfig
from contextifier_new.types import Chunk
from contextifier_new.chunking.strategies.base import BaseChunkingStrategy


class PlainChunkingStrategy(BaseChunkingStrategy):
    """
    Simple text splitting without structural awareness.

    Uses separators in order of preference:
    1. Double newline (paragraph breaks)
    2. Single newline
    3. Space
    4. Empty string (character-level)

    Always applies as the fallback when no other strategy matches.
    """

    def can_handle(
        self,
        text: str,
        config: ProcessingConfig,
        *,
        file_extension: str = "",
        **context: Any,
    ) -> bool:
        """Always returns True — this is the fallback strategy."""
        return True

    def chunk(
        self,
        text: str,
        config: ProcessingConfig,
        *,
        file_extension: str = "",
        include_position_metadata: bool = False,
        **context: Any,
    ) -> Union[List[str], List[Chunk]]:
        """
        Split text using recursive character splitting.

        Implementation will be provided in the concrete implementation phase.
        """
        # TODO: Implement recursive text splitting logic
        # (port from old text_chunker.py chunk_plain_text)
        raise NotImplementedError("Concrete implementation pending")

    @property
    def strategy_name(self) -> str:
        return "plain"

    @property
    def priority(self) -> int:
        return 100


__all__ = ["PlainChunkingStrategy"]
