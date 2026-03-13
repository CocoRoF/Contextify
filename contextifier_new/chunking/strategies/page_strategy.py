# contextifier_new/chunking/strategies/page_strategy.py
"""
PageChunkingStrategy — Split by page/slide markers.

Handles documents with [Page Number: N] or [Slide Number: N] tags.
Merges pages greedily until chunk_size is reached, then starts a new chunk.

Priority: 10 (high — page boundaries are the best split points for documents)
"""

from __future__ import annotations

from typing import Any, List, Union

from contextifier_new.config import ProcessingConfig
from contextifier_new.types import Chunk
from contextifier_new.chunking.strategies.base import BaseChunkingStrategy


class PageChunkingStrategy(BaseChunkingStrategy):
    """
    Split text at page/slide boundaries.

    Algorithm:
    1. Split text into page segments at page markers
    2. Merge pages greedily until chunk_size exceeded
    3. Allow up to 1.5x chunk_size for page boundary alignment
    4. Very large pages are recursively sub-split
    5. Overlap content is carried from the end of previous chunk
    """

    def can_handle(
        self,
        text: str,
        config: ProcessingConfig,
        *,
        file_extension: str = "",
        **context: Any,
    ) -> bool:
        """Handle if text contains page or slide markers."""
        # This will be connected to TagService patterns in implementation
        page_prefix = config.tags.page_prefix
        slide_prefix = config.tags.slide_prefix
        return page_prefix in text or slide_prefix in text

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
        Chunk text by page boundaries.

        Implementation will be provided in the concrete implementation phase.
        """
        # TODO: Implement page-based chunking logic
        # (port from old page_chunker.py with improved interface)
        raise NotImplementedError("Concrete implementation pending")

    @property
    def strategy_name(self) -> str:
        return "page"

    @property
    def priority(self) -> int:
        return 10


__all__ = ["PageChunkingStrategy"]
