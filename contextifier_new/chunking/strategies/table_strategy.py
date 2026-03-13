# contextifier_new/chunking/strategies/table_strategy.py
"""
TableChunkingStrategy — Table-aware chunking for spreadsheet-type content.

Handles CSV, XLSX, XLS files where content is primarily tabular.
Each table is chunked independently with header restoration.

Priority: 5 (highest — table files always use this strategy)
"""

from __future__ import annotations

from typing import Any, FrozenSet, List, Union

from contextifier_new.config import ProcessingConfig
from contextifier_new.types import Chunk, FileCategory, get_category
from contextifier_new.chunking.strategies.base import BaseChunkingStrategy

_TABLE_EXTENSIONS: FrozenSet[str] = frozenset({"csv", "tsv", "xlsx", "xls"})


class TableChunkingStrategy(BaseChunkingStrategy):
    """
    Split table-based content preserving table structure.

    Key invariant: Tables NEVER have overlap between chunks.
    This prevents data duplication in search/retrieval systems.

    Algorithm for each table:
    1. Parse into header rows + data rows
    2. Calculate available space per chunk (chunk_size - header_size)
    3. Accumulate data rows until space exceeded
    4. Each chunk gets header rows prepended
    5. [Table Chunk N/M] annotation added to each chunk
    """

    def can_handle(
        self,
        text: str,
        config: ProcessingConfig,
        *,
        file_extension: str = "",
        **context: Any,
    ) -> bool:
        """Handle table-based file types (CSV, TSV, XLSX, XLS)."""
        return file_extension.lower().lstrip(".") in _TABLE_EXTENSIONS

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
        Chunk table-based content.

        Implementation will be provided in the concrete implementation phase.
        """
        # TODO: Implement table-based chunking logic
        # (port from old sheet_processor.py + table_chunker.py)
        raise NotImplementedError("Concrete implementation pending")

    @property
    def strategy_name(self) -> str:
        return "table"

    @property
    def priority(self) -> int:
        return 5


__all__ = ["TableChunkingStrategy"]
