# contextifier_new/handlers/csv/handler.py
"""
CSVHandler — Handler for CSV (Comma-Separated Values) files (.csv ONLY).

CSV uses comma as the default delimiter. TSV (tab-separated) is handled
by a separate TSVHandler because delimiter differences affect parsing,
encoding detection heuristics, and downstream table formatting.

Pipeline:
    Convert:  Raw bytes → decoded text (encoding detection)
    Preprocess: Validate comma delimiter, clean data, handle quoting
    Metadata: Row count, column count, column names, encoding, delimiter
    Content:  Data as HTML table, optional data analysis summary
    Postprocess: Assemble with sheet tag and metadata block

Old issues resolved:
- Extra parameters on extract_text() — now uses config.format_options
- CSV and TSV no longer share a handler (different delimiter logic)
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier_new.handlers.base import BaseHandler
from contextifier_new.pipeline.converter import BaseConverter, NullConverter
from contextifier_new.pipeline.preprocessor import BasePreprocessor, NullPreprocessor
from contextifier_new.pipeline.metadata_extractor import (
    BaseMetadataExtractor,
    NullMetadataExtractor,
)
from contextifier_new.pipeline.content_extractor import (
    BaseContentExtractor,
    NullContentExtractor,
)
from contextifier_new.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor


class CSVHandler(BaseHandler):
    """Handler for CSV files (.csv only)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"csv"})

    @property
    def handler_name(self) -> str:
        return "CSV Handler"

    def create_converter(self) -> BaseConverter:
        # TODO: Implement CSVConverter (bytes → decoded string, comma delimiter)
        return NullConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        # TODO: Implement CSVPreprocessor (comma-delimited parsing, quoting)
        return NullPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        # TODO: Implement CSVMetadataExtractor (row/col count, columns, encoding)
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        # TODO: Implement CSVContentExtractor (data → HTML table)
        return NullContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["CSVHandler"]
