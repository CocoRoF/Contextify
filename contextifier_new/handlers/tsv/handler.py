# contextifier_new/handlers/tsv/handler.py
"""
TSVHandler — Handler for TSV (Tab-Separated Values) files (.tsv ONLY).

TSV uses tab as the delimiter, which is fundamentally different from CSV
in terms of parsing, quoting rules, and encoding heuristics.

Pipeline:
    Convert:  Raw bytes → decoded text (encoding detection)
    Preprocess: Validate tab delimiter, clean data, handle quoting
    Metadata: Row count, column count, column names, encoding, delimiter
    Content:  Data as HTML table, optional data analysis summary
    Postprocess: Assemble with sheet tag and metadata block

Key differences from CSVHandler:
- Tab delimiter (no ambiguity with commas in data)
- Different quoting conventions
- Often used in bioinformatics/data interchange contexts
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


class TSVHandler(BaseHandler):
    """Handler for TSV files (.tsv only)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"tsv"})

    @property
    def handler_name(self) -> str:
        return "TSV Handler"

    def create_converter(self) -> BaseConverter:
        # TODO: Implement TSVConverter (bytes → decoded string, tab delimiter)
        return NullConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        # TODO: Implement TSVPreprocessor (tab-delimited parsing)
        return NullPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        # TODO: Implement TSVMetadataExtractor
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        # TODO: Implement TSVContentExtractor (data → HTML table)
        return NullContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["TSVHandler"]
