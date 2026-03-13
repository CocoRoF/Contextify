# contextifier_new/handlers/xls/handler.py
"""
XLSHandler — Handler for legacy Excel XLS spreadsheets (.xls ONLY).

XLS is a BIFF (Binary Interchange File Format) binary format, fundamentally
different from XLSX (OOXML). Requires xlrd for reading or LibreOffice for
conversion to XLSX.

Pipeline:
    Convert:  Raw bytes → xlrd Workbook (or LibreOffice → XLSX → openpyxl)
    Preprocess: Detect sheets, data regions, merged cells
    Metadata: Author, creation date from BIFF compound document properties
    Content:  Sheet data as HTML tables (limited image/chart support)
    Postprocess: Assemble with sheet tags and metadata block

Key differences from XLSXHandler:
- BIFF binary format requires xlrd (limited to .xls)
- 65536 row / 256 column limit
- Limited or no embedded chart extraction (BIFF charts ≠ OOXML charts)
- Image extraction more limited than OOXML
- Metadata from OLE compound document properties (not OOXML core props)

Note: For chart extraction, XLS may internally convert to XLSX via
LibreOffice, but this is handled within the converter stage — the
handler interface remains identical to all other handlers.
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


class XLSHandler(BaseHandler):
    """Handler for legacy Excel files (.xls only)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"xls"})

    @property
    def handler_name(self) -> str:
        return "XLS Handler"

    def create_converter(self) -> BaseConverter:
        # TODO: Implement XLSConverter (bytes → xlrd Workbook, or → XLSX via LibreOffice)
        return NullConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        # TODO: Implement XLSPreprocessor
        return NullPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        # TODO: Implement XLSMetadataExtractor (OLE compound document properties)
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        # TODO: Implement XLSContentExtractor (sheets → HTML tables, limited charts/images)
        return NullContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["XLSHandler"]
