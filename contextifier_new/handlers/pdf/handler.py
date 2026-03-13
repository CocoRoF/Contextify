# contextifier_new/handlers/pdf/handler.py
"""
PDFHandler — Unified handler for PDF documents.

Pipeline:
    Convert:  Raw bytes → PyMuPDF Document (fitz)
    Preprocess: Clean/normalize pages, detect scanned pages
    Metadata: Author, title, creation date, page count
    Content:  Text per page, images, tables (via pdfplumber), charts
    Postprocess: Assemble with page tags and metadata block

Old issues resolved:
- Skipping file_converter.convert() — now always goes through converter
- Chart formatting duplicated — now uses ChartService
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


class PDFHandler(BaseHandler):
    """Handler for PDF files (.pdf)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"pdf"})

    @property
    def handler_name(self) -> str:
        return "PDF Handler"

    def create_converter(self) -> BaseConverter:
        # TODO: Implement PDFConverter (bytes → fitz.Document)
        return NullConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        # TODO: Implement PDFPreprocessor (clean pages, detect scanned)
        return NullPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        # TODO: Implement PDFMetadataExtractor
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        # TODO: Implement PDFContentExtractor
        return NullContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["PDFHandler"]
