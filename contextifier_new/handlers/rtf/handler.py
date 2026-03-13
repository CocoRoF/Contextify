# contextifier_new/handlers/rtf/handler.py
"""
RTFHandler — Unified handler for Rich Text Format documents.

Pipeline:
    Convert:  Raw bytes → RTF parsed structure (striprtf / LibreOffice)
    Preprocess: Clean RTF control codes, normalize whitespace
    Metadata: Basic metadata from RTF info group (author, title)
    Content:  Plain text extraction, basic table support
    Postprocess: Assemble with metadata block
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


class RTFHandler(BaseHandler):
    """Handler for RTF files (.rtf)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"rtf"})

    @property
    def handler_name(self) -> str:
        return "RTF Handler"

    def create_converter(self) -> BaseConverter:
        # TODO: Implement RTFConverter (bytes → parsed RTF / plain text)
        return NullConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        # TODO: Implement RTFPreprocessor (clean control codes)
        return NullPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        # TODO: Implement RTFMetadataExtractor
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        # TODO: Implement RTFContentExtractor
        return NullContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["RTFHandler"]
