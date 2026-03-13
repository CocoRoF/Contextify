# contextifier_new/handlers/hwpx/handler.py
"""
HWPXHandler — Unified handler for Hangul HWPX (ZIP-based XML) documents.

Pipeline:
    Convert:  Raw bytes → HWPX XML document (ZIP extraction)
    Preprocess: Parse XML sections, normalize
    Metadata: Author, title, creation date from HWPX properties
    Content:  Text from XML body, tables, embedded images
    Postprocess: Assemble with page tags and metadata block
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


class HWPXHandler(BaseHandler):
    """Handler for HWPX files (.hwpx)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"hwpx"})

    @property
    def handler_name(self) -> str:
        return "HWPX Handler"

    def create_converter(self) -> BaseConverter:
        # TODO: Implement HWPXConverter (bytes → ZIP → XML DOM)
        return NullConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        # TODO: Implement HWPXPreprocessor
        return NullPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        # TODO: Implement HWPXMetadataExtractor
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        # TODO: Implement HWPXContentExtractor
        return NullContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["HWPXHandler"]
