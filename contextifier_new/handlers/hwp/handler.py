# contextifier_new/handlers/hwp/handler.py
"""
HWPHandler — Unified handler for Hangul Word Processor (HWP) documents.

Pipeline:
    Convert:  Raw bytes → converted intermediate (HWP5 binary → pyhwp/hwp5)
    Preprocess: Normalize Korean text, clean control characters
    Metadata: Author, title, creation date, page count
    Content:  Text, tables, images
    Postprocess: Assemble with page tags and metadata block

Old issues resolved:
- HWP handler delegated to HWPX handler — now self-contained
- No cross-handler dependency
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


class HWPHandler(BaseHandler):
    """Handler for HWP files (.hwp)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"hwp"})

    @property
    def handler_name(self) -> str:
        return "HWP Handler"

    def create_converter(self) -> BaseConverter:
        # TODO: Implement HWPConverter (bytes → intermediate via pyhwp or hwp5)
        return NullConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        # TODO: Implement HWPPreprocessor (Korean text normalization)
        return NullPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        # TODO: Implement HWPMetadataExtractor
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        # TODO: Implement HWPContentExtractor
        return NullContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["HWPHandler"]
