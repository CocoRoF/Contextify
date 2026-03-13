# contextifier_new/handlers/ppt/handler.py
"""
PPTHandler — Handler for legacy PowerPoint PPT documents (.ppt ONLY).

PPT is an OLE2/CFBF binary format, fundamentally different from PPTX (OOXML).
Each format gets its own handler because they require completely different
parsing libraries and conversion logic.

Pipeline:
    Convert:  Raw bytes → LibreOffice conversion to PPTX, then python-pptx
    Preprocess: Normalize slides, detect slide structure
    Metadata: Author, title, creation date, slide count
    Content:  Slide text, tables, images, shapes
    Postprocess: Assemble with slide tags and metadata block

Note: .ppt requires LibreOffice for conversion to a parseable format.
This is fundamentally different from .pptx which can be parsed directly.
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


class PPTHandler(BaseHandler):
    """Handler for legacy PowerPoint files (.ppt only)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"ppt"})

    @property
    def handler_name(self) -> str:
        return "PPT Handler"

    def create_converter(self) -> BaseConverter:
        # TODO: Implement PPTConverter (bytes → LibreOffice → PPTX → python-pptx)
        return NullConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        # TODO: Implement PPTPreprocessor
        return NullPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        # TODO: Implement PPTMetadataExtractor
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        # TODO: Implement PPTContentExtractor
        return NullContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["PPTHandler"]
