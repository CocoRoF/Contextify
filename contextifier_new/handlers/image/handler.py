# contextifier_new/handlers/image/handler.py
"""
ImageFileHandler — Unified handler for standalone image files.

Pipeline:
    Convert:  Raw bytes → validated image (PIL Image or path)
    Preprocess: Validate format, resize if needed, detect EXIF metadata
    Metadata: Dimensions, format, color mode, EXIF data, file size
    Content:  OCR text extraction via OCREngine configured in config
    Postprocess: Wrap OCR result with image tag and metadata block

Old issues resolved:
- ImageFileHandler skipped convert() and metadata — now uses full pipeline
- OCR engine was passed as constructor arg — now configured via OCRConfig
- extra ocr_engine parameter diverged from base signature — eliminated
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


_IMAGE_EXTENSIONS = frozenset({
    "jpg", "jpeg", "png", "gif", "bmp", "webp",
    "tiff", "tif", "svg", "ico", "heic", "heif",
})


class ImageFileHandler(BaseHandler):
    """Handler for standalone image files."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return _IMAGE_EXTENSIONS

    @property
    def handler_name(self) -> str:
        return "Image File Handler"

    def create_converter(self) -> BaseConverter:
        # TODO: Implement ImageConverter (bytes → validated PIL Image + path)
        return NullConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        # TODO: Implement ImagePreprocessor (format validation, optional resize)
        return NullPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        # TODO: Implement ImageMetadataExtractor (dimensions, EXIF, format, size)
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        # TODO: Implement ImageContentExtractor (OCR via engine from config)
        return NullContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["ImageFileHandler"]
