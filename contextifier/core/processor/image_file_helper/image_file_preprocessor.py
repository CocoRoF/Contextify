# contextifier/core/processor/image_file_helper/image_file_preprocessor.py
"""
Image File Preprocessor - Process image file after conversion.

Processing Pipeline Position:
    1. ImageFileConverter.convert() → bytes (raw image data)
    2. ImageFilePreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. ImageFileMetadataExtractor.extract() → DocumentMetadata
    4. OCR processing (if OCR engine available)

Current Implementation:
    - Pass-through (Image uses raw bytes directly for OCR)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.image_file.preprocessor")


class ImageFilePreprocessor(BasePreprocessor):
    """
    Image File Preprocessor.

    Currently a pass-through implementation as image processing
    is handled by the OCR engine.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted image data.

        Args:
            converted_data: Image bytes from ImageFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the image data
        """
        metadata: Dict[str, Any] = {}

        if isinstance(converted_data, bytes):
            metadata['size_bytes'] = len(converted_data)
            # Try to detect image format from magic bytes
            if converted_data.startswith(b'\xff\xd8\xff'):
                metadata['format'] = 'jpeg'
            elif converted_data.startswith(b'\x89PNG'):
                metadata['format'] = 'png'
            elif converted_data.startswith(b'GIF'):
                metadata['format'] = 'gif'
            elif converted_data.startswith(b'BM'):
                metadata['format'] = 'bmp'
            elif converted_data.startswith(b'RIFF') and b'WEBP' in converted_data[:12]:
                metadata['format'] = 'webp'

        logger.debug("Image file preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the image bytes
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE - image bytes
            encoding="binary",
            extracted_resources={},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "Image File Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is image bytes."""
        return isinstance(data, bytes) and len(data) > 0


__all__ = ['ImageFilePreprocessor']
