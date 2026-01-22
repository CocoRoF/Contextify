# contextifier/core/processor/ppt_helper/ppt_preprocessor.py
"""
PPT Preprocessor - Process PPT/PPTX presentation after conversion.

Processing Pipeline Position:
    1. PPTFileConverter.convert() → pptx.Presentation
    2. PPTPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. PPTMetadataExtractor.extract() → DocumentMetadata
    4. Content extraction (slides, shapes, images, charts)

Current Implementation:
    - Pass-through (PPT uses python-pptx Presentation object directly)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.ppt.preprocessor")


class PPTPreprocessor(BasePreprocessor):
    """
    PPT/PPTX Presentation Preprocessor.

    Currently a pass-through implementation as PPT processing
    is handled during the content extraction phase using python-pptx.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted PPT presentation.

        Args:
            converted_data: pptx.Presentation object from PPTFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the presentation and any extracted resources
        """
        metadata: Dict[str, Any] = {}

        if hasattr(converted_data, 'slides'):
            metadata['slide_count'] = len(converted_data.slides)

        if hasattr(converted_data, 'slide_width'):
            metadata['slide_width'] = converted_data.slide_width
            metadata['slide_height'] = converted_data.slide_height

        logger.debug("PPT preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the Presentation
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE - pptx.Presentation
            encoding="utf-8",
            extracted_resources={},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "PPT Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is a PPT Presentation object."""
        return hasattr(data, 'slides') and hasattr(data, 'slide_layouts')


__all__ = ['PPTPreprocessor']
