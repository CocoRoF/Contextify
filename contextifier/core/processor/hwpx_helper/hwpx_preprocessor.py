# contextifier/core/processor/hwpx_helper/hwpx_preprocessor.py
"""
HWPX Preprocessor - Process HWPX ZIP document after conversion.

Processing Pipeline Position:
    1. HWPXFileConverter.convert() → zipfile.ZipFile
    2. HWPXPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. HWPXMetadataExtractor.extract() → DocumentMetadata
    4. Content extraction (sections, tables, images)

Current Implementation:
    - Pass-through (HWPX uses zipfile object directly)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.hwpx.preprocessor")


class HWPXPreprocessor(BasePreprocessor):
    """
    HWPX ZIP Document Preprocessor.

    Currently a pass-through implementation as HWPX processing
    is handled during the content extraction phase.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted HWPX ZIP document.

        Args:
            converted_data: zipfile.ZipFile object from HWPXFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the ZIP object and any extracted resources
        """
        metadata: Dict[str, Any] = {}

        if hasattr(converted_data, 'namelist'):
            try:
                files = converted_data.namelist()
                metadata['file_count'] = len(files)
                # Check for section files
                sections = [f for f in files if 'section' in f.lower() and f.endswith('.xml')]
                metadata['section_count'] = len(sections)
            except Exception:  # noqa: BLE001
                pass

        logger.debug("HWPX preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the ZipFile
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE - zipfile.ZipFile
            encoding="utf-8",
            extracted_resources={},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "HWPX Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is a ZipFile object."""
        return hasattr(data, 'namelist') and hasattr(data, 'open')


__all__ = ['HWPXPreprocessor']
