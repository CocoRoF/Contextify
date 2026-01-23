# contextifier/core/processor/ole_helper/ole_preprocessor.py
"""
OLE Preprocessor - Process OLE content after conversion.

Processing Pipeline Position:
    1. OLEFileConverter.convert() → olefile.OleFileIO
    2. OLEPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. Content extraction (OLE stream extraction, table parsing)

Current Implementation:
    - Pass-through (OLE file object is used directly by handler)
    - Metadata extraction (stream list, file info)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.ole.preprocessor")


class OLEPreprocessor(BasePreprocessor):
    """
    OLE Document Preprocessor.

    Extracts basic metadata from OLE file and passes through
    the OleFileIO object for content extraction.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted OLE content.

        Args:
            converted_data: olefile.OleFileIO object from OLEFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the OLE file object
        """
        import olefile
        
        metadata: Dict[str, Any] = {}
        
        if isinstance(converted_data, olefile.OleFileIO):
            # Extract basic OLE metadata
            try:
                # List available streams
                streams = converted_data.listdir()
                metadata['stream_count'] = len(streams)
                metadata['has_worddocument'] = converted_data.exists('WordDocument')
                metadata['has_1table'] = converted_data.exists('1Table')
                metadata['has_0table'] = converted_data.exists('0Table')
                metadata['has_data'] = converted_data.exists('Data')
                
                logger.debug(f"OLE file streams: {len(streams)} total")
            except Exception as e:
                logger.warning(f"Failed to extract OLE metadata: {e}")

        logger.debug("OLE preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the OLE file object
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE - the OLE file object
            encoding="utf-8",
            extracted_resources={},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "OLE Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is OLE file object."""
        try:
            import olefile
            return isinstance(data, olefile.OleFileIO)
        except ImportError:
            return False


__all__ = ['OLEPreprocessor']
