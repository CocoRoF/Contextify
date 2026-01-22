# contextifier/core/processor/hwp_helper/hwp_preprocessor.py
"""
HWP Preprocessor - Process HWP OLE document after conversion.

Processing Pipeline Position:
    1. HWPFileConverter.convert() → olefile.OleFileIO
    2. HWPPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. HWPMetadataExtractor.extract() → DocumentMetadata
    4. Content extraction (body text, tables, images)

Current Implementation:
    - Pass-through (HWP uses olefile object directly)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.hwp.preprocessor")


class HWPPreprocessor(BasePreprocessor):
    """
    HWP OLE Document Preprocessor.

    Currently a pass-through implementation as HWP processing
    is handled during the content extraction phase using olefile.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted HWP OLE document.

        Args:
            converted_data: olefile.OleFileIO object from HWPFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the OLE object and any extracted resources
        """
        metadata: Dict[str, Any] = {}

        if hasattr(converted_data, 'listdir'):
            try:
                streams = converted_data.listdir()
                metadata['stream_count'] = len(streams)
                # Check for common HWP streams
                has_body = any('BodyText' in '/'.join(s) for s in streams)
                has_docinfo = any('DocInfo' in '/'.join(s) for s in streams)
                metadata['has_body_text'] = has_body
                metadata['has_doc_info'] = has_docinfo
            except Exception:
                pass

        logger.debug("HWP preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the OLE object
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE - olefile.OleFileIO
            encoding="utf-8",
            extracted_resources={},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "HWP Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is an OLE file object."""
        return hasattr(data, 'listdir') and hasattr(data, 'openstream')


__all__ = ['HWPPreprocessor']
