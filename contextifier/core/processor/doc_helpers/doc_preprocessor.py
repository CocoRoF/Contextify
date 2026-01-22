# contextifier/core/processor/doc_helpers/doc_preprocessor.py
"""
DOC Preprocessor - Process DOC content after conversion.

Processing Pipeline Position:
    1. DOCFileConverter.convert() → (converted_obj, DocFormat)
    2. DOCPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. Content extraction (depends on format: RTF, OLE, HTML, DOCX)

Current Implementation:
    - Pass-through (DOC delegates to format-specific handlers)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.doc.preprocessor")


class DOCPreprocessor(BasePreprocessor):
    """
    DOC Document Preprocessor.

    Currently a pass-through implementation as DOC processing
    delegates to format-specific handlers (RTF, OLE, HTML, DOCX).
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted DOC content.

        Args:
            converted_data: Tuple of (converted_obj, DocFormat) from DOCFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the converted object
        """
        metadata: Dict[str, Any] = {}

        converted_obj = converted_data
        doc_format = None

        # Handle tuple return from DOCFileConverter
        if isinstance(converted_data, tuple) and len(converted_data) >= 2:
            converted_obj, doc_format = converted_data[0], converted_data[1]
            if hasattr(doc_format, 'value'):
                metadata['detected_format'] = doc_format.value
            else:
                metadata['detected_format'] = str(doc_format)

        logger.debug("DOC preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the converted object
        # For DOC, this is the format-specific object (OLE, BeautifulSoup, etc.)
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_obj,  # TRUE SOURCE - the converted object
            encoding="utf-8",
            extracted_resources={"doc_format": doc_format},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "DOC Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is DOC conversion result."""
        if isinstance(data, tuple) and len(data) >= 2:
            return True
        return data is not None


__all__ = ['DOCPreprocessor']
