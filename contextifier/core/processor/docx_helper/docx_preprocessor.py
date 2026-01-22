# contextifier/core/processor/docx_helper/docx_preprocessor.py
"""
DOCX Preprocessor - Process DOCX document after conversion.

Processing Pipeline Position:
    1. DOCXFileConverter.convert() → docx.Document
    2. DOCXPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. DOCXMetadataExtractor.extract() → DocumentMetadata
    4. Content extraction (paragraphs, tables, images)

Current Implementation:
    - Pass-through (DOCX uses python-docx Document object directly)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.docx.preprocessor")


class DOCXPreprocessor(BasePreprocessor):
    """
    DOCX Document Preprocessor.

    Currently a pass-through implementation as DOCX processing
    is handled during the content extraction phase using python-docx.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted DOCX document.

        Args:
            converted_data: docx.Document object from DOCXFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the document and any extracted resources
        """
        metadata: Dict[str, Any] = {}

        # Extract basic document info if available
        if hasattr(converted_data, 'core_properties'):
            props = converted_data.core_properties
            if hasattr(props, 'title') and props.title:
                metadata['title'] = props.title

        if hasattr(converted_data, 'paragraphs'):
            metadata['paragraph_count'] = len(converted_data.paragraphs)

        if hasattr(converted_data, 'tables'):
            metadata['table_count'] = len(converted_data.tables)

        logger.debug("DOCX preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the docx.Document
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE - docx.Document
            encoding="utf-8",
            extracted_resources={},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "DOCX Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is a DOCX Document object."""
        return hasattr(data, 'paragraphs') and hasattr(data, 'tables')


__all__ = ['DOCXPreprocessor']
