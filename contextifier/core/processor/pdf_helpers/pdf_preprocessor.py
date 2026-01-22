# contextifier/core/processor/pdf_helpers/pdf_preprocessor.py
"""
PDF Preprocessor - Process PDF document after conversion.

This preprocessor handles PDF-specific processing after the document
has been converted from binary to fitz.Document.

Processing Pipeline Position:
    1. PDFFileConverter.convert() → fitz.Document
    2. PDFPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. PDFMetadataExtractor.extract() → DocumentMetadata
    4. Content extraction (text, images, tables)

Current Implementation:
    - Pass-through (no special preprocessing needed for PDF)
    - PDF processing is done during content extraction phase

Future Enhancements:
    - Page rotation normalization
    - Damaged page recovery
    - Font embedding analysis
    - Document structure analysis
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.pdf.preprocessor")


class PDFPreprocessor(BasePreprocessor):
    """
    PDF Document Preprocessor.

    Currently a pass-through implementation as PDF processing
    is handled during the content extraction phase.

    The fitz.Document object from PDFFileConverter already provides
    a clean interface for accessing pages, text, and images.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted PDF document.

        Args:
            converted_data: fitz.Document object from PDFFileConverter
            **kwargs: Additional options
                - analyze_structure: Whether to analyze document structure
                - normalize_rotation: Whether to normalize page rotation

        Returns:
            PreprocessedData with the document and any extracted resources
        """
        # For now, PDF preprocessing is a pass-through
        # The fitz.Document is already in a workable state

        # Store the document reference for downstream processing
        metadata: Dict[str, Any] = {}

        # If it's a fitz.Document, extract some basic info
        if hasattr(converted_data, 'page_count'):
            metadata['page_count'] = converted_data.page_count
            metadata['is_encrypted'] = getattr(converted_data, 'is_encrypted', False)
            metadata['is_pdf'] = getattr(converted_data, 'is_pdf', True)

        logger.debug("PDF preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the fitz.Document
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE - fitz.Document
            encoding="binary",
            extracted_resources={"document": converted_data},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "PDF Preprocessor"

    def validate(self, data: Any) -> bool:
        """
        Validate if the data can be preprocessed.

        Args:
            data: fitz.Document object or bytes

        Returns:
            True if valid PDF document
        """
        # Check if it's a fitz.Document
        if hasattr(data, 'page_count') and hasattr(data, 'load_page'):
            return True
        return False


__all__ = ['PDFPreprocessor']
