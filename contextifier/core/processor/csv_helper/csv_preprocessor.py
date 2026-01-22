# contextifier/core/processor/csv_helper/csv_preprocessor.py
"""
CSV Preprocessor - Process CSV content after conversion.

Processing Pipeline Position:
    1. CSVFileConverter.convert() → (content: str, encoding: str)
    2. CSVPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. CSVMetadataExtractor.extract() → DocumentMetadata
    4. Content extraction (rows, columns)

Current Implementation:
    - Pass-through (CSV uses decoded string content directly)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.csv.preprocessor")


class CSVPreprocessor(BasePreprocessor):
    """
    CSV Content Preprocessor.

    Currently a pass-through implementation as CSV processing
    is handled during the content extraction phase.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted CSV content.

        Args:
            converted_data: Tuple of (content: str, encoding: str) from CSVFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the content and encoding
        """
        metadata: Dict[str, Any] = {}

        content = ""
        encoding = "utf-8"

        # Handle tuple return from CSVFileConverter
        if isinstance(converted_data, tuple) and len(converted_data) >= 2:
            content, encoding = converted_data[0], converted_data[1]
            metadata['detected_encoding'] = encoding
            if content:
                lines = content.split('\n')
                metadata['line_count'] = len(lines)
        elif isinstance(converted_data, str):
            content = converted_data
            metadata['line_count'] = len(content.split('\n'))

        logger.debug("CSV preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the processed string content
        return PreprocessedData(
            raw_content=content,
            clean_content=content,  # TRUE SOURCE - string content for CSV
            encoding=encoding,
            extracted_resources={},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "CSV Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is CSV content."""
        if isinstance(data, tuple) and len(data) >= 2:
            return isinstance(data[0], str)
        return isinstance(data, str)


__all__ = ['CSVPreprocessor']
