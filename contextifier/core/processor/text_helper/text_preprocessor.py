# contextifier/core/processor/text_helper/text_preprocessor.py
"""
Text Preprocessor - Process text content after conversion.

Processing Pipeline Position:
    1. TextFileConverter.convert() → str
    2. TextPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. TextMetadataExtractor.extract() → DocumentMetadata (if any)
    4. Content extraction

Current Implementation:
    - Pass-through (Text uses decoded string content directly)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.text.preprocessor")


class TextPreprocessor(BasePreprocessor):
    """
    Text Content Preprocessor.

    Currently a pass-through implementation as text processing
    is straightforward.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted text content.

        Args:
            converted_data: Text string from TextFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the content
        """
        metadata: Dict[str, Any] = {}

        content = ""
        encoding = kwargs.get("encoding", "utf-8")

        if isinstance(converted_data, str):
            content = converted_data
            metadata['char_count'] = len(content)
            metadata['line_count'] = len(content.split('\n'))
        elif isinstance(converted_data, bytes):
            content = converted_data.decode(encoding, errors='replace')
            metadata['char_count'] = len(content)
            metadata['line_count'] = len(content.split('\n'))

        logger.debug("Text preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the processed text/bytes
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE - bytes or str
            encoding=encoding,
            extracted_resources={},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "Text Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is text content."""
        return isinstance(data, (str, bytes))


__all__ = ['TextPreprocessor']
