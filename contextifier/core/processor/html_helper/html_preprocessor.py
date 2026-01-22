# contextifier/core/processor/html_helper/html_preprocessor.py
"""
HTML Preprocessor - Process HTML content after conversion.

Processing Pipeline Position:
    1. HTMLFileConverter.convert() → BeautifulSoup
    2. HTMLPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. Content extraction

Current Implementation:
    - Pass-through (HTML uses BeautifulSoup object directly)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.html.preprocessor")


class HTMLPreprocessor(BasePreprocessor):
    """
    HTML Content Preprocessor.

    Currently a pass-through implementation as HTML processing
    is handled using BeautifulSoup.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted HTML content.

        Args:
            converted_data: BeautifulSoup object from HTMLFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the BeautifulSoup object
        """
        metadata: Dict[str, Any] = {}

        if hasattr(converted_data, 'find_all'):
            # Count some basic elements
            metadata['table_count'] = len(converted_data.find_all('table'))
            metadata['image_count'] = len(converted_data.find_all('img'))
            metadata['link_count'] = len(converted_data.find_all('a'))

        logger.debug("HTML preprocessor: pass-through, metadata=%s", metadata)

        return PreprocessedData(
            raw_content=b"",
            clean_content=b"",
            encoding="utf-8",
            extracted_resources={"soup": converted_data},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "HTML Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is a BeautifulSoup object."""
        return hasattr(data, 'find_all') and hasattr(data, 'get_text')


__all__ = ['HTMLPreprocessor']
