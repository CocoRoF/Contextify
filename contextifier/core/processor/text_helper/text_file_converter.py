# libs/core/processor/text_helper/text_file_converter.py
"""
TextFileConverter - Text file format converter

Converts binary text data to string with encoding detection.
"""
from typing import Optional, BinaryIO

from contextifier.core.functions.file_converter import TextFileConverter as BaseTextFileConverter


class TextFileConverter(BaseTextFileConverter):
    """
    Text file converter.
    
    Converts binary text data to decoded string.
    Inherits from base TextFileConverter.
    """
    
    def __init__(self):
        """Initialize with common text encodings."""
        super().__init__(encodings=['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1', 'ascii'])
    
    def get_format_name(self) -> str:
        """Return format name."""
        enc = self._detected_encoding or 'unknown'
        return f"Text File ({enc})"
