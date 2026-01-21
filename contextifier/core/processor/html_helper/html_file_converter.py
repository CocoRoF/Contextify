# libs/core/processor/html_helper/html_file_converter.py
"""
HTMLFileConverter - HTML file format converter

Converts binary HTML data to BeautifulSoup object.
"""
from typing import Any, Optional, BinaryIO

from contextifier.core.functions.file_converter import BaseFileConverter


class HTMLFileConverter(BaseFileConverter):
    """
    HTML file converter using BeautifulSoup.
    
    Converts binary HTML data to BeautifulSoup object.
    """
    
    DEFAULT_ENCODINGS = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']
    
    def __init__(self, parser: str = 'html.parser'):
        """
        Initialize HTMLFileConverter.
        
        Args:
            parser: BeautifulSoup parser to use
        """
        self._parser = parser
        self._detected_encoding: Optional[str] = None
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        encoding: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Convert binary HTML data to BeautifulSoup object.
        
        Args:
            file_data: Raw binary HTML data
            file_stream: Ignored
            encoding: Specific encoding to use
            **kwargs: Additional options
            
        Returns:
            BeautifulSoup object
        """
        from bs4 import BeautifulSoup
        
        # Decode to text first
        text = self._decode(file_data, encoding)
        return BeautifulSoup(text, self._parser)
    
    def _decode(self, file_data: bytes, encoding: Optional[str] = None) -> str:
        """Decode bytes to string."""
        if encoding:
            try:
                self._detected_encoding = encoding
                return file_data.decode(encoding)
            except UnicodeDecodeError:
                pass
        
        for enc in self.DEFAULT_ENCODINGS:
            try:
                self._detected_encoding = enc
                return file_data.decode(enc)
            except UnicodeDecodeError:
                continue
        
        # Fallback
        self._detected_encoding = 'utf-8'
        return file_data.decode('utf-8', errors='replace')
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "HTML Document"
    
    def validate(self, file_data: bytes) -> bool:
        """Validate if data appears to be HTML."""
        if not file_data:
            return False
        
        header = file_data[:100].lower()
        return (
            b'<!doctype' in header or
            b'<html' in header or
            b'<head' in header or
            b'<body' in header
        )
