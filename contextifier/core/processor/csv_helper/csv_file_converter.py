# libs/core/processor/csv_helper/csv_file_converter.py
"""
CSVFileConverter - CSV file format converter

Converts binary CSV data to text string with encoding detection.
"""
from typing import Any, Optional, BinaryIO, Tuple

from contextifier.core.functions.file_converter import TextFileConverter


class CSVFileConverter(TextFileConverter):
    """
    CSV file converter.
    
    Converts binary CSV data to decoded text string.
    Extends TextFileConverter with BOM detection.
    """
    
    # BOM markers
    BOM_UTF8 = b'\xef\xbb\xbf'
    BOM_UTF16_LE = b'\xff\xfe'
    BOM_UTF16_BE = b'\xfe\xff'
    
    def __init__(self):
        """Initialize CSVFileConverter."""
        super().__init__(encodings=['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'iso-8859-1', 'latin-1'])
        self._delimiter: Optional[str] = None
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Convert binary CSV data to text string.
        
        Args:
            file_data: Raw binary CSV data
            file_stream: Ignored
            encoding: Specific encoding to use
            delimiter: CSV delimiter (for reference)
            **kwargs: Additional options
            
        Returns:
            Tuple of (decoded text, detected encoding)
        """
        self._delimiter = delimiter
        
        # Check for BOM
        bom_encoding = self._detect_bom(file_data)
        if bom_encoding:
            text = file_data.decode(bom_encoding)
            self._detected_encoding = bom_encoding
            return text, bom_encoding
        
        # Use parent's convert logic
        text = super().convert(file_data, file_stream, encoding, **kwargs)
        return text, self._detected_encoding or 'utf-8'
    
    def _detect_bom(self, file_data: bytes) -> Optional[str]:
        """Detect encoding from BOM."""
        if file_data.startswith(self.BOM_UTF8):
            return 'utf-8-sig'
        elif file_data.startswith(self.BOM_UTF16_LE):
            return 'utf-16-le'
        elif file_data.startswith(self.BOM_UTF16_BE):
            return 'utf-16-be'
        return None
    
    def get_format_name(self) -> str:
        """Return format name."""
        enc = self._detected_encoding or 'unknown'
        return f"CSV ({enc})"
