# libs/core/processor/docx_helper/docx_file_converter.py
"""
DOCXFileConverter - DOCX file format converter

Converts binary DOCX data to python-docx Document object.
"""
from io import BytesIO
from typing import Any, Optional, BinaryIO
import zipfile

from contextifier.core.functions.file_converter import BaseFileConverter


class DOCXFileConverter(BaseFileConverter):
    """
    DOCX file converter using python-docx.
    
    Converts binary DOCX data to Document object.
    """
    
    # ZIP magic number (DOCX is a ZIP file)
    ZIP_MAGIC = b'PK\x03\x04'
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> Any:
        """
        Convert binary DOCX data to Document object.
        
        Args:
            file_data: Raw binary DOCX data
            file_stream: Optional file stream
            **kwargs: Additional options
            
        Returns:
            docx.Document object
            
        Raises:
            Exception: If DOCX cannot be opened
        """
        from docx import Document
        
        stream = file_stream if file_stream is not None else BytesIO(file_data)
        stream.seek(0)
        return Document(stream)
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "DOCX Document"
    
    def validate(self, file_data: bytes) -> bool:
        """
        Validate if data is a valid DOCX (ZIP with specific structure).
        
        Args:
            file_data: Raw binary file data
            
        Returns:
            True if file appears to be a DOCX
        """
        if not file_data or len(file_data) < 4:
            return False
        
        if not file_data[:4] == self.ZIP_MAGIC:
            return False
        
        # Check for DOCX-specific content
        try:
            with zipfile.ZipFile(BytesIO(file_data), 'r') as zf:
                return '[Content_Types].xml' in zf.namelist()
        except zipfile.BadZipFile:
            return False
