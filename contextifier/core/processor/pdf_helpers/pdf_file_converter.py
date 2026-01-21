# libs/core/processor/pdf_helpers/pdf_file_converter.py
"""
PDFFileConverter - PDF file format converter

Converts binary PDF data to fitz.Document object using PyMuPDF.
"""
from typing import Any, Optional, BinaryIO

from contextifier.core.functions.file_converter import BaseFileConverter


class PDFFileConverter(BaseFileConverter):
    """
    PDF file converter using PyMuPDF (fitz).
    
    Converts binary PDF data to fitz.Document object.
    """
    
    # PDF magic number
    PDF_MAGIC = b'%PDF'
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> Any:
        """
        Convert binary PDF data to fitz.Document.
        
        Args:
            file_data: Raw binary PDF data
            file_stream: Optional file stream (not used, fitz prefers bytes)
            **kwargs: Additional options
            
        Returns:
            fitz.Document object
            
        Raises:
            RuntimeError: If PDF cannot be opened
        """
        import fitz
        return fitz.open(stream=file_data, filetype="pdf")
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "PDF Document"
    
    def validate(self, file_data: bytes) -> bool:
        """
        Validate if data is a valid PDF.
        
        Args:
            file_data: Raw binary file data
            
        Returns:
            True if file appears to be a PDF
        """
        if not file_data or len(file_data) < 4:
            return False
        return file_data[:4] == self.PDF_MAGIC
    
    def close(self, converted_object: Any) -> None:
        """
        Close the fitz.Document.
        
        Args:
            converted_object: fitz.Document to close
        """
        if converted_object is not None and hasattr(converted_object, 'close'):
            converted_object.close()
