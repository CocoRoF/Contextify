# libs/core/functions/file_converter.py
"""
BaseFileConverter - Abstract base class for file format conversion

Defines the interface for converting binary file data to a workable format.
Each handler can optionally implement a format-specific converter.

The converter's job is to transform raw binary data into a format-specific
object that the handler can work with (e.g., Document, Workbook, OLE file).

This is the FIRST step in the processing pipeline:
    Binary Data → FileConverter → Workable Object → Handler Processing

Usage:
    class PDFFileConverter(BaseFileConverter):
        def convert(self, file_data: bytes, file_stream: BinaryIO) -> Any:
            import fitz
            return fitz.open(stream=file_data, filetype="pdf")
        
        def get_format_name(self) -> str:
            return "PDF Document"
"""
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Optional, Union, BinaryIO


class BaseFileConverter(ABC):
    """
    Abstract base class for file format converters.
    
    Converts raw binary file data into a format-specific workable object.
    This is the first processing step before text extraction.
    
    Subclasses must implement:
    - convert(): Convert binary data to workable format
    - get_format_name(): Return human-readable format name
    """
    
    @abstractmethod
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> Any:
        """
        Convert binary file data to a workable format.
        
        Args:
            file_data: Raw binary file data
            file_stream: Optional file stream (BytesIO) for libraries that prefer streams
            **kwargs: Additional format-specific options
            
        Returns:
            Format-specific object (Document, Workbook, OLE file, etc.)
            
        Raises:
            ConversionError: If conversion fails
        """
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """
        Return human-readable format name.
        
        Returns:
            Format name string (e.g., "PDF Document", "DOCX Document")
        """
        pass
    
    def validate(self, file_data: bytes) -> bool:
        """
        Validate if the file data can be converted by this converter.
        
        Override this method to add format-specific validation.
        Default implementation returns True.
        
        Args:
            file_data: Raw binary file data
            
        Returns:
            True if file can be converted, False otherwise
        """
        return True
    
    def close(self, converted_object: Any) -> None:
        """
        Close/cleanup the converted object if needed.
        
        Override this method if the converted object needs explicit cleanup.
        Default implementation does nothing.
        
        Args:
            converted_object: The object returned by convert()
        """
        pass


class NullFileConverter(BaseFileConverter):
    """
    Null implementation of file converter.
    
    Used as default when no conversion is needed.
    Returns the original file data unchanged.
    """
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> bytes:
        """Return file data unchanged."""
        return file_data
    
    def get_format_name(self) -> str:
        """Return generic format name."""
        return "Raw Binary"


class PassThroughConverter(BaseFileConverter):
    """
    Pass-through converter that returns file stream.
    
    Used for handlers that work directly with BytesIO streams.
    """
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> BinaryIO:
        """Return BytesIO stream of file data."""
        if file_stream is not None:
            file_stream.seek(0)
            return file_stream
        return BytesIO(file_data)
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "Binary Stream"


class TextFileConverter(BaseFileConverter):
    """
    Converter for text-based files.
    
    Decodes binary data to text string using encoding detection.
    """
    
    DEFAULT_ENCODINGS = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1', 'ascii']
    
    def __init__(self, encodings: Optional[list] = None):
        """
        Initialize TextFileConverter.
        
        Args:
            encodings: List of encodings to try (default: common encodings)
        """
        self._encodings = encodings or self.DEFAULT_ENCODINGS
        self._detected_encoding: Optional[str] = None
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        encoding: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Convert binary data to text string.
        
        Args:
            file_data: Raw binary file data
            file_stream: Ignored for text conversion
            encoding: Specific encoding to use (None for auto-detect)
            **kwargs: Additional options
            
        Returns:
            Decoded text string
            
        Raises:
            UnicodeDecodeError: If decoding fails with all encodings
        """
        # Try specified encoding first
        if encoding:
            try:
                result = file_data.decode(encoding)
                self._detected_encoding = encoding
                return result
            except UnicodeDecodeError:
                pass
        
        # Try each encoding in order
        for enc in self._encodings:
            try:
                result = file_data.decode(enc)
                self._detected_encoding = enc
                return result
            except UnicodeDecodeError:
                continue
        
        # Fallback: decode with errors='replace'
        self._detected_encoding = 'utf-8'
        return file_data.decode('utf-8', errors='replace')
    
    def get_format_name(self) -> str:
        """Return format name with detected encoding."""
        if self._detected_encoding:
            return f"Text ({self._detected_encoding})"
        return "Text"
    
    @property
    def detected_encoding(self) -> Optional[str]:
        """Return the encoding detected during last conversion."""
        return self._detected_encoding
