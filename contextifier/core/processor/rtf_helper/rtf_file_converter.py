# libs/core/processor/rtf_helper/rtf_file_converter.py
"""
RTFFileConverter - RTF file format converter

Converts binary RTF data to RTFDocument object.
"""
from io import BytesIO
from typing import Any, Optional, BinaryIO, Set

from contextifier.core.functions.file_converter import BaseFileConverter


class RTFFileConverter(BaseFileConverter):
    """
    RTF file converter.
    
    Converts binary RTF data to RTFDocument object using RTFParser.
    """
    
    # RTF magic number
    RTF_MAGIC = b'{\\rtf'
    
    def __init__(self):
        """Initialize RTFFileConverter."""
        self._encoding = 'cp949'
        self._image_processor = None
        self._processed_images: Set[str] = set()
    
    def configure(
        self,
        encoding: str = 'cp949',
        image_processor: Any = None,
        processed_images: Optional[Set[str]] = None
    ) -> 'RTFFileConverter':
        """
        Configure the converter with additional options.
        
        Args:
            encoding: Default encoding (Korean documents typically use cp949)
            image_processor: Image processor instance
            processed_images: Set of already processed image hashes
            
        Returns:
            Self for method chaining
        """
        self._encoding = encoding
        self._image_processor = image_processor
        self._processed_images = processed_images if processed_images is not None else set()
        return self
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> Any:
        """
        Convert binary RTF data to RTFDocument object.
        
        Args:
            file_data: Raw binary RTF data
            file_stream: Optional file stream (ignored, uses file_data)
            **kwargs: Additional options:
                - encoding: Override default encoding
                - image_processor: Override configured image processor
                - processed_images: Override configured processed images set
            
        Returns:
            RTFDocument object
        """
        from contextifier.core.processor.rtf_helper import parse_rtf
        
        # Get options from kwargs or use configured values
        encoding = kwargs.get('encoding', self._encoding)
        image_processor = kwargs.get('image_processor', self._image_processor)
        processed_images = kwargs.get('processed_images', self._processed_images)
        
        # Parse RTF
        rtf_doc = parse_rtf(
            file_data,
            processed_images=processed_images,
            image_processor=image_processor
        )
        
        return rtf_doc
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "RTF Document"
    
    def validate(self, file_data: bytes) -> bool:
        """Validate if data is a valid RTF file."""
        if not file_data or len(file_data) < 5:
            return False
        return file_data[:5] == self.RTF_MAGIC
    
    def close(self, converted_object: Any) -> None:
        """
        Close/cleanup the converted object.
        
        RTFDocument doesn't need explicit cleanup.
        """
        pass


__all__ = ['RTFFileConverter']
