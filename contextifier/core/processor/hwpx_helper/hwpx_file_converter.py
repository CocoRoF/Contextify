# libs/core/processor/hwpx_helper/hwpx_file_converter.py
"""
HWPXFileConverter - HWPX file format converter

Converts binary HWPX data to ZipFile object.
"""
from io import BytesIO
from typing import Any, Optional, BinaryIO
import zipfile

from contextifier.core.functions.file_converter import BaseFileConverter


class HWPXFileConverter(BaseFileConverter):
    """
    HWPX file converter.
    
    Converts binary HWPX (ZIP format) data to ZipFile object.
    """
    
    # ZIP magic number
    ZIP_MAGIC = b'PK\x03\x04'
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> zipfile.ZipFile:
        """
        Convert binary HWPX data to ZipFile object.
        
        Args:
            file_data: Raw binary HWPX data
            file_stream: Optional file stream
            **kwargs: Additional options
            
        Returns:
            zipfile.ZipFile object
        """
        stream = file_stream if file_stream is not None else BytesIO(file_data)
        stream.seek(0)
        return zipfile.ZipFile(stream, 'r')
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "HWPX Document (ZIP/XML)"
    
    def validate(self, file_data: bytes) -> bool:
        """Validate if data is a valid ZIP file."""
        if not file_data or len(file_data) < 4:
            return False
        
        if file_data[:4] != self.ZIP_MAGIC:
            return False
        
        # Verify it's a valid ZIP
        try:
            with zipfile.ZipFile(BytesIO(file_data), 'r') as zf:
                # HWPX should have specific structure
                namelist = zf.namelist()
                return len(namelist) > 0
        except zipfile.BadZipFile:
            return False
    
    def close(self, converted_object: Any) -> None:
        """Close the ZipFile."""
        if converted_object is not None and hasattr(converted_object, 'close'):
            converted_object.close()
