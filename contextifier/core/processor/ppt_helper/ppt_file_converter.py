# libs/core/processor/ppt_helper/ppt_file_converter.py
"""
PPTFileConverter - PPT/PPTX file format converter

Converts binary PPT/PPTX data to python-pptx Presentation object.
"""
from io import BytesIO
from typing import Any, Optional, BinaryIO

from contextifier.core.functions.file_converter import BaseFileConverter


class PPTFileConverter(BaseFileConverter):
    """
    PPT/PPTX file converter using python-pptx.
    
    Converts binary PPT/PPTX data to Presentation object.
    """
    
    # ZIP magic number (PPTX is a ZIP file)
    ZIP_MAGIC = b'PK\x03\x04'
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> Any:
        """
        Convert binary PPT/PPTX data to Presentation object.
        
        Args:
            file_data: Raw binary PPT/PPTX data
            file_stream: Optional file stream
            **kwargs: Additional options
            
        Returns:
            pptx.Presentation object
        """
        from pptx import Presentation
        
        stream = file_stream if file_stream is not None else BytesIO(file_data)
        stream.seek(0)
        return Presentation(stream)
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "PPT/PPTX Presentation"
    
    def validate(self, file_data: bytes) -> bool:
        """Validate if data is a valid PPTX."""
        if not file_data or len(file_data) < 4:
            return False
        return file_data[:4] == self.ZIP_MAGIC
