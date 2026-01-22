# libs/core/processor/image_file_helper/image_file_converter.py
"""
ImageFileConverter - Image file format converter

Pass-through converter for image files.
Images are kept as binary data.
"""
from typing import Any, Optional, BinaryIO

from contextifier.core.functions.file_converter import NullFileConverter


class ImageFileConverter(NullFileConverter):
    """
    Image file converter.
    
    Images don't need conversion - returns raw bytes.
    This is a pass-through converter.
    """
    
    # Common image magic numbers
    MAGIC_JPEG = b'\xff\xd8\xff'
    MAGIC_PNG = b'\x89PNG\r\n\x1a\n'
    MAGIC_GIF = b'GIF8'
    MAGIC_BMP = b'BM'
    MAGIC_WEBP = b'RIFF'
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "Image File"
    
    def validate(self, file_data: bytes) -> bool:
        """Validate if data is an image."""
        if not file_data or len(file_data) < 4:
            return False
        
        return (
            file_data[:3] == self.MAGIC_JPEG or
            file_data[:8] == self.MAGIC_PNG or
            file_data[:4] == self.MAGIC_GIF or
            file_data[:2] == self.MAGIC_BMP or
            file_data[:4] == self.MAGIC_WEBP
        )
    
    def detect_image_type(self, file_data: bytes) -> Optional[str]:
        """
        Detect image type from binary data.
        
        Args:
            file_data: Raw binary image data
            
        Returns:
            Image type string (jpeg, png, gif, bmp, webp) or None
        """
        if not file_data or len(file_data) < 8:
            return None
        
        if file_data[:3] == self.MAGIC_JPEG:
            return "jpeg"
        elif file_data[:8] == self.MAGIC_PNG:
            return "png"
        elif file_data[:4] == self.MAGIC_GIF:
            return "gif"
        elif file_data[:2] == self.MAGIC_BMP:
            return "bmp"
        elif file_data[:4] == self.MAGIC_WEBP:
            return "webp"
        return None
