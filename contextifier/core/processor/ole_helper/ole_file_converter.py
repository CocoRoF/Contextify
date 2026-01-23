# contextifier/core/processor/ole_helper/ole_file_converter.py
"""
OLEFileConverter - OLE file format converter

Converts OLE compound document binary data to olefile.OleFileIO.
Specifically handles OLE-format (binary) DOC files.
"""
from io import BytesIO
from typing import Any, Optional, BinaryIO

from contextifier.core.functions.file_converter import BaseFileConverter


class OLEFileConverter(BaseFileConverter):
    """
    OLE file converter for compound document format.

    Converts binary OLE data to olefile.OleFileIO object.
    """

    # Magic number for OLE format
    MAGIC_OLE = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'

    def __init__(self):
        """Initialize OLEFileConverter."""
        pass

    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> Any:
        """
        Convert binary OLE data to olefile.OleFileIO.

        Args:
            file_data: Raw binary OLE data
            file_stream: Optional file stream
            **kwargs: Additional options

        Returns:
            olefile.OleFileIO object

        Raises:
            Exception: If conversion fails
        """
        import olefile
        
        # Verify OLE magic header
        if len(file_data) >= 8:
            if not file_data[:8].startswith(self.MAGIC_OLE):
                raise ValueError("Not a valid OLE file (magic header mismatch)")
        
        return olefile.OleFileIO(BytesIO(file_data))

    def get_format_name(self) -> str:
        """Return format name."""
        return "OLE Compound Document"

    def close(self, converted_object: Any) -> None:
        """Close the OLE file if needed."""
        if converted_object is not None:
            if hasattr(converted_object, 'close'):
                converted_object.close()


__all__ = ['OLEFileConverter']
