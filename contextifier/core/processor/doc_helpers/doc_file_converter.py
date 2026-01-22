# libs/core/processor/doc_helpers/doc_file_converter.py
"""
DOCFileConverter - DOC file format converter

Converts binary DOC data to appropriate format based on detection.
Supports RTF, OLE, HTML, and misnamed DOCX files.
"""
from io import BytesIO
from typing import Any, Optional, BinaryIO, Tuple
from enum import Enum
import zipfile

from contextifier.core.functions.file_converter import BaseFileConverter


class DocFormat(Enum):
    """Detected DOC file format."""
    RTF = "rtf"
    OLE = "ole"
    HTML = "html"
    DOCX = "docx"
    UNKNOWN = "unknown"


class DOCFileConverter(BaseFileConverter):
    """
    DOC file converter with format auto-detection.

    Detects actual format (RTF, OLE, HTML, DOCX) and converts accordingly.
    """

    # Magic numbers for format detection
    MAGIC_RTF = b'{\\rtf'
    MAGIC_OLE = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
    MAGIC_ZIP = b'PK\x03\x04'

    def __init__(self):
        """Initialize DOCFileConverter."""
        self._detected_format: DocFormat = DocFormat.UNKNOWN

    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> Tuple[Any, DocFormat]:
        """
        Convert binary DOC data to appropriate format.

        Args:
            file_data: Raw binary DOC data
            file_stream: Optional file stream
            **kwargs: Additional options

        Returns:
            Tuple of (converted object, detected format)
            - RTF: (bytes, DocFormat.RTF) - 원본 바이너리 반환 (RTFHandler에서 처리)
            - OLE: (olefile.OleFileIO, DocFormat.OLE)
            - HTML: (BeautifulSoup, DocFormat.HTML)
            - DOCX: (docx.Document, DocFormat.DOCX)

        Raises:
            Exception: If conversion fails
        """
        self._detected_format = self._detect_format(file_data)

        if self._detected_format == DocFormat.RTF:
            # RTF는 원본 바이너리 반환 - RTFHandler.extract_text()에서 처리
            return file_data, self._detected_format
        elif self._detected_format == DocFormat.OLE:
            return self._convert_ole(file_data), self._detected_format
        elif self._detected_format == DocFormat.HTML:
            return self._convert_html(file_data), self._detected_format
        elif self._detected_format == DocFormat.DOCX:
            return self._convert_docx(file_data), self._detected_format
        else:
            # Try OLE as fallback
            return self._convert_ole(file_data), DocFormat.OLE

    def _detect_format(self, file_data: bytes) -> DocFormat:
        """Detect actual file format from binary data."""
        if not file_data:
            return DocFormat.UNKNOWN

        header = file_data[:32] if len(file_data) >= 32 else file_data

        # Check RTF
        if header.startswith(self.MAGIC_RTF):
            return DocFormat.RTF

        # Check OLE
        if header.startswith(self.MAGIC_OLE):
            return DocFormat.OLE

        # Check ZIP (possible DOCX)
        if header.startswith(self.MAGIC_ZIP):
            try:
                with zipfile.ZipFile(BytesIO(file_data), 'r') as zf:
                    if '[Content_Types].xml' in zf.namelist():
                        return DocFormat.DOCX
            except zipfile.BadZipFile:
                pass

        # Check HTML
        header_lower = header.lower()
        if (header_lower.startswith(b'<!doctype') or
            header_lower.startswith(b'<html') or
            b'<html' in header_lower[:100]):
            return DocFormat.HTML

        # Check for BOM + RTF
        if header.startswith(b'\xef\xbb\xbf'):
            text_header = header[3:].decode('utf-8', errors='ignore').lower()
            if text_header.startswith('{\\rtf'):
                return DocFormat.RTF

        return DocFormat.UNKNOWN

    def _convert_ole(self, file_data: bytes) -> Any:
        """Convert OLE data."""
        import olefile
        return olefile.OleFileIO(BytesIO(file_data))

    def _convert_html(self, file_data: bytes) -> Any:
        """Convert HTML data."""
        from bs4 import BeautifulSoup
        # Decode with fallback
        try:
            text = file_data.decode('utf-8')
        except UnicodeDecodeError:
            text = file_data.decode('cp949', errors='replace')
        return BeautifulSoup(text, 'html.parser')

    def _convert_docx(self, file_data: bytes) -> Any:
        """Convert misnamed DOCX data."""
        from docx import Document
        return Document(BytesIO(file_data))

    def get_format_name(self) -> str:
        """Return detected format name."""
        format_names = {
            DocFormat.RTF: "RTF Document",
            DocFormat.OLE: "OLE Document (DOC)",
            DocFormat.HTML: "HTML Document",
            DocFormat.DOCX: "DOCX Document (misnamed)",
            DocFormat.UNKNOWN: "Unknown DOC Format",
        }
        return format_names.get(self._detected_format, "Unknown")

    @property
    def detected_format(self) -> DocFormat:
        """Return detected format after conversion."""
        return self._detected_format

    def close(self, converted_object: Any) -> None:
        """Close the converted object if needed."""
        if converted_object is not None:
            if hasattr(converted_object, 'close'):
                converted_object.close()
