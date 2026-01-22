# libs/core/processor/excel_helper/excel_file_converter.py
"""
ExcelFileConverter - Excel file format converter

Converts binary Excel data to Workbook object.
Supports both XLSX and XLS formats.
"""
from io import BytesIO
from typing import Any, Optional, BinaryIO, Union

from contextifier.core.functions.file_converter import BaseFileConverter


class XLSXFileConverter(BaseFileConverter):
    """
    XLSX file converter using openpyxl.
    
    Converts binary XLSX data to openpyxl Workbook object.
    """
    
    # ZIP magic number (XLSX is a ZIP file)
    ZIP_MAGIC = b'PK\x03\x04'
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        data_only: bool = True,
        **kwargs
    ) -> Any:
        """
        Convert binary XLSX data to Workbook object.
        
        Args:
            file_data: Raw binary XLSX data
            file_stream: Optional file stream
            data_only: If True, return calculated values instead of formulas
            **kwargs: Additional options
            
        Returns:
            openpyxl.Workbook object
        """
        from openpyxl import load_workbook
        
        stream = file_stream if file_stream is not None else BytesIO(file_data)
        stream.seek(0)
        return load_workbook(stream, data_only=data_only)
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "XLSX Workbook"
    
    def validate(self, file_data: bytes) -> bool:
        """Validate if data is a valid XLSX."""
        if not file_data or len(file_data) < 4:
            return False
        return file_data[:4] == self.ZIP_MAGIC


class XLSFileConverter(BaseFileConverter):
    """
    XLS file converter using xlrd.
    
    Converts binary XLS data to xlrd Workbook object.
    """
    
    # OLE magic number (XLS is an OLE file)
    OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> Any:
        """
        Convert binary XLS data to xlrd Workbook object.
        
        Args:
            file_data: Raw binary XLS data
            file_stream: Optional file stream (not used)
            **kwargs: Additional options
            
        Returns:
            xlrd.Book object
        """
        import xlrd
        return xlrd.open_workbook(file_contents=file_data)
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "XLS Workbook"
    
    def validate(self, file_data: bytes) -> bool:
        """Validate if data is a valid XLS."""
        if not file_data or len(file_data) < 8:
            return False
        return file_data[:8] == self.OLE_MAGIC


class ExcelFileConverter(BaseFileConverter):
    """
    Unified Excel file converter.
    
    Auto-detects format (XLSX/XLS) and uses appropriate converter.
    """
    
    def __init__(self):
        """Initialize with both converters."""
        self._xlsx_converter = XLSXFileConverter()
        self._xls_converter = XLSFileConverter()
        self._used_converter: Optional[BaseFileConverter] = None
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        extension: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Convert binary Excel data to Workbook object.
        
        Args:
            file_data: Raw binary Excel data
            file_stream: Optional file stream
            extension: File extension hint ('xlsx' or 'xls')
            **kwargs: Additional options
            
        Returns:
            Workbook object (openpyxl or xlrd)
        """
        # Determine format from extension or magic number
        if extension:
            ext = extension.lower().lstrip('.')
            if ext == 'xlsx':
                self._used_converter = self._xlsx_converter
            elif ext == 'xls':
                self._used_converter = self._xls_converter
        else:
            # Auto-detect
            if self._xlsx_converter.validate(file_data):
                self._used_converter = self._xlsx_converter
            elif self._xls_converter.validate(file_data):
                self._used_converter = self._xls_converter
            else:
                # Default to XLSX
                self._used_converter = self._xlsx_converter
        
        return self._used_converter.convert(file_data, file_stream, **kwargs)
    
    def get_format_name(self) -> str:
        """Return format name based on detected type."""
        if self._used_converter:
            return self._used_converter.get_format_name()
        return "Excel Workbook"
