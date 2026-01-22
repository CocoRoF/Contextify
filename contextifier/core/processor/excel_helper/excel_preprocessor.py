# contextifier/core/processor/excel_helper/excel_preprocessor.py
"""
Excel Preprocessor - Process Excel workbook after conversion.

Processing Pipeline Position:
    1. ExcelFileConverter.convert() → openpyxl.Workbook or xlrd.Book
    2. ExcelPreprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. ExcelMetadataExtractor.extract() → DocumentMetadata
    4. Content extraction (sheets, cells, images, charts)

Current Implementation:
    - Pass-through (Excel uses openpyxl/xlrd objects directly)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("contextify.excel.preprocessor")


class ExcelPreprocessor(BasePreprocessor):
    """
    Excel Workbook Preprocessor.

    Currently a pass-through implementation as Excel processing
    is handled during the content extraction phase using openpyxl/xlrd.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted Excel workbook.

        Args:
            converted_data: openpyxl.Workbook or xlrd.Book from ExcelFileConverter
            **kwargs: Additional options

        Returns:
            PreprocessedData with the workbook and any extracted resources
        """
        metadata: Dict[str, Any] = {}

        # Detect workbook type and extract info
        if hasattr(converted_data, 'sheetnames'):
            # openpyxl Workbook
            metadata['format'] = 'xlsx'
            metadata['sheet_count'] = len(converted_data.sheetnames)
            metadata['sheet_names'] = converted_data.sheetnames
        elif hasattr(converted_data, 'sheet_names'):
            # xlrd Book
            metadata['format'] = 'xls'
            metadata['sheet_count'] = converted_data.nsheets
            metadata['sheet_names'] = converted_data.sheet_names()

        logger.debug("Excel preprocessor: pass-through, metadata=%s", metadata)

        # clean_content is the TRUE SOURCE - contains the Workbook
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE - openpyxl.Workbook or xlrd.Book
            encoding="utf-8",
            extracted_resources={},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "Excel Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is an Excel Workbook object."""
        # openpyxl or xlrd
        return hasattr(data, 'sheetnames') or hasattr(data, 'sheet_names')


__all__ = ['ExcelPreprocessor']
