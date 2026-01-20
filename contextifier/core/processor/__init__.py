# libs/core/processor/__init__.py
"""
Processor - Document Type-specific Handler Module

Provides handlers for processing individual document formats.

Handler List:
- pdf_handler: PDF document processing (adaptive complexity-based)
- docx_handler: DOCX document processing
- doc_handler: DOC document processing (including RTF)
- ppt_handler: PPT/PPTX document processing
- excel_handler: Excel (XLSX/XLS) document processing
- hwp_processor: HWP document processing
- hwpx_processor: HWPX document processing
- csv_handler: CSV file processing
- text_handler: Text file processing
- html_reprocessor: HTML reprocessing

Helper Modules (subdirectories):
- csv_helper/: CSV processing helper
- docx_helper/: DOCX processing helper
- doc_helpers/: DOC/RTF processing helper
- excel_helper/: Excel processing helper
- hwp_helper/: HWP processing helper
- hwpx_helper/: HWPX processing helper
- pdf_helpers/: PDF processing helper
- ppt_helper/: PPT processing helper

Usage Example:
    from contextifier.core.processor import PDFHandler
    from contextifier.core.processor import DOCXHandler
    from contextifier.core.processor.pdf_helpers import extract_pdf_metadata
"""

# === PDF Handler ===
from contextifier.core.processor.pdf_handler import PDFHandler

# === Document Handlers ===
from contextifier.core.processor.docx_handler import DOCXHandler
from contextifier.core.processor.doc_handler import DOCHandler
from contextifier.core.processor.ppt_handler import PPTHandler

# === Data Handlers ===
from contextifier.core.processor.excel_handler import ExcelHandler
from contextifier.core.processor.csv_handler import CSVHandler
from contextifier.core.processor.text_handler import TextHandler

# === HWP Handlers ===
from contextifier.core.processor.hwp_handler import HWPHandler
from contextifier.core.processor.hwps_handler import HWPXHandler

# === Other Processors ===
# from contextifier.core.processor.html_reprocessor import ...  # HTML reprocessing

# === Helper Modules (subpackages) ===
from contextifier.core.processor import csv_helper
from contextifier.core.processor import docx_helper
from contextifier.core.processor import excel_helper
from contextifier.core.processor import hwp_helper
from contextifier.core.processor import hwpx_helper
from contextifier.core.processor import pdf_helpers
from contextifier.core.processor import ppt_helper

__all__ = [
    # PDF Handler
    "PDFHandler",
    # Document Handlers
    "DOCXHandler",
    "DOCHandler",
    "PPTHandler",
    # Data Handlers
    "ExcelHandler",
    "CSVHandler",
    "TextHandler",
    # HWP Handlers
    "HWPHandler",
    "HWPXHandler",
    # Helper subpackages
    "csv_helper",
    "docx_helper",
    "excel_helper",
    "hwp_helper",
    "hwpx_helper",
    "pdf_helpers",
    "ppt_helper",
]
