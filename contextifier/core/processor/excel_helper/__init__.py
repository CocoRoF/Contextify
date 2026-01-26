"""
Excel Helper Module

Handles extraction of elements (textboxes, charts, images, tables, etc.) from XLSX/XLS files.

Module Structure:
- excel_chart_constants: Chart type mapping constants
- excel_chart_extractor: Chart extraction (ChartExtractor)
- excel_table_extractor: Table extraction (XLSXTableExtractor, XLSTableExtractor)
- excel_table_processor: Table formatting (ExcelTableProcessor)
- excel_table_xlsx: XLSX table conversion (backward compatible)
- excel_table_xls: XLS table conversion (backward compatible)
- excel_textbox: Textbox extraction
- excel_metadata: Metadata extraction
- excel_image: Image extraction
- excel_layout_detector: Layout detection
"""

# === Textbox ===
from contextifier.core.processor.excel_helper.excel_textbox import extract_textboxes_from_xlsx

# === Metadata ===
from contextifier.core.processor.excel_helper.excel_metadata import (
    ExcelMetadataExtractor,
    XLSXMetadataExtractor,
    XLSMetadataExtractor,
)

# === Chart Extractor ===
from contextifier.core.processor.excel_helper.excel_chart_extractor import (
    ExcelChartExtractor,
    CHART_TYPE_MAP,
)

# === Image Processor (replaces excel_image.py utility functions) ===
from contextifier.core.processor.excel_helper.excel_image_processor import (
    ExcelImageProcessor,
)

# === Table Extractor (NEW - BaseTableExtractor implementation) ===
from contextifier.core.processor.excel_helper.excel_table_extractor import (
    ExcelTableExtractor,
    XLSXTableExtractor,
    XLSTableExtractor,
    ExcelTableExtractorConfig,
    ExcelTableRegionInfo,
    # Backward compatibility
    has_merged_cells_xlsx,
    has_merged_cells_xls,
)

# === Table Processor (NEW - Table formatting) ===
from contextifier.core.processor.excel_helper.excel_table_processor import (
    ExcelTableProcessor,
    ExcelTableProcessorConfig,
    # Backward compatibility - XLSX
    convert_xlsx_sheet_to_table,
    convert_xlsx_sheet_to_markdown,
    convert_xlsx_sheet_to_html,
    convert_xlsx_objects_to_tables,
    # Backward compatibility - XLS
    convert_xls_sheet_to_table,
    convert_xls_sheet_to_markdown,
    convert_xls_sheet_to_html,
    convert_xls_objects_to_tables,
)

# === Layout Detector ===
from contextifier.core.processor.excel_helper.excel_layout_detector import (
    layout_detect_range_xlsx,
    layout_detect_range_xls,
    object_detect_xlsx,
    object_detect_xls,
    LayoutRange,
)


__all__ = [
    # Textbox
    'extract_textboxes_from_xlsx',
    # Metadata
    'ExcelMetadataExtractor',
    'XLSXMetadataExtractor',
    'XLSMetadataExtractor',
    # Chart Constants
    'CHART_TYPE_MAP',
    # Chart Extractor
    'ExcelChartExtractor',
    # Image Processor
    'ExcelImageProcessor',
    # Table Extractor (NEW)
    'ExcelTableExtractor',
    'XLSXTableExtractor',
    'XLSTableExtractor',
    'ExcelTableExtractorConfig',
    'ExcelTableRegionInfo',
    # Table Processor (NEW)
    'ExcelTableProcessor',
    'ExcelTableProcessorConfig',
    # Table XLSX (backward compatible)
    'has_merged_cells_xlsx',
    'convert_xlsx_sheet_to_table',
    'convert_xlsx_sheet_to_markdown',
    'convert_xlsx_sheet_to_html',
    'convert_xlsx_objects_to_tables',
    # Table XLS (backward compatible)
    'has_merged_cells_xls',
    'convert_xls_sheet_to_table',
    'convert_xls_sheet_to_markdown',
    'convert_xls_sheet_to_html',
    'convert_xls_objects_to_tables',
    # Layout Detector
    'layout_detect_range_xlsx',
    'layout_detect_range_xls',
    'object_detect_xlsx',
    'object_detect_xls',
    'LayoutRange',
]
