# contextifier/core/processor/excel_helper/excel_table_extractor.py
"""
Excel Table Extractor - Excel Format-Specific Table Extraction

Implements table extraction for Excel files (XLSX/XLS).
Follows BaseTableExtractor interface from table_extractor.py.

Excel Table Structure:
- XLSX: openpyxl Worksheet with merged_cells.ranges for merge info
- XLS: xlrd Sheet with merged_cells tuples for merge info
- Both use LayoutRange for data region detection

2-Pass Approach:
1. Pass 1: Detect table regions using object_detect_xlsx/xls
2. Pass 2: Extract content from detected regions (TableData objects)

Usage:
    from contextifier.core.processor.excel_helper.excel_table_extractor import (
        ExcelTableExtractor,
        XLSXTableExtractor,
        XLSTableExtractor,
    )

    # For XLSX files
    extractor = XLSXTableExtractor()
    tables = extractor.extract_tables(worksheet)  # openpyxl Worksheet
    
    # For XLS files
    extractor = XLSTableExtractor()
    tables = extractor.extract_tables((sheet, workbook))  # xlrd Sheet, Workbook
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from contextifier.core.functions.table_extractor import (
    BaseTableExtractor,
    TableCell,
    TableData,
    TableRegion,
    TableExtractorConfig,
)
from contextifier.core.processor.excel_helper.excel_layout_detector import (
    layout_detect_range_xlsx,
    layout_detect_range_xls,
    object_detect_xlsx,
    object_detect_xls,
    LayoutRange,
)

logger = logging.getLogger("document-processor")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExcelTableExtractorConfig(TableExtractorConfig):
    """Configuration specific to Excel table extraction.
    
    Attributes:
        skip_empty_tables: Whether to skip tables with no data
        detect_header_row: Whether to detect header row automatically
        treat_first_row_as_header: Whether to treat first row as header
        max_rows: Maximum rows to process (for performance)
        max_cols: Maximum columns to process (for performance)
    """
    skip_empty_tables: bool = True
    detect_header_row: bool = True
    treat_first_row_as_header: bool = True
    max_rows: int = 1000
    max_cols: int = 100


# ============================================================================
# Data Classes for Table Region Tracking
# ============================================================================

@dataclass
class ExcelTableRegionInfo:
    """Additional information for Excel table region.
    
    Stores reference to the layout range for Pass 2.
    """
    sheet_name: str           # Sheet name
    layout: LayoutRange       # LayoutRange object for the table region
    has_merged_cells: bool    # Whether the region has merged cells
    region_index: int         # Index of this region within the sheet


# ============================================================================
# Base Excel Table Extractor Class
# ============================================================================

class ExcelTableExtractor(BaseTableExtractor):
    """Base class for Excel table extractors.
    
    Provides common functionality for XLSX and XLS table extraction.
    Subclasses should implement format-specific methods.
    """
    
    def __init__(self, config: Optional[ExcelTableExtractorConfig] = None):
        """Initialize Excel table extractor.
        
        Args:
            config: Excel table extraction configuration
        """
        self._config = config or ExcelTableExtractorConfig()
        super().__init__(self._config)
        # Cache for regions
        self._region_info_cache: Dict[int, ExcelTableRegionInfo] = {}
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format."""
        return format_type.lower() in ("xlsx", "xls", "xlsm", "xlsb")
    
    def detect_table_regions(self, content: Any) -> List[TableRegion]:
        """Detect table regions in Excel content.
        
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement detect_table_regions()")
    
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from a detected region.
        
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement extract_table_from_region()")


# ============================================================================
# XLSX Table Extractor
# ============================================================================

class XLSXTableExtractor(ExcelTableExtractor):
    """XLSX format-specific table extractor.
    
    Extracts tables from XLSX files using openpyxl.
    Implements BaseTableExtractor interface.
    
    Supports:
    - Cell merges (rowspan/colspan via merged_cells.ranges)
    - Multiple table regions per sheet (object detection)
    - Header row detection
    """
    
    def __init__(self, config: Optional[ExcelTableExtractorConfig] = None):
        """Initialize XLSX table extractor."""
        super().__init__(config)
        self._worksheet_cache: Any = None
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format."""
        return format_type.lower() in ("xlsx", "xlsm")
    
    def detect_table_regions(self, content: Any) -> List[TableRegion]:
        """Detect table regions in XLSX worksheet.
        
        Pass 1: Find all table regions using object detection.
        
        Args:
            content: openpyxl Worksheet object
            
        Returns:
            List of TableRegion objects
        """
        ws = content
        if ws is None:
            return []
        
        self._worksheet_cache = ws
        self._region_info_cache.clear()
        
        regions = []
        
        try:
            # Get sheet name
            sheet_name = ws.title if hasattr(ws, 'title') else "Sheet"
            
            # Detect object regions
            layout_regions = object_detect_xlsx(ws, None)
            
            if not layout_regions:
                # Fallback: try to detect single layout
                single_layout = layout_detect_range_xlsx(ws)
                if single_layout:
                    layout_regions = [single_layout]
            
            for idx, layout in enumerate(layout_regions):
                if not layout or not layout.is_valid():
                    continue
                
                # Check for merged cells in this region
                has_merged = self._has_merged_cells_in_region(ws, layout)
                
                # Create region info
                region_info = ExcelTableRegionInfo(
                    sheet_name=sheet_name,
                    layout=layout,
                    has_merged_cells=has_merged,
                    region_index=idx,
                )
                self._region_info_cache[idx] = region_info
                
                # Create TableRegion
                region = TableRegion(
                    start_offset=idx,
                    end_offset=idx + 1,
                    row_count=layout.row_count(),
                    col_count=layout.col_count(),
                    confidence=self._calculate_confidence(ws, layout),
                )
                regions.append(region)
            
            self.logger.debug(f"Detected {len(regions)} table regions in XLSX sheet '{sheet_name}'")
            
        except Exception as e:
            self.logger.warning(f"Error detecting XLSX table regions: {e}")
        
        return regions
    
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from a detected XLSX region.
        
        Pass 2: Extract actual table content from region.
        
        Args:
            content: openpyxl Worksheet object
            region: TableRegion identifying where the table is
            
        Returns:
            TableData object or None if extraction fails
        """
        ws = content if content is not None else self._worksheet_cache
        if ws is None:
            return None
        
        region_idx = region.start_offset
        region_info = self._region_info_cache.get(region_idx)
        
        if region_info is None:
            self.logger.warning(f"No region info found for index {region_idx}")
            return None
        
        layout = region_info.layout
        
        try:
            # Collect merge information
            merged_cells_info, skip_cells, merged_value_override = self._collect_merge_info(
                ws, layout
            )
            
            # Build TableData
            rows: List[List[TableCell]] = []
            has_data = False
            
            for row_idx in range(layout.min_row, layout.max_row + 1):
                row_cells: List[TableCell] = []
                
                for col_idx in range(layout.min_col, layout.max_col + 1):
                    # Skip merged cell parts
                    if (row_idx, col_idx) in skip_cells:
                        continue
                    
                    # Get cell value
                    cell = ws.cell(row=row_idx, column=col_idx)
                    cell_value = ""
                    
                    # Check for merged value override
                    if (row_idx, col_idx) in merged_value_override:
                        cell_value = str(merged_value_override[(row_idx, col_idx)]).strip()
                    elif cell.value is not None:
                        cell_value = str(cell.value).strip()
                    
                    if cell_value:
                        has_data = True
                    
                    # Get merge info
                    row_span = 1
                    col_span = 1
                    if (row_idx, col_idx) in merged_cells_info:
                        row_span, col_span = merged_cells_info[(row_idx, col_idx)]
                    
                    # Create TableCell
                    table_cell = TableCell(
                        content=cell_value,
                        row_span=row_span,
                        col_span=col_span,
                        is_header=(row_idx == layout.min_row and self._config.treat_first_row_as_header),
                        row_index=row_idx - layout.min_row,
                        col_index=col_idx - layout.min_col,
                    )
                    row_cells.append(table_cell)
                
                if row_cells:  # Include row even if empty (for table structure)
                    rows.append(row_cells)
            
            if not has_data and self._config.skip_empty_tables:
                return None
            
            # Create TableData
            table_data = TableData(
                rows=rows,
                num_rows=len(rows),
                num_cols=layout.col_count(),
                has_header=self._config.treat_first_row_as_header,
                source_format="xlsx",
                metadata={
                    "sheet_name": region_info.sheet_name,
                    "has_merged_cells": region_info.has_merged_cells,
                    "layout": {
                        "min_row": layout.min_row,
                        "max_row": layout.max_row,
                        "min_col": layout.min_col,
                        "max_col": layout.max_col,
                    }
                }
            )
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting XLSX table from region: {e}")
            return None
    
    def _has_merged_cells_in_region(self, ws, layout: LayoutRange) -> bool:
        """Check if there are merged cells in the given region."""
        try:
            if len(ws.merged_cells.ranges) == 0:
                return False
            
            for merged_range in ws.merged_cells.ranges:
                if (merged_range.min_row <= layout.max_row and
                    merged_range.max_row >= layout.min_row and
                    merged_range.min_col <= layout.max_col and
                    merged_range.max_col >= layout.min_col):
                    return True
            
            return False
        except Exception:
            return False
    
    def _collect_merge_info(
        self, 
        ws, 
        layout: LayoutRange
    ) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], Set[Tuple[int, int]], Dict[Tuple[int, int], Any]]:
        """Collect merge cell information for the region.
        
        Returns:
            Tuple of (merged_cells_info, skip_cells, merged_value_override)
            - merged_cells_info: Dict of (row, col) -> (rowspan, colspan)
            - skip_cells: Set of cells to skip (part of merged region)
            - merged_value_override: Dict of (row, col) -> value for cells where merge starts outside layout
        """
        merged_cells_info: Dict[Tuple[int, int], Tuple[int, int]] = {}
        skip_cells: Set[Tuple[int, int]] = set()
        merged_value_override: Dict[Tuple[int, int], Any] = {}
        
        try:
            for merged_range in ws.merged_cells.ranges:
                mr_min_row, mr_min_col = merged_range.min_row, merged_range.min_col
                mr_max_row, mr_max_col = merged_range.max_row, merged_range.max_col
                
                # Check if region overlaps with layout
                if not (mr_min_row <= layout.max_row and
                        mr_max_row >= layout.min_row and
                        mr_min_col <= layout.max_col and
                        mr_max_col >= layout.min_col):
                    continue
                
                # Check if merge start is inside layout
                start_in_layout = (layout.min_row <= mr_min_row <= layout.max_row and
                                   layout.min_col <= mr_min_col <= layout.max_col)
                
                if start_in_layout:
                    # Normal case: store merge info
                    rowspan = mr_max_row - mr_min_row + 1
                    colspan = mr_max_col - mr_min_col + 1
                    merged_cells_info[(mr_min_row, mr_min_col)] = (rowspan, colspan)
                    
                    # Mark other cells as skip
                    for r in range(mr_min_row, mr_max_row + 1):
                        for c in range(mr_min_col, mr_max_col + 1):
                            if r != mr_min_row or c != mr_min_col:
                                skip_cells.add((r, c))
                else:
                    # Merge start is outside layout - get value for first cell in layout
                    merged_value = ws.cell(row=mr_min_row, column=mr_min_col).value
                    if merged_value is not None:
                        first_row_in_layout = max(mr_min_row, layout.min_row)
                        first_col_in_layout = max(mr_min_col, layout.min_col)
                        merged_value_override[(first_row_in_layout, first_col_in_layout)] = merged_value
                    
                    # Mark other cells in layout as skip
                    for r in range(max(mr_min_row, layout.min_row), min(mr_max_row, layout.max_row) + 1):
                        for c in range(max(mr_min_col, layout.min_col), min(mr_max_col, layout.max_col) + 1):
                            if (r, c) not in merged_value_override:
                                skip_cells.add((r, c))
        
        except Exception as e:
            self.logger.warning(f"Error collecting merge info: {e}")
        
        return merged_cells_info, skip_cells, merged_value_override
    
    def _calculate_confidence(self, ws, layout: LayoutRange) -> float:
        """Calculate confidence score for a table region."""
        try:
            # Base confidence
            confidence = 0.7
            
            # Boost for reasonable size
            if layout.row_count() >= 2 and layout.col_count() >= 2:
                confidence += 0.1
            
            # Boost for having merged cells (indicates structured table)
            if self._has_merged_cells_in_region(ws, layout):
                confidence += 0.1
            
            # Boost for data density
            data_cells = 0
            total_cells = layout.cell_count()
            for row_idx in range(layout.min_row, min(layout.min_row + 10, layout.max_row + 1)):
                for col_idx in range(layout.min_col, min(layout.min_col + 10, layout.max_col + 1)):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    if cell.value is not None and str(cell.value).strip():
                        data_cells += 1
            
            density = data_cells / min(total_cells, 100)
            confidence += density * 0.1
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.5


# ============================================================================
# XLS Table Extractor
# ============================================================================

class XLSTableExtractor(ExcelTableExtractor):
    """XLS format-specific table extractor.
    
    Extracts tables from XLS files using xlrd.
    Implements BaseTableExtractor interface.
    
    Note: XLS extraction requires both sheet and workbook objects
    due to xlrd's cell value formatting requirements.
    
    Supports:
    - Cell merges (rowspan/colspan via merged_cells tuples)
    - Multiple table regions per sheet (object detection)
    - Header row detection
    """
    
    def __init__(self, config: Optional[ExcelTableExtractorConfig] = None):
        """Initialize XLS table extractor."""
        super().__init__(config)
        self._sheet_cache: Any = None
        self._workbook_cache: Any = None
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format."""
        return format_type.lower() == "xls"
    
    def detect_table_regions(self, content: Any) -> List[TableRegion]:
        """Detect table regions in XLS sheet.
        
        Pass 1: Find all table regions using object detection.
        
        Args:
            content: Tuple of (xlrd Sheet, xlrd Workbook) or just xlrd Sheet
            
        Returns:
            List of TableRegion objects
        """
        # Handle content format
        sheet, wb = self._parse_content(content)
        if sheet is None:
            return []
        
        self._sheet_cache = sheet
        self._workbook_cache = wb
        self._region_info_cache.clear()
        
        regions = []
        
        try:
            # Get sheet name
            sheet_name = sheet.name if hasattr(sheet, 'name') else "Sheet"
            
            # Detect object regions
            layout_regions = object_detect_xls(sheet, wb, None)
            
            if not layout_regions:
                # Fallback: try to detect single layout
                single_layout = layout_detect_range_xls(sheet)
                if single_layout:
                    layout_regions = [single_layout]
            
            for idx, layout in enumerate(layout_regions):
                if not layout or not layout.is_valid():
                    continue
                
                # Check for merged cells in this region
                has_merged = self._has_merged_cells_in_region(sheet, layout)
                
                # Create region info
                region_info = ExcelTableRegionInfo(
                    sheet_name=sheet_name,
                    layout=layout,
                    has_merged_cells=has_merged,
                    region_index=idx,
                )
                self._region_info_cache[idx] = region_info
                
                # Create TableRegion
                region = TableRegion(
                    start_offset=idx,
                    end_offset=idx + 1,
                    row_count=layout.row_count(),
                    col_count=layout.col_count(),
                    confidence=self._calculate_confidence(sheet, layout),
                )
                regions.append(region)
            
            self.logger.debug(f"Detected {len(regions)} table regions in XLS sheet '{sheet_name}'")
            
        except Exception as e:
            self.logger.warning(f"Error detecting XLS table regions: {e}")
        
        return regions
    
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from a detected XLS region.
        
        Pass 2: Extract actual table content from region.
        
        Args:
            content: Tuple of (xlrd Sheet, xlrd Workbook) or just xlrd Sheet
            region: TableRegion identifying where the table is
            
        Returns:
            TableData object or None if extraction fails
        """
        # Handle content format
        sheet, wb = self._parse_content(content)
        if sheet is None:
            sheet = self._sheet_cache
            wb = self._workbook_cache
        
        if sheet is None:
            return None
        
        region_idx = region.start_offset
        region_info = self._region_info_cache.get(region_idx)
        
        if region_info is None:
            self.logger.warning(f"No region info found for index {region_idx}")
            return None
        
        layout = region_info.layout
        
        try:
            # Collect merge information
            merged_cells_info, skip_cells = self._collect_merge_info(sheet, layout)
            
            # Build TableData
            rows: List[List[TableCell]] = []
            has_data = False
            
            # Convert 1-based layout to 0-based for xlrd
            for row_idx_0 in range(layout.min_row - 1, layout.max_row):
                row_idx_1 = row_idx_0 + 1  # For TableCell position
                row_cells: List[TableCell] = []
                
                for col_idx_0 in range(layout.min_col - 1, layout.max_col):
                    col_idx_1 = col_idx_0 + 1  # For TableCell position
                    
                    # Skip merged cell parts (using 0-based index)
                    if (row_idx_0, col_idx_0) in skip_cells:
                        continue
                    
                    # Get cell value
                    cell_value = ""
                    try:
                        value = sheet.cell_value(row_idx_0, col_idx_0)
                        if value:
                            cell_type = sheet.cell_type(row_idx_0, col_idx_0)
                            cell_value = self._format_xls_cell_value(value, cell_type, wb)
                            if cell_value:
                                has_data = True
                    except Exception:
                        pass
                    
                    # Get merge info (using 0-based index)
                    row_span = 1
                    col_span = 1
                    if (row_idx_0, col_idx_0) in merged_cells_info:
                        row_span, col_span = merged_cells_info[(row_idx_0, col_idx_0)]
                    
                    # Create TableCell
                    table_cell = TableCell(
                        content=cell_value,
                        row_span=row_span,
                        col_span=col_span,
                        is_header=(row_idx_0 == layout.min_row - 1 and self._config.treat_first_row_as_header),
                        row_index=row_idx_0 - (layout.min_row - 1),
                        col_index=col_idx_0 - (layout.min_col - 1),
                    )
                    row_cells.append(table_cell)
                
                if row_cells:  # Include row even if empty (for table structure)
                    rows.append(row_cells)
            
            if not has_data and self._config.skip_empty_tables:
                return None
            
            # Create TableData
            table_data = TableData(
                rows=rows,
                num_rows=len(rows),
                num_cols=layout.col_count(),
                has_header=self._config.treat_first_row_as_header,
                source_format="xls",
                metadata={
                    "sheet_name": region_info.sheet_name,
                    "has_merged_cells": region_info.has_merged_cells,
                    "layout": {
                        "min_row": layout.min_row,
                        "max_row": layout.max_row,
                        "min_col": layout.min_col,
                        "max_col": layout.max_col,
                    }
                }
            )
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting XLS table from region: {e}")
            return None
    
    def _parse_content(self, content: Any) -> Tuple[Any, Any]:
        """Parse content to get sheet and workbook.
        
        Args:
            content: Tuple of (sheet, workbook) or just sheet
            
        Returns:
            Tuple of (sheet, workbook)
        """
        if content is None:
            return None, None
        
        if isinstance(content, tuple) and len(content) == 2:
            return content[0], content[1]
        
        # Assume it's just the sheet, try to get cached workbook
        return content, self._workbook_cache
    
    def _has_merged_cells_in_region(self, sheet, layout: LayoutRange) -> bool:
        """Check if there are merged cells in the given region."""
        try:
            if len(sheet.merged_cells) == 0:
                return False
            
            # xlrd merged_cells is (rlo, rhi, clo, chi), 0-based, exclusive
            for (rlo, rhi, clo, chi) in sheet.merged_cells:
                # Convert to 1-based for comparison
                mr_min_row = rlo + 1
                mr_max_row = rhi  # exclusive, so equals max row
                mr_min_col = clo + 1
                mr_max_col = chi  # exclusive, so equals max col
                
                if (mr_min_row <= layout.max_row and
                    mr_max_row >= layout.min_row and
                    mr_min_col <= layout.max_col and
                    mr_max_col >= layout.min_col):
                    return True
            
            return False
        except Exception:
            return False
    
    def _collect_merge_info(
        self, 
        sheet, 
        layout: LayoutRange
    ) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], Set[Tuple[int, int]]]:
        """Collect merge cell information for the region.
        
        Note: Uses 0-based indexing for xlrd compatibility.
        
        Returns:
            Tuple of (merged_cells_info, skip_cells)
            - merged_cells_info: Dict of (row, col) -> (rowspan, colspan), 0-based
            - skip_cells: Set of cells to skip (part of merged region), 0-based
        """
        merged_cells_info: Dict[Tuple[int, int], Tuple[int, int]] = {}
        skip_cells: Set[Tuple[int, int]] = set()
        
        try:
            # xlrd merged_cells is (rlo, rhi, clo, chi), 0-based, exclusive
            for (rlo, rhi, clo, chi) in sheet.merged_cells:
                # Convert to 1-based for comparison with layout
                mr_min_row_1 = rlo + 1
                mr_max_row_1 = rhi  # exclusive
                mr_min_col_1 = clo + 1
                mr_max_col_1 = chi  # exclusive
                
                # Check if region overlaps with layout
                if not (mr_min_row_1 <= layout.max_row and
                        mr_max_row_1 >= layout.min_row and
                        mr_min_col_1 <= layout.max_col and
                        mr_max_col_1 >= layout.min_col):
                    continue
                
                # Calculate span (using 0-based xlrd values)
                rowspan = rhi - rlo
                colspan = chi - clo
                
                # Store using 0-based index
                merged_cells_info[(rlo, clo)] = (rowspan, colspan)
                
                # Mark other cells as skip (0-based)
                for r in range(rlo, rhi):
                    for c in range(clo, chi):
                        if r != rlo or c != clo:
                            skip_cells.add((r, c))
        
        except Exception as e:
            self.logger.warning(f"Error collecting XLS merge info: {e}")
        
        return merged_cells_info, skip_cells
    
    def _format_xls_cell_value(self, value, cell_type, wb) -> str:
        """Format XLS cell value to string.
        
        Args:
            value: Cell value
            cell_type: xlrd cell type
            wb: xlrd Workbook object
            
        Returns:
            Formatted string
        """
        try:
            import xlrd
            
            if cell_type == xlrd.XL_CELL_NUMBER:
                if value == int(value):
                    return str(int(value))
                else:
                    return str(value)
            elif cell_type == xlrd.XL_CELL_DATE:
                try:
                    date_tuple = xlrd.xldate_as_tuple(value, wb.datemode if wb else 0)
                    return f"{date_tuple[0]:04d}-{date_tuple[1]:02d}-{date_tuple[2]:02d}"
                except Exception:
                    return str(value)
            else:
                return str(value).strip()
        except Exception:
            return str(value).strip() if value else ""
    
    def _calculate_confidence(self, sheet, layout: LayoutRange) -> float:
        """Calculate confidence score for a table region."""
        try:
            # Base confidence
            confidence = 0.7
            
            # Boost for reasonable size
            if layout.row_count() >= 2 and layout.col_count() >= 2:
                confidence += 0.1
            
            # Boost for having merged cells (indicates structured table)
            if self._has_merged_cells_in_region(sheet, layout):
                confidence += 0.1
            
            # Boost for data density (using 0-based indexing)
            data_cells = 0
            total_cells = layout.cell_count()
            for row_idx in range(layout.min_row - 1, min(layout.min_row + 9, layout.max_row)):
                for col_idx in range(layout.min_col - 1, min(layout.min_col + 9, layout.max_col)):
                    try:
                        value = sheet.cell_value(row_idx, col_idx)
                        if value and str(value).strip():
                            data_cells += 1
                    except Exception:
                        pass
            
            density = data_cells / min(total_cells, 100)
            confidence += density * 0.1
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.5


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def has_merged_cells_xlsx(ws, layout: Optional[LayoutRange] = None) -> bool:
    """Check if XLSX worksheet has merged cells in the given region.
    
    Backward compatible function.
    
    Args:
        ws: openpyxl Worksheet object
        layout: LayoutRange to check (None for entire sheet)
        
    Returns:
        True if merged cells exist
    """
    extractor = XLSXTableExtractor()
    if layout is None:
        layout = layout_detect_range_xlsx(ws)
    if layout is None:
        return False
    return extractor._has_merged_cells_in_region(ws, layout)


def has_merged_cells_xls(sheet, layout: Optional[LayoutRange] = None) -> bool:
    """Check if XLS sheet has merged cells in the given region.
    
    Backward compatible function.
    
    Args:
        sheet: xlrd Sheet object
        layout: LayoutRange to check (None for entire sheet)
        
    Returns:
        True if merged cells exist
    """
    extractor = XLSTableExtractor()
    if layout is None:
        layout = layout_detect_range_xls(sheet)
    if layout is None:
        return False
    return extractor._has_merged_cells_in_region(sheet, layout)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Configuration
    "ExcelTableExtractorConfig",
    # Region Info
    "ExcelTableRegionInfo",
    # Extractors
    "ExcelTableExtractor",
    "XLSXTableExtractor",
    "XLSTableExtractor",
    # Backward compatibility
    "has_merged_cells_xlsx",
    "has_merged_cells_xls",
]
