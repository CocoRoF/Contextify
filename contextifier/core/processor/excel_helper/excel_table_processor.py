# contextifier/core/processor/excel_helper/excel_table_processor.py
"""
Excel Table Processor - Excel Format-Specific Table Processing

Handles conversion of Excel tables to various output formats (HTML, Markdown, Text).
Works with ExcelTableExtractor to extract and format tables.

Processor Responsibilities:
- TableData → HTML conversion (with merge cell support)
- TableData → Markdown conversion (no merge support)
- TableData → Plain Text conversion
- Direct worksheet/sheet → table string conversion (for backward compatibility)

Output Format Selection:
- Merged cells present → HTML format (rowspan/colspan support)
- No merged cells → Markdown format (simpler, more readable)

Usage:
    from contextifier.core.processor.excel_helper.excel_table_processor import (
        ExcelTableProcessor,
    )

    processor = ExcelTableProcessor()
    
    # From TableData
    html = processor.format_table(table_data)
    
    # Direct from worksheet (backward compatible)
    html = processor.format_xlsx_sheet(worksheet)
    html = processor.format_xls_sheet(sheet, workbook)
"""
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict, Set

from contextifier.core.functions.table_processor import (
    TableProcessor,
    TableProcessorConfig,
    TableOutputFormat,
)
from contextifier.core.functions.table_extractor import TableData, TableCell
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
class ExcelTableProcessorConfig(TableProcessorConfig):
    """Configuration specific to Excel table processing.
    
    Attributes:
        auto_select_format: Whether to auto-select format based on merge cells
        prefer_markdown: Whether to prefer Markdown when no merge cells
        escape_pipe_in_markdown: Whether to escape | in Markdown output
        escape_newline: Whether to convert newlines to <br> or space
    """
    auto_select_format: bool = True
    prefer_markdown: bool = True
    escape_pipe_in_markdown: bool = True
    escape_newline: bool = True


# ============================================================================
# Excel Table Processor Class
# ============================================================================

class ExcelTableProcessor:
    """Excel format-specific table processor.
    
    Handles conversion of Excel tables to various output formats.
    Uses TableData from ExcelTableExtractor or processes worksheets directly.
    
    Features:
    - Auto-selects HTML vs Markdown based on merge cells
    - Supports XLSX (openpyxl) and XLS (xlrd) formats
    - Handles merged cells with rowspan/colspan in HTML
    - Escapes special characters appropriately
    
    Usage:
        processor = ExcelTableProcessor()
        html = processor.format_table(table_data)
        # or
        markdown = processor.format_table_as_markdown(table_data)
    """
    
    def __init__(self, config: Optional[ExcelTableProcessorConfig] = None):
        """Initialize Excel table processor.
        
        Args:
            config: Excel table processing configuration
        """
        self.config = config or ExcelTableProcessorConfig()
        self._table_processor = TableProcessor(self.config)
        self.logger = logging.getLogger("document-processor")
    
    # ========================================================================
    # Main Public Methods - TableData Processing
    # ========================================================================
    
    def format_table(self, table_data: TableData) -> str:
        """Format TableData to output string.
        
        Automatically chooses format based on table structure:
        - Merged cells present: HTML
        - No merged cells: Markdown (if prefer_markdown) or HTML
        
        Args:
            table_data: TableData object from ExcelTableExtractor
            
        Returns:
            Formatted table string
        """
        if not table_data or not table_data.rows:
            return ""
        
        # Check if auto-select is enabled
        if self.config.auto_select_format:
            has_merges = self._has_merged_cells(table_data)
            if has_merges:
                return self.format_table_as_html(table_data)
            elif self.config.prefer_markdown:
                return self.format_table_as_markdown(table_data)
        
        # Default to configured format
        return self._table_processor.format_table(table_data)
    
    def format_table_as_html(self, table_data: TableData) -> str:
        """Format TableData to HTML.
        
        Supports merged cells with rowspan/colspan attributes.
        
        Args:
            table_data: TableData object
            
        Returns:
            HTML table string
        """
        if not table_data or not table_data.rows:
            return ""
        
        return self._format_table_to_html(table_data)
    
    def format_table_as_markdown(self, table_data: TableData) -> str:
        """Format TableData to Markdown.
        
        Note: Markdown doesn't support merged cells, they are flattened.
        
        Args:
            table_data: TableData object
            
        Returns:
            Markdown table string
        """
        if not table_data or not table_data.rows:
            return ""
        
        return self._format_table_to_markdown(table_data)
    
    def format_table_as_text(self, table_data: TableData) -> str:
        """Format TableData to plain text.
        
        Args:
            table_data: TableData object
            
        Returns:
            Tab-separated text representation
        """
        if not table_data or not table_data.rows:
            return ""
        
        return self._format_table_to_text(table_data)
    
    # ========================================================================
    # Direct Worksheet/Sheet Processing (Backward Compatible)
    # ========================================================================
    
    def format_xlsx_sheet(
        self, 
        ws, 
        layout: Optional[LayoutRange] = None
    ) -> str:
        """Format XLSX worksheet directly to table string.
        
        Convenience method for direct formatting without going through
        ExcelTableExtractor. Auto-selects HTML or Markdown based on merges.
        
        Args:
            ws: openpyxl Worksheet object
            layout: LayoutRange to format (None for auto-detect)
            
        Returns:
            Formatted table string
        """
        if layout is None:
            layout = layout_detect_range_xlsx(ws)
        
        if layout is None:
            self.logger.debug("No data found in worksheet")
            return ""
        
        if self._has_merged_cells_xlsx(ws, layout):
            self.logger.debug("Merged cells detected in XLSX, using HTML format")
            return self._format_xlsx_to_html(ws, layout)
        else:
            self.logger.debug("No merged cells in XLSX, using Markdown format")
            return self._format_xlsx_to_markdown(ws, layout)
    
    def format_xls_sheet(
        self, 
        sheet, 
        wb, 
        layout: Optional[LayoutRange] = None
    ) -> str:
        """Format XLS sheet directly to table string.
        
        Convenience method for direct formatting without going through
        ExcelTableExtractor. Auto-selects HTML or Markdown based on merges.
        
        Args:
            sheet: xlrd Sheet object
            wb: xlrd Workbook object
            layout: LayoutRange to format (None for auto-detect)
            
        Returns:
            Formatted table string
        """
        if layout is None:
            layout = layout_detect_range_xls(sheet)
        
        if layout is None:
            self.logger.debug("No data found in XLS sheet")
            return ""
        
        if self._has_merged_cells_xls(sheet, layout):
            self.logger.debug("Merged cells detected in XLS, using HTML format")
            return self._format_xls_to_html(sheet, wb, layout)
        else:
            self.logger.debug("No merged cells in XLS, using Markdown format")
            return self._format_xls_to_markdown(sheet, wb, layout)
    
    def format_xlsx_objects(
        self, 
        ws, 
        layout: Optional[LayoutRange] = None
    ) -> List[str]:
        """Format multiple table objects from XLSX worksheet.
        
        Detects individual table objects and formats each separately.
        
        Args:
            ws: openpyxl Worksheet object
            layout: LayoutRange to search in (None for entire sheet)
            
        Returns:
            List of formatted table strings
        """
        objects = object_detect_xlsx(ws, layout)
        
        if not objects:
            return []
        
        tables = []
        for obj_layout in objects:
            table_str = self.format_xlsx_sheet(ws, obj_layout)
            if table_str and table_str.strip() and self._has_table_data(table_str):
                tables.append(table_str)
        
        self.logger.debug(f"Converted {len(tables)} objects to tables (XLSX)")
        return tables
    
    def format_xls_objects(
        self, 
        sheet, 
        wb, 
        layout: Optional[LayoutRange] = None
    ) -> List[str]:
        """Format multiple table objects from XLS sheet.
        
        Detects individual table objects and formats each separately.
        
        Args:
            sheet: xlrd Sheet object
            wb: xlrd Workbook object
            layout: LayoutRange to search in (None for entire sheet)
            
        Returns:
            List of formatted table strings
        """
        objects = object_detect_xls(sheet, wb, layout)
        
        if not objects:
            return []
        
        tables = []
        for obj_layout in objects:
            table_str = self.format_xls_sheet(sheet, wb, obj_layout)
            if table_str and table_str.strip() and self._has_table_data(table_str):
                tables.append(table_str)
        
        self.logger.debug(f"Converted {len(tables)} objects to tables (XLS)")
        return tables
    
    # ========================================================================
    # Private Helper Methods - TableData Formatting
    # ========================================================================
    
    def _has_merged_cells(self, table_data: TableData) -> bool:
        """Check if TableData has merged cells."""
        for row in table_data.rows:
            for cell in row:
                if cell.row_span > 1 or cell.col_span > 1:
                    return True
        return False
    
    def _format_table_to_html(self, table_data: TableData) -> str:
        """Format TableData to HTML with merge support."""
        html_parts = ["<table border='1'>"]
        
        for row_idx, row in enumerate(table_data.rows):
            row_parts = ["<tr>"]
            
            for cell in row:
                # Skip cells that are part of a merge (span = 0)
                if cell.col_span == 0 or cell.row_span == 0:
                    continue
                
                # Determine tag
                tag = "th" if cell.is_header else "td"
                
                # Build attributes
                attrs = []
                if cell.row_span > 1:
                    attrs.append(f"rowspan='{cell.row_span}'")
                if cell.col_span > 1:
                    attrs.append(f"colspan='{cell.col_span}'")
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                
                # Escape and format content
                content = self._escape_html(cell.content)
                
                row_parts.append(f"<{tag}{attr_str}>{content}</{tag}>")
            
            row_parts.append("</tr>")
            html_parts.append("".join(row_parts))
        
        html_parts.append("</table>")
        
        return "\n".join(html_parts)
    
    def _format_table_to_markdown(self, table_data: TableData) -> str:
        """Format TableData to Markdown (no merge support)."""
        md_parts = []
        row_count = 0
        
        for row in table_data.rows:
            cells = []
            for cell in row:
                # Skip cells that are part of a merge
                if cell.col_span == 0 or cell.row_span == 0:
                    continue
                
                content = cell.content
                if self.config.escape_pipe_in_markdown:
                    content = content.replace("|", "\\|")
                if self.config.escape_newline:
                    content = content.replace("\n", " ")
                cells.append(content)
            
            if not cells:
                continue
            
            row_str = "| " + " | ".join(cells) + " |"
            md_parts.append(row_str)
            row_count += 1
            
            # Add separator after first row (header)
            if row_count == 1:
                separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                md_parts.append(separator)
        
        return "\n".join(md_parts)
    
    def _format_table_to_text(self, table_data: TableData) -> str:
        """Format TableData to plain text."""
        lines = []
        
        for row in table_data.rows:
            cells = []
            for cell in row:
                if cell.col_span == 0 or cell.row_span == 0:
                    continue
                content = cell.content.replace("\n", " ").replace("\t", " ")
                cells.append(content)
            
            if cells:
                lines.append("\t".join(cells))
        
        return "\n".join(lines)
    
    # ========================================================================
    # Private Helper Methods - Direct Worksheet/Sheet Formatting
    # ========================================================================
    
    def _has_merged_cells_xlsx(self, ws, layout: LayoutRange) -> bool:
        """Check if XLSX worksheet has merged cells in the region."""
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
    
    def _has_merged_cells_xls(self, sheet, layout: LayoutRange) -> bool:
        """Check if XLS sheet has merged cells in the region."""
        try:
            if len(sheet.merged_cells) == 0:
                return False
            
            for (rlo, rhi, clo, chi) in sheet.merged_cells:
                mr_min_row = rlo + 1
                mr_max_row = rhi
                mr_min_col = clo + 1
                mr_max_col = chi
                
                if (mr_min_row <= layout.max_row and
                    mr_max_row >= layout.min_row and
                    mr_min_col <= layout.max_col and
                    mr_max_col >= layout.min_col):
                    return True
            
            return False
        except Exception:
            return False
    
    def _format_xlsx_to_html(self, ws, layout: LayoutRange) -> str:
        """Format XLSX worksheet to HTML."""
        try:
            # Collect merge information
            merged_cells_info, skip_cells, merged_value_override = self._collect_xlsx_merge_info(
                ws, layout
            )
            
            html_parts = ["<table border='1'>"]
            has_data = False
            
            for row_idx in range(layout.min_row, layout.max_row + 1):
                row_parts = ["<tr>"]
                
                for col_idx in range(layout.min_col, layout.max_col + 1):
                    if (row_idx, col_idx) in skip_cells:
                        continue
                    
                    cell = ws.cell(row=row_idx, column=col_idx)
                    
                    # Get cell value
                    cell_value = ""
                    if (row_idx, col_idx) in merged_value_override:
                        cell_value = str(merged_value_override[(row_idx, col_idx)]).strip()
                    elif cell.value is not None:
                        cell_value = str(cell.value).strip()
                    
                    if cell_value:
                        has_data = True
                    
                    # Escape HTML
                    cell_value = self._escape_html(cell_value)
                    
                    # Tag
                    tag = "th" if row_idx == layout.min_row else "td"
                    
                    # Merge attributes
                    attrs = []
                    if (row_idx, col_idx) in merged_cells_info:
                        rowspan, colspan = merged_cells_info[(row_idx, col_idx)]
                        if rowspan > 1:
                            attrs.append(f"rowspan='{rowspan}'")
                        if colspan > 1:
                            attrs.append(f"colspan='{colspan}'")
                    
                    attr_str = " " + " ".join(attrs) if attrs else ""
                    row_parts.append(f"<{tag}{attr_str}>{cell_value}</{tag}>")
                
                row_parts.append("</tr>")
                html_parts.append("".join(row_parts))
            
            html_parts.append("</table>")
            
            return "\n".join(html_parts) if has_data else ""
            
        except Exception as e:
            self.logger.warning(f"Error formatting XLSX to HTML: {e}")
            return ""
    
    def _format_xlsx_to_markdown(self, ws, layout: LayoutRange) -> str:
        """Format XLSX worksheet to Markdown."""
        try:
            # Collect merge value override for merged cells starting outside layout
            merged_value_override = {}
            for merged_range in ws.merged_cells.ranges:
                mr_min_row, mr_min_col = merged_range.min_row, merged_range.min_col
                mr_max_row, mr_max_col = merged_range.max_row, merged_range.max_col
                
                if (mr_min_row <= layout.max_row and
                    mr_max_row >= layout.min_row and
                    mr_min_col <= layout.max_col and
                    mr_max_col >= layout.min_col):
                    
                    start_in_layout = (layout.min_row <= mr_min_row <= layout.max_row and
                                       layout.min_col <= mr_min_col <= layout.max_col)
                    
                    if not start_in_layout:
                        merged_value = ws.cell(row=mr_min_row, column=mr_min_col).value
                        if merged_value is not None:
                            first_row = max(mr_min_row, layout.min_row)
                            first_col = max(mr_min_col, layout.min_col)
                            merged_value_override[(first_row, first_col)] = merged_value
            
            md_parts = []
            row_count = 0
            
            for row_idx in range(layout.min_row, layout.max_row + 1):
                cells = []
                row_has_content = False
                
                for col_idx in range(layout.min_col, layout.max_col + 1):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    cell_value = ""
                    
                    if (row_idx, col_idx) in merged_value_override:
                        cell_value = str(merged_value_override[(row_idx, col_idx)]).strip()
                    elif cell.value is not None:
                        cell_value = str(cell.value).strip()
                    
                    if cell_value:
                        row_has_content = True
                    
                    # Escape for Markdown
                    cell_value = cell_value.replace("|", "\\|").replace("\n", " ")
                    cells.append(cell_value)
                
                if not row_has_content:
                    continue
                
                row_str = "| " + " | ".join(cells) + " |"
                md_parts.append(row_str)
                row_count += 1
                
                if row_count == 1:
                    separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                    md_parts.append(separator)
            
            return "\n".join(md_parts)
            
        except Exception as e:
            self.logger.warning(f"Error formatting XLSX to Markdown: {e}")
            return ""
    
    def _format_xls_to_html(self, sheet, wb, layout: LayoutRange) -> str:
        """Format XLS sheet to HTML."""
        try:
            # Collect merge information
            merged_cells_info, skip_cells = self._collect_xls_merge_info(sheet, layout)
            
            html_parts = ["<table border='1'>"]
            has_data = False
            
            for row_idx in range(layout.min_row - 1, layout.max_row):  # 0-based
                row_parts = ["<tr>"]
                
                for col_idx in range(layout.min_col - 1, layout.max_col):  # 0-based
                    if (row_idx, col_idx) in skip_cells:
                        continue
                    
                    cell_value = ""
                    try:
                        value = sheet.cell_value(row_idx, col_idx)
                        if value:
                            cell_type = sheet.cell_type(row_idx, col_idx)
                            cell_value = self._format_xls_cell_value(value, cell_type, wb)
                            if cell_value:
                                has_data = True
                    except Exception:
                        pass
                    
                    # Escape HTML
                    cell_value = self._escape_html(cell_value)
                    
                    # Tag
                    tag = "th" if row_idx == layout.min_row - 1 else "td"
                    
                    # Merge attributes
                    attrs = []
                    if (row_idx, col_idx) in merged_cells_info:
                        rowspan, colspan = merged_cells_info[(row_idx, col_idx)]
                        if rowspan > 1:
                            attrs.append(f"rowspan='{rowspan}'")
                        if colspan > 1:
                            attrs.append(f"colspan='{colspan}'")
                    
                    attr_str = " " + " ".join(attrs) if attrs else ""
                    row_parts.append(f"<{tag}{attr_str}>{cell_value}</{tag}>")
                
                row_parts.append("</tr>")
                html_parts.append("".join(row_parts))
            
            html_parts.append("</table>")
            
            return "\n".join(html_parts) if has_data else ""
            
        except Exception as e:
            self.logger.warning(f"Error formatting XLS to HTML: {e}")
            return ""
    
    def _format_xls_to_markdown(self, sheet, wb, layout: LayoutRange) -> str:
        """Format XLS sheet to Markdown."""
        try:
            md_parts = []
            row_count = 0
            
            for row_idx in range(layout.min_row - 1, layout.max_row):  # 0-based
                cells = []
                row_has_content = False
                
                for col_idx in range(layout.min_col - 1, layout.max_col):  # 0-based
                    cell_value = ""
                    try:
                        value = sheet.cell_value(row_idx, col_idx)
                        if value:
                            cell_type = sheet.cell_type(row_idx, col_idx)
                            cell_value = self._format_xls_cell_value(value, cell_type, wb)
                            if cell_value:
                                row_has_content = True
                    except Exception:
                        pass
                    
                    # Escape for Markdown
                    cell_value = cell_value.replace("|", "\\|").replace("\n", " ")
                    cells.append(cell_value)
                
                if not row_has_content:
                    continue
                
                row_str = "| " + " | ".join(cells) + " |"
                md_parts.append(row_str)
                row_count += 1
                
                if row_count == 1:
                    separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                    md_parts.append(separator)
            
            return "\n".join(md_parts)
            
        except Exception as e:
            self.logger.warning(f"Error formatting XLS to Markdown: {e}")
            return ""
    
    def _collect_xlsx_merge_info(
        self, 
        ws, 
        layout: LayoutRange
    ) -> Tuple[Dict, Set, Dict]:
        """Collect merge information for XLSX worksheet."""
        merged_cells_info = {}
        skip_cells = set()
        merged_value_override = {}
        
        try:
            for merged_range in ws.merged_cells.ranges:
                mr_min_row, mr_min_col = merged_range.min_row, merged_range.min_col
                mr_max_row, mr_max_col = merged_range.max_row, merged_range.max_col
                
                if not (mr_min_row <= layout.max_row and
                        mr_max_row >= layout.min_row and
                        mr_min_col <= layout.max_col and
                        mr_max_col >= layout.min_col):
                    continue
                
                start_in_layout = (layout.min_row <= mr_min_row <= layout.max_row and
                                   layout.min_col <= mr_min_col <= layout.max_col)
                
                if start_in_layout:
                    rowspan = mr_max_row - mr_min_row + 1
                    colspan = mr_max_col - mr_min_col + 1
                    merged_cells_info[(mr_min_row, mr_min_col)] = (rowspan, colspan)
                    
                    for r in range(mr_min_row, mr_max_row + 1):
                        for c in range(mr_min_col, mr_max_col + 1):
                            if r != mr_min_row or c != mr_min_col:
                                skip_cells.add((r, c))
                else:
                    merged_value = ws.cell(row=mr_min_row, column=mr_min_col).value
                    if merged_value is not None:
                        first_row = max(mr_min_row, layout.min_row)
                        first_col = max(mr_min_col, layout.min_col)
                        merged_value_override[(first_row, first_col)] = merged_value
                    
                    for r in range(max(mr_min_row, layout.min_row), min(mr_max_row, layout.max_row) + 1):
                        for c in range(max(mr_min_col, layout.min_col), min(mr_max_col, layout.max_col) + 1):
                            if (r, c) not in merged_value_override:
                                skip_cells.add((r, c))
        
        except Exception as e:
            self.logger.warning(f"Error collecting XLSX merge info: {e}")
        
        return merged_cells_info, skip_cells, merged_value_override
    
    def _collect_xls_merge_info(
        self, 
        sheet, 
        layout: LayoutRange
    ) -> Tuple[Dict, Set]:
        """Collect merge information for XLS sheet."""
        merged_cells_info = {}
        skip_cells = set()
        
        try:
            for (rlo, rhi, clo, chi) in sheet.merged_cells:
                mr_min_row = rlo + 1
                mr_max_row = rhi
                mr_min_col = clo + 1
                mr_max_col = chi
                
                if not (mr_min_row <= layout.max_row and
                        mr_max_row >= layout.min_row and
                        mr_min_col <= layout.max_col and
                        mr_max_col >= layout.min_col):
                    continue
                
                rowspan = rhi - rlo
                colspan = chi - clo
                merged_cells_info[(rlo, clo)] = (rowspan, colspan)
                
                for r in range(rlo, rhi):
                    for c in range(clo, chi):
                        if r != rlo or c != clo:
                            skip_cells.add((r, c))
        
        except Exception as e:
            self.logger.warning(f"Error collecting XLS merge info: {e}")
        
        return merged_cells_info, skip_cells
    
    # ========================================================================
    # Private Helper Methods - Utility
    # ========================================================================
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        
        if self.config.escape_newline:
            text = text.replace("\n", "<br>")
        
        return text
    
    def _format_xls_cell_value(self, value, cell_type, wb) -> str:
        """Format XLS cell value to string."""
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
    
    def _has_table_data(self, table_str: str) -> bool:
        """Check if table string has actual data."""
        if not table_str:
            return False
        
        lines = [line.strip() for line in table_str.strip().split('\n') if line.strip()]
        
        for line in lines:
            if '---' not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if parts:
                    return True
        
        return False


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def convert_xlsx_sheet_to_table(ws, layout: Optional[LayoutRange] = None) -> str:
    """Convert XLSX worksheet to table string.
    
    Backward compatible function.
    
    Args:
        ws: openpyxl Worksheet object
        layout: LayoutRange to convert (None for auto-detect)
        
    Returns:
        Table string (HTML or Markdown)
    """
    processor = ExcelTableProcessor()
    return processor.format_xlsx_sheet(ws, layout)


def convert_xlsx_sheet_to_markdown(ws, layout: Optional[LayoutRange] = None) -> str:
    """Convert XLSX worksheet to Markdown.
    
    Backward compatible function.
    """
    processor = ExcelTableProcessor()
    if layout is None:
        layout = layout_detect_range_xlsx(ws)
    if layout is None:
        return ""
    return processor._format_xlsx_to_markdown(ws, layout)


def convert_xlsx_sheet_to_html(ws, layout: Optional[LayoutRange] = None) -> str:
    """Convert XLSX worksheet to HTML.
    
    Backward compatible function.
    """
    processor = ExcelTableProcessor()
    if layout is None:
        layout = layout_detect_range_xlsx(ws)
    if layout is None:
        return ""
    return processor._format_xlsx_to_html(ws, layout)


def convert_xlsx_objects_to_tables(ws, layout: Optional[LayoutRange] = None) -> List[str]:
    """Convert XLSX worksheet objects to table strings.
    
    Backward compatible function.
    """
    processor = ExcelTableProcessor()
    return processor.format_xlsx_objects(ws, layout)


def convert_xls_sheet_to_table(sheet, wb, layout: Optional[LayoutRange] = None) -> str:
    """Convert XLS sheet to table string.
    
    Backward compatible function.
    """
    processor = ExcelTableProcessor()
    return processor.format_xls_sheet(sheet, wb, layout)


def convert_xls_sheet_to_markdown(sheet, wb, layout: Optional[LayoutRange] = None) -> str:
    """Convert XLS sheet to Markdown.
    
    Backward compatible function.
    """
    processor = ExcelTableProcessor()
    if layout is None:
        layout = layout_detect_range_xls(sheet)
    if layout is None:
        return ""
    return processor._format_xls_to_markdown(sheet, wb, layout)


def convert_xls_sheet_to_html(sheet, wb, layout: Optional[LayoutRange] = None) -> str:
    """Convert XLS sheet to HTML.
    
    Backward compatible function.
    """
    processor = ExcelTableProcessor()
    if layout is None:
        layout = layout_detect_range_xls(sheet)
    if layout is None:
        return ""
    return processor._format_xls_to_html(sheet, wb, layout)


def convert_xls_objects_to_tables(sheet, wb, layout: Optional[LayoutRange] = None) -> List[str]:
    """Convert XLS sheet objects to table strings.
    
    Backward compatible function.
    """
    processor = ExcelTableProcessor()
    return processor.format_xls_objects(sheet, wb, layout)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Configuration
    "ExcelTableProcessorConfig",
    # Processor
    "ExcelTableProcessor",
    # Backward compatibility - XLSX
    "convert_xlsx_sheet_to_table",
    "convert_xlsx_sheet_to_markdown",
    "convert_xlsx_sheet_to_html",
    "convert_xlsx_objects_to_tables",
    # Backward compatibility - XLS
    "convert_xls_sheet_to_table",
    "convert_xls_sheet_to_markdown",
    "convert_xls_sheet_to_html",
    "convert_xls_objects_to_tables",
]
