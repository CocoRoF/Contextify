# contextifier/core/processor/ppt_helper/ppt_table_processor.py
"""
PPT Table Processor - PPT/PPTX Format-Specific Table Processing

Handles conversion of PPT tables to various output formats (HTML, Text).
Works with PPTTableExtractor to extract and format tables.

Processor Responsibilities:
- TableData → HTML conversion (with merge cell support)
- TableData → Plain Text conversion
- Simple table detection and formatting

Usage:
    from contextifier.core.processor.ppt_helper.ppt_table_processor import (
        PPTTableProcessor,
    )

    processor = PPTTableProcessor()
    html = processor.format_table(table_data)
"""
import logging
from typing import Any, List, Optional

from contextifier.core.functions.table_processor import (
    TableProcessor,
    TableProcessorConfig,
    TableOutputFormat,
)
from contextifier.core.functions.table_extractor import TableData, TableCell

logger = logging.getLogger("document-processor")


class PPTTableProcessor:
    """PPT/PPTX format-specific table processor.
    
    Handles conversion of PPT tables to various output formats.
    Uses TableData from PPTTableExtractor.
    
    Usage:
        processor = PPTTableProcessor()
        html = processor.format_table(table_data)
        # or
        text = processor.format_table_as_text(table_data)
    """
    
    def __init__(self, config: Optional[TableProcessorConfig] = None):
        """Initialize PPT table processor.
        
        Args:
            config: Table processing configuration
        """
        self.config = config or TableProcessorConfig()
        self._table_processor = TableProcessor(self.config)
        self.logger = logging.getLogger("document-processor")
    
    def format_table(self, table_data: TableData) -> str:
        """Format TableData to output string.
        
        Automatically chooses format based on table structure:
        - Simple tables (1xN, Nx1): Plain text
        - Complex tables: HTML
        
        Args:
            table_data: TableData object from PPTTableExtractor
            
        Returns:
            Formatted table string
        """
        if not table_data or not table_data.rows:
            return ""
        
        # Check if it's a simple table
        if self._is_simple_table(table_data):
            return self._format_simple_table_as_text(table_data)
        
        # Complex table: use HTML
        return self._format_table_to_html(table_data)
    
    def format_table_as_html(self, table_data: TableData) -> str:
        """Format TableData to HTML.
        
        Args:
            table_data: TableData object
            
        Returns:
            HTML table string
        """
        if not table_data or not table_data.rows:
            return ""
        
        return self._format_table_to_html(table_data)
    
    def format_table_as_text(self, table_data: TableData) -> str:
        """Format TableData to plain text.
        
        Args:
            table_data: TableData object
            
        Returns:
            Plain text representation
        """
        if not table_data or not table_data.rows:
            return ""
        
        return self._format_table_to_text(table_data)
    
    def format_pptx_table(self, table: Any) -> str:
        """Format python-pptx Table object directly.
        
        Convenience method for direct table formatting without
        going through PPTTableExtractor.
        
        Args:
            table: python-pptx Table object
            
        Returns:
            Formatted table string (HTML or text)
        """
        try:
            num_rows = len(table.rows)
            num_cols = len(table.columns)
            
            # Simple table check
            if num_rows == 1 or num_cols == 1:
                return self._format_pptx_simple_table(table)
            
            # Complex table: HTML
            return self._format_pptx_table_to_html(table)
        
        except Exception as e:
            self.logger.warning(f"Error formatting PPT table: {e}")
            return ""
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _is_simple_table(self, table_data: TableData) -> bool:
        """Check if table is simple (1xN or Nx1)."""
        return table_data.num_rows == 1 or table_data.num_cols == 1
    
    def _format_simple_table_as_text(self, table_data: TableData) -> str:
        """Format simple table (1xN or Nx1) as plain text."""
        texts = []
        
        for row in table_data.rows:
            row_texts = []
            for cell in row:
                if cell.content and cell.col_span > 0 and cell.row_span > 0:
                    row_texts.append(cell.content)
            
            if row_texts:
                texts.append(" ".join(row_texts))
        
        return "\n".join(texts) if texts else ""
    
    def _format_table_to_html(self, table_data: TableData) -> str:
        """Format TableData to HTML with merge cell support."""
        if not table_data.rows:
            return ""
        
        html_parts = ["<table border='1'>"]
        
        for row_idx, row in enumerate(table_data.rows):
            html_parts.append("<tr>")
            
            for cell in row:
                # Skip merged-away cells (colspan=0 or rowspan=0)
                if cell.col_span == 0 or cell.row_span == 0:
                    continue
                
                # Determine tag (th for header, td for data)
                tag = "th" if cell.is_header else "td"
                
                # Build attributes
                attrs = []
                if cell.row_span > 1:
                    attrs.append(f"rowspan='{cell.row_span}'")
                if cell.col_span > 1:
                    attrs.append(f"colspan='{cell.col_span}'")
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                
                # Escape HTML content
                content = self._escape_html(cell.content)
                
                html_parts.append(f"<{tag}{attr_str}>{content}</{tag}>")
            
            html_parts.append("</tr>")
        
        html_parts.append("</table>")
        
        return "\n".join(html_parts)
    
    def _format_table_to_text(self, table_data: TableData) -> str:
        """Format TableData to plain text with pipe separators."""
        rows_text = []
        
        for row in table_data.rows:
            row_cells = []
            for cell in row:
                # Skip merged-away cells
                if cell.col_span == 0 or cell.row_span == 0:
                    continue
                row_cells.append(cell.content if cell.content else "")
            
            if any(c for c in row_cells):
                rows_text.append(" | ".join(row_cells))
        
        return "\n".join(rows_text) if rows_text else ""
    
    def _format_pptx_simple_table(self, table: Any) -> str:
        """Format simple python-pptx table as text."""
        try:
            texts = []
            for row in table.rows:
                row_texts = []
                for cell in row.cells:
                    cell_text = cell.text.strip() if cell.text else ""
                    if cell_text:
                        row_texts.append(cell_text)
                if row_texts:
                    texts.append(" ".join(row_texts))
            
            return "\n".join(texts) if texts else ""
        except Exception:
            return ""
    
    def _format_pptx_table_to_html(self, table: Any) -> str:
        """Format python-pptx table directly to HTML.
        
        Used for direct table formatting without TableData.
        """
        try:
            num_rows = len(table.rows)
            num_cols = len(table.columns)
            
            if num_rows == 0 or num_cols == 0:
                return ""
            
            # Build merge info
            cell_info = [[None for _ in range(num_cols)] for _ in range(num_rows)]
            
            for row_idx in range(num_rows):
                for col_idx in range(num_cols):
                    if cell_info[row_idx][col_idx] == 'skip':
                        continue
                    
                    cell = table.cell(row_idx, col_idx)
                    merge_info = self._get_pptx_cell_merge_info(
                        cell, table, row_idx, col_idx, num_rows, num_cols
                    )
                    
                    rowspan = merge_info['rowspan']
                    colspan = merge_info['colspan']
                    
                    # Mark merged cells
                    for r in range(row_idx, min(row_idx + rowspan, num_rows)):
                        for c in range(col_idx, min(col_idx + colspan, num_cols)):
                            if r == row_idx and c == col_idx:
                                cell_info[r][c] = {
                                    'rowspan': rowspan,
                                    'colspan': colspan,
                                    'text': cell.text.strip() if cell.text else ""
                                }
                            else:
                                cell_info[r][c] = 'skip'
            
            # Generate HTML
            html_parts = ["<table border='1'>"]
            
            for row_idx in range(num_rows):
                html_parts.append("<tr>")
                
                for col_idx in range(num_cols):
                    info = cell_info[row_idx][col_idx]
                    
                    if info == 'skip':
                        continue
                    
                    if info is None:
                        cell = table.cell(row_idx, col_idx)
                        info = {
                            'rowspan': 1,
                            'colspan': 1,
                            'text': cell.text.strip() if cell.text else ""
                        }
                    
                    tag = "th" if row_idx == 0 else "td"
                    
                    attrs = []
                    if info['rowspan'] > 1:
                        attrs.append(f"rowspan='{info['rowspan']}'")
                    if info['colspan'] > 1:
                        attrs.append(f"colspan='{info['colspan']}'")
                    
                    attr_str = " " + " ".join(attrs) if attrs else ""
                    text = self._escape_html(info['text'])
                    
                    html_parts.append(f"<{tag}{attr_str}>{text}</{tag}>")
                
                html_parts.append("</tr>")
            
            html_parts.append("</table>")
            
            return "\n".join(html_parts)
        
        except Exception as e:
            self.logger.warning(f"Error converting table to HTML: {e}")
            return self._format_pptx_simple_table(table)
    
    def _get_pptx_cell_merge_info(
        self,
        cell: Any,
        table: Any,
        row_idx: int,
        col_idx: int,
        num_rows: int,
        num_cols: int,
    ) -> dict:
        """Get merge info for python-pptx cell."""
        rowspan = 1
        colspan = 1
        
        try:
            # Method 1: Built-in attributes
            if hasattr(cell, 'is_merge_origin') and cell.is_merge_origin:
                if hasattr(cell, 'span_height'):
                    rowspan = cell.span_height
                if hasattr(cell, 'span_width'):
                    colspan = cell.span_width
                return {'rowspan': rowspan, 'colspan': colspan}
            
            if hasattr(cell, 'is_spanned') and cell.is_spanned:
                return {'rowspan': 0, 'colspan': 0}
            
            # Method 2: XML parsing
            tc = cell._tc
            
            grid_span = tc.get('gridSpan')
            if grid_span:
                colspan = int(grid_span)
            
            row_span_attr = tc.get('rowSpan')
            if row_span_attr:
                rowspan = int(row_span_attr)
            
            # Method 3: Reference comparison
            if colspan == 1:
                for c in range(col_idx + 1, num_cols):
                    next_cell = table.cell(row_idx, c)
                    if next_cell._tc is cell._tc:
                        colspan += 1
                    else:
                        break
            
            if rowspan == 1:
                for r in range(row_idx + 1, num_rows):
                    next_cell = table.cell(r, col_idx)
                    if next_cell._tc is cell._tc:
                        rowspan += 1
                    else:
                        break
        
        except Exception:
            pass
        
        return {'rowspan': rowspan, 'colspan': colspan}
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace("\n", "<br>")
        
        return text


# ============================================================================
# Convenience Functions (Backward Compatibility)
# ============================================================================

def convert_table_to_html(table: Any) -> str:
    """Convert python-pptx table to HTML.
    
    Backward compatibility wrapper for ppt_table.convert_table_to_html.
    """
    processor = PPTTableProcessor()
    return processor._format_pptx_table_to_html(table)


def extract_table_as_text(table: Any) -> str:
    """Extract python-pptx table as plain text.
    
    Backward compatibility wrapper for ppt_table.extract_table_as_text.
    """
    processor = PPTTableProcessor()
    return processor._format_pptx_simple_table(table)


def extract_simple_table_as_text(table: Any) -> str:
    """Extract simple table (1xN, Nx1) as text.
    
    Backward compatibility wrapper.
    """
    return extract_table_as_text(table)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main class
    'PPTTableProcessor',
    # Convenience functions
    'convert_table_to_html',
    'extract_table_as_text',
    'extract_simple_table_as_text',
]
