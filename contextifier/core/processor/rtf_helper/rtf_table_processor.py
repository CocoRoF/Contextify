# contextifier/core/processor/rtf_helper/rtf_table_processor.py
"""
RTF Table Processor - RTF Format-Specific Table Processing

Handles conversion of RTF tables to various output formats (HTML, Text).
Works with RTFTableExtractor to extract and format tables.

Processor Responsibilities:
- RTFTable → HTML conversion
- RTFTable → Plain Text conversion
- TableData → HTML/Text conversion (via TableProcessor)

Usage:
    from contextifier.core.processor.rtf_helper.rtf_table_processor import (
        RTFTableProcessor,
    )

    processor = RTFTableProcessor()
    html = processor.format_rtf_table(rtf_table)
"""
import logging
import re
from typing import List, Optional, Tuple

from contextifier.core.functions.table_processor import (
    TableProcessor,
    TableProcessorConfig,
    TableOutputFormat,
)
from contextifier.core.functions.table_extractor import TableData

logger = logging.getLogger("document-processor")


class RTFTableProcessor:
    """RTF format-specific table processor.
    
    Handles conversion of RTF tables to various output formats.
    Uses RTFTable's merge info for accurate HTML rendering.
    
    Usage:
        processor = RTFTableProcessor()
        html = processor.format_rtf_table(rtf_table)
        # or
        html = processor.format_table_data(table_data)
    """
    
    def __init__(self, config: Optional[TableProcessorConfig] = None):
        """Initialize RTF table processor.
        
        Args:
            config: Table processing configuration
        """
        self.config = config or TableProcessorConfig()
        self._table_processor = TableProcessor(self.config)
        self.logger = logging.getLogger("document-processor")
    
    def format_rtf_table(self, rtf_table: "RTFTable") -> str:
        """Format RTFTable to output string.
        
        Args:
            rtf_table: RTFTable object
            
        Returns:
            Formatted table string (HTML by default)
        """
        # Import here to avoid circular imports
        from contextifier.core.processor.rtf_helper.rtf_table_extractor import RTFTable
        
        if not isinstance(rtf_table, RTFTable):
            self.logger.warning(f"Expected RTFTable, got {type(rtf_table)}")
            return ""
        
        if not rtf_table.rows:
            return ""
        
        # Check if it's a real table (2+ columns)
        if rtf_table.is_real_table():
            return self._rtf_table_to_html(rtf_table)
        else:
            return self._rtf_table_to_text_list(rtf_table)
    
    def format_table_data(self, table_data: TableData) -> str:
        """Format TableData to output string.
        
        Delegates to the standard TableProcessor.
        
        Args:
            table_data: TableData object
            
        Returns:
            Formatted table string
        """
        return self._table_processor.format_table(table_data)
    
    def _rtf_table_to_html(self, rtf_table: "RTFTable") -> str:
        """Convert RTFTable to HTML with merge cell support.
        
        Args:
            rtf_table: RTFTable object
            
        Returns:
            HTML table string
        """
        if not rtf_table.rows:
            return ""
        
        merge_info = self._calculate_merge_info(rtf_table)
        html_parts = ['<table border="1">']
        
        for row_idx, row in enumerate(rtf_table.rows):
            html_parts.append('<tr>')
            
            for col_idx, cell in enumerate(row):
                if col_idx < len(merge_info[row_idx]):
                    colspan, rowspan = merge_info[row_idx][col_idx]
                    
                    # Skip merged-away cells
                    if colspan == 0 or rowspan == 0:
                        continue
                    
                    cell_text = re.sub(r'\s+', ' ', cell.text).strip()
                    
                    attrs = []
                    if colspan > 1:
                        attrs.append(f'colspan="{colspan}"')
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                    
                    attr_str = ' ' + ' '.join(attrs) if attrs else ''
                    html_parts.append(f'<td{attr_str}>{cell_text}</td>')
                else:
                    cell_text = re.sub(r'\s+', ' ', cell.text).strip()
                    html_parts.append(f'<td>{cell_text}</td>')
            
            html_parts.append('</tr>')
        
        html_parts.append('</table>')
        return '\n'.join(html_parts)
    
    def _rtf_table_to_text_list(self, rtf_table: "RTFTable") -> str:
        """Convert 1-column RTFTable to text list.
        
        - 1x1 table: Return cell content only (container table)
        - nx1 table: Return rows separated by blank lines
        
        Args:
            rtf_table: RTFTable object
            
        Returns:
            Text string
        """
        if not rtf_table.rows:
            return ""
        
        # 1x1 table: return content only (container table)
        if len(rtf_table.rows) == 1 and len(rtf_table.rows[0]) == 1:
            return rtf_table.rows[0][0].text
        
        # nx1 table: return rows separated by blank lines
        lines = []
        for row in rtf_table.rows:
            if row:
                cell_text = row[0].text
                if cell_text:
                    lines.append(cell_text)
        
        return '\n\n'.join(lines)
    
    def _calculate_merge_info(
        self, 
        rtf_table: "RTFTable"
    ) -> List[List[Tuple[int, int]]]:
        """Calculate colspan and rowspan for each cell.
        
        Args:
            rtf_table: RTFTable object
            
        Returns:
            2D list of (colspan, rowspan) tuples
        """
        if not rtf_table.rows:
            return []
        
        num_rows = len(rtf_table.rows)
        max_cols = max(len(row) for row in rtf_table.rows) if rtf_table.rows else 0
        
        if max_cols == 0:
            return []
        
        # Initialize with (1, 1) for all cells
        merge_info: List[List[Tuple[int, int]]] = [
            [(1, 1) for _ in range(max_cols)] for _ in range(num_rows)
        ]
        
        # Process horizontal merges
        for row_idx, row in enumerate(rtf_table.rows):
            col_idx = 0
            while col_idx < len(row):
                cell = row[col_idx]
                
                if cell.h_merge_first:
                    colspan = 1
                    for next_col in range(col_idx + 1, len(row)):
                        if row[next_col].h_merge_cont:
                            colspan += 1
                            merge_info[row_idx][next_col] = (0, 0)
                        else:
                            break
                    merge_info[row_idx][col_idx] = (colspan, 1)
                
                col_idx += 1
        
        # Process vertical merges
        for col_idx in range(max_cols):
            row_idx = 0
            while row_idx < num_rows:
                if col_idx >= len(rtf_table.rows[row_idx]):
                    row_idx += 1
                    continue
                
                cell = rtf_table.rows[row_idx][col_idx]
                
                if cell.v_merge_first:
                    rowspan = 1
                    for next_row in range(row_idx + 1, num_rows):
                        if col_idx < len(rtf_table.rows[next_row]) and rtf_table.rows[next_row][col_idx].v_merge_cont:
                            rowspan += 1
                            merge_info[next_row][col_idx] = (0, 0)
                        else:
                            break
                    
                    current_colspan = merge_info[row_idx][col_idx][0]
                    merge_info[row_idx][col_idx] = (current_colspan, rowspan)
                    row_idx += rowspan
                elif cell.v_merge_cont:
                    merge_info[row_idx][col_idx] = (0, 0)
                    row_idx += 1
                else:
                    row_idx += 1
        
        return merge_info


# =============================================================================
# Convenience Functions
# =============================================================================

def rtf_table_to_html(rtf_table: "RTFTable") -> str:
    """Convert RTFTable to HTML.
    
    Convenience function for backward compatibility.
    
    Args:
        rtf_table: RTFTable object
        
    Returns:
        HTML table string
    """
    processor = RTFTableProcessor()
    return processor._rtf_table_to_html(rtf_table)


def rtf_table_to_text(rtf_table: "RTFTable") -> str:
    """Convert RTFTable to text.
    
    Convenience function for backward compatibility.
    
    Args:
        rtf_table: RTFTable object
        
    Returns:
        Text string
    """
    processor = RTFTableProcessor()
    return processor._rtf_table_to_text_list(rtf_table)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main class
    'RTFTableProcessor',
    # Convenience functions
    'rtf_table_to_html',
    'rtf_table_to_text',
]
