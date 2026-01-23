# libs/core/functions/table_processor.py
"""
Table Processor - Common Table Processing Module

Provides common table processing utilities for formatting tables.
This module handles HTML, Markdown, and Text conversion of TableData.

Module Components:
- TableOutputFormat: Output format options enum
- TableProcessorConfig: Configuration for table processing
- TableProcessor: Main table processing class with formatting methods

Usage Example:
    from contextifier.core.functions.table_processor import (
        TableProcessor,
        TableProcessorConfig,
        TableOutputFormat,
    )

    processor = TableProcessor()
    html = processor.format_table(table_data)
"""
import logging
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from contextifier.core.functions.table_extractor import TableData, TableCell

logger = logging.getLogger("document-processor")


class TableOutputFormat(Enum):
    """Table output format options."""
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"


@dataclass
class TableProcessorConfig:
    """Configuration for table processing.
    
    Attributes:
        output_format: Output format for tables (HTML, Markdown, Text)
        clean_whitespace: Whether to clean excessive whitespace in cells
        preserve_merged_cells: Whether to preserve merged cell attributes in HTML
    """
    output_format: TableOutputFormat = TableOutputFormat.HTML
    clean_whitespace: bool = True
    preserve_merged_cells: bool = True


class TableProcessor:
    """Main table processing class.
    
    Handles formatting of TableData into various output formats.
    Uses format-specific extractors for actual extraction work.
    
    Usage:
        processor = TableProcessor(config)
        html = processor.format_table(table_data)
        markdown = processor.format_table_as_markdown(table_data)
    """
    
    def __init__(self, config: Optional[TableProcessorConfig] = None):
        """Initialize the table processor.
        
        Args:
            config: Table processing configuration
        """
        self.config = config or TableProcessorConfig()
        self.logger = logging.getLogger("document-processor")
    
    def format_table(self, table: TableData) -> str:
        """Format a table according to the configured output format.
        
        Args:
            table: TableData to format
            
        Returns:
            Formatted table string (HTML, Markdown, or Text)
        """
        if self.config.output_format == TableOutputFormat.HTML:
            return self.format_table_as_html(table)
        elif self.config.output_format == TableOutputFormat.MARKDOWN:
            return self.format_table_as_markdown(table)
        else:
            return self.format_table_as_text(table)
    
    def format_table_as_html(self, table: TableData) -> str:
        """Format table as HTML.
        
        Args:
            table: TableData to format
            
        Returns:
            HTML table string
        """
        if not table.rows:
            return ""
        
        html_parts = ["<table>"]
        
        # Add column width specifications if available
        if table.col_widths_percent:
            html_parts.append("  <colgroup>")
            for width_pct in table.col_widths_percent:
                html_parts.append(f'    <col style="width: {width_pct:.1f}%">')
            html_parts.append("  </colgroup>")
        
        for row_idx, row in enumerate(table.rows):
            html_parts.append("  <tr>")
            
            for cell in row:
                # Determine tag (th for header, td otherwise)
                tag = "th" if cell.is_header else "td"
                
                # Build attributes
                attrs = []
                if self.config.preserve_merged_cells:
                    if cell.row_span > 1:
                        attrs.append(f'rowspan="{cell.row_span}"')
                    if cell.col_span > 1:
                        attrs.append(f'colspan="{cell.col_span}"')
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                
                # Handle nested table
                if cell.nested_table:
                    nested_html = self.format_table_as_html(cell.nested_table)
                    html_parts.append(f"    <{tag}{attr_str}>{nested_html}</{tag}>")
                else:
                    # Clean content
                    content = self._clean_cell_content(cell.content)
                    html_parts.append(f"    <{tag}{attr_str}>{content}</{tag}>")
            
            html_parts.append("  </tr>")
        
        html_parts.append("</table>")
        
        return "\n".join(html_parts)
    
    def format_table_as_markdown(self, table: TableData) -> str:
        """Format table as Markdown.
        
        Note: Markdown tables don't support merged cells, so they are flattened.
        
        Args:
            table: TableData to format
            
        Returns:
            Markdown table string
        """
        if not table.rows:
            return ""
        
        lines = []
        
        for row_idx, row in enumerate(table.rows):
            cells = [self._clean_cell_content(cell.content) for cell in row]
            line = "| " + " | ".join(cells) + " |"
            lines.append(line)
            
            # Add separator after header row
            if row_idx == 0 and table.has_header:
                separator = "| " + " | ".join(["---"] * len(row)) + " |"
                lines.append(separator)
        
        return "\n".join(lines)
    
    def format_table_as_text(self, table: TableData) -> str:
        """Format table as plain text with tab separators.
        
        Args:
            table: TableData to format
            
        Returns:
            Tab-separated text
        """
        if not table.rows:
            return ""
        
        lines = []
        for row in table.rows:
            cells = [self._clean_cell_content(cell.content) for cell in row]
            lines.append("\t".join(cells))
        
        return "\n".join(lines)
    
    def _clean_cell_content(self, content: str) -> str:
        """Clean cell content.
        
        Args:
            content: Raw cell content
            
        Returns:
            Cleaned content
        """
        if not content:
            return ""
        
        if self.config.clean_whitespace:
            # Normalize whitespace
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
        
        return content


def create_table_processor(
    config: Optional[TableProcessorConfig] = None
) -> TableProcessor:
    """Factory function to create a TableProcessor.
    
    Args:
        config: Table processing configuration
        
    Returns:
        Configured TableProcessor instance
    """
    return TableProcessor(config)


# Default configuration
DEFAULT_PROCESSOR_CONFIG = TableProcessorConfig()
