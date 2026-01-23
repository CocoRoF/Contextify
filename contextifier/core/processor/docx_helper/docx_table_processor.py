# contextifier/core/processor/docx_helper/docx_table_processor.py
"""
DOCX Table Processor - DOCX Format-Specific Table Processing

Handles conversion of DOCX tables to various output formats (HTML, Text).
Works with DOCXTableExtractor to extract and format tables.

Processor Responsibilities:
- TableData → HTML conversion
- TableData → Plain Text conversion
- Special case handling (1x1 tables, single column tables)

Usage:
    from contextifier.core.processor.docx_helper.docx_table_processor import (
        DOCXTableProcessor,
        process_table_element,
    )

    processor = DOCXTableProcessor()
    html = processor.process_table(table_elem, doc)
"""
import logging
import traceback
from typing import Optional

from docx import Document
from docx.oxml.ns import qn

from contextifier.core.functions.table_processor import (
    TableProcessor,
    TableProcessorConfig,
    TableOutputFormat,
)
from contextifier.core.processor.docx_helper.docx_constants import NAMESPACES

logger = logging.getLogger("document-processor")


class DOCXTableProcessor:
    """DOCX format-specific table processor.
    
    Handles conversion of DOCX table elements to various output formats.
    Uses DOCXTableExtractor for extraction and TableProcessor for formatting.
    
    Usage:
        processor = DOCXTableProcessor()
        html = processor.process_table(table_elem, doc)
    """
    
    def __init__(self, config: Optional[TableProcessorConfig] = None):
        """Initialize DOCX table processor.
        
        Args:
            config: Table processing configuration
        """
        self.config = config or TableProcessorConfig()
        self._table_processor = TableProcessor(self.config)
        self.logger = logging.getLogger("document-processor")
    
    def process_table(self, table_elem, doc: Document) -> str:
        """Process a DOCX table element to formatted output.
        
        Args:
            table_elem: table XML element (w:tbl)
            doc: python-docx Document object
            
        Returns:
            Formatted table string (HTML by default)
        """
        try:
            # Import here to avoid circular imports
            from contextifier.core.processor.docx_helper.docx_table_extractor import (
                DOCXTableExtractor,
                DOCXTableExtractorConfig,
            )
            
            # Create extractor with config that allows all tables
            extractor_config = DOCXTableExtractorConfig(
                skip_single_cell_tables=True,
                skip_single_column_tables=True,
                min_rows=1,
                min_cols=1
            )
            extractor = DOCXTableExtractor(extractor_config)
            
            # Extract table data directly from element
            table_data = extractor._extract_table_data(table_elem, doc, 0)
            
            if table_data is None:
                # Fallback for single-cell or single-column tables
                return self._process_special_table(table_elem)
            
            # Use TableProcessor to format
            return self._table_processor.format_table(table_data)
            
        except Exception as e:
            self.logger.warning(f"Error processing table element: {e}")
            self.logger.debug(traceback.format_exc())
            return extract_table_as_text(table_elem)
    
    def _process_special_table(self, table_elem) -> str:
        """Process special cases (1x1, single column) tables.
        
        Args:
            table_elem: table XML element
            
        Returns:
            Text content or empty string
        """
        rows = table_elem.findall('w:tr', NAMESPACES)
        if not rows:
            return ""
        
        num_rows = len(rows)
        
        # Get column count
        tblGrid = table_elem.find('w:tblGrid', NAMESPACES)
        if tblGrid is not None:
            num_cols = len(tblGrid.findall('w:gridCol', NAMESPACES))
        else:
            num_cols = 0
            for cell in rows[0].findall('w:tc', NAMESPACES):
                tcPr = cell.find('w:tcPr', NAMESPACES)
                colspan = 1
                if tcPr is not None:
                    gs = tcPr.find('w:gridSpan', NAMESPACES)
                    if gs is not None:
                        try:
                            colspan = int(gs.get(qn('w:val'), 1))
                        except (ValueError, TypeError):
                            colspan = 1
                num_cols += colspan
        
        # 1x1 table: return content only
        if num_rows == 1 and num_cols == 1:
            cells = rows[0].findall('w:tc', NAMESPACES)
            if cells:
                return extract_cell_text(cells[0])
            return ""
        
        # Single column table: return cells joined by newlines
        if num_cols == 1:
            text_items = []
            for row in rows:
                cells = row.findall('w:tc', NAMESPACES)
                for cell in cells:
                    content = extract_cell_text(cell)
                    if content:
                        text_items.append(content)
            return "\n\n".join(text_items) if text_items else ""
        
        return ""


# ============================================================================
# Utility Functions
# ============================================================================

def extract_cell_text(cell_elem) -> str:
    """Extract text content from a cell element.

    Args:
        cell_elem: Cell XML element

    Returns:
        Cell text content
    """
    texts = []

    for p in cell_elem.findall('.//w:p', NAMESPACES):
        p_texts = []
        for t in p.findall('.//w:t', NAMESPACES):
            if t.text:
                p_texts.append(t.text)
        if p_texts:
            texts.append(''.join(p_texts))

    return '\n'.join(texts)


def extract_table_as_text(table_elem) -> str:
    """Extract table as plain text (fallback).

    Args:
        table_elem: table XML element

    Returns:
        Plain text table string
    """
    try:
        rows_text = []

        for row in table_elem.findall('w:tr', NAMESPACES):
            row_cells = []
            for cell in row.findall('w:tc', NAMESPACES):
                cell_text = extract_cell_text(cell)
                row_cells.append(cell_text.replace('\n', ' '))
            if any(c.strip() for c in row_cells):
                rows_text.append(" | ".join(row_cells))

        return "\n".join(rows_text) if rows_text else ""

    except Exception:
        return ""


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def process_table_element(table_elem, doc: Document) -> str:
    """
    Table 요소를 HTML로 변환합니다.
    
    이 함수는 기존 코드와의 호환성을 위해 유지됩니다.
    내부적으로 DOCXTableProcessor를 사용합니다.
    
    Args:
        table_elem: table XML 요소
        doc: python-docx Document 객체

    Returns:
        HTML 테이블 문자열, 또는 단순 텍스트
    """
    processor = DOCXTableProcessor()
    return processor.process_table(table_elem, doc)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main class
    'DOCXTableProcessor',
    # Utility functions
    'extract_cell_text',
    'extract_table_as_text',
    # Backward compatibility
    'process_table_element',
]
