# contextifier/core/processor/docx_helper/docx_table_extractor.py
"""
DOCX Table Extractor

Extracts tables from DOCX documents using the BaseTableExtractor interface.
Converts DOCX table elements to TableData objects for further processing.

Key Features:
- 2-Pass extraction (detect_table_regions → extract_table_from_region)
- Full support for rowspan/colspan (vMerge/gridSpan)
- Column width calculation
- Header row detection
- Nested table support

OOXML Table Structure:
- w:tblGrid: Table grid column definitions
- w:tr: Table row
- w:tc: Table cell
- w:tcPr/w:gridSpan: colspan (horizontal merge)
- w:tcPr/w:vMerge val="restart": rowspan start
- w:tcPr/w:vMerge (no val): rowspan continue (merged cell)
"""
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from docx.document import Document as DocumentClass
from docx.oxml.ns import qn

from contextifier.core.functions.table_extractor import (
    BaseTableExtractor,
    TableCell,
    TableData,
    TableRegion,
    TableExtractorConfig,
)
from contextifier.core.processor.docx_helper.docx_constants import NAMESPACES

logger = logging.getLogger("document-processor")


class DOCXTableExtractor(BaseTableExtractor):
    """
    DOCX-specific table extractor implementation.
    
    Extracts tables from DOCX documents and converts them to TableData objects.
    Supports complex table structures including merged cells (rowspan/colspan).
    
    Usage:
        extractor = DOCXTableExtractor()
        tables = extractor.extract_tables(doc)
        
        # Or with 2-pass approach:
        regions = extractor.detect_table_regions(doc)
        for region in regions:
            table_data = extractor.extract_table_from_region(doc, region)
    """
    
    def __init__(self, config: Optional[TableExtractorConfig] = None):
        """Initialize the DOCX table extractor.
        
        Args:
            config: Table extraction configuration
        """
        super().__init__(config)
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format.
        
        Args:
            format_type: Format identifier
            
        Returns:
            True if format is 'docx'
        """
        return format_type.lower() == 'docx'
    
    def detect_table_regions(self, content: Any) -> List[TableRegion]:
        """Detect table regions in a DOCX document.
        
        Pass 1: Scans the document body for table elements and records their positions.
        
        Args:
            content: Document object (python-docx Document) or document body element
            
        Returns:
            List of TableRegion objects with table position information
        """
        regions = []
        
        try:
            # Handle both Document object and body element
            if isinstance(content, DocumentClass):
                body = content.element.body
            else:
                body = content
            
            # Enumerate body elements to find tables
            for idx, elem in enumerate(body):
                local_tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                
                if local_tag == 'tbl':
                    # Found a table element
                    rows = elem.findall('w:tr', NAMESPACES)
                    num_rows = len(rows)
                    
                    # Calculate column count from tblGrid or first row
                    num_cols = self._estimate_column_count(elem, rows)
                    
                    region = TableRegion(
                        start_offset=idx,  # Use element index as position
                        end_offset=idx + 1,
                        row_count=num_rows,
                        col_count=num_cols,
                        confidence=1.0,  # DOCX tables are explicit
                        metadata={
                            'element_index': idx,
                            'table_element': elem  # Store reference to element
                        }
                    )
                    regions.append(region)
            
            self.logger.debug(f"Detected {len(regions)} table regions in DOCX")
            
        except Exception as e:
            self.logger.error(f"Error detecting table regions: {e}")
            self.logger.debug(traceback.format_exc())
        
        return regions
    
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from a detected region.
        
        Pass 2: Extracts actual table content from the detected region.
        
        Args:
            content: Document object or document body element
            region: TableRegion containing table position info
            
        Returns:
            TableData object or None if extraction fails
        """
        try:
            # Get table element from region metadata
            table_elem = region.metadata.get('table_element')
            
            if table_elem is None:
                # Fallback: find element by index
                if isinstance(content, DocumentClass):
                    body = content.element.body
                else:
                    body = content
                
                elem_idx = region.metadata.get('element_index', region.start_offset)
                elements = list(body)
                
                if elem_idx < len(elements):
                    table_elem = elements[elem_idx]
                else:
                    return None
            
            # Handle both Document object and body element for doc reference
            doc = content if isinstance(content, DocumentClass) else None
            
            return self._extract_table_data(table_elem, doc)
            
        except Exception as e:
            self.logger.error(f"Error extracting table from region: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def extract_table_from_element(
        self, 
        table_elem: Any, 
        doc: Optional[Document] = None
    ) -> Optional[TableData]:
        """Extract table data directly from a table element.
        
        Convenience method for direct element processing without 2-pass approach.
        
        Args:
            table_elem: Table XML element
            doc: Optional Document object for context
            
        Returns:
            TableData object or None if extraction fails
        """
        try:
            return self._extract_table_data(table_elem, doc)
        except Exception as e:
            self.logger.error(f"Error extracting table from element: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def _extract_table_data(
        self, 
        table_elem: Any, 
        doc: Optional[Document] = None
    ) -> Optional[TableData]:
        """Internal method to extract TableData from a table element.
        
        Args:
            table_elem: Table XML element
            doc: Optional Document object
            
        Returns:
            TableData object
        """
        rows_elem = table_elem.findall('w:tr', NAMESPACES)
        if not rows_elem:
            return None
        
        num_rows = len(rows_elem)
        
        # Calculate column count and widths
        num_cols = self._estimate_column_count(table_elem, rows_elem)
        col_widths = self._calculate_column_widths(table_elem, num_cols)
        
        # Calculate all rowspans and cell positions
        rowspan_map, cell_grid_col = self._calculate_all_rowspans(
            table_elem, rows_elem, num_rows
        )
        
        # Build TableCell grid
        table_rows: List[List[TableCell]] = []
        
        for row_idx, row in enumerate(rows_elem):
            cells_elem = row.findall('w:tc', NAMESPACES)
            row_cells: List[TableCell] = []
            
            for cell_idx, cell in enumerate(cells_elem):
                # Get cell properties
                tcPr = cell.find('w:tcPr', NAMESPACES)
                colspan = 1
                is_vmerge_continue = False
                
                if tcPr is not None:
                    # Get colspan (gridSpan)
                    gs = tcPr.find('w:gridSpan', NAMESPACES)
                    if gs is not None:
                        try:
                            colspan = int(gs.get(qn('w:val'), 1))
                        except (ValueError, TypeError):
                            colspan = 1
                    
                    # Check vMerge status
                    vMerge = tcPr.find('w:vMerge', NAMESPACES)
                    if vMerge is not None:
                        val = vMerge.get(qn('w:val'))
                        if val != 'restart':
                            is_vmerge_continue = True
                
                # Skip cells that are merged (continue cells)
                if is_vmerge_continue:
                    continue
                
                # Get grid column position
                if cell_idx < len(cell_grid_col[row_idx]):
                    start_col, end_col = cell_grid_col[row_idx][cell_idx]
                else:
                    start_col = cell_idx
                
                # Get rowspan from pre-calculated map
                rowspan = rowspan_map.get((row_idx, start_col), 1)
                
                # Extract cell content
                content = self._extract_cell_text(cell)
                
                # Create TableCell
                table_cell = TableCell(
                    content=content,
                    row_span=rowspan,
                    col_span=colspan,
                    is_header=(row_idx == 0 and self.config.include_header_row),
                    row_index=row_idx,
                    col_index=start_col,
                    nested_table=None  # TODO: Handle nested tables if needed
                )
                row_cells.append(table_cell)
            
            if row_cells:
                table_rows.append(row_cells)
        
        # Handle special cases (1x1 or single column tables)
        actual_rows = len(table_rows)
        actual_cols = num_cols
        
        # Create TableData
        table_data = TableData(
            rows=table_rows,
            num_rows=actual_rows,
            num_cols=actual_cols,
            has_header=self.config.include_header_row and actual_rows > 0,
            start_offset=0,
            end_offset=0,
            source_format='docx',
            metadata={},
            col_widths_percent=col_widths
        )
        
        return table_data
    
    def _estimate_column_count(
        self, 
        table_elem: Any, 
        rows: List[Any]
    ) -> int:
        """Estimate the number of columns in the table.
        
        Args:
            table_elem: Table XML element
            rows: List of row elements
            
        Returns:
            Number of columns
        """
        # Try to get from tblGrid first
        tblGrid = table_elem.find('w:tblGrid', NAMESPACES)
        if tblGrid is not None:
            grid_cols = tblGrid.findall('w:gridCol', NAMESPACES)
            if grid_cols:
                return len(grid_cols)
        
        # Fallback: calculate from first row
        if not rows:
            return 0
        
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
        
        return num_cols
    
    def _calculate_column_widths(
        self, 
        table_elem: Any, 
        num_cols: int
    ) -> List[float]:
        """Calculate column widths as percentages.
        
        Args:
            table_elem: Table XML element
            num_cols: Number of columns
            
        Returns:
            List of column widths as percentages
        """
        widths = []
        
        tblGrid = table_elem.find('w:tblGrid', NAMESPACES)
        if tblGrid is not None:
            grid_cols = tblGrid.findall('w:gridCol', NAMESPACES)
            
            # Extract widths in twips
            raw_widths = []
            for col in grid_cols:
                w = col.get(qn('w:w'))
                if w:
                    try:
                        raw_widths.append(int(w))
                    except ValueError:
                        raw_widths.append(0)
                else:
                    raw_widths.append(0)
            
            # Convert to percentages
            total_width = sum(raw_widths)
            if total_width > 0:
                widths = [(w / total_width) * 100 for w in raw_widths]
        
        # Fallback: equal widths
        if not widths and num_cols > 0:
            widths = [100.0 / num_cols] * num_cols
        
        return widths
    
    def _calculate_all_rowspans(
        self, 
        table_elem: Any, 
        rows: List[Any], 
        num_rows: int
    ) -> Tuple[Dict[Tuple[int, int], int], List[List[Tuple[int, int]]]]:
        """Calculate rowspans for all cells with vMerge restart.
        
        Uses improved algorithm (v3) for accurate merge tracking:
        1. Collect all cell information
        2. Use merge_info matrix to track cell ownership
        3. Connect continue cells to restart cells above
        4. Calculate rowspan by counting owned cells below
        
        Args:
            table_elem: Table XML element
            rows: List of row elements
            num_rows: Number of rows
            
        Returns:
            Tuple of (rowspan_map, cell_grid_col)
            - rowspan_map: Dict[(row_idx, grid_col), rowspan]
            - cell_grid_col: List[List[(start_col, end_col)]]
        """
        rowspan_map: Dict[Tuple[int, int], int] = {}
        
        # Collect all cell info
        all_cells_info: List[List[Tuple[int, str]]] = []
        
        for row in rows:
            cells = row.findall('w:tc', NAMESPACES)
            row_info = []
            for cell in cells:
                tcPr = cell.find('w:tcPr', NAMESPACES)
                colspan = 1
                vmerge_status = 'none'
                
                if tcPr is not None:
                    gs = tcPr.find('w:gridSpan', NAMESPACES)
                    if gs is not None:
                        try:
                            colspan = int(gs.get(qn('w:val'), 1))
                        except (ValueError, TypeError):
                            colspan = 1
                    
                    vMerge = tcPr.find('w:vMerge', NAMESPACES)
                    if vMerge is not None:
                        val = vMerge.get(qn('w:val'))
                        vmerge_status = 'restart' if val == 'restart' else 'continue'
                
                row_info.append((colspan, vmerge_status))
            all_cells_info.append(row_info)
        
        # Step 1: Calculate grid column positions for all cells
        max_cols = 30
        cell_grid_col: List[List[Tuple[int, int]]] = []
        
        # merge_info[row][col] = (owner_row, owner_col, colspan)
        merge_info: List[List[Optional[Tuple[int, int, int]]]] = [
            [None] * max_cols for _ in range(num_rows)
        ]
        
        for row_idx, row_info in enumerate(all_cells_info):
            grid_col = 0
            row_grid_cols: List[Tuple[int, int]] = []
            
            for cell_idx, (colspan, vmerge_status) in enumerate(row_info):
                # Skip already occupied columns (from vMerge above)
                while grid_col < max_cols and merge_info[row_idx][grid_col] is not None:
                    grid_col += 1
                
                # Expand if needed
                while grid_col + colspan > max_cols:
                    for r in range(num_rows):
                        merge_info[r].extend([None] * 10)
                    max_cols += 10
                
                start_col = grid_col
                end_col = grid_col + colspan - 1
                row_grid_cols.append((start_col, end_col))
                
                if vmerge_status == 'restart':
                    # Restart cell: mark current row only
                    for c in range(start_col, start_col + colspan):
                        merge_info[row_idx][c] = (row_idx, start_col, colspan)
                
                elif vmerge_status == 'continue':
                    # Continue cell: link to cell above
                    for prev_row in range(row_idx - 1, -1, -1):
                        if merge_info[prev_row][start_col] is not None:
                            owner = merge_info[prev_row][start_col]
                            for c in range(start_col, start_col + colspan):
                                merge_info[row_idx][c] = owner
                            break
                    else:
                        # Not found - set to current (edge case)
                        for c in range(start_col, start_col + colspan):
                            merge_info[row_idx][c] = (row_idx, start_col, colspan)
                else:
                    # Normal cell
                    for c in range(start_col, start_col + colspan):
                        merge_info[row_idx][c] = (row_idx, start_col, colspan)
                
                grid_col += colspan
            
            cell_grid_col.append(row_grid_cols)
        
        # Step 2: Calculate rowspans for restart cells
        for row_idx, row_info in enumerate(all_cells_info):
            for cell_idx, (colspan, vmerge_status) in enumerate(row_info):
                if cell_idx >= len(cell_grid_col[row_idx]):
                    continue
                start_col, end_col = cell_grid_col[row_idx][cell_idx]
                
                if vmerge_status == 'restart':
                    # Count cells below with same owner
                    rowspan = 1
                    for next_row in range(row_idx + 1, num_rows):
                        if start_col < max_cols and merge_info[next_row][start_col] == (row_idx, start_col, colspan):
                            rowspan += 1
                        else:
                            break
                    rowspan_map[(row_idx, start_col)] = rowspan
                
                elif vmerge_status == 'none':
                    rowspan_map[(row_idx, start_col)] = 1
        
        return rowspan_map, cell_grid_col
    
    def _extract_cell_text(self, cell_elem: Any) -> str:
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


# Factory function
def create_docx_table_extractor(
    config: Optional[TableExtractorConfig] = None
) -> DOCXTableExtractor:
    """Create a DOCX table extractor instance.
    
    Args:
        config: Table extraction configuration
        
    Returns:
        Configured DOCXTableExtractor instance
    """
    return DOCXTableExtractor(config)


__all__ = [
    'DOCXTableExtractor',
    'create_docx_table_extractor',
]
