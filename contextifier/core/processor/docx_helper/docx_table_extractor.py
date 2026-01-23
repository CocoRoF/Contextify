# contextifier/core/processor/docx_helper/docx_table_extractor.py
"""
DOCX Table Extractor - DOCX Format-Specific Table Extraction

Implements table extraction for DOCX files (Office Open XML format).
Follows BaseTableExtractor interface from table_extractor.py.

DOCX Table Structure (OOXML):
- w:tblGrid: 테이블의 그리드 열 정의
- w:tr: 테이블 행
- w:tc: 테이블 셀
- w:tcPr/w:gridSpan: colspan (가로 병합)
- w:tcPr/w:vMerge val="restart": rowspan 시작
- w:tcPr/w:vMerge (val 없음): rowspan 계속 (병합된 셀)

2-Pass Approach:
1. Pass 1: Detect table regions (w:tbl elements in document)
2. Pass 2: Extract content from detected regions (TableData objects)

Usage:
    from contextifier.core.processor.docx_helper.docx_table_extractor import (
        DOCXTableExtractor,
    )

    extractor = DOCXTableExtractor()
    tables = extractor.extract_tables(doc)  # doc is python-docx Document
"""
import logging
import traceback
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
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


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DOCXTableExtractorConfig(TableExtractorConfig):
    """Configuration specific to DOCX table extraction.
    
    Attributes:
        skip_single_cell_tables: Whether to skip 1x1 tables (often containers)
        skip_single_column_tables: Whether to skip single column tables
        extract_nested_tables: Whether to extract tables nested inside cells
    """
    skip_single_cell_tables: bool = True
    skip_single_column_tables: bool = True
    extract_nested_tables: bool = True


# ============================================================================
# Legacy Data Class (for backward compatibility)
# ============================================================================

@dataclass
class TableCellInfo:
    """테이블 셀 정보를 저장하는 데이터 클래스 (레거시 호환용)
    
    Note: 새 코드에서는 table_extractor.TableCell 사용 권장
    """
    grid_row: int       # 그리드 상의 행 위치 (0-based)
    grid_col: int       # 그리드 상의 열 위치 (0-based)
    rowspan: int        # 실제 rowspan 값
    colspan: int        # 실제 colspan 값 (gridSpan)
    content: str        # 셀 내용
    is_merged_away: bool  # True면 다른 셀에 병합되어 렌더링하지 않음


# ============================================================================
# DOCXTableExtractor Class (BaseTableExtractor 인터페이스 구현)
# ============================================================================

class DOCXTableExtractor(BaseTableExtractor):
    """DOCX format-specific table extractor.
    
    Extracts tables from DOCX files by parsing Office Open XML structure.
    Implements BaseTableExtractor interface from table_extractor.py.
    
    Supports:
    - Cell merges (colspan via gridSpan, rowspan via vMerge)
    - Nested tables
    - Header row detection
    - Column width calculation
    """
    
    def __init__(self, config: Optional[DOCXTableExtractorConfig] = None):
        """Initialize DOCX table extractor.
        
        Args:
            config: DOCX table extraction configuration
        """
        self._config = config or DOCXTableExtractorConfig()
        super().__init__(self._config)
        # Cache for document
        self._doc_cache: Optional[Document] = None
        self._table_elements_cache: Optional[List] = None
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format."""
        return format_type.lower() in ("docx", "docm")
    
    # ========================================================================
    # BaseTableExtractor Interface Implementation
    # ========================================================================
    
    def detect_table_regions(self, content: Any) -> List[TableRegion]:
        """Detect table regions in DOCX document.
        
        Pass 1: Find all w:tbl elements in the document body.
        For DOCX, tables are explicitly marked in XML, so detection is straightforward.
        
        Args:
            content: python-docx Document object, file path, or bytes
            
        Returns:
            List of TableRegion objects (start_offset = table index)
        """
        doc = self._get_document(content)
        if doc is None:
            return []
        
        regions = []
        
        try:
            # Find all table elements in document body
            body = doc.element.body
            table_elements = body.findall('.//w:tbl', NAMESPACES)
            
            # Cache for later use
            self._doc_cache = doc
            self._table_elements_cache = table_elements
            
            for idx, table_elem in enumerate(table_elements):
                rows = table_elem.findall('w:tr', NAMESPACES)
                if not rows:
                    continue
                
                row_count = len(rows)
                col_count = self._get_column_count(table_elem, rows)
                
                # Skip single-cell tables if configured
                if self._config.skip_single_cell_tables:
                    if row_count == 1 and col_count == 1:
                        continue
                
                # Skip single-column tables if configured
                if self._config.skip_single_column_tables:
                    if col_count == 1:
                        continue
                
                # For DOCX, we use table index as offset (XML doesn't have byte offsets)
                region = TableRegion(
                    start_offset=idx,  # Table index
                    end_offset=idx,
                    row_count=row_count,
                    col_count=col_count,
                    confidence=1.0  # XML structure is explicit
                )
                regions.append(region)
            
            self.logger.debug(f"Detected {len(regions)} table regions in DOCX")
            
        except Exception as e:
            self.logger.warning(f"Error detecting DOCX table regions: {e}")
        
        return regions
    
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from a detected region.
        
        Pass 2: Parse the w:tbl element to build TableData.
        
        Args:
            content: python-docx Document object, file path, or bytes
            region: TableRegion identifying the table (start_offset = table index)
            
        Returns:
            TableData object or None
        """
        doc = self._get_document(content)
        if doc is None:
            return None
        
        try:
            # Use cache if available
            if self._table_elements_cache is not None and self._doc_cache is doc:
                table_elements = self._table_elements_cache
            else:
                body = doc.element.body
                table_elements = body.findall('.//w:tbl', NAMESPACES)
            
            table_idx = region.start_offset
            if table_idx >= len(table_elements):
                self.logger.warning(f"Table index {table_idx} out of range")
                return None
            
            table_elem = table_elements[table_idx]
            return self._extract_table_data(table_elem, doc, table_idx)
            
        except Exception as e:
            self.logger.warning(f"Error extracting DOCX table: {e}")
            return None
    
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    def _get_document(self, content: Any) -> Optional[Document]:
        """Get python-docx Document object from various input types.
        
        Args:
            content: Document object, file path, or bytes
            
        Returns:
            python-docx Document object or None
        """
        try:
            if hasattr(content, 'element'):
                # Already a Document object
                return content
            elif isinstance(content, (str, bytes, BytesIO)):
                if isinstance(content, bytes):
                    content = BytesIO(content)
                return Document(content)
            else:
                self.logger.warning(f"Unsupported content type: {type(content)}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Error loading DOCX document: {e}")
            return None
    
    def _get_column_count(self, table_elem, rows) -> int:
        """Calculate column count from tblGrid or first row.
        
        Args:
            table_elem: w:tbl XML element
            rows: List of w:tr elements
            
        Returns:
            Number of columns
        """
        # Try tblGrid first
        tblGrid = table_elem.find('w:tblGrid', NAMESPACES)
        if tblGrid is not None:
            grid_cols = tblGrid.findall('w:gridCol', NAMESPACES)
            if grid_cols:
                return len(grid_cols)
        
        # Fallback: count from first row
        if rows:
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
        
        return 0
    
    def _extract_table_data(
        self, 
        table_elem, 
        doc: Document, 
        table_idx: int
    ) -> Optional[TableData]:
        """Extract TableData from a table XML element.
        
        Uses rowspan/colspan calculation to build accurate TableCell objects.
        
        Args:
            table_elem: w:tbl XML element
            doc: python-docx Document object
            table_idx: Index of this table in document
            
        Returns:
            TableData object or None
        """
        try:
            rows_elem = table_elem.findall('w:tr', NAMESPACES)
            if not rows_elem:
                return None
            
            num_rows = len(rows_elem)
            num_cols = self._get_column_count(table_elem, rows_elem)
            
            # Skip single cell tables
            if self._config.skip_single_cell_tables:
                if num_rows == 1 and num_cols == 1:
                    return None
            
            # Skip single column tables
            if self._config.skip_single_column_tables:
                if num_cols == 1:
                    return None
            
            # Calculate rowspan information using existing algorithm
            rowspan_map, cell_grid_col = self._calculate_all_rowspans(
                table_elem, rows_elem, num_rows
            )
            
            # Build TableCell objects
            table_rows: List[List[TableCell]] = []
            
            for row_idx, row_elem in enumerate(rows_elem):
                cells_elem = row_elem.findall('w:tc', NAMESPACES)
                row_cells: List[TableCell] = []
                
                for cell_idx, cell_elem in enumerate(cells_elem):
                    # Get cell properties
                    tcPr = cell_elem.find('w:tcPr', NAMESPACES)
                    colspan = 1
                    is_vmerge_continue = False
                    
                    if tcPr is not None:
                        # Get gridSpan (colspan)
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
                    
                    # Skip merged-away cells
                    if is_vmerge_continue:
                        continue
                    
                    # Get grid column position
                    if cell_idx < len(cell_grid_col[row_idx]):
                        start_col, _ = cell_grid_col[row_idx][cell_idx]
                    else:
                        start_col = cell_idx
                    
                    # Get rowspan
                    rowspan = rowspan_map.get((row_idx, start_col), 1)
                    
                    # Extract cell content
                    cell_content = self._extract_cell_text(cell_elem)
                    
                    # Check for nested table
                    nested_table = None
                    if self._config.extract_nested_tables:
                        nested_tbl = cell_elem.find('.//w:tbl', NAMESPACES)
                        if nested_tbl is not None:
                            nested_table = self._extract_table_data(nested_tbl, doc, -1)
                    
                    # Create TableCell (using table_extractor.TableCell)
                    cell = TableCell(
                        content=cell_content,
                        row_span=rowspan,
                        col_span=colspan,
                        is_header=(row_idx == 0 and self._config.include_header_row),
                        row_index=row_idx,
                        col_index=start_col,
                        nested_table=nested_table
                    )
                    row_cells.append(cell)
                
                if row_cells:
                    table_rows.append(row_cells)
            
            if not table_rows:
                return None
            
            # Calculate column widths
            col_widths_percent = self._calculate_column_widths(table_elem, num_cols)
            
            return TableData(
                rows=table_rows,
                num_rows=len(table_rows),
                num_cols=num_cols,
                has_header=self._config.include_header_row,
                start_offset=table_idx,
                end_offset=table_idx,
                source_format="docx",
                col_widths_percent=col_widths_percent
            )
            
        except Exception as e:
            self.logger.warning(f"Error extracting table data: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def _calculate_all_rowspans(
        self, 
        table_elem, 
        rows, 
        num_rows: int
    ) -> Tuple[Dict[Tuple[int, int], int], List[List[Tuple[int, int]]]]:
        """Calculate rowspan values for all vMerge restart cells.
        
        Algorithm:
        1. Collect all cell info (colspan, vmerge_status)
        2. Use merge_info matrix to track which grid_col belongs to which cell
        3. Connect continue cells to the same grid_col position in upper rows
        4. Calculate rowspan for restart cells
        
        Args:
            table_elem: table XML element
            rows: List of row elements
            num_rows: Number of rows
            
        Returns:
            Tuple of (rowspan_map, cell_grid_col)
            - rowspan_map: Dict[(row_idx, grid_col)] -> rowspan value
            - cell_grid_col: List[List[(start_col, end_col)]] for each cell
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

        # Calculate grid_col for all cells
        max_cols = 30
        cell_grid_col: List[List[Tuple[int, int]]] = []
        merge_info: List[List[Optional[Tuple[int, int, int]]]] = [
            [None] * max_cols for _ in range(num_rows)
        ]

        for row_idx, row_info in enumerate(all_cells_info):
            grid_col = 0
            row_grid_cols: List[Tuple[int, int]] = []

            for cell_idx, (colspan, vmerge_status) in enumerate(row_info):
                # Skip already occupied columns
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
                    for c in range(start_col, start_col + colspan):
                        merge_info[row_idx][c] = (row_idx, start_col, colspan)
                elif vmerge_status == 'continue':
                    for prev_row in range(row_idx - 1, -1, -1):
                        if merge_info[prev_row][start_col] is not None:
                            owner = merge_info[prev_row][start_col]
                            for c in range(start_col, start_col + colspan):
                                merge_info[row_idx][c] = owner
                            break
                    else:
                        for c in range(start_col, start_col + colspan):
                            merge_info[row_idx][c] = (row_idx, start_col, colspan)
                else:
                    for c in range(start_col, start_col + colspan):
                        merge_info[row_idx][c] = (row_idx, start_col, colspan)

                grid_col += colspan

            cell_grid_col.append(row_grid_cols)

        # Calculate rowspan for restart cells
        for row_idx, row_info in enumerate(all_cells_info):
            for cell_idx, (colspan, vmerge_status) in enumerate(row_info):
                if cell_idx >= len(cell_grid_col[row_idx]):
                    continue
                start_col, end_col = cell_grid_col[row_idx][cell_idx]

                if vmerge_status == 'restart':
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
    
    def _extract_cell_text(self, cell_elem) -> str:
        """Extract text content from a cell element.
        
        Args:
            cell_elem: w:tc XML element
            
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
    
    def _calculate_column_widths(self, table_elem, num_cols: int) -> List[float]:
        """Calculate column widths as percentages from tblGrid.
        
        Args:
            table_elem: w:tbl XML element
            num_cols: Total number of columns
            
        Returns:
            List of column widths as percentages
        """
        try:
            tblGrid = table_elem.find('w:tblGrid', NAMESPACES)
            if tblGrid is None:
                if num_cols > 0:
                    return [100.0 / num_cols] * num_cols
                return []
            
            grid_cols = tblGrid.findall('w:gridCol', NAMESPACES)
            if not grid_cols:
                if num_cols > 0:
                    return [100.0 / num_cols] * num_cols
                return []
            
            # Extract widths (in twips or EMU)
            widths = []
            for col in grid_cols:
                w = col.get(qn('w:w'))
                if w:
                    try:
                        widths.append(float(w))
                    except ValueError:
                        widths.append(0)
                else:
                    widths.append(0)
            
            # Convert to percentages
            total = sum(widths)
            if total > 0:
                return [round(w / total * 100, 1) for w in widths]
            else:
                return [100.0 / len(widths)] * len(widths)
                
        except Exception as e:
            self.logger.debug(f"Error calculating column widths: {e}")
            if num_cols > 0:
                return [100.0 / num_cols] * num_cols
            return []


# ============================================================================
# Extractor Helper Functions (Backward Compatibility)
# ============================================================================

def calculate_all_rowspans(table_elem, rows, num_rows: int):
    """Calculate rowspan values (backward compatibility wrapper).
    
    This function wraps DOCXTableExtractor._calculate_all_rowspans for compatibility.
    
    Args:
        table_elem: table XML element
        rows: List of row elements
        num_rows: Number of rows
        
    Returns:
        Tuple of (rowspan_map, cell_grid_col)
    """
    extractor = DOCXTableExtractor()
    return extractor._calculate_all_rowspans(table_elem, rows, num_rows)


def estimate_column_count(first_row) -> int:
    """Estimate column count from first row.
    
    Args:
        first_row: First row XML element

    Returns:
        Estimated column count
    """
    cells = first_row.findall('w:tc', NAMESPACES)
    total_cols = 0
    for cell in cells:
        colspan = 1
        tcPr = cell.find('w:tcPr', NAMESPACES)
        if tcPr is not None:
            gs = tcPr.find('w:gridSpan', NAMESPACES)
            if gs is not None:
                try:
                    colspan = int(gs.get(qn('w:val'), 1))
                except (ValueError, TypeError):
                    colspan = 1
        total_cols += colspan
    return total_cols


# ============================================================================
# Processor Functions (Delegated to docx_table_processor)
# ============================================================================

# These functions are now provided by docx_table_processor.py
# Re-exported here for backward compatibility
from contextifier.core.processor.docx_helper.docx_table_processor import (
    process_table_element,
    extract_cell_text,
    extract_table_as_text,
)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main class (BaseTableExtractor implementation)
    'DOCXTableExtractor',
    'DOCXTableExtractorConfig',
    # Legacy data class
    'TableCellInfo',
    # Extractor helper functions
    'calculate_all_rowspans',
    'estimate_column_count',
    # Processor functions (re-exported from docx_table_processor)
    'process_table_element',
    'extract_cell_text',
    'extract_table_as_text',
]
