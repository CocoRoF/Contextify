# contextifier/core/processor/rtf_helper/rtf_table_extractor.py
"""
RTF Table Extractor - RTF Format-Specific Table Extraction

Implements table extraction for RTF files.
Follows BaseTableExtractor interface from table_extractor.py.

RTF Table Structure:
- \\trowd: Table row start (row definition)
- \\cellxN: Cell boundary position (N in twips)
- \\clmgf: Horizontal merge start
- \\clmrg: Horizontal merge continue  
- \\clvmgf: Vertical merge start
- \\clvmrg: Vertical merge continue
- \\intbl: Paragraph in cell
- \\cell: Cell end marker
- \\row: Row end marker

2-Pass Approach:
1. Pass 1: Detect table regions by finding \\trowd...\\row patterns
2. Pass 2: Extract content from detected regions (TableData objects)

Usage:
    from contextifier.core.processor.rtf_helper.rtf_table_extractor import (
        RTFTableExtractor,
    )

    extractor = RTFTableExtractor()
    tables = extractor.extract_tables(rtf_content)
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from contextifier.core.functions.table_extractor import (
    BaseTableExtractor,
    TableCell,
    TableData,
    TableRegion,
    TableExtractorConfig,
)
from contextifier.core.processor.rtf_helper.rtf_decoder import (
    decode_hex_escapes,
)
from contextifier.core.processor.rtf_helper.rtf_text_cleaner import (
    clean_rtf_text,
)
from contextifier.core.processor.rtf_helper.rtf_region_finder import (
    find_excluded_regions,
    is_in_excluded_region,
)

logger = logging.getLogger("document-processor")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RTFTableExtractorConfig(TableExtractorConfig):
    """Configuration specific to RTF table extraction.
    
    Attributes:
        encoding: Default encoding for RTF content
        row_gap_threshold: Maximum chars between rows to consider same table
        skip_single_column_tables: Whether to skip single column tables
    """
    encoding: str = "cp949"
    row_gap_threshold: int = 150
    skip_single_column_tables: bool = True


# =============================================================================
# Legacy Data Models (Backward Compatibility)
# =============================================================================

class RTFCellInfo(NamedTuple):
    """RTF cell information with merge info (legacy format).
    
    Note: New code should use TableCell from table_extractor.py
    """
    text: str              # Cell text content
    h_merge_first: bool    # Horizontal merge start (clmgf)
    h_merge_cont: bool     # Horizontal merge continue (clmrg)
    v_merge_first: bool    # Vertical merge start (clvmgf)
    v_merge_cont: bool     # Vertical merge continue (clvmrg)
    right_boundary: int    # Cell right boundary (twips)


@dataclass
class RTFTable:
    """RTF table structure (legacy format).
    
    Note: New code should use TableData from table_extractor.py
    This class is kept for backward compatibility.
    """
    rows: List[List[RTFCellInfo]] = field(default_factory=list)
    col_count: int = 0
    position: int = 0      # Start position in document
    end_position: int = 0  # End position in document
    
    def is_real_table(self) -> bool:
        """
        Determine if this is a real table.
        
        n rows x 1 column is considered a list, not a table.
        """
        if not self.rows:
            return False
        
        effective_cols = self._get_effective_col_count()
        return effective_cols >= 2
    
    def _get_effective_col_count(self) -> int:
        """Calculate effective column count (excluding empty columns)."""
        if not self.rows:
            return 0
        
        effective_counts = []
        for row in self.rows:
            non_empty_cells = []
            for i, cell in enumerate(row):
                if cell.h_merge_cont:
                    continue
                if cell.text.strip() or cell.v_merge_first:
                    non_empty_cells.append(i)
            
            if non_empty_cells:
                effective_counts.append(max(non_empty_cells) + 1)
        
        return max(effective_counts) if effective_counts else 0
    
    def to_table_data(self) -> TableData:
        """Convert RTFTable to TableData.
        
        Returns:
            TableData object compatible with table_extractor interface
        """
        if not self.rows:
            return TableData()
        
        # Calculate merge info
        merge_info = self._calculate_merge_info()
        
        # Build TableCell rows
        table_rows: List[List[TableCell]] = []
        
        for row_idx, row in enumerate(self.rows):
            table_row: List[TableCell] = []
            
            for col_idx, cell in enumerate(row):
                if col_idx < len(merge_info[row_idx]):
                    colspan, rowspan = merge_info[row_idx][col_idx]
                    
                    # Skip merged-away cells
                    if colspan == 0 or rowspan == 0:
                        continue
                    
                    cell_text = re.sub(r'\s+', ' ', cell.text).strip()
                    
                    table_cell = TableCell(
                        content=cell_text,
                        row_span=rowspan,
                        col_span=colspan,
                        is_header=False,
                        row_index=row_idx,
                        col_index=col_idx,
                    )
                    table_row.append(table_cell)
                else:
                    cell_text = re.sub(r'\s+', ' ', cell.text).strip()
                    table_cell = TableCell(
                        content=cell_text,
                        row_span=1,
                        col_span=1,
                        is_header=False,
                        row_index=row_idx,
                        col_index=col_idx,
                    )
                    table_row.append(table_cell)
            
            if table_row:
                table_rows.append(table_row)
        
        return TableData(
            rows=table_rows,
            num_rows=len(table_rows),
            num_cols=self.col_count,
            has_header=False,
            start_offset=self.position,
            end_offset=self.end_position,
            source_format="rtf",
        )
    
    def _calculate_merge_info(self) -> List[List[Tuple[int, int]]]:
        """Calculate colspan and rowspan for each cell."""
        if not self.rows:
            return []
        
        num_rows = len(self.rows)
        max_cols = max(len(row) for row in self.rows) if self.rows else 0
        
        if max_cols == 0:
            return []
        
        # Initialize with (1, 1) for all cells
        merge_info: List[List[Tuple[int, int]]] = [
            [(1, 1) for _ in range(max_cols)] for _ in range(num_rows)
        ]
        
        # Process horizontal merges
        for row_idx, row in enumerate(self.rows):
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
                if col_idx >= len(self.rows[row_idx]):
                    row_idx += 1
                    continue
                
                cell = self.rows[row_idx][col_idx]
                
                if cell.v_merge_first:
                    rowspan = 1
                    for next_row in range(row_idx + 1, num_rows):
                        if col_idx < len(self.rows[next_row]) and self.rows[next_row][col_idx].v_merge_cont:
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
# RTFTableExtractor Class (BaseTableExtractor 인터페이스 구현)
# =============================================================================

class RTFTableExtractor(BaseTableExtractor):
    """RTF format-specific table extractor.
    
    Extracts tables from RTF files by parsing control words.
    Implements BaseTableExtractor interface from table_extractor.py.
    
    Supports:
    - Cell merges (horizontal: clmgf/clmrg, vertical: clvmgf/clvmrg)
    - Multiple tables in document
    - Excluded region handling (headers, footers, footnotes)
    """
    
    def __init__(self, config: Optional[RTFTableExtractorConfig] = None):
        """Initialize RTF table extractor.
        
        Args:
            config: RTF table extraction configuration
        """
        self._config = config or RTFTableExtractorConfig()
        super().__init__(self._config)
        # Cache for parsed data
        self._content_cache: Optional[str] = None
        self._rtf_tables_cache: Optional[List[RTFTable]] = None
        self._regions_cache: Optional[List[Tuple[int, int, RTFTable]]] = None
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format."""
        return format_type.lower() in ("rtf",)
    
    # =========================================================================
    # BaseTableExtractor Interface Implementation
    # =========================================================================
    
    def detect_table_regions(self, content: Any) -> List[TableRegion]:
        """Detect table regions in RTF content.
        
        Pass 1: Find all \\trowd...\\row patterns in the document.
        Groups consecutive rows into table regions.
        
        Args:
            content: RTF string content
            
        Returns:
            List of TableRegion objects
        """
        if not isinstance(content, str):
            self.logger.warning(f"RTF content must be string, got {type(content)}")
            return []
        
        # Parse tables and cache results
        self._content_cache = content
        rtf_tables, table_regions = self._extract_rtf_tables(
            content, 
            self._config.encoding
        )
        self._rtf_tables_cache = rtf_tables
        self._regions_cache = table_regions
        
        # Convert to TableRegion objects
        regions: List[TableRegion] = []
        
        for start_pos, end_pos, rtf_table in table_regions:
            # Skip single column tables if configured
            if self._config.skip_single_column_tables:
                if not rtf_table.is_real_table():
                    continue
            
            region = TableRegion(
                start_offset=start_pos,
                end_offset=end_pos,
                row_count=len(rtf_table.rows),
                col_count=rtf_table.col_count,
                confidence=1.0  # RTF structure is explicit
            )
            regions.append(region)
        
        self.logger.debug(f"Detected {len(regions)} table regions in RTF")
        return regions
    
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from a detected region.
        
        Pass 2: Parse the table content to build TableData.
        
        Args:
            content: RTF string content
            region: TableRegion identifying the table
            
        Returns:
            TableData object or None
        """
        if not isinstance(content, str):
            return None
        
        # Use cache if available
        if (self._content_cache == content and 
            self._regions_cache is not None):
            # Find matching RTFTable from cache
            for start_pos, end_pos, rtf_table in self._regions_cache:
                if start_pos == region.start_offset and end_pos == region.end_offset:
                    return rtf_table.to_table_data()
        
        # Re-parse if cache miss
        rtf_tables, table_regions = self._extract_rtf_tables(
            content, 
            self._config.encoding
        )
        
        for start_pos, end_pos, rtf_table in table_regions:
            if start_pos == region.start_offset and end_pos == region.end_offset:
                return rtf_table.to_table_data()
        
        return None
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _extract_rtf_tables(
        self,
        content: str,
        encoding: str
    ) -> Tuple[List[RTFTable], List[Tuple[int, int, RTFTable]]]:
        """Extract RTFTable objects from content.
        
        Args:
            content: RTF string content
            encoding: Encoding to use
            
        Returns:
            Tuple of (table list, table region list)
        """
        tables: List[RTFTable] = []
        table_regions: List[Tuple[int, int, RTFTable]] = []
        
        # Find excluded regions (header, footer, footnote, etc.)
        excluded_regions = find_excluded_regions(content)
        
        # Step 1: Find all \row positions
        row_positions = []
        for match in re.finditer(r'\\row(?![a-z])', content):
            row_positions.append(match.end())
        
        if not row_positions:
            return tables, table_regions
        
        # Step 2: Find \trowd before each \row
        all_rows: List[Tuple[int, int, str]] = []
        for i, row_end in enumerate(row_positions):
            if i == 0:
                search_start = 0
            else:
                search_start = row_positions[i - 1]
            
            segment = content[search_start:row_end]
            trowd_match = re.search(r'\\trowd', segment)
            
            if trowd_match:
                row_start = search_start + trowd_match.start()
                
                # Skip rows in excluded regions
                if is_in_excluded_region(row_start, excluded_regions):
                    self.logger.debug(f"Skipping table row at {row_start} (in excluded region)")
                    continue
                
                row_text = content[row_start:row_end]
                all_rows.append((row_start, row_end, row_text))
        
        if not all_rows:
            return tables, table_regions
        
        # Group consecutive rows into tables
        table_groups = self._group_rows_into_tables(all_rows)
        
        self.logger.debug(f"Found {len(table_groups)} table groups")
        
        # Parse each table group
        for start_pos, end_pos, table_rows in table_groups:
            rtf_table = self._parse_table_with_merge(table_rows, encoding)
            if rtf_table and rtf_table.rows:
                rtf_table.position = start_pos
                rtf_table.end_position = end_pos
                tables.append(rtf_table)
                table_regions.append((start_pos, end_pos, rtf_table))
        
        self.logger.debug(f"Extracted {len(tables)} tables")
        return tables, table_regions
    
    def _group_rows_into_tables(
        self, 
        all_rows: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, List[str]]]:
        """Group consecutive rows into table groups.
        
        Args:
            all_rows: List of (start, end, row_text) tuples
            
        Returns:
            List of (start, end, [row_texts]) tuples
        """
        table_groups: List[Tuple[int, int, List[str]]] = []
        current_table: List[str] = []
        current_start = -1
        current_end = -1
        prev_end = -1
        
        for row_start, row_end, row_text in all_rows:
            # Rows within threshold chars are same table
            if prev_end == -1 or row_start - prev_end < self._config.row_gap_threshold:
                if current_start == -1:
                    current_start = row_start
                current_table.append(row_text)
                current_end = row_end
            else:
                if current_table:
                    table_groups.append((current_start, current_end, current_table))
                current_table = [row_text]
                current_start = row_start
                current_end = row_end
            prev_end = row_end
        
        if current_table:
            table_groups.append((current_start, current_end, current_table))
        
        return table_groups
    
    def _parse_table_with_merge(
        self, 
        rows: List[str], 
        encoding: str
    ) -> Optional[RTFTable]:
        """Parse table rows to RTFTable object with merge support.
        
        Args:
            rows: Table row text list
            encoding: Encoding to use
            
        Returns:
            RTFTable object
        """
        table = RTFTable()
        
        for row_text in rows:
            cells = self._extract_cells_with_merge(row_text, encoding)
            if cells:
                table.rows.append(cells)
                if len(cells) > table.col_count:
                    table.col_count = len(cells)
        
        return table if table.rows else None
    
    def _extract_cells_with_merge(
        self, 
        row_text: str, 
        encoding: str
    ) -> List[RTFCellInfo]:
        """Extract cell content and merge information from table row.
        
        Args:
            row_text: Table row RTF text
            encoding: Encoding to use
            
        Returns:
            List of RTFCellInfo
        """
        cells: List[RTFCellInfo] = []
        
        # Step 1: Parse cell definitions (attributes before cellx)
        cell_defs: List[Dict[str, Any]] = []
        
        # Find first \cell that is not \cellx
        first_cell_idx = self._find_first_cell_marker(row_text)
        def_part = row_text[:first_cell_idx]
        
        current_def = {
            'h_merge_first': False,
            'h_merge_cont': False,
            'v_merge_first': False,
            'v_merge_cont': False,
            'right_boundary': 0
        }
        
        cell_def_pattern = r'\\cl(?:mgf|mrg|vmgf|vmrg)|\\cellx(-?\d+)'
        
        for match in re.finditer(cell_def_pattern, def_part):
            token = match.group()
            if token == '\\clmgf':
                current_def['h_merge_first'] = True
            elif token == '\\clmrg':
                current_def['h_merge_cont'] = True
            elif token == '\\clvmgf':
                current_def['v_merge_first'] = True
            elif token == '\\clvmrg':
                current_def['v_merge_cont'] = True
            elif token.startswith('\\cellx'):
                if match.group(1):
                    current_def['right_boundary'] = int(match.group(1))
                cell_defs.append(current_def.copy())
                current_def = {
                    'h_merge_first': False,
                    'h_merge_cont': False,
                    'v_merge_first': False,
                    'v_merge_cont': False,
                    'right_boundary': 0
                }
        
        # Step 2: Extract cell texts
        cell_texts = self._extract_cell_texts(row_text, encoding)
        
        # Step 3: Match cell definitions with content
        for i, cell_text in enumerate(cell_texts):
            if i < len(cell_defs):
                cell_def = cell_defs[i]
            else:
                cell_def = {
                    'h_merge_first': False,
                    'h_merge_cont': False,
                    'v_merge_first': False,
                    'v_merge_cont': False,
                    'right_boundary': 0
                }
            
            cells.append(RTFCellInfo(
                text=cell_text,
                h_merge_first=cell_def['h_merge_first'],
                h_merge_cont=cell_def['h_merge_cont'],
                v_merge_first=cell_def['v_merge_first'],
                v_merge_cont=cell_def['v_merge_cont'],
                right_boundary=cell_def['right_boundary']
            ))
        
        return cells
    
    def _find_first_cell_marker(self, row_text: str) -> int:
        """Find first \\cell marker (not \\cellx).
        
        Args:
            row_text: Row text
            
        Returns:
            Position of first \\cell marker
        """
        pos = 0
        while True:
            idx = row_text.find('\\cell', pos)
            if idx == -1:
                return len(row_text)
            if idx + 5 < len(row_text) and row_text[idx + 5] == 'x':
                pos = idx + 1
                continue
            return idx
    
    def _extract_cell_texts(
        self, 
        row_text: str, 
        encoding: str
    ) -> List[str]:
        """Extract cell texts from row.
        
        Args:
            row_text: Table row RTF text
            encoding: Encoding to use
            
        Returns:
            List of cell texts
        """
        cell_texts: List[str] = []
        
        # Step 1: Find all \cell positions (not \cellx)
        cell_positions: List[int] = []
        pos = 0
        while True:
            idx = row_text.find('\\cell', pos)
            if idx == -1:
                break
            next_pos = idx + 5
            if next_pos < len(row_text) and row_text[next_pos] == 'x':
                pos = idx + 1
                continue
            cell_positions.append(idx)
            pos = idx + 1
        
        if not cell_positions:
            return cell_texts
        
        # Step 2: Find last \cellx before first \cell
        first_cell_pos = cell_positions[0]
        def_part = row_text[:first_cell_pos]
        
        last_cellx_end = 0
        for match in re.finditer(r'\\cellx-?\d+', def_part):
            last_cellx_end = match.end()
        
        # Step 3: Extract each cell content
        prev_end = last_cellx_end
        for cell_end in cell_positions:
            cell_content = row_text[prev_end:cell_end]
            
            # RTF decoding and cleaning
            decoded = decode_hex_escapes(cell_content, encoding)
            clean = clean_rtf_text(decoded, encoding)
            cell_texts.append(clean)
            
            prev_end = cell_end + 5  # len('\\cell') = 5
        
        return cell_texts


# =============================================================================
# Backward Compatibility Functions
# =============================================================================

def extract_tables_with_positions(
    content: str,
    encoding: str = "cp949"
) -> Tuple[List[RTFTable], List[Tuple[int, int, RTFTable]]]:
    """
    Extract tables from RTF content with position information.
    
    This function is kept for backward compatibility.
    New code should use RTFTableExtractor class.
    
    Args:
        content: RTF string content
        encoding: Encoding to use
        
    Returns:
        Tuple of (table list, table region list [(start, end, table), ...])
    """
    config = RTFTableExtractorConfig(encoding=encoding)
    extractor = RTFTableExtractor(config)
    return extractor._extract_rtf_tables(content, encoding)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main class (BaseTableExtractor implementation)
    'RTFTableExtractor',
    'RTFTableExtractorConfig',
    # Legacy data classes
    'RTFCellInfo',
    'RTFTable',
    # Backward compatibility functions
    'extract_tables_with_positions',
]
