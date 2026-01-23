# contextifier/core/processor/ole_helper/ole_table_properties.py
"""
OLE Table Properties Parser

Parses TAPX (Table Properties) from OLE DOC format binary streams.
Extracts cell merge information (colspan/rowspan) from sprmTDefTable.

MS-DOC Specification Reference:
- sprmTDefTable (0xD608): Defines table structure
  - itcMac: Number of columns
  - rgdxaCenter[itcMac+1]: Column boundaries in twips
  - rgtc[itcMac]: Cell properties (TC) array
  
- TC (Table Cell) structure:
  - tcgrf (2 bytes): Cell flags
    - bit 5 (0x20): fVertMerge - cell is part of vertical merge
    - bit 6 (0x40): fVertRestart - cell starts vertical merge
    - bit 0 (0x01): fFirstMerged - cell starts horizontal merge
    - bit 1 (0x02): fMerged - cell continues horizontal merge

This module is OLE-specific - it parses binary structures only found in
OLE Compound Document format DOC files (Word 97-2003).
"""
import logging
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger("document-processor")


@dataclass
class TCProperties:
    """Table Cell Properties from TC structure."""
    
    col_index: int
    width_twips: int = 0
    tcgrf: int = 0
    
    # Horizontal merge flags
    f_first_merged: bool = False  # Starts horizontal merge (colspan)
    f_merged: bool = False        # Continues horizontal merge
    
    # Vertical merge flags
    f_vert_restart: bool = False  # Starts vertical merge (rowspan)
    f_vert_merge: bool = False    # Continues vertical merge
    
    @classmethod
    def from_bytes(cls, data: bytes, col_index: int) -> "TCProperties":
        """Parse TC structure from bytes.
        
        Args:
            data: 20-byte TC structure
            col_index: Column index
            
        Returns:
            TCProperties instance
        """
        if len(data) < 4:
            return cls(col_index=col_index)
        
        tcgrf = struct.unpack('<H', data[0:2])[0]
        width = struct.unpack('<H', data[2:4])[0]
        
        return cls(
            col_index=col_index,
            width_twips=width,
            tcgrf=tcgrf,
            f_first_merged=(tcgrf >> 0) & 1 == 1,
            f_merged=(tcgrf >> 1) & 1 == 1,
            f_vert_merge=(tcgrf >> 5) & 1 == 1,
            f_vert_restart=(tcgrf >> 6) & 1 == 1,
        )
    
    def is_merged_horizontally(self) -> bool:
        """Check if cell is merged with previous cell horizontally."""
        return self.f_merged and not self.f_first_merged
    
    def starts_horizontal_merge(self) -> bool:
        """Check if cell starts a horizontal merge (colspan)."""
        return self.f_first_merged
    
    def is_merged_vertically(self) -> bool:
        """Check if cell is merged with cell above."""
        return self.f_vert_merge and not self.f_vert_restart
    
    def starts_vertical_merge(self) -> bool:
        """Check if cell starts a vertical merge (rowspan)."""
        return self.f_vert_restart


@dataclass
class RowDefinition:
    """Table row definition from sprmTDefTable."""
    
    row_index: int
    file_offset: int  # Offset of sprmTDefTable in WordDocument stream
    col_count: int
    col_boundaries: List[int] = field(default_factory=list)  # twips
    col_widths: List[int] = field(default_factory=list)  # twips
    cells: List[TCProperties] = field(default_factory=list)
    
    def get_colspan(self, col_index: int) -> int:
        """Calculate colspan for a cell at given column.
        
        Counts consecutive fMerged cells following this cell.
        """
        if col_index >= len(self.cells):
            return 1
        
        cell = self.cells[col_index]
        if not cell.starts_horizontal_merge():
            return 1
        
        # Count following merged cells
        colspan = 1
        for i in range(col_index + 1, len(self.cells)):
            if self.cells[i].is_merged_horizontally():
                colspan += 1
            else:
                break
        
        return colspan


@dataclass
class TableStructure:
    """Complete table structure parsed from OLE DOC."""
    
    rows: List[RowDefinition] = field(default_factory=list)
    max_cols: int = 0
    
    def calculate_rowspans(self) -> Dict[Tuple[int, int], int]:
        """Calculate rowspan for each cell position.
        
        Returns:
            Dictionary mapping (row, col) to rowspan value
        """
        rowspans: Dict[Tuple[int, int], int] = {}
        
        if not self.rows:
            return rowspans
        
        # For each column, track vertical merges
        for col_idx in range(self.max_cols):
            merge_start_row = -1
            merge_count = 0
            
            for row_idx, row in enumerate(self.rows):
                if col_idx >= len(row.cells):
                    # Column doesn't exist in this row - end any merge
                    if merge_start_row >= 0 and merge_count > 1:
                        rowspans[(merge_start_row, col_idx)] = merge_count
                    merge_start_row = -1
                    merge_count = 0
                    continue
                
                cell = row.cells[col_idx]
                
                if cell.starts_vertical_merge():
                    # End previous merge if any
                    if merge_start_row >= 0 and merge_count > 1:
                        rowspans[(merge_start_row, col_idx)] = merge_count
                    # Start new merge
                    merge_start_row = row_idx
                    merge_count = 1
                elif cell.is_merged_vertically() and merge_start_row >= 0:
                    # Continue merge
                    merge_count += 1
                else:
                    # End merge
                    if merge_start_row >= 0 and merge_count > 1:
                        rowspans[(merge_start_row, col_idx)] = merge_count
                    merge_start_row = -1
                    merge_count = 0
            
            # End merge at end of table
            if merge_start_row >= 0 and merge_count > 1:
                rowspans[(merge_start_row, col_idx)] = merge_count
        
        return rowspans


class OLETablePropertiesParser:
    """Parser for OLE DOC table properties from WordDocument stream.
    
    Parses binary TAPX structures specific to OLE Compound Document format.
    """
    
    # SPRM codes
    SPRM_T_DEF_TABLE = 0xD608      # Table definition
    SPRM_P_F_IN_TABLE = 0x2416    # Paragraph in table
    SPRM_P_F_TTP = 0x2417         # Table terminating paragraph
    SPRM_T_DXA_LEFT = 0x9601      # Table left indent
    SPRM_T_DXA_GAP_HALF = 0x9602  # Table gap
    
    TC_SIZE = 20  # Size of TC structure in bytes
    
    def __init__(self, word_doc_stream: bytes):
        """Initialize parser with WordDocument stream data.
        
        Args:
            word_doc_stream: Raw bytes of WordDocument stream
        """
        self.data = word_doc_stream
    
    def find_table_definitions(self) -> List[int]:
        """Find all sprmTDefTable occurrences in the stream.
        
        Returns:
            List of byte offsets where sprmTDefTable is found
        """
        pattern = struct.pack('<H', self.SPRM_T_DEF_TABLE)  # 0x08 0xD6
        positions = []
        
        pos = 0
        while True:
            pos = self.data.find(pattern, pos)
            if pos == -1:
                break
            positions.append(pos)
            pos += 1
        
        return positions
    
    def parse_row_definition(self, offset: int, row_index: int = 0) -> Optional[RowDefinition]:
        """Parse sprmTDefTable at given offset.
        
        Args:
            offset: Byte offset in WordDocument stream
            row_index: Row index to assign
            
        Returns:
            RowDefinition or None if parsing fails
        """
        if offset + 3 > len(self.data):
            return None
        
        # Verify it's sprmTDefTable
        sprm = struct.unpack('<H', self.data[offset:offset+2])[0]
        if sprm != self.SPRM_T_DEF_TABLE:
            return None
        
        cb = self.data[offset + 2]  # Operand size
        if cb < 2:
            return None
        
        operand_start = offset + 3
        if operand_start + cb > len(self.data):
            return None
        
        operand = self.data[operand_start:operand_start + cb]
        
        # Parse operand structure:
        # [0]: Unknown/padding (usually 0x00)
        # [1]: itcMac (column count)
        # [2..]: Column boundaries, then TC array
        
        if len(operand) < 4:
            return None
        
        # Try both interpretations
        itc_mac = operand[1] if operand[0] == 0 else operand[0]
        boundary_start = 2 if operand[0] == 0 else 1
        
        # Sanity check
        if itc_mac < 1 or itc_mac > 50:
            logger.debug(f"Invalid column count {itc_mac} at offset {hex(offset)}")
            return None
        
        # Read column boundaries
        col_boundaries = []
        for i in range(itc_mac + 1):
            pos = boundary_start + i * 2
            if pos + 2 > len(operand):
                break
            dxa = struct.unpack('<h', operand[pos:pos+2])[0]
            col_boundaries.append(dxa)
        
        if len(col_boundaries) < itc_mac + 1:
            return None
        
        # Calculate widths
        col_widths = []
        for i in range(len(col_boundaries) - 1):
            width = col_boundaries[i+1] - col_boundaries[i]
            col_widths.append(width)
        
        # Parse TC array
        tc_start = boundary_start + (itc_mac + 1) * 2
        cells = []
        
        for i in range(itc_mac):
            tc_offset = tc_start + i * self.TC_SIZE
            if tc_offset + self.TC_SIZE <= len(operand):
                tc_data = operand[tc_offset:tc_offset + self.TC_SIZE]
                tc = TCProperties.from_bytes(tc_data, i)
                cells.append(tc)
            else:
                # Not enough data for full TC
                remaining = len(operand) - tc_offset
                if remaining >= 4:
                    tc_data = operand[tc_offset:tc_offset + remaining]
                    tc = TCProperties.from_bytes(tc_data, i)
                    cells.append(tc)
        
        return RowDefinition(
            row_index=row_index,
            file_offset=offset,
            col_count=itc_mac,
            col_boundaries=col_boundaries,
            col_widths=col_widths,
            cells=cells,
        )
    
    def parse_table_structure(self) -> Optional[TableStructure]:
        """Parse complete table structure from the document.
        
        Returns:
            TableStructure containing all row definitions
        """
        positions = self.find_table_definitions()
        
        if not positions:
            return None
        
        rows = []
        max_cols = 0
        
        for idx, pos in enumerate(positions):
            row = self.parse_row_definition(pos, idx)
            if row:
                rows.append(row)
                max_cols = max(max_cols, row.col_count)
        
        if not rows:
            return None
        
        return TableStructure(rows=rows, max_cols=max_cols)
    
    def get_merge_info_for_rows(
        self, 
        num_rows: int
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
        """Get colspan and rowspan info for table cells.
        
        Args:
            num_rows: Expected number of rows in the table
            
        Returns:
            Tuple of (colspan_map, rowspan_map)
            Each map is (row, col) -> span value
        """
        structure = self.parse_table_structure()
        
        if not structure or not structure.rows:
            return {}, {}
        
        colspan_map: Dict[Tuple[int, int], int] = {}
        
        # Calculate colspans from row definitions
        for row in structure.rows:
            for col_idx, cell in enumerate(row.cells):
                if cell.starts_horizontal_merge():
                    colspan = row.get_colspan(col_idx)
                    if colspan > 1:
                        colspan_map[(row.row_index, col_idx)] = colspan
        
        # Calculate rowspans
        rowspan_map = structure.calculate_rowspans()
        
        return colspan_map, rowspan_map


# Backward compatibility alias
DOCTablePropertiesParser = OLETablePropertiesParser


def extract_table_merge_info(
    word_doc_stream: bytes
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    """Convenience function to extract merge info from WordDocument stream.
    
    Args:
        word_doc_stream: Raw bytes of WordDocument stream
        
    Returns:
        Tuple of (colspan_map, rowspan_map)
    """
    parser = OLETablePropertiesParser(word_doc_stream)
    structure = parser.parse_table_structure()
    
    if not structure:
        return {}, {}
    
    colspan_map: Dict[Tuple[int, int], int] = {}
    
    # Calculate colspans from explicit flags
    for row in structure.rows:
        for col_idx, cell in enumerate(row.cells):
            if cell.starts_horizontal_merge():
                colspan = row.get_colspan(col_idx)
                if colspan > 1:
                    colspan_map[(row.row_index, col_idx)] = colspan
    
    # Calculate colspans from column width comparison
    # If a row has fewer columns but same total width, cells span multiple columns
    colspan_map_from_widths = _infer_colspan_from_widths(structure)
    colspan_map.update(colspan_map_from_widths)
    
    # Calculate rowspans
    rowspan_map = structure.calculate_rowspans()
    
    return colspan_map, rowspan_map


def _infer_colspan_from_widths(structure: TableStructure) -> Dict[Tuple[int, int], int]:
    """Infer colspan by comparing column widths across rows.
    
    When a row has fewer columns but the same total width, cells must span
    multiple columns. This is detected by matching cell widths to the
    reference (max column count) row's column boundaries.
    
    Args:
        structure: Parsed table structure
        
    Returns:
        Dictionary mapping (row, col) to colspan
    """
    if not structure.rows or structure.max_cols < 2:
        return {}
    
    colspan_map: Dict[Tuple[int, int], int] = {}
    
    # Find the reference row (row with max columns)
    ref_row = None
    for row in structure.rows:
        if row.col_count == structure.max_cols:
            ref_row = row
            break
    
    if not ref_row:
        return {}
    
    # Reference column boundaries
    ref_boundaries = ref_row.col_boundaries
    
    # Check each row with fewer columns
    for row in structure.rows:
        if row.col_count >= structure.max_cols:
            continue
        
        # Match this row's cell widths to reference columns
        row_boundaries = row.col_boundaries
        
        # For each cell in this row, find how many ref columns it spans
        for cell_idx in range(row.col_count):
            cell_start = row_boundaries[cell_idx]
            cell_end = row_boundaries[cell_idx + 1]
            cell_width = cell_end - cell_start
            
            # Find matching columns in reference row
            # A cell spans from first matching boundary to last
            start_col = None
            end_col = None
            
            # Find which reference column this cell starts at
            for ref_idx in range(len(ref_boundaries)):
                if abs(ref_boundaries[ref_idx] - cell_start) < 50:  # 50 twips tolerance
                    start_col = ref_idx
                    break
            
            # Find which reference column this cell ends at
            for ref_idx in range(len(ref_boundaries)):
                if abs(ref_boundaries[ref_idx] - cell_end) < 50:
                    end_col = ref_idx
                    break
            
            if start_col is not None and end_col is not None:
                colspan = end_col - start_col
                if colspan > 1:
                    colspan_map[(row.row_index, cell_idx)] = colspan
    
    return colspan_map
