# contextifier_new/chunking/constants.py
"""
Chunking Module Constants

Defines regex patterns, dataclasses, and thresholds used across the chunking
subsystem. All tag patterns are kept consistent with TagService defaults.

Ported and cleaned from contextifier/chunking/constants.py with a cleaner
structure and explicit type annotations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, List


# ============================================================================
# Code Language Mapping
# ============================================================================

# Extension → langchain Language enum name (string-based to avoid hard dep)
CODE_LANGUAGE_MAP: dict[str, str] = {
    "py": "PYTHON",
    "js": "JS",
    "ts": "TS",
    "java": "JAVA",
    "cpp": "CPP",
    "c": "CPP",
    "cs": "CSHARP",
    "go": "GO",
    "rs": "RUST",
    "php": "PHP",
    "rb": "RUBY",
    "swift": "SWIFT",
    "kt": "KOTLIN",
    "scala": "SCALA",
    "html": "HTML",
    "jsx": "JS",
    "tsx": "TS",
}


# ============================================================================
# Protected Region Patterns (blocks that must NEVER be split)
# ============================================================================

# HTML table (with any attributes)
HTML_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)

# Chart block — [chart]...[/chart]
CHART_BLOCK_PATTERN = re.compile(r"\[chart\].*?\[/chart\]", re.DOTALL | re.IGNORECASE)

# Textbox block — [textbox]...[/textbox]
TEXTBOX_BLOCK_PATTERN = re.compile(r"\[textbox\].*?\[/textbox\]", re.DOTALL | re.IGNORECASE)

# Image tag — [image:path] or [Image: {path}]
IMAGE_TAG_PATTERN = re.compile(r"\[(?i:image)\s*:\s*\{?[^\]\}]+\}?\]")

# Page/Slide/Sheet tag patterns
PAGE_TAG_PATTERN = re.compile(r"\[Page Number:\s*\d+(?:\s*\(OCR(?:\+Ref)?\))?\]")
SLIDE_TAG_PATTERN = re.compile(r"\[Slide Number:\s*\d+(?:\s*\(OCR(?:\+Ref)?\))?\]")
SHEET_TAG_PATTERN = re.compile(r"\[Sheet:\s*[^\]]+\]")

# Document metadata block — <Document-Metadata>...</Document-Metadata>
METADATA_BLOCK_PATTERN = re.compile(
    r"<Document-Metadata>.*?</Document-Metadata>", re.DOTALL
)

# Data analysis block — [Data Analysis]...[/Data Analysis] or Korean variant
DATA_ANALYSIS_PATTERN = re.compile(
    r"\[(?:Data Analysis|데이터 분석)\].*?\[/(?:Data Analysis|데이터 분석)\]",
    re.DOTALL,
)

# Markdown tables
MARKDOWN_TABLE_PATTERN = re.compile(
    r"(?:^|\n)(\|[^\n]+\|\n\|[-:|\s]+\|\n(?:\|[^\n]+\|(?:\n|$))+)"
)
MARKDOWN_TABLE_ROW_PATTERN = re.compile(r"\|[^\n]+\|")
MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"^\|[\s\-:]+\|[\s\-:|]*$", re.MULTILINE)
MARKDOWN_TABLE_HEADER_PATTERN = re.compile(r"^(\|[^\n]+\|\n)(\|[-:|\s]+\|)")

# Aggregated: all protected patterns (order = priority)
ALL_PROTECTED_PATTERNS: list[re.Pattern[str]] = [
    CHART_BLOCK_PATTERN,
    TEXTBOX_BLOCK_PATTERN,
    HTML_TABLE_PATTERN,
    METADATA_BLOCK_PATTERN,
    DATA_ANALYSIS_PATTERN,
    IMAGE_TAG_PATTERN,
    PAGE_TAG_PATTERN,
    SLIDE_TAG_PATTERN,
    SHEET_TAG_PATTERN,
]


# ============================================================================
# Table Chunking Thresholds
# ============================================================================

TABLE_WRAPPER_OVERHEAD: int = 30       # <table border='1'>\n</table>
ROW_OVERHEAD: int = 12                 # <tr>\n</tr>
CELL_OVERHEAD: int = 10                # <td></td> or <th></th>
CHUNK_INDEX_OVERHEAD: int = 30         # [Table chunk 1/10]\n
TABLE_SIZE_THRESHOLD_MULTIPLIER: float = 1.2  # 1.2× of chunk_size

# Extensions that are inherently table-based
TABLE_EXTENSIONS: FrozenSet[str] = frozenset({"csv", "tsv", "xlsx", "xls"})


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(frozen=True)
class TableRow:
    """A single table row (HTML or Markdown)."""

    html: str
    is_header: bool
    cell_count: int
    char_length: int


@dataclass(frozen=True)
class ParsedTable:
    """Parsed HTML table."""

    header_rows: List[TableRow]
    data_rows: List[TableRow]
    total_cols: int
    original_html: str
    header_html: str
    header_size: int


@dataclass(frozen=True)
class ParsedMarkdownTable:
    """Parsed Markdown table."""

    header_row: str
    separator_row: str
    data_rows: List[str]
    total_cols: int
    original_text: str
    header_text: str
    header_size: int


__all__ = [
    # Patterns
    "HTML_TABLE_PATTERN",
    "CHART_BLOCK_PATTERN",
    "TEXTBOX_BLOCK_PATTERN",
    "IMAGE_TAG_PATTERN",
    "PAGE_TAG_PATTERN",
    "SLIDE_TAG_PATTERN",
    "SHEET_TAG_PATTERN",
    "METADATA_BLOCK_PATTERN",
    "DATA_ANALYSIS_PATTERN",
    "MARKDOWN_TABLE_PATTERN",
    "MARKDOWN_TABLE_ROW_PATTERN",
    "MARKDOWN_TABLE_SEPARATOR_PATTERN",
    "MARKDOWN_TABLE_HEADER_PATTERN",
    "ALL_PROTECTED_PATTERNS",
    # Thresholds
    "TABLE_WRAPPER_OVERHEAD",
    "ROW_OVERHEAD",
    "CELL_OVERHEAD",
    "CHUNK_INDEX_OVERHEAD",
    "TABLE_SIZE_THRESHOLD_MULTIPLIER",
    "TABLE_EXTENSIONS",
    # Language mapping
    "CODE_LANGUAGE_MAP",
    # Dataclasses
    "TableRow",
    "ParsedTable",
    "ParsedMarkdownTable",
]
