# contextifier/core/processor/ole_helper/ole_table_processor.py
"""
OLE Table Processor - OLE Format-Specific Table Processing

Handles conversion of OLE/DOC tables to various output formats (HTML, Text).
Works with OLETableExtractor to extract and format tables.

Processor Responsibilities:
- Full document processing with table detection and HTML conversion
- Text cleaning and noise removal
- Post-processing of extracted content

Usage:
    from contextifier.core.processor.ole_helper.ole_table_processor import (
        OLETableProcessor,
        extract_content_with_tables_from_word_stream,
    )

    processor = OLETableProcessor()
    html = processor.extract_content_with_tables(word_stream_data)
"""
import logging
import re
from typing import Optional

from contextifier.core.functions.table_processor import (
    TableProcessor,
    TableProcessorConfig,
)

logger = logging.getLogger("document-processor")


class OLETableProcessor:
    """OLE format-specific table processor.
    
    Handles conversion of OLE/DOC document content with tables to HTML.
    Uses OLETableExtractor for extraction and TableProcessor for formatting.
    
    Usage:
        processor = OLETableProcessor()
        html = processor.extract_content_with_tables(word_stream_data)
    """
    
    def __init__(self, config: Optional[TableProcessorConfig] = None):
        """Initialize OLE table processor.
        
        Args:
            config: Table processing configuration
        """
        self.config = config or TableProcessorConfig()
        self._table_processor = TableProcessor(self.config)
        self.logger = logging.getLogger("document-processor")
    
    def extract_content_with_tables(self, content: bytes) -> str:
        """Extract full content with tables converted to HTML.
        
        Main method for DOC processing. Detects tables and extracts
        them as HTML while preserving non-table content as plain text.
        
        Args:
            content: Raw bytes from WordDocument stream
            
        Returns:
            Extracted text with tables in HTML format
        """
        # Import here to avoid circular imports
        from contextifier.core.processor.ole_helper.ole_table_extractor import (
            OLETableExtractor,
            OLETableExtractorConfig,
        )
        from contextifier.core.processor.ole_helper.ole_encoding import OLEEncoder
        
        extractor = OLETableExtractor()
        encoder = OLEEncoder()
        
        # Detect table regions
        regions = extractor.detect_table_regions(content)
        
        if not regions:
            # No tables - extract as plain text
            text, _ = encoder.decode(content)
            return text
        
        # Sort regions by offset
        regions.sort(key=lambda r: r.start_offset)
        
        result_parts = []
        current_pos = 0
        
        for region in regions:
            # Add text before this table
            if region.start_offset > current_pos:
                text_segment = content[current_pos:region.start_offset]
                text, _ = encoder.decode(text_segment)
                if text.strip():
                    result_parts.append(text.strip())
            
            # Extract and format table
            if region.is_confident(extractor._config.confidence_threshold):
                table = extractor.extract_table_from_region(content, region)
                if table and table.is_valid(extractor._config.min_rows, extractor._config.min_cols):
                    table_html = self._table_processor.format_table(table)
                    result_parts.append(f"\n{table_html}\n")
            
            current_pos = region.end_offset + 1
        
        # Add remaining text after last table
        if current_pos < len(content):
            text_segment = content[current_pos:]
            text, _ = encoder.decode(text_segment)
            if text.strip():
                result_parts.append(text.strip())
        
        result = "\n".join(result_parts)
        
        # Post-process to clean remaining noise
        result = self._clean_extracted_text(result)
        
        return result
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text by removing noise patterns.
        
        Args:
            text: Extracted text that may contain noise
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are mostly noise
            if self._is_noise_line(line):
                continue
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        
        # Remove excessive blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def _is_noise_line(self, line: str) -> bool:
        """Check if a line is likely noise.
        
        Args:
            line: Line to check
            
        Returns:
            True if line appears to be noise
        """
        if not line or not line.strip():
            return False  # Keep empty lines
        
        stripped = line.strip()
        
        # Keep HTML table tags
        if stripped.startswith('<table') or stripped.startswith('</table'):
            return False
        if stripped.startswith('<tr') or stripped.startswith('</tr'):
            return False
        if stripped.startswith('<td') or stripped.startswith('</td'):
            return False
        if stripped.startswith('<th') or stripped.startswith('</th'):
            return False
        
        # Keep HYPERLINK lines
        if 'HYPERLINK' in stripped or stripped.startswith('http'):
            return False
        
        # Pattern: repeated Korean characters like "폪폪폪폪폪폪폪폪" or "뫌뫌뫌"
        # Normal Korean rarely has 3+ consecutive identical characters
        if re.search(r'(.)\1{2,}', stripped):
            # Check if repeated char is Korean
            match = re.search(r'(.)\1{2,}', stripped)
            if match:
                repeated_char = match.group(1)
                code = ord(repeated_char)
                if 0xAC00 <= code <= 0xD7AF:  # Korean syllable
                    # Korean words rarely have 3+ repeated syllables
                    return True
        
        # Pattern: alternating digit + Korean like "6뀀6됀6똀6저6쨀6찀6퐀6"
        if re.match(r'^[0-9][^\x00-\x7F][0-9]', stripped):
            alternating = True
            for i, char in enumerate(stripped):
                if i % 2 == 0:
                    if not char.isdigit():
                        alternating = False
                        break
            if alternating and len(stripped) >= 4:
                return True
        
        # Pattern: lines with colons/semicolons alternating with non-ASCII
        if re.match(r'^[:;<>=][^\x00-\x7F]', stripped):
            punct_count = len(re.findall(r'[:;<>=]', stripped))
            if punct_count >= len(stripped) * 0.3:
                return True
        
        # Pattern: lines with lots of numbers and symbols with scattered non-ASCII
        if re.match(r'^[0-9:<>=;?@\s]*[^\x00-\x7F]+[0-9:<>=;?@\s]*$', stripped):
            digit_symbol_count = len(re.findall(r'[0-9:<>=;?@\s]', stripped))
            if digit_symbol_count > len(stripped) * 0.4:
                return True
        
        # Count character types
        ascii_count = 0
        korean_count = 0
        rare_korean_count = 0
        rare_count = 0
        digit_count = 0
        jamo_count = 0
        punct_count = 0
        
        for char in stripped:
            code = ord(char)
            if 0x30 <= code <= 0x39:
                digit_count += 1
                ascii_count += 1
            elif char in ':<>=;?@':
                punct_count += 1
                ascii_count += 1
            elif 0x20 <= code <= 0x7E:
                ascii_count += 1
            elif 0xAC00 <= code <= 0xD7AF:
                korean_count += 1
                if char in '뀀됀똀퐀찀쨀숀츀퀀':
                    rare_korean_count += 1
            elif 0x1100 <= code <= 0x11FF:
                jamo_count += 1
            elif 0x3130 <= code <= 0x318F:
                jamo_count += 1
            elif code > 0x007F:
                if not (0xAC00 <= code <= 0xD7AF):
                    rare_count += 1
        
        total = len(stripped)
        if total == 0:
            return False
        
        # Pattern: Korean chars that are known noise patterns
        if rare_korean_count > korean_count * 0.5 and rare_korean_count >= 2:
            return True
        
        # Pattern: digit-korean alternating pattern
        if digit_count >= 2 and (korean_count > 0 or rare_count > 0):
            non_digit_non_space = [c for c in stripped if not c.isdigit() and not c.isspace()]
            if len(non_digit_non_space) <= digit_count and total > 4:
                return True
        
        # Very short lines with only rare chars
        if total <= 3 and rare_count > 0:
            return True
        
        # Lines with isolated Jamo
        if jamo_count > korean_count and jamo_count > 2:
            return True
        
        # Lines that are mostly rare Unicode
        if rare_count / total > 0.4:
            return True
        
        # Short lines with high noise ratio
        if total <= 10 and rare_count / total > 0.25:
            return True
        
        # Lines that are mostly digits/symbols
        if total > 5 and (digit_count + rare_count) / total > 0.6 and korean_count == 0:
            return True
        
        # Pattern: single rare char lines
        if total == 1 and rare_count == 1:
            return True
        
        return False


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def extract_content_with_tables_from_word_stream(
    data: bytes,
    config: Optional[TableProcessorConfig] = None
) -> str:
    """Convenience function to extract content with tables as HTML.
    
    Args:
        data: Raw bytes from WordDocument stream
        config: Optional processing config
        
    Returns:
        Text content with tables in HTML format
    """
    processor = OLETableProcessor(config)
    return processor.extract_content_with_tables(data)


# Backward compatibility alias
DOCTableProcessor = OLETableProcessor


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main class
    'OLETableProcessor',
    # Backward compatibility
    'DOCTableProcessor',
    'extract_content_with_tables_from_word_stream',
]
