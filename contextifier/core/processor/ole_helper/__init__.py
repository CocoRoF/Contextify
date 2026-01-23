# contextifier/core/processor/ole_helper/__init__.py
"""
OLE Helper Module

Contains utilities for processing OLE Compound Document format DOC files
(Microsoft Word 97-2003 binary format).

This module handles:
- OLE-specific encoding detection and decoding (FIB parsing, single-byte/UTF-16LE)
- OLE-specific table extraction (0x07 cell markers, sprmTDefTable)
- OLE-specific table processing (TableData → HTML conversion)
- Binary structure parsing specific to OLE DOC format

Note: This module is separate from the generic doc_helpers because:
- OLE DOC is just one of 5 DOC format types (RTF, OLE, HTML, DOCX, UNKNOWN)
- RTF and HTML DOC files don't use these binary structures
- DOCX uses XML-based structures (handled by docx_helper)

Usage:
    from contextifier.core.processor.ole_helper import (
        OLEEncoder,
        OLETableExtractor,
        OLETableProcessor,
        OLETablePropertiesParser,
    )
    
    # Or use backward compatibility aliases:
    from contextifier.core.processor.ole_helper import (
        DOCEncoder,          # Alias for OLEEncoder
        DOCTableExtractor,   # Alias for OLETableExtractor
        DOCTableProcessor,   # Alias for OLETableProcessor
    )
"""

# File converter
from contextifier.core.processor.ole_helper.ole_file_converter import (
    OLEFileConverter,
)

# Image processor
from contextifier.core.processor.ole_helper.ole_image_processor import (
    OLEImageProcessor,
)

# Preprocessor
from contextifier.core.processor.ole_helper.ole_preprocessor import (
    OLEPreprocessor,
)

# Encoding utilities
from contextifier.core.processor.ole_helper.ole_encoding import (
    # Main classes (OLE-prefixed)
    OLEEncoder,
    OLEEncodingType,
    OLEEncodingInfo,
    OLEEncodingConfig,
    # Convenience functions
    detect_ole_encoding,
    decode_word_stream,
    decode_cell_content,
    get_supported_codepages,
    # Backward compatibility aliases (DOC-prefixed)
    DOCEncoder,
    DOCEncodingType,
    DOCEncodingInfo,
    DOCEncodingConfig,
    detect_doc_encoding,
)

# Table extraction utilities (Extractor: bytes → TableData)
from contextifier.core.processor.ole_helper.ole_table_extractor import (
    # Main classes (OLE-prefixed)
    OLETableExtractor,
    OLETableExtractorConfig,
    CellMarkerInfo,
    # Convenience functions
    detect_tables_in_word_stream,
    extract_tables_from_word_stream,
    extract_content_with_tables_from_word_stream,
    # Backward compatibility aliases (DOC-prefixed)
    DOCTableExtractor,
    DOCTableExtractorConfig,
)

# Table processing utilities (Processor: TableData → HTML)
from contextifier.core.processor.ole_helper.ole_table_processor import (
    # Main classes (OLE-prefixed)
    OLETableProcessor,
    # Backward compatibility alias
    DOCTableProcessor,
)

# Table properties parser
from contextifier.core.processor.ole_helper.ole_table_properties import (
    # Main classes (OLE-prefixed)
    OLETablePropertiesParser,
    TCProperties,
    RowDefinition,
    TableStructure,
    # Convenience functions
    extract_table_merge_info,
    # Backward compatibility alias
    DOCTablePropertiesParser,
)

__all__ = [
    # OLE File Converter
    "OLEFileConverter",
    # OLE Image Processor
    "OLEImageProcessor",
    # OLE Preprocessor
    "OLEPreprocessor",
    # OLE Encoding (primary names)
    "OLEEncoder",
    "OLEEncodingType",
    "OLEEncodingInfo",
    "OLEEncodingConfig",
    "detect_ole_encoding",
    "decode_word_stream",
    "decode_cell_content",
    "get_supported_codepages",
    # OLE Encoding (backward compatibility)
    "DOCEncoder",
    "DOCEncodingType",
    "DOCEncodingInfo",
    "DOCEncodingConfig",
    "detect_doc_encoding",
    # OLE Table Extraction (primary names)
    "OLETableExtractor",
    "OLETableExtractorConfig",
    "CellMarkerInfo",
    "detect_tables_in_word_stream",
    "extract_tables_from_word_stream",
    "extract_content_with_tables_from_word_stream",
    # OLE Table Extraction (backward compatibility)
    "DOCTableExtractor",
    "DOCTableExtractorConfig",
    # OLE Table Processing (primary names)
    "OLETableProcessor",
    # OLE Table Processing (backward compatibility)
    "DOCTableProcessor",
    # OLE Table Properties (primary names)
    "OLETablePropertiesParser",
    "TCProperties",
    "RowDefinition",
    "TableStructure",
    "extract_table_merge_info",
    # OLE Table Properties (backward compatibility)
    "DOCTablePropertiesParser",
]
