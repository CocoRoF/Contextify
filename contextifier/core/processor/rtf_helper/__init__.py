# libs/core/processor/rtf_helper/__init__.py
"""
RTF Helper Module

Provides utilities for RTF document processing.

Module structure:
- rtf_constants: RTF-related constants
- rtf_models: RTF data models (RTFCellInfo, RTFTable, RTFContentPart, RTFDocument)
- rtf_parser: RTF parsing
- rtf_decoder: RTF encoding/decoding utilities
- rtf_content_extractor: RTF content extraction
- rtf_table_extractor: RTF table extraction
- rtf_metadata_extractor: RTF metadata extraction
- rtf_region_finder: RTF region finding
- rtf_text_cleaner: RTF text cleaning
- rtf_file_converter: RTF file converter
- rtf_image_processor: RTF image processor (includes binary preprocessing)
"""

# RTF Helper Components

# Constants
from contextifier.core.processor.rtf_helper.rtf_constants import (
    SHAPE_PROPERTY_NAMES,
    EXCLUDE_DESTINATION_KEYWORDS,
    SKIP_DESTINATIONS,
    IMAGE_DESTINATIONS,
    CODEPAGE_ENCODING_MAP,
    DEFAULT_ENCODINGS,
)

# Models
from contextifier.core.processor.rtf_helper.rtf_models import (
    RTFCellInfo,
    RTFTable,
    RTFContentPart,
    RTFDocument,
)

# Parser
from contextifier.core.processor.rtf_helper.rtf_parser import (
    RTFParser,
    parse_rtf,
)

# Decoder
from contextifier.core.processor.rtf_helper.rtf_decoder import (
    detect_encoding,
    decode_content,
    decode_hex_escapes,
)

# Content Extractor
from contextifier.core.processor.rtf_helper.rtf_content_extractor import (
    extract_inline_content,
    extract_text_legacy,
)

# Table Extractor
from contextifier.core.processor.rtf_helper.rtf_table_extractor import (
    extract_tables_with_positions,
)

# Region Finder
from contextifier.core.processor.rtf_helper.rtf_region_finder import (
    find_excluded_regions,
    is_in_excluded_region,
)

# Text Cleaner
from contextifier.core.processor.rtf_helper.rtf_text_cleaner import (
    clean_rtf_text,
    remove_shprslt_blocks,
)

# Image Processor (includes binary preprocessing)
from contextifier.core.processor.rtf_helper.rtf_image_processor import (
    RTFImageProcessor,
    preprocess_rtf_binary,
)

# Metadata
from contextifier.core.processor.rtf_helper.rtf_metadata_extractor import (
    DOCMetadataExtractor as RTFMetadataExtractor,
    RTFSourceInfo,
)

__all__ = [
    # Constants
    'SHAPE_PROPERTY_NAMES',
    'EXCLUDE_DESTINATION_KEYWORDS',
    'SKIP_DESTINATIONS',
    'IMAGE_DESTINATIONS',
    'CODEPAGE_ENCODING_MAP',
    'DEFAULT_ENCODINGS',
    # Models
    'RTFCellInfo',
    'RTFTable',
    'RTFContentPart',
    'RTFDocument',
    # Parser
    'RTFParser',
    'parse_rtf',
    # Decoder
    'detect_encoding',
    'decode_content',
    'decode_hex_escapes',
    # Content Extractor
    'extract_inline_content',
    'extract_text_legacy',
    # Table Extractor
    'extract_tables_with_positions',
    # Region Finder
    'find_excluded_regions',
    'is_in_excluded_region',
    # Text Cleaner
    'clean_rtf_text',
    'remove_shprslt_blocks',
    # Image Processor
    'RTFImageProcessor',
    'preprocess_rtf_binary',
    # Metadata
    'RTFMetadataExtractor',
    'RTFSourceInfo',
]
