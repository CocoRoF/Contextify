# libs/core/processor/doc_helpers/__init__.py
"""
DOC Helper 모듈

DOC 문서 처리에 필요한 유틸리티를 제공합니다.

RTF 관련 모듈들은 rtf_helper로 이동했습니다.
RTF 처리가 필요한 경우 rtf_helper를 사용하세요:
    from contextifier.core.processor import rtf_helper
    from contextifier.core.processor.rtf_helper import RTFParser

모듈 구성:
- doc_file_converter: DOC 파일 변환기 (모든 DOC 포맷 공통)
- doc_image_processor: DOC 이미지 처리기 (모든 DOC 포맷 공통)

OLE 포맷 전용 모듈은 별도 폴더로 분리되었습니다:
    from contextifier.core.processor import ole_helper
    from contextifier.core.processor.ole_helper import OLEEncoder

Note: DOC 파일은 5가지 형식이 있습니다 (RTF, OLE, HTML, DOCX, UNKNOWN)
      ole_helper 폴더는 doc_helpers와 같은 레벨에 있으며 OLE 형식 전용 코드를 담고 있습니다.
      doc_file_converter와 doc_image_processor는 모든 형식에 공통으로 적용됩니다.
"""

# Generic DOC components (applicable to all DOC formats)
from contextifier.core.processor.doc_helpers.doc_file_converter import DOCFileConverter
from contextifier.core.processor.doc_helpers.doc_image_processor import DOCImageProcessor

# OLE-specific components (from ole_helper - same level as doc_helpers)
# Re-exported here for backward compatibility
from contextifier.core.processor.ole_helper import (
    # Encoding (OLE-specific)
    OLEEncoder,
    OLEEncodingConfig,
    OLEEncodingType,
    OLEEncodingInfo,
    detect_ole_encoding,
    get_supported_codepages,
    decode_word_stream,
    decode_cell_content,
    # Table extraction (OLE-specific)
    OLETableExtractor,
    OLETableExtractorConfig,
    CellMarkerInfo,
    detect_tables_in_word_stream,
    extract_tables_from_word_stream,
    extract_content_with_tables_from_word_stream,
    # Table properties (OLE-specific)
    OLETablePropertiesParser,
    extract_table_merge_info,
    TCProperties,
    RowDefinition,
    TableStructure,
    # Backward compatibility aliases (DOC-prefixed names)
    DOCEncoder,
    DOCEncodingConfig,
    DOCEncodingType,
    DOCEncodingInfo,
    detect_doc_encoding,
    DOCTableExtractor,
    DOCTableExtractorConfig,
    DOCTablePropertiesParser,
)

__all__ = [
    # Generic DOC components
    'DOCFileConverter',
    'DOCImageProcessor',
    # OLE Encoding (OLE-prefixed - new names)
    'OLEEncoder',
    'OLEEncodingConfig',
    'OLEEncodingType',
    'OLEEncodingInfo',
    'detect_ole_encoding',
    'get_supported_codepages',
    'decode_word_stream',
    'decode_cell_content',
    # OLE Table Extraction (OLE-prefixed - new names)
    'OLETableExtractor',
    'OLETableExtractorConfig',
    'CellMarkerInfo',
    'detect_tables_in_word_stream',
    'extract_tables_from_word_stream',
    'extract_content_with_tables_from_word_stream',
    # OLE Table Properties (OLE-prefixed - new names)
    'OLETablePropertiesParser',
    'extract_table_merge_info',
    'TCProperties',
    'RowDefinition',
    'TableStructure',
    # Backward compatibility aliases (DOC-prefixed - legacy names)
    'DOCEncoder',
    'DOCEncodingConfig',
    'DOCEncodingType',
    'DOCEncodingInfo',
    'detect_doc_encoding',
    'DOCTableExtractor',
    'DOCTableExtractorConfig',
    'DOCTablePropertiesParser',
]
