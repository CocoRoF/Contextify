# libs/core/functions/encoding.py
"""
Encoding - Abstract Encoding Interface Module

Provides abstract base class and configuration for encoding operations.
Format-specific implementations should be placed in respective helper modules:
- DOC: doc_helpers/doc_encoding.py
- CSV: csv_helper/csv_encoding.py

Module Components:
- ENCODING_CANDIDATES: Common encoding candidates list
- EncodingConfig: Configuration dataclass for encoding operations  
- BaseEncoder: Abstract base class for format-specific encoders

Usage Example:
    # For format-specific encoding, use the appropriate encoder:
    from contextifier.core.processor.doc_helpers.doc_encoding import DOCEncoder
    from contextifier.core.processor.csv_helper.csv_encoding import read_file_with_encoding
    
    # For creating custom encoder:
    from contextifier.core.functions.encoding import BaseEncoder, EncodingConfig
    
    class MyEncoder(BaseEncoder):
        def decode(self, data: bytes) -> Tuple[str, str]:
            # Format-specific decoding logic
            pass
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger("document-processor")


# Common encoding candidates for various document types
# Can be used by format-specific encoders as reference
ENCODING_CANDIDATES = [
    'utf-8', 
    'utf-8-sig', 
    'cp949',      # Korean Windows
    'euc-kr',     # Korean legacy
    'utf-16-le',  # Word documents
    'utf-16-be',
    'cp1252',     # Western Windows
    'iso-8859-1', # Latin-1
    'latin-1',    # Alias for iso-8859-1
]


@dataclass
class EncodingConfig:
    """Configuration for encoding operations.
    
    Base configuration class that can be extended by format-specific configs.
    Provides common settings for encoding detection and decoding.
    
    Attributes:
        preferred_encoding: Preferred encoding to try first
        use_chardet: Whether to use chardet library for detection
        chardet_confidence_threshold: Minimum confidence for chardet detection
        encoding_candidates: List of encodings to try in order
        fallback_encoding: Final fallback encoding (should never fail)
    """
    preferred_encoding: Optional[str] = None
    use_chardet: bool = True
    chardet_confidence_threshold: float = 0.7
    encoding_candidates: List[str] = field(default_factory=lambda: ENCODING_CANDIDATES.copy())
    fallback_encoding: str = 'latin-1'


class BaseEncoder(ABC):
    """Abstract base class for format-specific encoders.
    
    Each document format may have specific encoding requirements.
    Implement subclasses for format-specific decoding logic.
    
    Implementations:
        - DOCEncoder: doc_helpers/doc_encoding.py (DOC binary format, UTF-16LE)
        - CSV: csv_helper/csv_encoding.py (text format, various encodings)
    
    Implementation Guidelines:
        - Override decode() for format-specific decoding
        - Override supports_format() to declare supported formats
        - Extend EncodingConfig for format-specific settings
        - Handle format-specific byte patterns and structure
    
    Example:
        class MyFormatEncoder(BaseEncoder):
            def decode(self, data: bytes) -> Tuple[str, str]:
                # 1. Detect encoding (format-specific logic)
                # 2. Decode binary to text
                # 3. Return (text, encoding_name)
                return decoded_text, "utf-8"
            
            def supports_format(self, format_type: str) -> bool:
                return format_type.lower() == "myformat"
    """
    
    def __init__(self, config: Optional[EncodingConfig] = None):
        """Initialize the encoder.
        
        Args:
            config: Encoding configuration
        """
        self.config = config or EncodingConfig()
        self.logger = logging.getLogger("document-processor")
    
    @abstractmethod
    def decode(self, data: bytes) -> Tuple[str, str]:
        """Decode binary data to string.
        
        Args:
            data: Binary data to decode
            
        Returns:
            Tuple of (decoded_text, detected_encoding)
        """
        pass
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this encoder supports the given format.
        
        Args:
            format_type: Format identifier (e.g., "doc", "csv")
            
        Returns:
            True if format is supported
        """
        return False


# Default configuration instance
DEFAULT_ENCODING_CONFIG = EncodingConfig()
