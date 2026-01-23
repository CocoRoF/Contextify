# contextifier/core/processor/ole_helper/ole_encoding.py
"""
OLE Encoding - OLE Format-Specific Encoding Utilities

Provides encoding utilities specifically for OLE Compound Documents (Word 97-2003).
Supports multiple encodings based on FIB (File Information Block) flags.

DOC Binary Format Encoding Types:
1. Double-byte encodings (fFarEast=True):
   - UTF-16LE: Universal Unicode encoding
   - Text stored as 2 bytes per character

2. Single-byte encodings (fFarEast=False):
   - CP1252: Western European (English, French, German, etc.)
   - CP1250: Central European (Polish, Czech, Hungarian, etc.)
   - CP1251: Cyrillic (Russian, Ukrainian, etc.)
   - CP1253: Greek
   - CP1254: Turkish
   - CP1255: Hebrew
   - CP1256: Arabic
   - CP1257: Baltic (Lithuanian, Latvian, etc.)
   - CP874: Thai
   
3. Multi-byte CJK encodings (alternative to UTF-16LE):
   - CP949: Korean (Extended Wansung)
   - CP932: Japanese (Shift-JIS)
   - CP936: Simplified Chinese (GBK)
   - CP950: Traditional Chinese (Big5)

FIB Flags for Encoding Detection:
- fFarEast (offset 0x0A, bit 8): East Asian language flag
- lid (offset 0x06): Language ID for codepage selection
- lidFE (offset 0x1C6): Far East language ID

Usage Example:
    from contextifier.core.processor.ole_helper.ole_encoding import (
        OLEEncoder,
        decode_word_stream,
    )

    encoder = OLEEncoder()
    text = encoder.decode(word_stream_data)  # Auto-detects encoding
"""
import logging
import re
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict

from contextifier.core.functions.encoding import BaseEncoder, EncodingConfig

logger = logging.getLogger("document-processor")


# Language ID to Codepage mapping
# Reference: https://docs.microsoft.com/en-us/openspecs/office_standards/ms-oe376
LANG_ID_TO_CODEPAGE: Dict[int, str] = {
    # Western European
    0x0409: 'cp1252',  # English (US)
    0x0809: 'cp1252',  # English (UK)
    0x0C09: 'cp1252',  # English (Australia)
    0x1009: 'cp1252',  # English (Canada)
    0x1409: 'cp1252',  # English (New Zealand)
    0x0407: 'cp1252',  # German (Germany)
    0x0807: 'cp1252',  # German (Switzerland)
    0x0C07: 'cp1252',  # German (Austria)
    0x040C: 'cp1252',  # French (France)
    0x080C: 'cp1252',  # French (Belgium)
    0x100C: 'cp1252',  # French (Switzerland)
    0x0410: 'cp1252',  # Italian (Italy)
    0x0810: 'cp1252',  # Italian (Switzerland)
    0x0C0A: 'cp1252',  # Spanish (Spain)
    0x080A: 'cp1252',  # Spanish (Mexico)
    0x0413: 'cp1252',  # Dutch (Netherlands)
    0x0813: 'cp1252',  # Dutch (Belgium)
    0x0416: 'cp1252',  # Portuguese (Brazil)
    0x0816: 'cp1252',  # Portuguese (Portugal)
    0x041D: 'cp1252',  # Swedish
    0x0414: 'cp1252',  # Norwegian (Bokmål)
    0x0814: 'cp1252',  # Norwegian (Nynorsk)
    0x0406: 'cp1252',  # Danish
    0x040B: 'cp1252',  # Finnish
    0x040F: 'cp1252',  # Icelandic
    0x0421: 'cp1252',  # Indonesian
    0x043E: 'cp1252',  # Malay
    
    # Central European
    0x0415: 'cp1250',  # Polish
    0x0405: 'cp1250',  # Czech
    0x040E: 'cp1250',  # Hungarian
    0x041B: 'cp1250',  # Slovak
    0x0424: 'cp1250',  # Slovenian
    0x041A: 'cp1250',  # Croatian
    0x0418: 'cp1250',  # Romanian
    0x0C1A: 'cp1250',  # Serbian (Latin)
    
    # Cyrillic
    0x0419: 'cp1251',  # Russian
    0x0422: 'cp1251',  # Ukrainian
    0x0423: 'cp1251',  # Belarusian
    0x0402: 'cp1251',  # Bulgarian
    0x081A: 'cp1251',  # Serbian (Cyrillic)
    0x042F: 'cp1251',  # Macedonian
    
    # Greek
    0x0408: 'cp1253',  # Greek
    
    # Turkish
    0x041F: 'cp1254',  # Turkish
    0x042C: 'cp1254',  # Azerbaijani (Latin)
    
    # Hebrew
    0x040D: 'cp1255',  # Hebrew
    
    # Arabic
    0x0401: 'cp1256',  # Arabic (Saudi Arabia)
    0x0801: 'cp1256',  # Arabic (Iraq)
    0x0C01: 'cp1256',  # Arabic (Egypt)
    0x1001: 'cp1256',  # Arabic (Libya)
    0x1401: 'cp1256',  # Arabic (Algeria)
    0x1801: 'cp1256',  # Arabic (Morocco)
    0x1C01: 'cp1256',  # Arabic (Tunisia)
    0x2001: 'cp1256',  # Arabic (Oman)
    0x2401: 'cp1256',  # Arabic (Yemen)
    0x2801: 'cp1256',  # Arabic (Syria)
    0x2C01: 'cp1256',  # Arabic (Jordan)
    0x3001: 'cp1256',  # Arabic (Lebanon)
    0x3401: 'cp1256',  # Arabic (Kuwait)
    0x3801: 'cp1256',  # Arabic (UAE)
    0x3C01: 'cp1256',  # Arabic (Bahrain)
    0x4001: 'cp1256',  # Arabic (Qatar)
    0x0429: 'cp1256',  # Persian
    0x0420: 'cp1256',  # Urdu
    
    # Baltic
    0x0427: 'cp1257',  # Lithuanian
    0x0426: 'cp1257',  # Latvian
    0x0425: 'cp1257',  # Estonian
    
    # Thai
    0x041E: 'cp874',   # Thai
    
    # Vietnamese
    0x042A: 'cp1258',  # Vietnamese
    
    # East Asian (CJK) - These use different handling
    0x0412: 'cp949',   # Korean
    0x0411: 'cp932',   # Japanese
    0x0804: 'cp936',   # Chinese (Simplified)
    0x1004: 'cp936',   # Chinese (Singapore)
    0x0404: 'cp950',   # Chinese (Traditional, Taiwan)
    0x0C04: 'cp950',   # Chinese (Hong Kong)
    0x1404: 'cp950',   # Chinese (Macau)
}

# Codepage primary language (first byte of lid)
PRIMARY_LANG_TO_CODEPAGE: Dict[int, str] = {
    0x09: 'cp1252',  # English
    0x07: 'cp1252',  # German
    0x0C: 'cp1252',  # French
    0x10: 'cp1252',  # Italian
    0x0A: 'cp1252',  # Spanish
    0x13: 'cp1252',  # Dutch
    0x16: 'cp1252',  # Portuguese
    0x1D: 'cp1252',  # Swedish
    0x14: 'cp1252',  # Norwegian
    0x06: 'cp1252',  # Danish
    0x0B: 'cp1252',  # Finnish
    0x15: 'cp1250',  # Polish
    0x05: 'cp1250',  # Czech
    0x0E: 'cp1250',  # Hungarian
    0x1B: 'cp1250',  # Slovak
    0x24: 'cp1250',  # Slovenian
    0x1A: 'cp1250',  # Croatian/Serbian
    0x18: 'cp1250',  # Romanian
    0x19: 'cp1251',  # Russian
    0x22: 'cp1251',  # Ukrainian
    0x02: 'cp1251',  # Bulgarian
    0x08: 'cp1253',  # Greek
    0x1F: 'cp1254',  # Turkish
    0x0D: 'cp1255',  # Hebrew
    0x01: 'cp1256',  # Arabic
    0x29: 'cp1256',  # Persian
    0x27: 'cp1257',  # Lithuanian
    0x26: 'cp1257',  # Latvian
    0x1E: 'cp874',   # Thai
    0x2A: 'cp1258',  # Vietnamese
    0x12: 'cp949',   # Korean
    0x11: 'cp932',   # Japanese
    0x04: 'cp936',   # Chinese
}

# Unicode ranges for Korean characters
KOREAN_SYLLABLE_START = 0xAC00  # 가
KOREAN_SYLLABLE_END = 0xD7AF    # 힣
KOREAN_JAMO_START = 0x1100
KOREAN_JAMO_END = 0x11FF
KOREAN_COMPAT_JAMO_START = 0x3130
KOREAN_COMPAT_JAMO_END = 0x318F

# CJK ranges
CJK_UNIFIED_START = 0x4E00
CJK_UNIFIED_END = 0x9FFF


class OLEEncodingType(Enum):
    """OLE file encoding types."""
    UTF16LE = "utf-16-le"      # Double-byte Unicode
    SINGLE_BYTE = "single"     # Single-byte codepage (CP1252, CP1251, etc.)
    MULTI_BYTE_CJK = "cjk"     # Multi-byte CJK (CP949, CP932, CP936, CP950)


@dataclass
class OLEEncodingInfo:
    """Information about OLE file encoding.
    
    Attributes:
        encoding_type: Type of encoding (UTF16LE, SINGLE_BYTE, MULTI_BYTE_CJK)
        codepage: Detected codepage name (e.g., 'cp1252', 'utf-16-le')
        lid: Language ID from FIB
        is_far_east: fFarEast flag value
        bytes_per_char: Bytes per character (1, 2, or variable)
        is_complex: Whether document uses complex script features
    """
    encoding_type: OLEEncodingType
    codepage: str
    lid: int = 0
    is_far_east: bool = False
    bytes_per_char: int = 2
    is_complex: bool = False


@dataclass
class OLEEncodingConfig(EncodingConfig):
    """Configuration for OLE encoding operations.
    
    Attributes:
        skip_control_chars: Whether to skip DOC control characters
        preserve_line_breaks: Whether to preserve line break characters
        min_text_length: Minimum length for text segments
        fallback_encodings: List of encodings to try if auto-detection fails
        force_encoding: Force a specific encoding (None for auto-detection)
    """
    skip_control_chars: bool = True
    preserve_line_breaks: bool = True
    min_text_length: int = 2
    fallback_encodings: List[str] = field(default_factory=lambda: [
        'utf-16-le', 'cp1252', 'cp949', 'cp932', 'cp936', 'cp950',
        'cp1250', 'cp1251', 'cp1253', 'cp1254', 'cp1256'
    ])
    force_encoding: Optional[str] = None


class OLEEncoder(BaseEncoder):
    """OLE format-specific encoder with automatic encoding detection.
    
    Handles text decoding for Word 97-2003 documents by:
    1. Parsing FIB (File Information Block) to detect encoding
    2. Using appropriate decoder based on detected encoding
    3. Falling back to heuristic detection if FIB parsing fails
    
    Supported encoding types:
    - UTF-16LE: Default for East Asian documents (fFarEast=True)
    - Single-byte codepages: CP1252, CP1250, CP1251, etc.
    - Multi-byte CJK: CP949, CP932, CP936, CP950
    """
    
    def __init__(self, config: Optional[OLEEncodingConfig] = None):
        """Initialize OLE encoder.
        
        Args:
            config: OLE encoding configuration
        """
        super().__init__(config or OLEEncodingConfig())
        self._ole_config = config or OLEEncodingConfig()
        self._encoding_info: Optional[OLEEncodingInfo] = None
        self._fib_parsed = False
    
    def decode(self, data: bytes) -> Tuple[str, str]:
        """Decode OLE binary data to text with automatic encoding detection.
        
        Process:
        1. Parse FIB to detect document encoding
        2. Select appropriate extraction method based on encoding
        3. Extract and clean text
        
        Args:
            data: Binary data from WordDocument stream
            
        Returns:
            Tuple of (decoded_text, encoding_name)
        """
        if not data:
            return "", "utf-16-le"
        
        # Force encoding if configured
        if self._ole_config.force_encoding:
            encoding = self._ole_config.force_encoding
            text = self._extract_with_encoding(data, encoding)
            return text, encoding
        
        # Detect encoding from FIB
        self._encoding_info = self._detect_encoding_from_fib(data)
        
        # Extract text based on encoding type
        if self._encoding_info.encoding_type == OLEEncodingType.UTF16LE:
            text = self._extract_text_utf16le(data)
        elif self._encoding_info.encoding_type == OLEEncodingType.SINGLE_BYTE:
            text = self._extract_text_single_byte(data, self._encoding_info.codepage)
        else:  # MULTI_BYTE_CJK
            text = self._extract_text_multibyte_cjk(data, self._encoding_info.codepage)
        
        # If extraction failed, try fallback encodings
        if not text or self._is_garbled(text):
            text, encoding = self._try_fallback_encodings(data)
            return text, encoding
        
        return text, self._encoding_info.codepage
    
    def _detect_encoding_from_fib(self, data: bytes) -> OLEEncodingInfo:
        """Detect encoding from FIB (File Information Block).
        
        Uses a multi-strategy approach:
        1. Read FIB flags (fFarEast, lid)
        2. Sample text area bytes to verify encoding
        3. Use fallback decoding with quality scoring
        
        Args:
            data: WordDocument stream data
            
        Returns:
            OLEEncodingInfo with detected encoding details
        """
        # Default: UTF-16LE (safest fallback for modern documents)
        default_info = OLEEncodingInfo(
            encoding_type=OLEEncodingType.UTF16LE,
            codepage='utf-16-le',
            bytes_per_char=2
        )
        
        if len(data) < 0x60:  # Minimum FIB size
            return default_info
        
        try:
            # Check Word signature
            w_ident = struct.unpack('<H', data[0:2])[0]
            if w_ident not in (0xA5EC, 0xA5DC):
                logger.warning(f"Invalid Word signature: {hex(w_ident)}")
                return default_info
            
            # Read lid (Language ID) at offset 0x06
            lid = struct.unpack('<H', data[0x06:0x08])[0]
            
            # Read flags at offset 0x0A
            flags = struct.unpack('<H', data[0x0A:0x0C])[0]
            is_far_east = bool(flags & 0x0100)  # Bit 8
            is_complex = bool(flags & 0x0004)   # Bit 2
            
            logger.debug(
                f"FIB encoding info: lid={hex(lid)}, "
                f"fFarEast={is_far_east}, fComplex={is_complex}"
            )
            
            # Read text area boundaries
            fc_min = struct.unpack('<I', data[0x18:0x1C])[0]
            ccp_text = struct.unpack('<I', data[0x4C:0x50])[0]
            
            # Use trial decoding approach - most reliable method
            detected_encoding = self._detect_encoding_by_trial(
                data, fc_min, ccp_text, is_far_east, lid
            )
            
            if detected_encoding:
                return detected_encoding
            
            # Fall back to FIB-based detection
            if is_far_east:
                return OLEEncodingInfo(
                    encoding_type=OLEEncodingType.UTF16LE,
                    codepage='utf-16-le',
                    lid=lid,
                    is_far_east=True,
                    bytes_per_char=2
                )
            else:
                codepage = self._get_codepage_for_lid(lid)
                return OLEEncodingInfo(
                    encoding_type=OLEEncodingType.SINGLE_BYTE,
                    codepage=codepage,
                    lid=lid,
                    is_far_east=False,
                    bytes_per_char=1,
                    is_complex=is_complex
                )
                
        except (struct.error, ValueError) as e:
            logger.warning(f"Error parsing FIB: {e}")
            return default_info
    
    def _detect_encoding_by_trial(
        self, 
        data: bytes, 
        fc_min: int,
        ccp_text: int,
        fib_far_east: bool,
        lid: int
    ) -> Optional[OLEEncodingInfo]:
        """Detect encoding by trying different decodings and scoring results.
        
        This method tries both UTF-16LE and single-byte encoding on the
        text area and returns the one that produces better quality text.
        
        Args:
            data: Binary data
            fc_min: Text start offset from FIB
            ccp_text: Character count from FIB
            fib_far_east: fFarEast flag from FIB
            lid: Language ID from FIB
            
        Returns:
            OLEEncodingInfo if clearly detected, None otherwise
        """
        if fc_min >= len(data) or ccp_text == 0:
            return None
        
        # Calculate text boundaries
        text_start = fc_min if fc_min >= 0x200 else 0x200
        
        # Get codepage for single-byte attempt
        codepage = self._get_codepage_for_lid(lid)
        
        # Try 1: Single-byte encoding (text area = ccp_text bytes)
        single_byte_end = min(text_start + ccp_text, len(data))
        single_byte_data = data[text_start:single_byte_end]
        
        single_byte_text = ""
        single_byte_score = 0
        if len(single_byte_data) > 0:
            try:
                single_byte_text = single_byte_data.decode(codepage, errors='ignore')
                single_byte_score = self._score_text_quality(single_byte_text)
            except:
                pass
        
        # Try 2: UTF-16LE encoding (text area = ccp_text * 2 bytes)
        utf16_end = min(text_start + ccp_text * 2, len(data))
        utf16_data = data[text_start:utf16_end]
        
        utf16_text = ""
        utf16_score = 0
        if len(utf16_data) >= 2:
            # Extract valid UTF-16LE text
            utf16_text = self._extract_text_utf16le_sample(utf16_data)
            utf16_score = self._score_text_quality(utf16_text)
        
        logger.debug(
            f"Trial decode scores: single_byte({codepage})={single_byte_score:.2f}, "
            f"utf16le={utf16_score:.2f}"
        )
        
        # Decision based on scores
        score_diff = abs(utf16_score - single_byte_score)
        
        # If one is clearly better (score diff > 0.2)
        if score_diff > 0.2:
            if utf16_score > single_byte_score:
                # Check if it contains Korean/CJK
                has_cjk = any(0xAC00 <= ord(c) <= 0xD7AF or 0x4E00 <= ord(c) <= 0x9FFF 
                             for c in utf16_text[:500])
                return OLEEncodingInfo(
                    encoding_type=OLEEncodingType.UTF16LE,
                    codepage='utf-16-le',
                    lid=lid,
                    is_far_east=has_cjk or fib_far_east,
                    bytes_per_char=2
                )
            else:
                return OLEEncodingInfo(
                    encoding_type=OLEEncodingType.SINGLE_BYTE,
                    codepage=codepage,
                    lid=lid,
                    is_far_east=False,
                    bytes_per_char=1
                )
        
        # If scores are close, use additional heuristics
        # Check for Korean characters - strong indicator of UTF-16LE
        if any(0xAC00 <= ord(c) <= 0xD7AF for c in utf16_text[:500]):
            return OLEEncodingInfo(
                encoding_type=OLEEncodingType.UTF16LE,
                codepage='utf-16-le',
                lid=lid,
                is_far_east=True,
                bytes_per_char=2
            )
        
        # If both scores are reasonable and similar, trust FIB
        return None
    
    def _extract_text_utf16le_sample(self, data: bytes) -> str:
        """Extract a sample of UTF-16LE text for scoring.
        
        Args:
            data: Binary data (should be text area)
            
        Returns:
            Extracted text sample
        """
        chars = []
        i = 0
        while i < len(data) - 1 and len(chars) < 500:
            low = data[i]
            high = data[i + 1]
            
            # Valid UTF-16LE characters
            if high == 0x00:
                if 0x20 <= low <= 0x7E or low in (0x0D, 0x0A, 0x09):
                    chars.append(chr(low))
            elif 0xAC <= high <= 0xD7:  # Korean
                code = low | (high << 8)
                chars.append(chr(code))
            elif 0x4E <= high <= 0x9F:  # CJK
                code = low | (high << 8)
                chars.append(chr(code))
            
            i += 2
        
        return ''.join(chars)
    
    def _score_text_quality(self, text: str) -> float:
        """Score the quality of decoded text.
        
        Higher score = more likely to be correct decoding.
        
        Scoring criteria:
        - Printable ASCII characters
        - Korean syllables
        - CJK characters
        - Common punctuation
        - Penalize control characters and rare Unicode
        
        Args:
            text: Decoded text
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        total = len(text)
        if total == 0:
            return 0.0
        
        # Count character types
        printable_ascii = 0
        korean = 0
        cjk = 0
        whitespace = 0
        common_punct = 0
        bad_chars = 0
        
        for char in text:
            code = ord(char)
            
            if char in ' \n\t\r':
                whitespace += 1
            elif 0x20 <= code <= 0x7E:  # Printable ASCII
                printable_ascii += 1
            elif 0xAC00 <= code <= 0xD7AF:  # Korean syllables
                korean += 1
            elif 0x4E00 <= code <= 0x9FFF:  # CJK
                cjk += 1
            elif code in (0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026):
                common_punct += 1
            elif 0x00 <= code <= 0x1F and code not in (0x09, 0x0A, 0x0D):
                bad_chars += 1  # Control characters
            elif 0xE000 <= code <= 0xF8FF:  # Private use area
                bad_chars += 2
            elif 0xD800 <= code <= 0xDFFF:  # Surrogates
                bad_chars += 2
        
        # Calculate score
        good_chars = printable_ascii + korean + cjk + whitespace + common_punct
        score = good_chars / total if total > 0 else 0
        
        # Penalty for bad characters
        bad_ratio = bad_chars / total if total > 0 else 0
        if bad_ratio > 0.1:
            score *= (1 - bad_ratio)
        
        # Bonus for having substantial readable content
        if good_chars >= 50:
            score = min(1.0, score * 1.1)
        
        return score
    
    def _get_codepage_for_lid(self, lid: int) -> str:
        """Get codepage name for a given Language ID.
        
        Args:
            lid: Language ID from FIB
            
        Returns:
            Codepage name (e.g., 'cp1252')
        """
        # Try exact lid match first
        if lid in LANG_ID_TO_CODEPAGE:
            return LANG_ID_TO_CODEPAGE[lid]
        
        # Try primary language (lower byte)
        primary_lang = lid & 0xFF
        if primary_lang in PRIMARY_LANG_TO_CODEPAGE:
            return PRIMARY_LANG_TO_CODEPAGE[primary_lang]
        
        # Default to Western European
        return 'cp1252'
    
    def _extract_text_utf16le(self, data: bytes) -> str:
        """Extract text from UTF-16LE encoded document.
        
        Used for Far East (Korean, Japanese, Chinese) documents.
        Text is stored as 2 bytes per character in Little Endian format.
        
        Args:
            data: Binary data
            
        Returns:
            Extracted text
        """
        text_parts = []
        i = 0
        
        while i < len(data) - 1:
            # Find text starting point: printable ASCII in UTF-16LE format
            if self._is_text_start_utf16le(data, i):
                # Found text start - collect continuous text bytes
                unicode_bytes, end_pos = self._collect_text_bytes_utf16le(data, i)
                
                if len(unicode_bytes) >= 8:  # Minimum 4 characters (8 bytes)
                    text = self._decode_collected_bytes_utf16le(unicode_bytes)
                    if text and len(text) >= 4 and not text.startswith('\\'):
                        text_parts.append(text)
                
                i = end_pos
            else:
                i += 1
        
        return self._merge_text_parts(text_parts)
    
    def _extract_text_single_byte(self, data: bytes, codepage: str) -> str:
        """Extract text from single-byte encoded document.
        
        Used for Western European (CP1252), Central European (CP1250),
        Cyrillic (CP1251), Greek (CP1253), Turkish (CP1254), etc.
        
        Each character is stored as 1 byte.
        
        Args:
            data: Binary data
            codepage: Codepage to use (e.g., 'cp1252')
            
        Returns:
            Extracted text
        """
        text_parts = []
        
        # Parse FIB to get text boundaries
        text_start, text_end = self._get_text_area_from_fib(data)
        
        if text_start >= text_end:
            return ""
        
        # Extract text area
        text_data = data[text_start:text_end]
        
        # For single-byte encodings, try direct decode first
        try:
            # Decode the text area
            decoded = text_data.decode(codepage, errors='ignore')
            
            # Clean control characters and extract readable segments
            cleaned = self._clean_single_byte_text(decoded)
            
            if cleaned:
                return cleaned
                
        except (UnicodeDecodeError, LookupError) as e:
            logger.warning(f"Failed to decode with {codepage}: {e}")
        
        # Fallback: segment-based extraction
        return self._extract_text_segments_single_byte(data, text_start, text_end, codepage)
    
    def _extract_text_segments_single_byte(
        self, 
        data: bytes, 
        text_start: int, 
        text_end: int, 
        codepage: str
    ) -> str:
        """Extract text segments from single-byte encoded document.
        
        This method finds valid text starting points and extracts segments.
        
        Args:
            data: Binary data
            text_start: Start offset of text area
            text_end: End offset of text area
            codepage: Codepage to use
            
        Returns:
            Extracted text
        """
        text_parts = []
        i = text_start
        
        while i < text_end:
            # Find printable text start
            if self._is_text_start_single_byte(data, i):
                # Collect consecutive printable bytes
                segment_bytes, end_pos = self._collect_text_bytes_single_byte(
                    data, i, text_end
                )
                
                if len(segment_bytes) >= 4:  # Minimum length
                    try:
                        text = bytes(segment_bytes).decode(codepage, errors='ignore')
                        text = text.strip()
                        if text and not self._is_noise_text(text):
                            text_parts.append(text)
                    except:
                        pass
                
                i = end_pos
            else:
                i += 1
        
        return self._merge_text_parts(text_parts)
    
    def _extract_text_multibyte_cjk(self, data: bytes, codepage: str) -> str:
        """Extract text from multi-byte CJK encoded document.
        
        Used for older CJK documents that don't use UTF-16LE:
        - CP949: Korean (Extended Wansung)
        - CP932: Japanese (Shift-JIS)
        - CP936: Chinese Simplified (GBK)
        - CP950: Chinese Traditional (Big5)
        
        Args:
            data: Binary data
            codepage: CJK codepage to use
            
        Returns:
            Extracted text
        """
        # Parse FIB to get text boundaries
        text_start, text_end = self._get_text_area_from_fib(data)
        
        if text_start >= text_end:
            return ""
        
        text_data = data[text_start:text_end]
        
        try:
            # Try direct decode
            decoded = text_data.decode(codepage, errors='ignore')
            cleaned = self._clean_single_byte_text(decoded)
            
            if cleaned:
                return cleaned
                
        except (UnicodeDecodeError, LookupError) as e:
            logger.warning(f"Failed to decode with {codepage}: {e}")
        
        return ""
    
    def _get_text_area_from_fib(self, data: bytes) -> Tuple[int, int]:
        """Parse FIB to get text area boundaries.
        
        DOC format stores text position info in FIB:
        - fcMin (offset 0x18): Start of text in file
        - ccpText (offset 0x4C): Character count for main document text
        
        For single-byte documents, ccpText is actual byte count.
        For UTF-16LE documents, multiply by 2 for byte count.
        
        Args:
            data: WordDocument stream data
            
        Returns:
            Tuple of (text_start_offset, text_end_offset)
        """
        if len(data) < 0x60:
            return 0, len(data)
        
        try:
            # fcMin: Text start offset
            fc_min = struct.unpack('<I', data[0x18:0x1C])[0]
            
            # ccpText: Character count
            ccp_text = struct.unpack('<I', data[0x4C:0x50])[0]
            
            # Determine bytes per char
            if self._encoding_info and self._encoding_info.bytes_per_char == 1:
                # Single-byte encoding
                text_end = fc_min + ccp_text
            else:
                # UTF-16LE (2 bytes per char)
                text_end = fc_min + (ccp_text * 2)
            
            # Sanity checks
            text_start = fc_min if fc_min >= 0x200 else 0x200
            if text_end > len(data):
                text_end = len(data)
            
            return text_start, text_end
            
        except (struct.error, ValueError):
            return 0, len(data)
    
    def _is_text_start_utf16le(self, data: bytes, pos: int) -> bool:
        """Check if position is a valid UTF-16LE text starting point."""
        if pos + 1 >= len(data):
            return False
        
        low_byte = data[pos]
        high_byte = data[pos + 1]
        
        # Printable ASCII: 0x20-0x7E with high byte 0x00
        if 0x20 <= low_byte <= 0x7E and high_byte == 0x00:
            return True
        
        # Korean Syllables (AC00-D7AF)
        if 0xAC <= high_byte <= 0xD7:
            return True
        
        return False
    
    def _is_text_start_single_byte(self, data: bytes, pos: int) -> bool:
        """Check if position is a valid single-byte text starting point."""
        if pos >= len(data):
            return False
        
        byte = data[pos]
        
        # Printable ASCII: 0x20-0x7E
        if 0x20 <= byte <= 0x7E:
            return True
        
        # Extended ASCII (0x80-0xFF) - may be valid in codepage
        if 0x80 <= byte <= 0xFF:
            return True
        
        return False
    
    def _collect_text_bytes_utf16le(self, data: bytes, start: int) -> Tuple[List[int], int]:
        """Collect continuous UTF-16LE text bytes."""
        unicode_bytes = []
        j = start
        
        while j < len(data) - 1:
            low_byte = data[j]
            high_byte = data[j + 1]
            
            if self._is_valid_text_char_utf16le(low_byte, high_byte):
                unicode_bytes.extend([low_byte, high_byte])
                j += 2
            else:
                break
        
        return unicode_bytes, j
    
    def _collect_text_bytes_single_byte(
        self, 
        data: bytes, 
        start: int, 
        end: int
    ) -> Tuple[List[int], int]:
        """Collect continuous single-byte text bytes."""
        collected = []
        j = start
        
        while j < end:
            byte = data[j]
            
            if self._is_valid_text_byte_single(byte):
                collected.append(byte)
                j += 1
            else:
                break
        
        return collected, j
    
    def _is_valid_text_char_utf16le(self, low_byte: int, high_byte: int) -> bool:
        """Check if a UTF-16LE character is valid text."""
        # ASCII with null high byte
        if high_byte == 0x00:
            if 0x20 <= low_byte <= 0x7E:
                return True
            if low_byte in (0x0D, 0x0A, 0x09):
                return True
            return False
        
        # Korean Syllables (AC00-D7AF)
        if 0xAC <= high_byte <= 0xD7:
            return True
        
        # Korean Jamo (1100-11FF)
        if high_byte == 0x11:
            return True
        
        # Korean Compatibility Jamo (3130-318F)
        if high_byte == 0x31 and 0x30 <= low_byte <= 0x8F:
            return True
        
        # CJK Unified Ideographs (4E00-9FFF)
        if 0x4E <= high_byte <= 0x9F:
            return True
        
        # CJK Extension A (3400-4DBF)
        if 0x34 <= high_byte <= 0x4D:
            return True
        
        # Common punctuation and symbols (2000-2FFF)
        if 0x20 <= high_byte <= 0x2F:
            return True
        
        # Fullwidth forms (FF00-FFEF)
        if high_byte == 0xFF and low_byte <= 0xEF:
            return True
        
        return False
    
    def _is_valid_text_byte_single(self, byte: int) -> bool:
        """Check if a byte is valid text in single-byte encoding."""
        # Printable ASCII
        if 0x20 <= byte <= 0x7E:
            return True
        
        # Control chars (CR, LF, TAB)
        if byte in (0x0D, 0x0A, 0x09):
            return True
        
        # Extended ASCII (codepage-specific characters)
        if 0x80 <= byte <= 0xFF:
            return True
        
        return False
    
    def _decode_collected_bytes_utf16le(self, unicode_bytes: List[int]) -> str:
        """Decode collected bytes as UTF-16LE text."""
        try:
            text = bytes(unicode_bytes).decode('utf-16-le', errors='ignore')
            text = text.strip()
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
            return text
        except Exception:
            return ""
    
    def _clean_single_byte_text(self, text: str) -> str:
        """Clean decoded single-byte text.
        
        Removes control characters and binary noise while preserving
        readable text content.
        
        Args:
            text: Decoded text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove null characters
        text = text.replace('\x00', '')
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove other control characters except tab and newline
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'[ \t]{10,}', ' ', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Extract meaningful text segments
        segments = []
        current_segment = []
        noise_count = 0
        
        for char in text:
            if char.isprintable() or char in '\n\t':
                current_segment.append(char)
                noise_count = 0
            else:
                noise_count += 1
                if noise_count > 5 and current_segment:
                    segment_text = ''.join(current_segment).strip()
                    if len(segment_text) >= 3:
                        segments.append(segment_text)
                    current_segment = []
        
        # Don't forget the last segment
        if current_segment:
            segment_text = ''.join(current_segment).strip()
            if len(segment_text) >= 3:
                segments.append(segment_text)
        
        return '\n'.join(segments)
    
    def _try_fallback_encodings(self, data: bytes) -> Tuple[str, str]:
        """Try fallback encodings when auto-detection fails.
        
        Args:
            data: Binary data
            
        Returns:
            Tuple of (decoded_text, encoding_used)
        """
        text_start, text_end = self._get_text_area_from_fib(data)
        text_data = data[text_start:text_end] if text_start < text_end else data
        
        best_text = ""
        best_encoding = "utf-16-le"
        best_score = 0
        
        for encoding in self._ole_config.fallback_encodings:
            try:
                if encoding == 'utf-16-le':
                    # For UTF-16LE, use the special extraction
                    text = self._extract_text_utf16le(data)
                else:
                    text = text_data.decode(encoding, errors='ignore')
                    text = self._clean_single_byte_text(text)
                
                if text:
                    score = self._score_decoded_text(text)
                    if score > best_score:
                        best_score = score
                        best_text = text
                        best_encoding = encoding
                        
            except (UnicodeDecodeError, LookupError):
                continue
        
        return best_text, best_encoding
    
    def _score_decoded_text(self, text: str) -> float:
        """Score decoded text quality.
        
        Higher score = more likely correct encoding.
        
        Args:
            text: Decoded text
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        total = len(text)
        if total == 0:
            return 0.0
        
        # Count character types
        ascii_printable = 0
        korean = 0
        cjk = 0
        common_punct = 0
        rare_chars = 0
        
        for char in text:
            code = ord(char)
            
            if 0x20 <= code <= 0x7E or char in '\n\t':
                ascii_printable += 1
            elif 0xAC00 <= code <= 0xD7AF:  # Korean syllables
                korean += 1
            elif 0x4E00 <= code <= 0x9FFF:  # CJK
                cjk += 1
            elif code in (0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026):
                common_punct += 1
            elif 0xE000 <= code <= 0xF8FF or 0xD800 <= code <= 0xDFFF:
                rare_chars += 2  # Private use / surrogates are bad
            elif code > 0x3000:
                rare_chars += 0.5
        
        # Score based on readable character ratio
        readable = ascii_printable + korean + cjk + common_punct
        score = readable / total if total > 0 else 0
        
        # Penalty for rare characters
        if rare_chars / total > 0.1:
            score *= 0.5
        
        return score
    
    def _is_garbled(self, text: str) -> bool:
        """Check if text appears to be garbled (wrong encoding).
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears garbled
        """
        if not text:
            return True
        
        score = self._score_decoded_text(text)
        return score < 0.3
    
    def _extract_with_encoding(self, data: bytes, encoding: str) -> str:
        """Extract text using a specific encoding.
        
        Args:
            data: Binary data
            encoding: Encoding to use
            
        Returns:
            Extracted text
        """
        if encoding == 'utf-16-le':
            return self._extract_text_utf16le(data)
        else:
            text_start, text_end = self._get_text_area_from_fib(data)
            text_data = data[text_start:text_end] if text_start < text_end else data
            
            try:
                text = text_data.decode(encoding, errors='ignore')
                return self._clean_single_byte_text(text)
            except:
                return ""
    
    def _merge_text_parts(self, text_parts: List[str]) -> str:
        """Merge text parts, removing duplicates and filtering noise.
        
        Args:
            text_parts: List of text segments
            
        Returns:
            Merged text string
        """
        if not text_parts:
            return ""
        
        # Remove duplicates while preserving order, and filter noise
        seen = set()
        unique_parts = []
        for part in text_parts:
            # Skip if already seen
            if part in seen:
                continue
            
            # Skip if too short
            if len(part) <= 3:
                continue
            
            # Filter out noise patterns
            if self._is_noise_text(part):
                continue
            
            seen.add(part)
            unique_parts.append(part)
        
        result = '\n'.join(unique_parts)
        
        # Clean excessive newlines
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def _is_noise_text(self, text: str) -> bool:
        """Check if text is likely noise from binary data.
        
        Noise patterns include:
        - Text with mostly non-printable or rare Unicode chars
        - Text with specific DOC binary patterns decoded as text
        - Very high ratio of uncommon characters
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be noise
        """
        if not text:
            return True
        
        # Quick check: if text is very short, apply stricter rules
        if len(text) <= 6:
            score = self._score_decoded_text(text)
            return score < 0.6
        
        # Use the scoring function
        score = self._score_decoded_text(text)
        return score < 0.4
    
    def decode_cell_content(self, data: bytes, start: int, end: int) -> str:
        """Decode a specific cell content range for table extraction.
        
        This method is optimized for table cell content extraction.
        It uses a more lenient extraction that allows short text (even 1 character),
        which is important for table cells that may contain single words.
        
        Args:
            data: Full binary data
            start: Start offset
            end: End offset (exclusive)
            
        Returns:
            Decoded cell text
        """
        if start >= end or start < 0 or end > len(data):
            return ""
        
        cell_data = data[start:end]
        
        # Detect encoding if not already done
        if not self._encoding_info:
            self._encoding_info = self._detect_encoding_from_fib(data)
        
        # For cell content, use appropriate decoder based on encoding
        if self._encoding_info.encoding_type == OLEEncodingType.UTF16LE:
            text = self._decode_cell_direct_utf16le(cell_data)
        else:
            text = self._decode_cell_direct_single_byte(
                cell_data, 
                self._encoding_info.codepage
            )
        
        return text.strip()
    
    def _decode_cell_direct_utf16le(self, data: bytes) -> str:
        """Direct UTF-16LE decoding for cell content.
        
        Unlike _extract_text_utf16le which requires minimum text length,
        this method decodes all valid characters without length restrictions.
        
        Args:
            data: Cell binary data
            
        Returns:
            Decoded text
        """
        if len(data) < 2:
            return ""
        
        # Collect all valid UTF-16LE characters
        unicode_bytes = []
        i = 0
        
        while i < len(data) - 1:
            low_byte = data[i]
            high_byte = data[i + 1]
            
            if self._is_valid_text_char_utf16le(low_byte, high_byte):
                unicode_bytes.extend([low_byte, high_byte])
                i += 2
            else:
                # Skip invalid bytes, but preserve what we've collected
                i += 2
        
        if not unicode_bytes:
            return ""
        
        # Decode collected bytes
        try:
            text = bytes(unicode_bytes).decode('utf-16-le', errors='ignore')
            # Filter out control characters except newline/carriage return
            filtered = []
            for c in text:
                if c.isprintable() or c in '\n\r\t':
                    filtered.append(c)
                elif c == '\r':
                    filtered.append('\n')
            return ''.join(filtered)
        except:
            return ""
    
    def _decode_cell_direct_single_byte(self, data: bytes, codepage: str) -> str:
        """Direct single-byte decoding for cell content.
        
        Args:
            data: Cell binary data
            codepage: Codepage to use
            
        Returns:
            Decoded text
        """
        if len(data) < 1:
            return ""
        
        try:
            text = data.decode(codepage, errors='ignore')
            # Filter out control characters
            filtered = []
            for c in text:
                if c.isprintable() or c in '\n\r\t':
                    filtered.append(c)
                elif c == '\r':
                    filtered.append('\n')
            return ''.join(filtered)
        except:
            return ""
    
    def get_encoding_info(self) -> Optional[OLEEncodingInfo]:
        """Get the detected encoding information.
        
        Returns:
            OLEEncodingInfo if encoding was detected, None otherwise
        """
        return self._encoding_info
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this encoder supports the given format."""
        return format_type.lower() in ('doc', 'ole')


# Backward compatibility aliases
DOCEncoder = OLEEncoder
DOCEncodingType = OLEEncodingType
DOCEncodingInfo = OLEEncodingInfo
DOCEncodingConfig = OLEEncodingConfig


def decode_word_stream(data: bytes, config: Optional[OLEEncodingConfig] = None) -> str:
    """Convenience function to decode WordDocument stream.
    
    Automatically detects encoding and extracts text.
    
    Args:
        data: Binary data from WordDocument stream
        config: Optional OLE encoding config
        
    Returns:
        Decoded text
    """
    encoder = OLEEncoder(config)
    text, _ = encoder.decode(data)
    return text


def decode_cell_content(data: bytes, start: int, end: int) -> str:
    """Convenience function to decode cell content.
    
    Args:
        data: Full binary data
        start: Start offset
        end: End offset
        
    Returns:
        Decoded cell text
    """
    encoder = OLEEncoder()
    return encoder.decode_cell_content(data, start, end)


def detect_ole_encoding(data: bytes) -> OLEEncodingInfo:
    """Detect encoding of an OLE DOC file from its binary data.
    
    Args:
        data: WordDocument stream data
        
    Returns:
        OLEEncodingInfo with detected encoding details
    """
    encoder = OLEEncoder()
    return encoder._detect_encoding_from_fib(data)


# Backward compatibility alias
detect_doc_encoding = detect_ole_encoding


def get_supported_codepages() -> Dict[str, str]:
    """Get list of supported codepages.
    
    Returns:
        Dictionary mapping codepage names to descriptions
    """
    return {
        'utf-16-le': 'Unicode UTF-16 Little Endian (East Asian)',
        'cp1252': 'Windows Western European (English, French, German)',
        'cp1250': 'Windows Central European (Polish, Czech, Hungarian)',
        'cp1251': 'Windows Cyrillic (Russian, Ukrainian)',
        'cp1253': 'Windows Greek',
        'cp1254': 'Windows Turkish',
        'cp1255': 'Windows Hebrew',
        'cp1256': 'Windows Arabic',
        'cp1257': 'Windows Baltic (Lithuanian, Latvian)',
        'cp1258': 'Windows Vietnamese',
        'cp874': 'Windows Thai',
        'cp949': 'Korean Extended Wansung',
        'cp932': 'Japanese Shift-JIS',
        'cp936': 'Simplified Chinese GBK',
        'cp950': 'Traditional Chinese Big5',
    }
