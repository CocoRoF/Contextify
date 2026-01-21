# contextifier/core/processor/rtf_helper/rtf_image_processor.py
"""
RTF Image Processor

Provides RTF-specific image processing that inherits from ImageProcessor.
Handles RTF embedded images (pict, shppict, blipuid, bin).

Includes binary preprocessing functionality for RTF files:
- \\binN tag processing (skip N bytes of raw binary data)
- \\pict group image extraction
- Image saving and tag generation
"""
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend

logger = logging.getLogger("contextify.image_processor.rtf")


# === Image Format Constants ===

# Magic numbers for image format detection
IMAGE_SIGNATURES = {
    b'\xff\xd8\xff': 'jpeg',
    b'\x89PNG\r\n\x1a\n': 'png',
    b'GIF87a': 'gif',
    b'GIF89a': 'gif',
    b'BM': 'bmp',
    b'\xd7\xcd\xc6\x9a': 'wmf',
    b'\x01\x00\x09\x00': 'wmf',
    b'\x01\x00\x00\x00': 'emf',
}

# RTF image type mapping
RTF_IMAGE_TYPES = {
    'jpegblip': 'jpeg',
    'pngblip': 'png',
    'wmetafile': 'wmf',
    'emfblip': 'emf',
    'dibitmap': 'bmp',
    'wbitmap': 'bmp',
}

# Supported image formats for saving
SUPPORTED_IMAGE_FORMATS = {'jpeg', 'png', 'gif', 'bmp'}


@dataclass
class RTFBinaryRegion:
    """RTF binary data region information."""
    start_pos: int
    end_pos: int
    bin_type: str  # "bin" or "pict"
    data_size: int
    image_format: str = ""
    image_data: bytes = b""


@dataclass
class RTFBinaryProcessResult:
    """RTF binary processing result."""
    clean_content: bytes
    binary_regions: List[RTFBinaryRegion] = field(default_factory=list)
    image_tags: Dict[int, str] = field(default_factory=dict)


class RTFImageProcessor(ImageProcessor):
    """
    RTF-specific image processor.
    
    Inherits from ImageProcessor and provides RTF-specific processing.
    
    Handles:
    - RTF embedded images (pict, shppict, blipuid)
    - WMF/EMF metafiles
    - JPEG/PNG/BMP embedded images
    
    Example:
        processor = RTFImageProcessor()
        
        # Process RTF picture
        tag = processor.process_image(image_data)
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
    ):
        """
        Initialize RTFImageProcessor.
        
        Args:
            directory_path: Image save directory
            tag_prefix: Tag prefix for image references
            tag_suffix: Tag suffix for image references
            storage_backend: Storage backend for saving images
        """
        super().__init__(
            directory_path=directory_path,
            tag_prefix=tag_prefix,
            tag_suffix=tag_suffix,
            storage_backend=storage_backend,
        )
        self._processed_images: Set[str] = set()
        self.logger = logger
    
    def process_rtf_image(
        self,
        image_data: bytes,
        blipuid: Optional[str] = None,
        image_format: Optional[str] = None
    ) -> Optional[str]:
        """
        Process an RTF embedded image.
        
        Args:
            image_data: Raw image data
            blipuid: Optional BLIPUID for deduplication
            image_format: Image format hint (png, jpg, wmf, emf, etc.)
            
        Returns:
            Image tag string or None
        """
        if not image_data:
            return None
        
        # Check for duplicates using hash
        import hashlib
        image_hash = hashlib.md5(image_data).hexdigest()
        
        if image_hash in self._processed_images:
            self.logger.debug(f"Skipping duplicate RTF image: {image_hash[:8]}")
            return None
        
        self._processed_images.add(image_hash)
        
        # Convert metafiles if needed
        if image_format in ('wmf', 'emf'):
            converted = self._convert_metafile(image_data, image_format)
            if converted:
                image_data = converted
        
        return self.save_image(image_data)
    
    def _convert_metafile(self, data: bytes, format_type: str) -> Optional[bytes]:
        """
        Attempt to convert WMF/EMF metafile to PNG.
        
        Args:
            data: Metafile data
            format_type: 'wmf' or 'emf'
            
        Returns:
            Converted PNG data or None
        """
        try:
            from PIL import Image
            import io
            
            # Try to open with PIL
            img = Image.open(io.BytesIO(data))
            output = io.BytesIO()
            img.save(output, format='PNG')
            return output.getvalue()
        except Exception:
            # Metafile conversion not supported
            return None
    
    def reset_processed_images(self) -> None:
        """Reset the set of processed images."""
        self._processed_images.clear()
    
    @property
    def processed_images(self) -> Set[str]:
        """Get the set of processed image hashes."""
        return self._processed_images

    # === Binary Preprocessing Methods ===
    
    def preprocess_binary(self, content: bytes) -> Tuple[bytes, Dict[int, str]]:
        """
        Preprocess RTF binary content.
        
        Processes \\bin tags and \\pict groups, extracts images,
        saves them locally, and returns clean content with image tags.
        
        Args:
            content: RTF file binary content
            
        Returns:
            Tuple of (clean_content, position->image_tag dict)
        """
        result = self._process_binary_content(content)
        return result.clean_content, result.image_tags
    
    def _process_binary_content(self, content: bytes) -> RTFBinaryProcessResult:
        """
        Process RTF binary content internally.
        
        Args:
            content: RTF file binary content
            
        Returns:
            RTFBinaryProcessResult with clean content and image tags
        """
        image_tags: Dict[int, str] = {}
        
        # Step 1: Find \\bin tag regions
        bin_regions = self._find_bin_regions(content)
        
        # Step 2: Find \\pict regions (excluding bin regions)
        pict_regions = self._find_pict_regions(content, bin_regions)
        
        # Step 3: Merge and sort all regions
        all_regions = bin_regions + pict_regions
        all_regions.sort(key=lambda r: r.start_pos)
        
        # Step 4: Process images and generate tags
        for region in all_regions:
            if not region.image_data:
                continue
            
            if region.image_format in SUPPORTED_IMAGE_FORMATS:
                image_tag = self.save_image(region.image_data)
                if image_tag:
                    image_tags[region.start_pos] = f"\n{image_tag}\n"
                    logger.info(
                        f"Saved RTF image: {image_tag} "
                        f"(format={region.image_format}, size={region.data_size})"
                    )
                else:
                    image_tags[region.start_pos] = ""
            else:
                image_tags[region.start_pos] = ""
        
        # Step 5: Remove binary data from content
        clean_content = self._remove_binary_data(content, all_regions, image_tags)
        
        return RTFBinaryProcessResult(
            clean_content=clean_content,
            binary_regions=all_regions,
            image_tags=image_tags
        )
    
    def _find_bin_regions(self, content: bytes) -> List[RTFBinaryRegion]:
        """Find \\binN tags and identify binary regions."""
        regions = []
        pattern = rb'\\bin(\d+)'
        
        for match in re.finditer(pattern, content):
            try:
                bin_size = int(match.group(1))
                bin_tag_start = match.start()
                bin_tag_end = match.end()
                
                data_start = bin_tag_end
                if data_start < len(content) and content[data_start:data_start+1] == b' ':
                    data_start += 1
                
                data_end = data_start + bin_size
                
                if data_end <= len(content):
                    binary_data = content[data_start:data_end]
                    image_format = self._detect_image_format(binary_data)
                    
                    # Find parent \\shppict group
                    group_start = bin_tag_start
                    group_end = data_end
                    
                    search_start = max(0, bin_tag_start - 500)
                    search_area = content[search_start:bin_tag_start]
                    
                    shppict_pos = search_area.rfind(b'\\shppict')
                    if shppict_pos != -1:
                        abs_pos = search_start + shppict_pos
                        brace_pos = abs_pos
                        while brace_pos > 0 and content[brace_pos:brace_pos+1] != b'{':
                            brace_pos -= 1
                        group_start = brace_pos
                        
                        depth = 1
                        j = data_end
                        while j < len(content) and depth > 0:
                            if content[j:j+1] == b'{':
                                depth += 1
                            elif content[j:j+1] == b'}':
                                depth -= 1
                            j += 1
                        group_end = j
                    
                    regions.append(RTFBinaryRegion(
                        start_pos=group_start,
                        end_pos=group_end,
                        bin_type="bin",
                        data_size=bin_size,
                        image_format=image_format,
                        image_data=binary_data
                    ))
            except (ValueError, IndexError):
                continue
        
        return regions
    
    def _find_pict_regions(
        self,
        content: bytes,
        exclude_regions: List[RTFBinaryRegion]
    ) -> List[RTFBinaryRegion]:
        """Find hex-encoded \\pict regions."""
        regions = []
        
        bin_tag_positions = {r.start_pos for r in exclude_regions if r.bin_type == "bin"}
        excluded_ranges = [(r.start_pos, r.end_pos) for r in exclude_regions]
        
        def is_excluded(pos: int) -> bool:
            return any(start <= pos < end for start, end in excluded_ranges)
        
        def has_bin_nearby(pict_pos: int) -> bool:
            return any(pict_pos < bp < pict_pos + 200 for bp in bin_tag_positions)
        
        try:
            text_content = content.decode('cp1252', errors='replace')
            pict_pattern = r'\\pict\s*((?:\\[a-zA-Z]+\d*\s*)*)'
            
            for match in re.finditer(pict_pattern, text_content):
                start_pos = match.start()
                
                if is_excluded(start_pos) or has_bin_nearby(start_pos):
                    continue
                
                attrs = match.group(1)
                image_format = ""
                for rtf_type, fmt in RTF_IMAGE_TYPES.items():
                    if rtf_type in attrs:
                        image_format = fmt
                        break
                
                # Extract hex data
                hex_start = match.end()
                hex_data = []
                i = hex_start
                
                while i < len(text_content):
                    ch = text_content[i]
                    if ch in '0123456789abcdefABCDEF':
                        hex_data.append(ch)
                    elif ch in ' \t\r\n':
                        pass
                    elif ch == '}':
                        break
                    elif ch == '\\':
                        if text_content[i:i+4] == '\\bin':
                            hex_data = []
                            break
                        while i < len(text_content) and text_content[i] not in ' \t\r\n}':
                            i += 1
                        continue
                    else:
                        break
                    i += 1
                
                hex_str = ''.join(hex_data)
                
                if len(hex_str) >= 32:
                    try:
                        image_data = bytes.fromhex(hex_str)
                        if not image_format:
                            image_format = self._detect_image_format(image_data)
                        
                        if image_format:
                            regions.append(RTFBinaryRegion(
                                start_pos=start_pos,
                                end_pos=i,
                                bin_type="pict",
                                data_size=len(image_data),
                                image_format=image_format,
                                image_data=image_data
                            ))
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"Error finding pict regions: {e}")
        
        return regions
    
    def _detect_image_format(self, data: bytes) -> str:
        """Detect image format from binary data."""
        if not data or len(data) < 4:
            return ""
        
        for signature, format_name in IMAGE_SIGNATURES.items():
            if data.startswith(signature):
                return format_name
        
        if len(data) >= 2 and data[0:2] == b'\xff\xd8':
            return 'jpeg'
        
        return ""
    
    def _remove_binary_data(
        self,
        content: bytes,
        regions: List[RTFBinaryRegion],
        image_tags: Dict[int, str]
    ) -> bytes:
        """Remove binary data regions from content."""
        if not regions:
            return content
        
        sorted_regions = sorted(regions, key=lambda r: r.start_pos, reverse=True)
        result = bytearray(content)
        
        for region in sorted_regions:
            replacement = b''
            if region.start_pos in image_tags:
                tag = image_tags[region.start_pos]
                if tag:
                    replacement = tag.encode('ascii', errors='replace')
            result[region.start_pos:region.end_pos] = replacement
        
        return bytes(result)


def preprocess_rtf_binary(
    content: bytes,
    processed_images: Optional[Set[str]] = None,
    image_processor: ImageProcessor = None
) -> Tuple[bytes, Dict[int, str]]:
    """
    Preprocess RTF content to handle binary data.
    
    Removes \\bin tag binary data and extracts images,
    saving them locally and converting to image tags.
    
    Call this before RTF parsing to prevent text corruption
    from binary data.
    
    Args:
        content: RTF file binary content
        processed_images: Set of processed image hashes (optional)
        image_processor: Image processor instance
        
    Returns:
        Tuple of (clean_content, position->image_tag dict)
    """
    if image_processor is None:
        # Create default processor
        processor = RTFImageProcessor()
    elif isinstance(image_processor, RTFImageProcessor):
        processor = image_processor
    else:
        # Wrap existing ImageProcessor
        processor = RTFImageProcessor(
            directory_path=image_processor.config.directory_path,
            tag_prefix=image_processor.config.tag_prefix,
            tag_suffix=image_processor.config.tag_suffix,
            storage_backend=image_processor.storage_backend,
        )
    
    if processed_images:
        processor._processed_images = processed_images
    
    return processor.preprocess_binary(content)


__all__ = ['RTFImageProcessor', 'preprocess_rtf_binary', 'RTFBinaryRegion', 'RTFBinaryProcessResult']
