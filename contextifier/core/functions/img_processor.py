# contextifier/core/functions/img_processor.py
"""
Image Processing Module

Provides functionality to save image data to various storage backends
and convert to tag format. Uses Strategy pattern for storage backends.

This is the BASE class for all image processors.
Format-specific processors (PDFImageProcessor, DOCXImageProcessor, etc.)
should inherit from ImageProcessor and override process_image() method.

Main Features:
- Base ImageProcessor class with pluggable storage backend
- Save image data to specified storage (Local, MinIO, S3, etc.)
- Return saved path in custom tag format
- Duplicate image detection and handling
- Support for various image formats
- Extensible for format-specific processing

Usage Example:
    from contextifier.core.functions.img_processor import ImageProcessor
    from contextifier.core.functions.storage_backend import (
        LocalStorageBackend,
        MinIOStorageBackend,
    )

    # Use with default settings (local storage)
    processor = ImageProcessor()
    tag = processor.save_image(image_bytes)
    # Result: "[Image:temp/images/abc123.png]"

    # Use with MinIO storage (when implemented)
    minio_backend = MinIOStorageBackend(endpoint="localhost:9000", bucket="images")
    processor = ImageProcessor(storage_backend=minio_backend)
    
    # Custom tag format
    processor = ImageProcessor(
        directory_path="output/images",
        tag_prefix="<img src='",
        tag_suffix="'>"
    )
    tag = processor.save_image(image_bytes)
    # Result: "<img src='output/images/abc123.png'>"
    
    # Inherit for format-specific processing
    class PDFImageProcessor(ImageProcessor):
        def process_image(self, image_data: bytes, **kwargs) -> Optional[str]:
            xref = kwargs.get('xref')
            custom_name = f"pdf_xref_{xref}" if xref else None
            return self.save_image(image_data, custom_name=custom_name)
"""
import hashlib
import io
import logging
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from contextifier.core.functions.storage_backend import (
    BaseStorageBackend,
    LocalStorageBackend,
    StorageType,
    get_default_backend,
)

logger = logging.getLogger("contextify.image_processor")


class ImageFormat(Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    GIF = "gif"
    BMP = "bmp"
    WEBP = "webp"
    TIFF = "tiff"
    UNKNOWN = "unknown"


class NamingStrategy(Enum):
    """Image file naming strategies."""
    HASH = "hash"           # Content-based hash (prevents duplicates)
    UUID = "uuid"           # Unique UUID
    SEQUENTIAL = "sequential"  # Sequential numbering
    TIMESTAMP = "timestamp"    # Timestamp-based


@dataclass
class ImageProcessorConfig:
    """
    ImageProcessor Configuration.

    Attributes:
        directory_path: Directory path or bucket prefix for saving images
        tag_prefix: Tag prefix (e.g., "[Image:")
        tag_suffix: Tag suffix (e.g., "]")
        naming_strategy: File naming strategy
        default_format: Default image format
        create_directory: Auto-create directory if not exists
        use_absolute_path: Use absolute path in tags
        hash_algorithm: Hash algorithm (for hash strategy)
        max_filename_length: Maximum filename length
    """
    directory_path: str = "temp/images"
    tag_prefix: str = "[Image:"
    tag_suffix: str = "]"
    naming_strategy: NamingStrategy = NamingStrategy.HASH
    default_format: ImageFormat = ImageFormat.PNG
    create_directory: bool = True
    use_absolute_path: bool = False
    hash_algorithm: str = "sha256"
    max_filename_length: int = 64


class ImageProcessor:
    """
    Base Image Processing Class.
    
    Saves image data using a pluggable storage backend and returns
    the saved path in the specified tag format.
    
    This is the BASE CLASS for all format-specific image processors.
    Subclasses should override process_image() for format-specific handling.
    
    Args:
        directory_path: Image save directory (default: "temp/images")
        tag_prefix: Tag prefix (default: "[Image:")
        tag_suffix: Tag suffix (default: "]")
        naming_strategy: File naming strategy (default: HASH)
        storage_backend: Storage backend instance (default: LocalStorageBackend)
        config: ImageProcessorConfig object (takes precedence)
        
    Examples:
        >>> # Default usage (local storage)
        >>> processor = ImageProcessor()
        >>> tag = processor.save_image(image_bytes)
        "[Image:temp/images/a1b2c3d4.png]"
        
        >>> # Custom directory and tags
        >>> processor = ImageProcessor(
        ...     directory_path="images",
        ...     tag_prefix="![image](",
        ...     tag_suffix=")"
        ... )
        >>> tag = processor.save_image(image_bytes)
        "![image](images/a1b2c3d4.png)"
        
        >>> # Subclass for format-specific processing
        >>> class PDFImageProcessor(ImageProcessor):
        ...     def process_image(self, image_data, **kwargs):
        ...         xref = kwargs.get('xref')
        ...         return self.save_image(image_data, custom_name=f"pdf_{xref}")
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        naming_strategy: Union[NamingStrategy, str] = NamingStrategy.HASH,
        storage_backend: Optional[BaseStorageBackend] = None,
        config: Optional[ImageProcessorConfig] = None,
    ):
        # Set config
        if config:
            self.config = config
        else:
            if isinstance(naming_strategy, str):
                naming_strategy = NamingStrategy(naming_strategy.lower())
            
            self.config = ImageProcessorConfig(
                directory_path=directory_path,
                tag_prefix=tag_prefix,
                tag_suffix=tag_suffix,
                naming_strategy=naming_strategy,
            )
        
        # Set storage backend (default: local)
        self._storage_backend = storage_backend or get_default_backend()
        
        # Track processed image hashes (for duplicate prevention)
        self._processed_hashes: Dict[str, str] = {}
        
        # Sequential counter (for sequential strategy)
        self._sequential_counter: int = 0
        
        # Logger
        self._logger = logging.getLogger("contextify.image_processor.ImageProcessor")
        
        # Create directory if using local storage
        if self.config.create_directory:
            self._ensure_storage_ready()
    
    @property
    def storage_backend(self) -> BaseStorageBackend:
        """Get the current storage backend."""
        return self._storage_backend
    
    @storage_backend.setter
    def storage_backend(self, backend: BaseStorageBackend) -> None:
        """
        Set storage backend.
        
        Args:
            backend: New storage backend instance
        """
        self._storage_backend = backend
        if self.config.create_directory:
            self._ensure_storage_ready()
    
    @property
    def storage_type(self) -> StorageType:
        """Get the current storage type."""
        return self._storage_backend.storage_type
    
    def _ensure_storage_ready(self) -> None:
        """Ensure storage is ready."""
        self._storage_backend.ensure_ready(self.config.directory_path)
    
    def _compute_hash(self, data: bytes) -> str:
        """Compute hash of image data."""
        hasher = hashlib.new(self.config.hash_algorithm)
        hasher.update(data)
        return hasher.hexdigest()[:32]
    
    def _detect_format(self, data: bytes) -> ImageFormat:
        """Detect format from image data using magic bytes."""
        if len(data) < 12:
            return ImageFormat.UNKNOWN
        
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return ImageFormat.PNG
        elif data[:2] == b'\xff\xd8':
            return ImageFormat.JPEG
        elif data[:6] in (b'GIF87a', b'GIF89a'):
            return ImageFormat.GIF
        elif data[:2] == b'BM':
            return ImageFormat.BMP
        elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return ImageFormat.WEBP
        elif data[:4] in (b'II*\x00', b'MM\x00*'):
            return ImageFormat.TIFF
        else:
            return ImageFormat.UNKNOWN
    
    def _generate_filename(
        self,
        data: bytes,
        image_format: ImageFormat,
        custom_name: Optional[str] = None
    ) -> str:
        """Generate filename based on naming strategy."""
        if custom_name:
            if not any(custom_name.lower().endswith(f".{fmt.value}") 
                      for fmt in ImageFormat if fmt != ImageFormat.UNKNOWN):
                ext = (image_format.value if image_format != ImageFormat.UNKNOWN 
                       else self.config.default_format.value)
                return f"{custom_name}.{ext}"
            return custom_name
        
        ext = (image_format.value if image_format != ImageFormat.UNKNOWN 
               else self.config.default_format.value)
        
        strategy = self.config.naming_strategy
        
        if strategy == NamingStrategy.HASH:
            base = self._compute_hash(data)
        elif strategy == NamingStrategy.UUID:
            base = str(uuid.uuid4())[:16]
        elif strategy == NamingStrategy.SEQUENTIAL:
            self._sequential_counter += 1
            base = f"image_{self._sequential_counter:06d}"
        elif strategy == NamingStrategy.TIMESTAMP:
            import time
            base = f"img_{int(time.time() * 1000)}"
        else:
            base = self._compute_hash(data)
        
        filename = f"{base}.{ext}"
        
        if len(filename) > self.config.max_filename_length:
            max_base_len = self.config.max_filename_length - len(ext) - 1
            filename = f"{base[:max_base_len]}.{ext}"
        
        return filename
    
    def _build_file_path(self, filename: str) -> str:
        """Build full file path from filename."""
        return os.path.join(self.config.directory_path, filename)
    
    def _build_tag(self, file_path: str) -> str:
        """Build tag from file path."""
        if self.config.use_absolute_path:
            path_str = str(Path(file_path).absolute())
        else:
            path_str = self._storage_backend.build_url(file_path)
        
        path_str = path_str.replace("\\", "/")
        return f"{self.config.tag_prefix}{path_str}{self.config.tag_suffix}"
    
    def save_image(
        self,
        image_data: bytes,
        custom_name: Optional[str] = None,
        processed_images: Optional[Set[str]] = None,
        skip_duplicate: bool = True,
    ) -> Optional[str]:
        """
        Save image data and return tag.
        
        Args:
            image_data: Image binary data
            custom_name: Custom filename (extension optional)
            processed_images: Set of processed image paths (for external duplicate tracking)
            skip_duplicate: If True, skip saving duplicate images
            
        Returns:
            Image tag string, or None on failure
            
        Examples:
            >>> processor = ImageProcessor()
            >>> tag = processor.save_image(png_bytes)
            "[Image:temp/images/abc123.png]"
        """
        if not image_data:
            self._logger.warning("Empty image data provided")
            return None
        
        try:
            # Detect image format
            image_format = self._detect_format(image_data)
            
            # Compute hash
            image_hash = self._compute_hash(image_data)
            
            # Check for duplicates
            if skip_duplicate and image_hash in self._processed_hashes:
                existing_path = self._processed_hashes[image_hash]
                self._logger.debug(f"Duplicate image detected: {existing_path}")
                return self._build_tag(existing_path)
            
            # Generate filename
            filename = self._generate_filename(image_data, image_format, custom_name)
            file_path = self._build_file_path(filename)
            
            # Check external duplicate tracking
            if processed_images is not None and file_path in processed_images:
                self._logger.debug(f"Image already processed: {file_path}")
                return self._build_tag(file_path)
            
            # Ensure storage is ready
            self._ensure_storage_ready()
            
            # Save using storage backend
            if not self._storage_backend.save(image_data, file_path):
                return None
            
            self._logger.debug(f"Image saved: {file_path}")
            
            # Update tracking
            self._processed_hashes[image_hash] = file_path
            if processed_images is not None:
                processed_images.add(file_path)
            
            return self._build_tag(file_path)
        
        except Exception as e:
            self._logger.error(f"Failed to save image: {e}")
            return None
    
    def process_image(
        self,
        image_data: bytes,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save image data.
        
        This is the main method for format-specific image processing.
        Subclasses should override this method to provide format-specific
        processing logic before saving.
        
        Default implementation simply saves the image.
        
        Args:
            image_data: Raw image binary data
            **kwargs: Format-specific options (e.g., xref, page_num, sheet_name)
            
        Returns:
            Image tag string, or None on failure
            
        Examples:
            >>> processor = ImageProcessor()
            >>> tag = processor.process_image(png_bytes)
            "[Image:temp/images/abc123.png]"
            
            >>> # Subclass example
            >>> class PDFImageProcessor(ImageProcessor):
            ...     def process_image(self, image_data, **kwargs):
            ...         xref = kwargs.get('xref')
            ...         custom_name = f"pdf_xref_{xref}" if xref else None
            ...         return self.save_image(image_data, custom_name=custom_name)
        """
        custom_name = kwargs.get('custom_name')
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_embedded_image(
        self,
        image_data: bytes,
        image_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process embedded image from document.
        
        Override in subclasses for format-specific embedded image handling.
        Default implementation just saves the image.
        
        Args:
            image_data: Image binary data
            image_name: Original image name in document
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        return self.save_image(image_data, custom_name=image_name)
    
    def process_chart_image(
        self,
        chart_data: bytes,
        chart_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process chart as image.
        
        Override in subclasses for format-specific chart image handling.
        Default implementation just saves the image.
        
        Args:
            chart_data: Chart image binary data
            chart_name: Chart name
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        return self.save_image(chart_data, custom_name=chart_name)
    
    def save_image_from_pil(
        self,
        pil_image,
        image_format: Optional[ImageFormat] = None,
        custom_name: Optional[str] = None,
        processed_images: Optional[Set[str]] = None,
        quality: int = 95,
    ) -> Optional[str]:
        """
        Save PIL Image object and return tag.
        
        Args:
            pil_image: PIL Image object
            image_format: Image format to save
            custom_name: Custom filename
            processed_images: Set of processed image paths
            quality: JPEG quality (1-100)
            
        Returns:
            Image tag string, or None on failure
        """
        try:
            from PIL import Image
            
            if not isinstance(pil_image, Image.Image):
                self._logger.error("Invalid PIL Image object")
                return None
            
            fmt = image_format or ImageFormat.PNG
            if fmt == ImageFormat.UNKNOWN:
                fmt = self.config.default_format
            
            buffer = io.BytesIO()
            save_format = fmt.value.upper()
            if save_format == "JPG":
                save_format = "JPEG"
            
            save_kwargs = {}
            if save_format == "JPEG":
                save_kwargs["quality"] = quality
            elif save_format == "PNG":
                save_kwargs["compress_level"] = 6
            
            pil_image.save(buffer, format=save_format, **save_kwargs)
            image_data = buffer.getvalue()
            
            return self.save_image(image_data, custom_name, processed_images)
        
        except Exception as e:
            self._logger.error(f"Failed to save PIL image: {e}")
            return None
    
    def get_processed_count(self) -> int:
        """Return number of processed images."""
        return len(self._processed_hashes)
    
    def get_processed_paths(self) -> List[str]:
        """Return all processed image paths."""
        return list(self._processed_hashes.values())
    
    def clear_cache(self) -> None:
        """Clear internal duplicate tracking cache."""
        self._processed_hashes.clear()
        self._sequential_counter = 0
    
    def cleanup(self, delete_files: bool = False) -> int:
        """
        Clean up resources.
        
        Args:
            delete_files: If True, delete saved files
            
        Returns:
            Number of deleted files
        """
        deleted = 0
        if delete_files:
            for path in self._processed_hashes.values():
                if self._storage_backend.delete(path):
                    deleted += 1
        self.clear_cache()
        return deleted
    
    def get_pattern_string(self) -> str:
        """
        Get regex pattern string for matching image tags.
        
        Returns:
            Regex pattern string
        """
        import re
        prefix = re.escape(self.config.tag_prefix)
        suffix = re.escape(self.config.tag_suffix)
        
        if not self.config.tag_suffix:
            capture = r'(\S+)'
        else:
            first_char = self.config.tag_suffix[0]
            capture = f'([^{re.escape(first_char)}]+)'
        
        return f'{prefix}{capture}{suffix}'


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_IMAGE_CONFIG = {
    "directory_path": "temp/images",
    "tag_prefix": "[Image:",
    "tag_suffix": "]",
    "naming_strategy": NamingStrategy.HASH,
}


# ============================================================================
# Factory Function
# ============================================================================

def create_image_processor(
    directory_path: Optional[str] = None,
    tag_prefix: Optional[str] = None,
    tag_suffix: Optional[str] = None,
    naming_strategy: Optional[Union[NamingStrategy, str]] = None,
    storage_backend: Optional[BaseStorageBackend] = None,
) -> ImageProcessor:
    """
    Create a new ImageProcessor instance.
    
    Args:
        directory_path: Image save directory
        tag_prefix: Tag prefix
        tag_suffix: Tag suffix
        naming_strategy: File naming strategy
        storage_backend: Storage backend instance
        
    Returns:
        ImageProcessor instance
    """
    if naming_strategy is not None and isinstance(naming_strategy, str):
        naming_strategy = NamingStrategy(naming_strategy.lower())
    
    return ImageProcessor(
        directory_path=directory_path or DEFAULT_IMAGE_CONFIG["directory_path"],
        tag_prefix=tag_prefix or DEFAULT_IMAGE_CONFIG["tag_prefix"],
        tag_suffix=tag_suffix or DEFAULT_IMAGE_CONFIG["tag_suffix"],
        naming_strategy=naming_strategy or DEFAULT_IMAGE_CONFIG["naming_strategy"],
        storage_backend=storage_backend,
    )


def save_image_to_file(
    image_data: bytes,
    directory_path: str = "temp",
    tag_prefix: str = "[Image:",
    tag_suffix: str = "]",
    processed_images: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Save image to file and return tag.
    
    Convenience function for quick image saving using local storage.
    
    Args:
        image_data: Image binary data
        directory_path: Save directory
        tag_prefix: Tag prefix
        tag_suffix: Tag suffix
        processed_images: Set for duplicate tracking
        
    Returns:
        Image tag string, or None on failure
    """
    processor = ImageProcessor(
        directory_path=directory_path,
        tag_prefix=tag_prefix,
        tag_suffix=tag_suffix,
    )
    return processor.save_image(image_data, processed_images=processed_images)


__all__ = [
    # Main class
    "ImageProcessor",
    # Config
    "ImageProcessorConfig",
    # Enums
    "ImageFormat",
    "NamingStrategy",
    # Factory function
    "create_image_processor",
    "DEFAULT_IMAGE_CONFIG",
    # Convenience function
    "save_image_to_file",
]
