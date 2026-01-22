# contextifier/core/processor/image_file_helper/image_file_image_processor.py
"""
Image File Image Processor

Provides image-file-specific processing that inherits from ImageProcessor.
Handles standalone image files (jpg, png, gif, bmp, webp, etc.).
"""
import logging
from typing import Any, Optional

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend

logger = logging.getLogger("contextify.image_processor.image_file")


class ImageFileImageProcessor(ImageProcessor):
    """
    Image file-specific image processor.
    
    Inherits from ImageProcessor and provides image file-specific processing.
    Handles standalone image files that are the document themselves.
    
    Handles:
    - Standalone image files (jpg, jpeg, png, gif, bmp, webp)
    - Image saving with metadata preservation
    - Format conversion if needed
    
    Example:
        processor = ImageFileImageProcessor()
        
        # Process standalone image
        tag = processor.process_image(image_data, source_path="/path/to/image.png")
        
        # Process with original filename
        tag = processor.process_standalone_image(image_data, original_name="photo.jpg")
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
        preserve_original_name: bool = False,
    ):
        """
        Initialize ImageFileImageProcessor.
        
        Args:
            directory_path: Image save directory
            tag_prefix: Tag prefix for image references
            tag_suffix: Tag suffix for image references
            storage_backend: Storage backend for saving images
            preserve_original_name: Whether to preserve original filename
        """
        super().__init__(
            directory_path=directory_path,
            tag_prefix=tag_prefix,
            tag_suffix=tag_suffix,
            storage_backend=storage_backend,
        )
        self._preserve_original_name = preserve_original_name
    
    @property
    def preserve_original_name(self) -> bool:
        """Whether to preserve original filename."""
        return self._preserve_original_name
    
    def process_image(
        self,
        image_data: bytes,
        source_path: Optional[str] = None,
        original_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save image file data.
        
        Args:
            image_data: Raw image binary data
            source_path: Original file path
            original_name: Original filename
            **kwargs: Additional options
            
        Returns:
            Image tag string or None if processing failed
        """
        # Use original name if preserve option is set
        custom_name = None
        if self._preserve_original_name and original_name:
            import os
            custom_name = os.path.splitext(original_name)[0]
        elif source_path:
            import os
            custom_name = os.path.splitext(os.path.basename(source_path))[0]
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_standalone_image(
        self,
        image_data: bytes,
        original_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process standalone image file.
        
        Specialized method for processing image files that are the document.
        
        Args:
            image_data: Raw image binary data
            original_name: Original filename
            **kwargs: Additional options
            
        Returns:
            Image tag string or None if processing failed
        """
        return self.process_image(
            image_data,
            original_name=original_name,
            **kwargs
        )
