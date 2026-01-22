# contextifier/core/processor/text_helper/text_image_processor.py
"""
Text Image Processor

Provides text-specific image processing that inherits from ImageProcessor.
Text files do not contain embedded images, so this is a minimal implementation.
"""
import logging
from typing import Any, Optional

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend

logger = logging.getLogger("contextify.image_processor.text")


class TextImageProcessor(ImageProcessor):
    """
    Text-specific image processor.
    
    Inherits from ImageProcessor and provides text-specific processing.
    Text files do not contain embedded images, so this processor
    provides a consistent interface without additional functionality.
    
    This class exists to maintain interface consistency across all handlers.
    
    Example:
        processor = TextImageProcessor()
        
        # No images in text files, but interface is consistent
        tag = processor.process_image(image_data)  # Falls back to base implementation
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
    ):
        """
        Initialize TextImageProcessor.
        
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
    
    def process_image(
        self,
        image_data: bytes,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save image data.
        
        Text files do not contain embedded images, so this method
        delegates to the base implementation.
        
        Args:
            image_data: Raw image binary data
            **kwargs: Additional options
            
        Returns:
            Image tag string or None if processing failed
        """
        return super().process_image(image_data, **kwargs)
