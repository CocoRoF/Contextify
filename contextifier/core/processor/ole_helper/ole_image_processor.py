# contextifier/core/processor/ole_helper/ole_image_processor.py
"""
OLE Image Processor

Provides OLE-specific image processing that inherits from ImageProcessor.
Handles images from OLE compound documents (binary DOC files).
"""
import logging
from typing import Any, Dict, Optional, Set

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend

logger = logging.getLogger("contextify.image_processor.ole")


class OLEImageProcessor(ImageProcessor):
    """
    OLE-specific image processor.
    
    Inherits from ImageProcessor and provides OLE-specific processing.
    
    Handles:
    - OLE compound document images (Pictures stream, embedded objects)
    - WMF/EMF metafiles
    - Embedded images from Word binary format
    
    Example:
        processor = OLEImageProcessor()
        
        # Process OLE embedded image
        tag = processor.process_image(ole_data, stream_name="Pictures/image1.png")
        
        # Process OLE Pictures stream
        tag = processor.process_ole_image(ole_data, stream_name="Pictures/image1.png")
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
    ):
        """
        Initialize OLEImageProcessor.
        
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
        self._processed_streams: Set[str] = set()
    
    def process_image(
        self,
        image_data: bytes,
        stream_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save OLE image data.
        
        Args:
            image_data: Raw image binary data
            stream_name: OLE stream name
            **kwargs: Additional options
            
        Returns:
            Image tag string or None if processing failed
        """
        # Custom naming based on stream name
        custom_name = None
        
        if stream_name:
            # Use stream name for OLE images (deduplication)
            if stream_name in self._processed_streams:
                logger.debug(f"Skipping duplicate OLE image: {stream_name}")
                return None
            self._processed_streams.add(stream_name)
            
            import os
            custom_name = f"ole_{os.path.basename(stream_name).split('.')[0]}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_ole_image(
        self,
        image_data: bytes,
        stream_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process OLE compound document embedded image.
        
        Args:
            image_data: Raw image binary data from OLE stream
            stream_name: Name of the OLE stream
            **kwargs: Additional options
            
        Returns:
            Image tag string or None if processing failed
        """
        return self.process_image(
            image_data,
            stream_name=stream_name,
            **kwargs
        )
    
    def reset_tracking(self) -> None:
        """Reset processed image tracking for new document."""
        self._processed_streams.clear()


__all__ = ['OLEImageProcessor']
