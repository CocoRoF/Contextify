# contextifier/core/processor/doc_helpers/doc_image_processor.py
"""
DOC Image Processor

Provides DOC-specific image processing that inherits from ImageProcessor.
Handles images from RTF, OLE compound documents, and HTML-formatted DOC files.
"""
import logging
from typing import Any, Dict, Optional, Set

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend

logger = logging.getLogger("contextify.image_processor.doc")


class DOCImageProcessor(ImageProcessor):
    """
    DOC-specific image processor.
    
    Inherits from ImageProcessor and provides DOC-specific processing.
    
    Handles:
    - RTF embedded images (pict, shppict, blipuid)
    - OLE compound document images (Pictures stream, embedded objects)
    - HTML-format DOC images (base64 encoded)
    - WMF/EMF metafiles
    
    Example:
        processor = DOCImageProcessor()
        
        # Process RTF picture
        tag = processor.process_image(image_data, source="rtf", blipuid="abc123")
        
        # Process OLE embedded image
        tag = processor.process_ole_image(ole_data, stream_name="Pictures/image1.png")
        
        # Process HTML base64 image
        tag = processor.process_html_image(base64_data, src_attr="data:image/png;base64,...")
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
    ):
        """
        Initialize DOCImageProcessor.
        
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
        self._processed_blipuids: Set[str] = set()
    
    def process_image(
        self,
        image_data: bytes,
        source: Optional[str] = None,
        blipuid: Optional[str] = None,
        stream_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save DOC image data.
        
        Args:
            image_data: Raw image binary data
            source: Image source type ("rtf", "ole", "html")
            blipuid: RTF BLIP unique ID (for deduplication)
            stream_name: OLE stream name
            **kwargs: Additional options
            
        Returns:
            Image tag string or None if processing failed
        """
        # Custom naming based on source
        custom_name = None
        
        if source == "rtf" and blipuid:
            # Use blipuid for RTF images (deduplication key)
            if blipuid in self._processed_blipuids:
                logger.debug(f"Skipping duplicate RTF image: {blipuid}")
                return None
            self._processed_blipuids.add(blipuid)
            custom_name = f"rtf_{blipuid[:16]}"
        elif source == "ole" and stream_name:
            # Use stream name for OLE images
            import os
            custom_name = f"ole_{os.path.basename(stream_name).split('.')[0]}"
        elif source == "html":
            custom_name = None  # Use hash-based naming
        
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
            source="ole",
            stream_name=stream_name,
            **kwargs
        )
    
    def process_rtf_image(
        self,
        image_data: bytes,
        blipuid: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process RTF embedded image.
        
        Args:
            image_data: Raw image binary data from RTF
            blipuid: BLIP unique ID for deduplication
            **kwargs: Additional options
            
        Returns:
            Image tag string or None if processing failed
        """
        return self.process_image(
            image_data,
            source="rtf",
            blipuid=blipuid,
            **kwargs
        )
    
    def process_html_image(
        self,
        image_data: bytes,
        src_attr: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process HTML-format DOC base64 image.
        
        Args:
            image_data: Decoded image binary data
            src_attr: Original src attribute value
            **kwargs: Additional options
            
        Returns:
            Image tag string or None if processing failed
        """
        return self.process_image(
            image_data,
            source="html",
            **kwargs
        )
    
    def reset_tracking(self) -> None:
        """Reset processed image tracking for new document."""
        self._processed_blipuids.clear()
