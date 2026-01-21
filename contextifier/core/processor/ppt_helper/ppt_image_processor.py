# contextifier/core/processor/ppt_helper/ppt_image_processor.py
"""
PPT Image Processor

Provides PPT/PPTX-specific image processing that inherits from ImageProcessor.
Handles slide images, shape images, and embedded pictures.
"""
import logging
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend

if TYPE_CHECKING:
    from pptx import Presentation
    from pptx.slide import Slide
    from pptx.shapes.base import BaseShape

logger = logging.getLogger("contextify.image_processor.ppt")


class PPTImageProcessor(ImageProcessor):
    """
    PPT/PPTX-specific image processor.
    
    Inherits from ImageProcessor and provides PPT-specific processing.
    
    Handles:
    - Picture shapes
    - Embedded images
    - Group shape images
    - Background images
    
    Example:
        processor = PPTImageProcessor()
        
        # Process slide image
        tag = processor.process_image(image_data, slide_num=1)
        
        # Process from shape
        tag = processor.process_picture_shape(shape)
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
    ):
        """
        Initialize PPTImageProcessor.
        
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
        slide_num: Optional[int] = None,
        shape_id: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save PPT image data.
        
        Args:
            image_data: Raw image binary data
            slide_num: Source slide number (for naming)
            shape_id: Shape ID (for naming)
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = None
        if slide_num is not None:
            if shape_id is not None:
                custom_name = f"ppt_slide{slide_num}_shape{shape_id}"
            else:
                custom_name = f"ppt_slide{slide_num}"
        elif shape_id is not None:
            custom_name = f"ppt_shape{shape_id}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_picture_shape(
        self,
        shape: "BaseShape",
        slide_num: Optional[int] = None,
    ) -> Optional[str]:
        """
        Process python-pptx picture shape.
        
        Args:
            shape: Picture shape object
            slide_num: Source slide number
            
        Returns:
            Image tag string, or None on failure
        """
        try:
            if not hasattr(shape, 'image'):
                return None
            
            image = shape.image
            image_data = image.blob
            
            if not image_data:
                return None
            
            shape_id = shape.shape_id if hasattr(shape, 'shape_id') else None
            
            return self.process_image(
                image_data,
                slide_num=slide_num,
                shape_id=shape_id
            )
            
        except Exception as e:
            self._logger.warning(f"Failed to process picture shape: {e}")
            return None
    
    def process_embedded_image(
        self,
        image_data: bytes,
        image_name: Optional[str] = None,
        slide_num: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process embedded PPT image.
        
        Args:
            image_data: Image binary data
            image_name: Original image filename
            slide_num: Source slide number
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = image_name
        if custom_name is None and slide_num is not None:
            custom_name = f"ppt_embed_slide{slide_num}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_group_shape_images(
        self,
        group_shape: "BaseShape",
        slide_num: Optional[int] = None,
    ) -> list:
        """
        Process all images in a group shape.
        
        Args:
            group_shape: Group shape containing other shapes
            slide_num: Source slide number
            
        Returns:
            List of image tags
        """
        tags = []
        
        try:
            if not hasattr(group_shape, 'shapes'):
                return tags
            
            for shape in group_shape.shapes:
                if hasattr(shape, 'image'):
                    tag = self.process_picture_shape(shape, slide_num)
                    if tag:
                        tags.append(tag)
                elif hasattr(shape, 'shapes'):
                    # Nested group
                    nested_tags = self.process_group_shape_images(shape, slide_num)
                    tags.extend(nested_tags)
                    
        except Exception as e:
            self._logger.warning(f"Failed to process group shape: {e}")
        
        return tags


__all__ = ["PPTImageProcessor"]
