# contextifier/core/processor/hwpx_helper/hwpx_image_processor.py
"""
HWPX Image Processor

Provides HWPX-specific image processing that inherits from ImageProcessor.
Handles images in HWPX (ZIP/XML based) Korean document format.

This class consolidates all HWPX image extraction logic including:
- BinData images extraction from ZIP
- Remaining images processing
- Image filtering by extension
"""
import logging
import os
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
import zipfile

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend

logger = logging.getLogger("contextify.image_processor.hwpx")

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = frozenset(['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'])


class HWPXImageProcessor(ImageProcessor):
    """
    HWPX-specific image processor.
    
    Inherits from ImageProcessor and provides HWPX-specific processing.
    
    Handles:
    - BinData images in HWPX ZIP structure
    - Embedded images
    - Referenced images via bin_item_map
    
    Example:
        processor = HWPXImageProcessor()
        
        # Process image from ZIP
        with zipfile.ZipFile(file_stream, 'r') as zf:
            tag = processor.process_from_zip(zf, "BinData/image1.png")
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
    ):
        """
        Initialize HWPXImageProcessor.
        
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
        bin_item_id: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save HWPX image data.
        
        Args:
            image_data: Raw image binary data
            bin_item_id: BinItem ID from HWPX
            image_path: Original path in ZIP (for naming)
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = None
        if bin_item_id is not None:
            custom_name = f"hwpx_{bin_item_id}"
        elif image_path is not None:
            # Extract filename from path
            filename = image_path.split('/')[-1] if '/' in image_path else image_path
            # Remove extension and sanitize
            name_base = filename.rsplit('.', 1)[0] if '.' in filename else filename
            custom_name = f"hwpx_{name_base}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_from_zip(
        self,
        zf: zipfile.ZipFile,
        image_path: str,
        bin_item_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Process image from HWPX ZIP archive.
        
        Args:
            zf: ZipFile object
            image_path: Path to image in ZIP
            bin_item_id: BinItem ID
            
        Returns:
            Image tag string, or None on failure
        """
        try:
            with zf.open(image_path) as f:
                image_data = f.read()
            
            return self.process_image(
                image_data,
                bin_item_id=bin_item_id,
                image_path=image_path
            )
            
        except Exception as e:
            self._logger.warning(f"Failed to process image from ZIP {image_path}: {e}")
            return None
    
    def process_embedded_image(
        self,
        image_data: bytes,
        image_name: Optional[str] = None,
        bin_item_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process embedded HWPX image.
        
        Args:
            image_data: Image binary data
            image_name: Original image filename
            bin_item_id: BinItem ID
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = image_name
        if custom_name is None and bin_item_id is not None:
            custom_name = f"hwpx_embed_{bin_item_id}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_bindata_images(
        self,
        zf: zipfile.ZipFile,
        bin_item_map: Dict[str, str],
        exclude_processed: Optional[Set[str]] = None,
    ) -> Dict[str, str]:
        """
        Process all BinData images from HWPX.
        
        Args:
            zf: ZipFile object
            bin_item_map: Mapping of bin_item_id to path
            exclude_processed: Set of already processed IDs to skip
            
        Returns:
            Dictionary mapping bin_item_id to image tag
        """
        exclude = exclude_processed or set()
        result = {}
        
        for bin_id, image_path in bin_item_map.items():
            if bin_id in exclude:
                continue
            
            tag = self.process_from_zip(zf, image_path, bin_item_id=bin_id)
            if tag:
                result[bin_id] = tag
        
        return result
    
    def process_images(
        self,
        zf: zipfile.ZipFile,
        image_files: List[str],
    ) -> str:
        """
        Extract images from HWPX zip and save locally.

        Args:
            zf: Open ZipFile object
            image_files: List of image file paths to process

        Returns:
            Image tag strings joined by newlines
        """
        results = []

        for img_path in image_files:
            ext = os.path.splitext(img_path)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                tag = self.process_from_zip(zf, img_path)
                if tag:
                    results.append(tag)

        return "\n\n".join(results)
    
    def get_remaining_images(
        self,
        zf: zipfile.ZipFile,
        processed_images: Set[str],
    ) -> List[str]:
        """
        Return list of image files not yet processed.

        Args:
            zf: Open ZipFile object
            processed_images: Set of already processed image paths

        Returns:
            List of unprocessed image file paths
        """
        image_files = [
            f for f in zf.namelist()
            if f.startswith("BinData/") and not f.endswith("/")
        ]

        remaining_images = []
        for img in image_files:
            if img not in processed_images:
                remaining_images.append(img)

        return remaining_images
    
    def process_remaining_images(
        self,
        zf: zipfile.ZipFile,
        processed_images: Set[str],
    ) -> str:
        """
        Process all images not yet processed.

        Args:
            zf: Open ZipFile object
            processed_images: Set of already processed image paths

        Returns:
            Image tag strings joined by newlines
        """
        remaining = self.get_remaining_images(zf, processed_images)
        return self.process_images(zf, remaining)


__all__ = ["HWPXImageProcessor"]
