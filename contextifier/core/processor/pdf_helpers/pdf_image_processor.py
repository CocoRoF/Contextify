# contextifier/core/processor/pdf_helpers/pdf_image_processor.py
"""
PDF Image Processor

Provides PDF-specific image processing that inherits from ImageProcessor.
Handles XRef images, inline images, and page rendering for complex regions.

This class consolidates all PDF image extraction logic including:
- XRef-based image extraction
- Page region rendering
- Image filtering by size/position
"""
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend

if TYPE_CHECKING:
    import fitz

logger = logging.getLogger("contextify.image_processor.pdf")


class PDFImageProcessor(ImageProcessor):
    """
    PDF-specific image processor.
    
    Inherits from ImageProcessor and provides PDF-specific processing.
    
    Handles:
    - XRef images (embedded images with XRef references)
    - Inline images
    - Page region rendering for complex areas
    - Image extraction from PyMuPDF objects
    
    Example:
        processor = PDFImageProcessor()
        
        # Process XRef image
        tag = processor.process_image(image_data, xref=123)
        
        # Process page region
        tag = processor.process_page_region(page, rect)
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
        dpi: int = 150,
    ):
        """
        Initialize PDFImageProcessor.
        
        Args:
            directory_path: Image save directory
            tag_prefix: Tag prefix for image references
            tag_suffix: Tag suffix for image references
            storage_backend: Storage backend for saving images
            dpi: DPI for page rendering
        """
        super().__init__(
            directory_path=directory_path,
            tag_prefix=tag_prefix,
            tag_suffix=tag_suffix,
            storage_backend=storage_backend,
        )
        self._dpi = dpi
    
    @property
    def dpi(self) -> int:
        """DPI for page rendering."""
        return self._dpi
    
    @dpi.setter
    def dpi(self, value: int) -> None:
        """Set DPI for page rendering."""
        self._dpi = value
    
    def process_image(
        self,
        image_data: bytes,
        xref: Optional[int] = None,
        page_num: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save PDF image data.
        
        Args:
            image_data: Raw image binary data
            xref: Image XRef number (for naming)
            page_num: Page number (for naming)
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        # Generate custom name based on XRef or page
        custom_name = None
        if xref is not None:
            custom_name = f"pdf_xref_{xref}"
        elif page_num is not None:
            custom_name = f"pdf_page_{page_num}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_xref_image(
        self,
        doc: "fitz.Document",
        xref: int,
    ) -> Optional[str]:
        """
        Extract and save image by XRef number.
        
        Args:
            doc: PyMuPDF document object
            xref: Image XRef number
            
        Returns:
            Image tag string, or None on failure
        """
        try:
            import fitz
            
            image_dict = doc.extract_image(xref)
            if not image_dict:
                return None
            
            image_data = image_dict.get("image")
            if not image_data:
                return None
            
            return self.process_image(image_data, xref=xref)
            
        except Exception as e:
            self._logger.warning(f"Failed to extract XRef image {xref}: {e}")
            return None
    
    def process_page_region(
        self,
        page: "fitz.Page",
        rect: "fitz.Rect",
        region_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Render and save a page region as image.
        
        Used for complex regions that can't be represented as text.
        
        Args:
            page: PyMuPDF page object
            rect: Region rectangle to render
            region_name: Optional name for the region
            
        Returns:
            Image tag string, or None on failure
        """
        try:
            import fitz
            
            # Calculate zoom for DPI
            zoom = self._dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Clip to region
            clip = rect
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            image_data = pix.tobytes("png")
            
            custom_name = region_name or f"pdf_page{page.number}_region"
            return self.save_image(image_data, custom_name=custom_name)
            
        except Exception as e:
            self._logger.warning(f"Failed to render page region: {e}")
            return None
    
    def process_embedded_image(
        self,
        image_data: bytes,
        image_name: Optional[str] = None,
        xref: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process embedded PDF image.
        
        Args:
            image_data: Image binary data
            image_name: Original image name
            xref: Image XRef number
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = image_name
        if custom_name is None and xref is not None:
            custom_name = f"pdf_embedded_{xref}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def render_page(
        self,
        page: "fitz.Page",
        alpha: bool = False,
    ) -> Optional[str]:
        """
        Render entire page as image.
        
        Args:
            page: PyMuPDF page object
            alpha: Include alpha channel
            
        Returns:
            Image tag string, or None on failure
        """
        try:
            import fitz
            
            zoom = self._dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=alpha)
            image_data = pix.tobytes("png")
            
            custom_name = f"pdf_page_{page.number + 1}_full"
            return self.save_image(image_data, custom_name=custom_name)
            
        except Exception as e:
            self._logger.warning(f"Failed to render page: {e}")
            return None
    
    def extract_images_from_page(
        self,
        page: "fitz.Page",
        page_num: int,
        doc: "fitz.Document",
        processed_images: Set[int],
        table_bboxes: List[Tuple[float, float, float, float]],
        min_image_size: int = 50,
        min_image_area: int = 2500
    ) -> List[Dict[str, Any]]:
        """
        Extract images from PDF page.
        
        This consolidates the logic from pdf_image.py extract_images_from_page().
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            doc: PyMuPDF document object
            processed_images: Set of already processed image xrefs
            table_bboxes: List of table bounding boxes to exclude
            min_image_size: Minimum image dimension
            min_image_area: Minimum image area
            
        Returns:
            List of dicts with 'content', 'bbox', 'page_num' keys
        """
        from contextifier.core.processor.pdf_helpers.pdf_utils import (
            find_image_position,
            is_inside_any_bbox,
        )
        
        elements = []
        
        try:
            image_list = page.get_images()
            
            for img_info in image_list:
                xref = img_info[0]
                
                if xref in processed_images:
                    continue
                
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    
                    image_bytes = base_image.get("image")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    if width < min_image_size or height < min_image_size:
                        continue
                    if width * height < min_image_area:
                        continue
                    
                    img_bbox = find_image_position(page, xref)
                    if img_bbox is None:
                        continue
                    
                    if is_inside_any_bbox(img_bbox, table_bboxes, threshold=0.7):
                        continue
                    
                    # Use format-specific process_image method
                    image_tag = self.process_image(image_bytes, xref=xref, page_num=page_num)
                    
                    if image_tag:
                        processed_images.add(xref)
                        elements.append({
                            'content': f'\n{image_tag}\n',
                            'bbox': img_bbox,
                            'page_num': page_num
                        })
                
                except Exception as e:
                    logger.debug(f"[PDF] Error extracting image xref={xref}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"[PDF] Error extracting images: {e}")
        
        return elements


__all__ = ["PDFImageProcessor"]
