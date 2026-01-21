# contextifier/core/processor/excel_helper/excel_image_processor.py
"""
Excel Image Processor

Provides Excel-specific image processing that inherits from ImageProcessor.
Handles embedded images, chart images, and drawing images for XLSX/XLS files.

This class consolidates all Excel image extraction logic including:
- XLSX ZIP-based image extraction
- openpyxl Image object processing
- Sheet image extraction
"""
import os
import logging
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend

if TYPE_CHECKING:
    from openpyxl.workbook import Workbook
    from openpyxl.worksheet.worksheet import Worksheet
    from openpyxl.drawing.image import Image

logger = logging.getLogger("contextify.image_processor.excel")

# Image formats supported by PIL
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']

# Unsupported formats (EMF, WMF, etc.)
UNSUPPORTED_IMAGE_EXTENSIONS = ['.emf', '.wmf']


class ExcelImageProcessor(ImageProcessor):
    """
    Excel-specific image processor.
    
    Inherits from ImageProcessor and provides Excel-specific processing.
    
    Handles:
    - Embedded worksheet images
    - Drawing images
    - Chart images
    - Shape images
    
    Example:
        processor = ExcelImageProcessor()
        
        # Process worksheet image
        tag = processor.process_image(image_data, sheet_name="Sheet1")
        
        # Process from openpyxl Image object
        tag = processor.process_openpyxl_image(image_obj)
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
    ):
        """
        Initialize ExcelImageProcessor.
        
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
        sheet_name: Optional[str] = None,
        image_index: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save Excel image data.
        
        Args:
            image_data: Raw image binary data
            sheet_name: Source sheet name (for naming)
            image_index: Image index in sheet (for naming)
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = None
        if sheet_name is not None:
            safe_sheet = sheet_name.replace(' ', '_').replace('/', '_')
            if image_index is not None:
                custom_name = f"excel_{safe_sheet}_{image_index}"
            else:
                custom_name = f"excel_{safe_sheet}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_openpyxl_image(
        self,
        image: "Image",
        sheet_name: Optional[str] = None,
        image_index: Optional[int] = None,
    ) -> Optional[str]:
        """
        Process openpyxl Image object.
        
        Args:
            image: openpyxl Image object
            sheet_name: Source sheet name
            image_index: Image index
            
        Returns:
            Image tag string, or None on failure
        """
        try:
            # Get image data from openpyxl Image
            if hasattr(image, '_data'):
                image_data = image._data()
            elif hasattr(image, 'ref'):
                # For embedded images with reference
                image_data = image.ref.blob
            else:
                self._logger.warning("Cannot extract data from openpyxl Image")
                return None
            
            return self.process_image(
                image_data,
                sheet_name=sheet_name,
                image_index=image_index
            )
            
        except Exception as e:
            self._logger.warning(f"Failed to process openpyxl image: {e}")
            return None
    
    def process_embedded_image(
        self,
        image_data: bytes,
        image_name: Optional[str] = None,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process embedded Excel image.
        
        Args:
            image_data: Image binary data
            image_name: Original image filename
            sheet_name: Source sheet name
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = image_name
        if custom_name is None and sheet_name is not None:
            safe_sheet = sheet_name.replace(' ', '_').replace('/', '_')
            custom_name = f"excel_embed_{safe_sheet}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_chart_image(
        self,
        chart_data: bytes,
        chart_name: Optional[str] = None,
        sheet_name: Optional[str] = None,
        chart_index: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process Excel chart as image.
        
        Args:
            chart_data: Chart image binary data
            chart_name: Chart title/name
            sheet_name: Source sheet name
            chart_index: Chart index in sheet
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = chart_name
        if custom_name is None:
            if sheet_name is not None:
                safe_sheet = sheet_name.replace(' ', '_').replace('/', '_')
                if chart_index is not None:
                    custom_name = f"excel_chart_{safe_sheet}_{chart_index}"
                else:
                    custom_name = f"excel_chart_{safe_sheet}"
            elif chart_index is not None:
                custom_name = f"excel_chart_{chart_index}"
        
        return self.save_image(chart_data, custom_name=custom_name)
    
    def extract_images_from_xlsx(
        self,
        file_path: str,
    ) -> Dict[str, bytes]:
        """
        Extract images from XLSX file (direct ZIP access).
        Excludes formats not supported by PIL (EMF, WMF, etc.).

        Args:
            file_path: Path to XLSX file

        Returns:
            {image_path: image_bytes} dictionary
        """
        images = {}

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                for name in zf.namelist():
                    if name.startswith('xl/media/'):
                        ext = os.path.splitext(name)[1].lower()
                        if ext in SUPPORTED_IMAGE_EXTENSIONS:
                            images[name] = zf.read(name)
                        elif ext in UNSUPPORTED_IMAGE_EXTENSIONS:
                            logger.debug(f"Skipping unsupported image format: {name}")

            return images

        except Exception as e:
            logger.warning(f"Error extracting images from XLSX: {e}")
            return {}
    
    def get_sheet_images(
        self,
        ws: "Worksheet",
        images_data: Dict[str, bytes],
        file_path: str,
    ) -> List[Tuple[bytes, str]]:
        """
        Get images contained in a sheet.

        Args:
            ws: openpyxl Worksheet object
            images_data: Image dictionary from extract_images_from_xlsx
            file_path: Path to XLSX file

        Returns:
            [(image_bytes, anchor_info)] list
        """
        result = []

        try:
            # Use openpyxl's _images attribute
            if hasattr(ws, '_images') and ws._images:
                for img in ws._images:
                    try:
                        if hasattr(img, '_data') and callable(img._data):
                            img_data = img._data()
                            anchor = str(img.anchor) if hasattr(img, 'anchor') else ""
                            result.append((img_data, anchor))
                    except Exception as e:
                        logger.debug(f"Error accessing image data: {e}")

            # Use directly extracted images (if not obtained above)
            if not result and images_data:
                for name, data in images_data.items():
                    result.append((data, name))

            return result

        except Exception as e:
            logger.warning(f"Error getting sheet images: {e}")
            return []
    
    def process_sheet_images(
        self,
        ws: "Worksheet",
        sheet_name: str,
        images_data: Optional[Dict[str, bytes]] = None,
        file_path: Optional[str] = None,
    ) -> str:
        """
        Process all images in a sheet.

        Args:
            ws: openpyxl Worksheet object
            sheet_name: Sheet name
            images_data: Pre-extracted image dictionary
            file_path: Path to XLSX file

        Returns:
            Joined image tag strings
        """
        results = []
        
        if images_data is None and file_path:
            images_data = self.extract_images_from_xlsx(file_path)
        
        images_data = images_data or {}
        sheet_images = self.get_sheet_images(ws, images_data, file_path or "")
        
        for idx, (img_data, anchor) in enumerate(sheet_images):
            tag = self.process_image(img_data, sheet_name=sheet_name, image_index=idx)
            if tag:
                results.append(tag)
        
        return "\n\n".join(results)


__all__ = ["ExcelImageProcessor"]
