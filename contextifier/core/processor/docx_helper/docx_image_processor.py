# contextifier/core/processor/docx_helper/docx_image_processor.py
"""
DOCX Image Processor

Provides DOCX-specific image processing that inherits from ImageProcessor.
Handles embedded images, drawing elements (image/diagram), and relationship-based images.

This class consolidates all DOCX image and drawing extraction logic including:
- Drawing/picture element extraction (blip)
- Diagram text extraction from drawings
- Legacy VML pict element processing
- Relationship-based image loading
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from docx.oxml.ns import qn

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend
from contextifier.core.processor.docx_helper.docx_constants import ElementType

if TYPE_CHECKING:
    from docx import Document
    from docx.opc.part import Part

logger = logging.getLogger("contextify.image_processor.docx")

# DOCX XML namespaces
NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
}


class DOCXImageProcessor(ImageProcessor):
    """
    DOCX-specific image processor.
    
    Inherits from ImageProcessor and provides DOCX-specific processing.
    
    Handles:
    - Embedded images via relationships
    - Drawing/picture elements
    - Inline images in runs
    - Shape images
    
    Example:
        processor = DOCXImageProcessor()
        
        # Process relationship-based image
        tag = processor.process_image(image_data, rel_id="rId1")
        
        # Process from part
        tag = processor.process_image_part(image_part)
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
    ):
        """
        Initialize DOCXImageProcessor.
        
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
        rel_id: Optional[str] = None,
        image_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save DOCX image data.
        
        Args:
            image_data: Raw image binary data
            rel_id: Relationship ID (for naming)
            image_name: Original image name
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = image_name
        if custom_name is None and rel_id is not None:
            custom_name = f"docx_{rel_id}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_image_part(
        self,
        image_part: "Part",
        rel_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Process image from OOXML part.
        
        Args:
            image_part: OOXML Part containing image data
            rel_id: Relationship ID
            
        Returns:
            Image tag string, or None on failure
        """
        try:
            image_data = image_part.blob
            if not image_data:
                return None
            
            # Try to get original filename
            image_name = None
            if hasattr(image_part, 'partname'):
                partname = str(image_part.partname)
                if '/' in partname:
                    image_name = partname.split('/')[-1]
            
            return self.process_image(
                image_data,
                rel_id=rel_id,
                image_name=image_name
            )
            
        except Exception as e:
            self._logger.warning(f"Failed to process image part: {e}")
            return None
    
    def process_embedded_image(
        self,
        image_data: bytes,
        image_name: Optional[str] = None,
        embed_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process embedded DOCX image.
        
        Args:
            image_data: Image binary data
            image_name: Original image filename
            embed_id: Embed relationship ID
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = image_name
        if custom_name is None and embed_id is not None:
            custom_name = f"docx_embed_{embed_id}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_drawing_image(
        self,
        image_data: bytes,
        drawing_id: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process DOCX drawing/picture element image.
        
        Args:
            image_data: Image binary data
            drawing_id: Drawing element ID
            description: Image description/alt text
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = None
        if drawing_id is not None:
            custom_name = f"docx_drawing_{drawing_id}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def extract_from_drawing(
        self,
        graphic_data,
        doc: "Document",
        processed_images: Set[str],
    ) -> Tuple[str, bool]:
        """
        Extract image from Drawing graphic data element.
        
        This is the core DOCX image extraction logic that was previously
        in docx_image.py extract_image_from_drawing() function.
        
        Args:
            graphic_data: graphicData XML element
            doc: python-docx Document object
            processed_images: Set of processed image paths (deduplication)
            
        Returns:
            (image_tag, is_image) tuple. image_tag is the tag string or empty,
            is_image indicates if an image was found.
        """
        try:
            # Find blip element (image reference)
            blip = graphic_data.find('.//a:blip', NAMESPACES)
            if blip is None:
                return "", False

            # Get relationship ID
            r_embed = blip.get(qn('r:embed'))
            r_link = blip.get(qn('r:link'))
            rId = r_embed or r_link
            
            if not rId:
                return "", False

            # Find image part from relationship
            try:
                rel = doc.part.rels.get(rId)
                if rel is None:
                    return "", False

                # Extract image data
                if hasattr(rel, 'target_part') and hasattr(rel.target_part, 'blob'):
                    image_data = rel.target_part.blob
                    
                    # Save using process_image with rel_id
                    image_tag = self.process_image(
                        image_data, 
                        rel_id=rId,
                        processed_images=processed_images
                    )
                    
                    if image_tag:
                        return f"\n{image_tag}\n", True

                return "[Unknown Image]", True

            except Exception as e:
                logger.warning(f"Error extracting image from relationship: {e}")
                return "[Unknown Image]", True

        except Exception as e:
            logger.warning(f"Error extracting image from drawing: {e}")
            return "", False
    
    def extract_from_pict(
        self,
        pict_elem,
        doc: "Document",
        processed_images: Set[str],
    ) -> str:
        """
        Extract image from legacy VML pict element.
        
        This is the core DOCX VML image extraction logic that was previously
        in docx_image.py process_pict_element() function.
        
        Args:
            pict_elem: pict XML element
            doc: python-docx Document object
            processed_images: Set of processed image paths (deduplication)
            
        Returns:
            Image tag string or placeholder
        """
        try:
            # Find VML imagedata
            ns_v = 'urn:schemas-microsoft-com:vml'
            ns_r = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

            imagedata = pict_elem.find('.//{%s}imagedata' % ns_v)
            if imagedata is None:
                return "[Unknown Image]"

            rId = imagedata.get('{%s}id' % ns_r)
            if not rId:
                return "[Unknown Image]"

            try:
                rel = doc.part.rels.get(rId)
                if rel and hasattr(rel, 'target_part') and hasattr(rel.target_part, 'blob'):
                    image_data = rel.target_part.blob
                    image_tag = self.process_image(
                        image_data,
                        rel_id=rId,
                        processed_images=processed_images
                    )
                    if image_tag:
                        return f"\n{image_tag}\n"
            except Exception:
                pass

            return "[Unknown Image]"

        except Exception as e:
            logger.warning(f"Error processing pict element: {e}")
            return ""
    
    def process_drawing_element(
        self,
        drawing_elem,
        doc: "Document",
        processed_images: Set[str],
        chart_callback: Optional[Callable[[], str]] = None,
    ) -> Tuple[str, Optional[ElementType]]:
        """
        Process Drawing element (image, chart, diagram).
        
        Main entry point for handling all drawing elements in DOCX.
        Branches to appropriate handler based on content type.
        
        Args:
            drawing_elem: drawing XML element
            doc: python-docx Document object
            processed_images: Set of processed image paths (deduplication)
            chart_callback: Callback function to get next chart content
            
        Returns:
            (content, element_type) tuple
        """
        try:
            # Check inline or anchor
            inline = drawing_elem.find('.//wp:inline', NAMESPACES)
            anchor = drawing_elem.find('.//wp:anchor', NAMESPACES)

            container = inline if inline is not None else anchor
            if container is None:
                return "", None

            # Check graphic data
            graphic = container.find('.//a:graphic', NAMESPACES)
            if graphic is None:
                return "", None

            graphic_data = graphic.find('a:graphicData', NAMESPACES)
            if graphic_data is None:
                return "", None

            uri = graphic_data.get('uri', '')

            # Image case
            if 'picture' in uri.lower():
                content, is_image = self.extract_from_drawing(
                    graphic_data, doc, processed_images
                )
                return (content, ElementType.IMAGE) if is_image else ("", None)

            # Chart case - delegate to callback
            if 'chart' in uri.lower():
                if chart_callback:
                    chart_content = chart_callback()
                    return chart_content, ElementType.CHART
                return "", ElementType.CHART

            # Diagram case
            if 'diagram' in uri.lower():
                return self.extract_diagram(graphic_data)

            return "", None

        except Exception as e:
            logger.warning(f"Error processing drawing element: {e}")
            return "", None
    
    def extract_diagram(
        self,
        graphic_data,
    ) -> Tuple[str, Optional[ElementType]]:
        """
        Extract diagram information from Drawing.
        
        Args:
            graphic_data: graphicData XML element
            
        Returns:
            (content, element_type) tuple
        """
        try:
            texts = []
            ns_a = 'http://schemas.openxmlformats.org/drawingml/2006/main'
            for t_elem in graphic_data.findall('.//{%s}t' % ns_a):
                if t_elem.text:
                    texts.append(t_elem.text.strip())

            if texts:
                return f"[Diagram: {' / '.join(texts)}]", ElementType.DIAGRAM

            return "[Diagram]", ElementType.DIAGRAM

        except Exception as e:
            logger.warning(f"Error extracting diagram: {e}")
            return "[Diagram]", ElementType.DIAGRAM


__all__ = ["DOCXImageProcessor"]
