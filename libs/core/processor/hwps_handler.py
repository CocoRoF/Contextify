# libs/core/processor/hwpx_processor.py
"""
HWPX Handler - HWPX (ZIP/XML 기반) 문서 처리기

Class-based handler for HWPX files inheriting from BaseHandler.
"""
import zipfile
import logging
from typing import Dict, Any, Set

from libs.core.processor.base_handler import BaseHandler
from libs.core.processor.hwp_helper import MetadataHelper
from libs.core.processor.hwpx_helper import (
    extract_hwpx_metadata,
    parse_bin_item_map,
    parse_hwpx_section,
    process_hwpx_images,
    get_remaining_images,
    extract_charts_from_hwpx,
)

logger = logging.getLogger("document-processor")


class HWPXHandler(BaseHandler):
    """HWPX (ZIP/XML 기반 한글 문서) 처리 핸들러 클래스"""
    
    def extract_text(
        self,
        file_path: str,
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """HWPX 파일에서 텍스트를 추출합니다."""
        text_content = []
        
        try:
            if not zipfile.is_zipfile(file_path):
                self.logger.error(f"Not a valid Zip file: {file_path}")
                return ""
            
            with zipfile.ZipFile(file_path, 'r') as zf:
                if extract_metadata:
                    metadata = extract_hwpx_metadata(zf)
                    metadata_text = MetadataHelper.format_metadata(metadata)
                    if metadata_text:
                        text_content.append(metadata_text)
                        text_content.append("")
                
                bin_item_map = parse_bin_item_map(zf)
                
                section_files = [
                    f for f in zf.namelist() 
                    if f.startswith("Contents/section") and f.endswith(".xml")
                ]
                section_files.sort(key=lambda x: int(x.replace("Contents/section", "").replace(".xml", "")))
                
                processed_images: Set[str] = set()
                
                for sec_file in section_files:
                    with zf.open(sec_file) as f:
                        xml_content = f.read()
                        section_text = parse_hwpx_section(xml_content, zf, bin_item_map, processed_images, image_processor=self.image_processor)
                        text_content.append(section_text)
                
                remaining_images = get_remaining_images(zf, processed_images)
                if remaining_images:
                    image_text = process_hwpx_images(zf, remaining_images, image_processor=self.image_processor)
                    if image_text:
                        text_content.append("\n\n=== Extracted Images (Not Inline) ===\n")
                        text_content.append(image_text)
                
                chart_texts = extract_charts_from_hwpx(zf)
                if chart_texts:
                    text_content.extend(chart_texts)
        
        except Exception as e:
            self.logger.error(f"Error processing HWPX file: {e}")
            return f"Error processing HWPX file: {str(e)}"
        
        return "\n".join(text_content)
