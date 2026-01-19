# libs/core/processor/docx_handler.py
"""
DOCX Handler - DOCX 문서 처리기

주요 기능:
- 메타데이터 추출 (제목, 작성자, 주제, 키워드, 작성일, 수정일 등)
- 텍스트 추출 (python-docx를 통한 직접 파싱)
- 테이블 추출 (HTML 형식 보존, rowspan/colspan 지원)
- 인라인 이미지 추출 및 로컬 저장
- 차트 데이터 추출 (OOXML DrawingML Chart 파싱)
- 다이어그램 처리

모든 처리는 python-docx를 통한 직접 Binary 파싱으로 수행됩니다.
이미지 OCR은 별도의 후처리 단계에서 수행됩니다.

Class-based Handler:
- DOCXHandler 클래스가 BaseHandler를 상속받아 config/image_processor를 관리
- 내부 메서드들은 self를 통해 접근
"""
import logging
import traceback
from typing import Any, Dict, Optional, Set

from docx import Document
from lxml import etree

# Base handler
from libs.core.processor.base_handler import BaseHandler
from libs.core.functions.img_processor import ImageProcessor

# docx_helper에서 필요한 것들 import
from libs.core.processor.docx_helper import (
    # Constants
    ElementType,
    # Metadata
    extract_docx_metadata,
    format_metadata,
    # Table
    process_table_element,
    # Paragraph
    process_paragraph_element,
)

logger = logging.getLogger("document-processor")


# ============================================================================
# DOCXHandler Class
# ============================================================================

class DOCXHandler(BaseHandler):
    """
    DOCX 문서 처리 핸들러
    
    BaseHandler를 상속받아 config와 image_processor를 인스턴스 레벨에서 관리합니다.
    
    Usage:
        handler = DOCXHandler(config=config, image_processor=image_processor)
        text = handler.extract_text(file_path)
    """
    
    def extract_text(
        self,
        file_path: str,
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        DOCX 파일에서 텍스트를 추출합니다.
        
        Args:
            file_path: DOCX 파일 경로
            extract_metadata: 메타데이터 추출 여부
            **kwargs: 추가 옵션
            
        Returns:
            추출된 텍스트 (인라인 이미지 태그, 테이블 HTML 포함)
        """
        self.logger.info(f"DOCX processing: {file_path}")
        return self._extract_docx_enhanced(file_path, extract_metadata)
    
    def _extract_docx_enhanced(
        self,
        file_path: str,
        extract_metadata: bool = True
    ) -> str:
        """
        고도화된 DOCX 처리.
        
        - 문서 순서 보존 (body 요소 순회)
        - 메타데이터 추출
        - 인라인 이미지 추출 및 로컬 저장
        - 테이블 HTML 형식 보존 (셀 병합 지원)
        - 차트 데이터 추출
        - 페이지 구분 처리
        """
        self.logger.info(f"Enhanced DOCX processing: {file_path}")

        try:
            doc = Document(file_path)
            result_parts = []
            processed_images: Set[str] = set()
            current_page = 1
            total_tables = 0
            total_images = 0
            total_charts = 0

            # 메타데이터 추출
            if extract_metadata:
                metadata = extract_docx_metadata(doc)
                metadata_str = format_metadata(metadata)
                if metadata_str:
                    result_parts.append(metadata_str + "\n\n")
                    self.logger.info(f"DOCX metadata extracted: {list(metadata.keys())}")

            # 페이지 1 시작
            result_parts.append(f"<페이지 번호> {current_page} </페이지 번호>\n")

            # body 요소를 문서 순서대로 순회
            for body_elem in doc.element.body:
                local_tag = etree.QName(body_elem).localname

                if local_tag == 'p':
                    # Paragraph 처리 - image_processor 전달
                    content, has_page_break, img_count, chart_count = process_paragraph_element(
                        body_elem, doc, processed_images, file_path,
                        image_processor=self.image_processor
                    )

                    if has_page_break:
                        current_page += 1
                        result_parts.append(f"\n<페이지 번호> {current_page} </페이지 번호>\n")

                    if content.strip():
                        result_parts.append(content + "\n")

                    total_images += img_count
                    total_charts += chart_count

                elif local_tag == 'tbl':
                    table_html = process_table_element(body_elem, doc)
                    if table_html:
                        total_tables += 1
                        result_parts.append("\n" + table_html + "\n\n")

                elif local_tag == 'sectPr':
                    continue

            result = "".join(result_parts)
            self.logger.info(f"Enhanced DOCX processing completed: {current_page} pages, "
                           f"{total_tables} tables, {total_images} images, {total_charts} charts")

            return result

        except Exception as e:
            self.logger.error(f"Error in enhanced DOCX processing: {e}")
            self.logger.debug(traceback.format_exc())
            return self._extract_docx_simple_text(file_path)
    
    def _extract_docx_simple_text(self, file_path: str) -> str:
        """간단한 텍스트 추출 (폴백용)."""
        try:
            doc = Document(file_path)
            result_parts = []

            for para in doc.paragraphs:
                if para.text.strip():
                    result_parts.append(para.text)

            for table in doc.tables:
                for row in table.rows:
                    row_texts = []
                    for cell in row.cells:
                        row_texts.append(cell.text.strip())
                    if any(t for t in row_texts):
                        result_parts.append(" | ".join(row_texts))

            return "\n".join(result_parts)

        except Exception as e:
            self.logger.error(f"Error in simple DOCX text extraction: {e}")
            return f"[DOCX 파일 처리 실패: {str(e)}]"


# ============================================================================
# Legacy Function Interface (for backward compatibility)
# ============================================================================

def extract_text_from_docx(
    file_path: str,
    current_config: Dict[str, Any] = None,
    extract_default_metadata: bool = True
) -> str:
    """
    DOCX 파일에서 텍스트를 추출합니다 (레거시 함수 인터페이스).
    
    새 코드에서는 DOCXHandler 클래스를 직접 사용하는 것을 권장합니다.

    Args:
        file_path: DOCX 파일 경로
        current_config: 설정 딕셔너리
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트 (인라인 이미지 태그, 테이블 HTML 포함)
    """
    if current_config is None:
        current_config = {}

    image_processor = current_config.get("image_processor")
    
    handler = DOCXHandler(config=current_config, image_processor=image_processor)
    return handler.extract_text(file_path, extract_metadata=extract_default_metadata)


__all__ = ["DOCXHandler", "extract_text_from_docx"]
