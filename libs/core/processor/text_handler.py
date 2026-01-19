# libs/core/processor/text_handler.py
"""
Text Handler - 텍스트 파일 처리기

Class-based handler for text files inheriting from BaseHandler.
"""
import logging
from typing import List, Optional

from libs.core.processor.base_handler import BaseHandler
from libs.core.functions.utils import clean_text, clean_code_text

logger = logging.getLogger("document-processor")


DEFAULT_ENCODINGS = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1', 'ascii']


class TextHandler(BaseHandler):
    """텍스트 파일 처리 핸들러 클래스"""
    
    def extract_text(
        self,
        file_path: str,
        extract_metadata: bool = True,
        file_type: Optional[str] = None,
        encodings: Optional[List[str]] = None,
        is_code: bool = False,
        **kwargs
    ) -> str:
        """
        텍스트 파일에서 텍스트를 추출합니다.
        
        Args:
            file_path: 파일 경로
            extract_metadata: 메타데이터 추출 여부 (텍스트 파일에서는 무시)
            file_type: 파일 타입 (확장자)
            encodings: 시도할 인코딩 목록
            is_code: 코드 파일 여부
            **kwargs: 추가 옵션
            
        Returns:
            추출된 텍스트
        """
        enc = encodings or DEFAULT_ENCODINGS
        
        for e in enc:
            try:
                with open(file_path, 'r', encoding=e) as f:
                    text = f.read()
                self.logger.info(f"Successfully read {file_path} with {e} encoding")
                return clean_code_text(text) if is_code else clean_text(text)
            except UnicodeDecodeError:
                self.logger.debug(f"Failed to read {file_path} with {e}, trying next...")
                continue
            except Exception as ex:
                self.logger.error(f"Error reading file {file_path} with {e}: {ex}")
                continue
        
        raise Exception(f"Could not read file {file_path} with any supported encoding")
