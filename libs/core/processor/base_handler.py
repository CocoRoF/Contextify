# libs/core/processor/base_handler.py
"""
BaseHandler - 문서 처리 핸들러 추상 기본 클래스

모든 문서 핸들러의 기본 인터페이스를 정의합니다.
DocumentProcessor에서 전달받은 config와 ImageProcessor를
인스턴스 레벨에서 관리하여 내부 메서드들이 재사용할 수 있도록 합니다.

사용 예:
    class PDFHandler(BaseHandler):
        def extract_text(self, file_path: str, extract_metadata: bool = True) -> str:
            # self.config, self.image_processor 사용 가능
            ...
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from libs.core.functions.img_processor import ImageProcessor

logger = logging.getLogger("document-processor")


class BaseHandler(ABC):
    """
    문서 핸들러 추상 기본 클래스
    
    모든 핸들러는 이 클래스를 상속받아 구현합니다.
    config와 image_processor는 생성 시 전달받아 인스턴스 변수로 저장되며,
    모든 내부 메서드에서 self.config, self.image_processor로 접근할 수 있습니다.
    
    Attributes:
        config: DocumentProcessor에서 전달받은 설정 딕셔너리
        image_processor: DocumentProcessor에서 전달받은 ImageProcessor 인스턴스
        logger: 로깅 인스턴스
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        image_processor: Optional[ImageProcessor] = None
    ):
        """
        BaseHandler 초기화
        
        Args:
            config: 설정 딕셔너리 (DocumentProcessor에서 전달)
            image_processor: ImageProcessor 인스턴스 (DocumentProcessor에서 전달)
        """
        self._config = config or {}
        self._image_processor = image_processor or ImageProcessor()
        self._logger = logging.getLogger(f"document-processor.{self.__class__.__name__}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """설정 딕셔너리"""
        return self._config
    
    @property
    def image_processor(self) -> ImageProcessor:
        """ImageProcessor 인스턴스"""
        return self._image_processor
    
    @property
    def logger(self) -> logging.Logger:
        """로거 인스턴스"""
        return self._logger
    
    @abstractmethod
    def extract_text(
        self,
        file_path: str,
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        파일에서 텍스트를 추출합니다.
        
        Args:
            file_path: 파일 경로
            extract_metadata: 메타데이터 추출 여부
            **kwargs: 추가 옵션
            
        Returns:
            추출된 텍스트
        """
        pass
    
    def save_image(self, image_data: bytes, processed_images: Optional[set] = None) -> Optional[str]:
        """
        이미지를 저장하고 태그를 반환합니다.
        
        편의 메서드로, self.image_processor.save_image()를 래핑합니다.
        
        Args:
            image_data: 이미지 바이너리 데이터
            processed_images: 처리된 이미지 해시 집합 (중복 방지용)
            
        Returns:
            이미지 태그 문자열 또는 None
        """
        return self._image_processor.save_image(image_data, processed_images=processed_images)


__all__ = ["BaseHandler"]
