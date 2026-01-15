# libs/ocr/base.py
# OCR 모델 추상 클래스 정의
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger("ocr-base")


class BaseOCR(ABC):
    """
    OCR 처리를 위한 추상 기본 클래스.

    모든 OCR 모델 구현체는 이 클래스를 상속받아야 합니다.
    """

    # 기본 프롬프트 (서브클래스에서 오버라이드 가능)
    DEFAULT_PROMPT = (
        "Extract meaningful information from this image.\n\n"
        "**If the image contains a TABLE:**\n"
        "- Convert to HTML table format (<table>, <tr>, <td>, <th>)\n"
        "- Use 'rowspan' and 'colspan' attributes for merged cells\n"
        "- Preserve all cell content exactly as shown\n"
        "- Example:\n"
        "  <table>\n"
        "    <tr><th colspan=\"2\">Header</th></tr>\n"
        "    <tr><td rowspan=\"2\">Merged</td><td>A</td></tr>\n"
        "    <tr><td>B</td></tr>\n"
        "  </table>\n\n"
        "**If the image contains TEXT (non-table):**\n"
        "- Extract all text exactly as shown\n"
        "- Keep layout, hierarchy, and structure\n\n"
        "**If the image contains DATA (charts, graphs, diagrams):**\n"
        "- Extract the data and its meaning\n"
        "- Describe trends, relationships, or key insights\n\n"
        "**If the image is decorative or has no semantic meaning:**\n"
        "- Simply state what it is in one short sentence\n"
        "- Example: 'A decorative geometric shape' or 'Company logo'\n"
        "- Do NOT over-analyze decorative elements\n\n"
        "**Rules:**\n"
        "- Output in Korean (except HTML tags)\n"
        "- Tables MUST use HTML format with proper rowspan/colspan\n"
        "- Be concise - only include what is semantically meaningful\n"
        "- No filler words or unnecessary descriptions"
    )

    # 간단한 프롬프트 (vllm 등에서 사용)
    SIMPLE_PROMPT = "Describe the contents of this image."

    def __init__(self, llm_client: Any, prompt: Optional[str] = None):
        """
        OCR 모델 초기화.

        Args:
            llm_client: LangChain LLM 클라이언트 (Vision 모델 지원 필수)
            prompt: 사용자 정의 프롬프트 (None이면 기본 프롬프트 사용)
        """
        self.llm_client = llm_client
        self.prompt = prompt if prompt is not None else self.DEFAULT_PROMPT

    @property
    @abstractmethod
    def provider(self) -> str:
        """OCR 프로바이더 이름 반환 (예: 'openai', 'anthropic')"""
        pass

    @abstractmethod
    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        """
        LLM에 전달할 메시지 content 구성.

        Args:
            b64_image: Base64 인코딩된 이미지
            mime_type: 이미지 MIME 타입

        Returns:
            LangChain HumanMessage에 전달할 content 리스트
        """
        pass

    async def convert_image_to_text(self, image_path: str) -> Optional[str]:
        """
        이미지를 텍스트로 변환.

        Args:
            image_path: 로컬 이미지 파일 경로

        Returns:
            이미지에서 추출된 텍스트 또는 None (실패 시)
        """
        from libs.ocr.ocr_processor import (
            _b64_from_file,
            _get_mime_type,
        )
        from langchain_core.messages import HumanMessage

        try:
            b64_image = _b64_from_file(image_path)
            mime_type = _get_mime_type(image_path)

            content = self.build_message_content(b64_image, mime_type)
            message = HumanMessage(content=content)

            response = await self.llm_client.ainvoke([message])
            result = response.content.strip()

            # 결과를 [그림:...] 형식으로 감싸기
            result = f"[그림:{result}]"

            logger.info(f"[{self.provider.upper()}] 이미지 텍스트 변환 완료")
            return result

        except Exception as e:
            logger.error(f"[{self.provider.upper()}] 이미지 텍스트 변환 실패: {e}")
            return f"[이미지 변환 오류: {str(e)}]"

    async def process_text(self, text: str) -> str:
        """
        텍스트 내 이미지 태그를 감지하고 OCR 처리하여 텍스트로 대체.

        Args:
            text: [Image:{path}] 태그가 포함된 텍스트

        Returns:
            이미지 태그가 OCR 결과로 대체된 텍스트
        """
        import re
        from libs.ocr.ocr_processor import (
            extract_image_tags,
            load_image_from_path,
        )

        if not self.llm_client:
            logger.warning(f"[{self.provider.upper()}] LLM 클라이언트가 없어 OCR 처리를 건너뜁니다")
            return text

        image_paths = extract_image_tags(text)

        if not image_paths:
            logger.debug(f"[{self.provider.upper()}] 텍스트에 이미지 태그가 없습니다")
            return text

        logger.info(f"[{self.provider.upper()}] {len(image_paths)}개의 이미지 태그 감지됨")

        result_text = text

        for img_path in image_paths:
            tag_pattern = re.compile(r'\[[Ii]mage:' + re.escape(img_path) + r'\]')

            local_path = load_image_from_path(img_path)

            if local_path is None:
                logger.warning(f"[{self.provider.upper()}] 이미지 로드 실패, 원본 태그 유지: {img_path}")
                continue

            ocr_result = await self.convert_image_to_text(local_path)

            if ocr_result is None or ocr_result.startswith("[이미지 변환 오류:"):
                logger.warning(f"[{self.provider.upper()}] 이미지 변환 실패, 원본 태그 유지: {img_path}")
                continue

            result_text = tag_pattern.sub(ocr_result, result_text)
            logger.info(f"[{self.provider.upper()}] 태그 대체 완료: {img_path[:50]}...")

        return result_text

    async def process_batch_texts(self, texts: list) -> list:
        """
        여러 텍스트에 대해 OCR 처리를 수행.

        Args:
            texts: 텍스트 목록

        Returns:
            OCR 처리된 텍스트 목록
        """
        results = []
        for text in texts:
            processed = await self.process_text(text)
            results.append(processed)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider='{self.provider}')"
