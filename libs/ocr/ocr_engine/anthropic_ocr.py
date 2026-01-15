# libs/ocr/ocr_engine/anthropic_ocr.py
# Anthropic Claude Vision 모델을 사용한 OCR 클래스
import logging
from typing import Any, Optional

from libs.ocr.base import BaseOCR

logger = logging.getLogger("ocr-anthropic")

# 기본 모델
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


class AnthropicOCR(BaseOCR):
    """
    Anthropic Claude Vision 모델을 사용한 OCR 처리 클래스.

    지원 모델: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-sonnet-4 등

    Example:
        ```python
        from libs.ocr.ocr_engine import AnthropicOCR

        # 방법 1: api_key와 model로 초기화
        ocr = AnthropicOCR(api_key="sk-ant-...", model="claude-sonnet-4-20250514")

        # 방법 2: 기존 LLM 클라이언트 사용
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, api_key="sk-ant-...")
        ocr = AnthropicOCR(llm_client=llm)

        # 단일 이미지 변환
        result = await ocr.convert_image_to_text("/path/to/image.png")
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        """
        Anthropic OCR 초기화.

        Args:
            api_key: Anthropic API 키 (llm_client가 없을 경우 필수)
            model: 사용할 모델명 (기본값: claude-sonnet-4-20250514)
            llm_client: 기존 LangChain Anthropic 클라이언트 (있으면 api_key, model 무시)
            prompt: 사용자 정의 프롬프트 (None이면 기본 프롬프트 사용)
            temperature: 생성 온도 (기본값: 0.0)
            max_tokens: 최대 토큰 수 (기본값: 4096)
        """
        if llm_client is None:
            if api_key is None:
                raise ValueError("api_key 또는 llm_client 중 하나는 필수입니다.")

            from langchain_anthropic import ChatAnthropic

            llm_client = ChatAnthropic(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info(f"[Anthropic OCR] 클라이언트 생성 완료: model={model}")

        super().__init__(llm_client=llm_client, prompt=prompt)
        self.model = model
        logger.info("[Anthropic OCR] 초기화 완료")

    @property
    def provider(self) -> str:
        return "anthropic"

    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": b64_image
                }
            },
            {"type": "text", "text": self.prompt}
        ]
