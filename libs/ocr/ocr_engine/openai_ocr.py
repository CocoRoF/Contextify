# libs/ocr/ocr_engine/openai_ocr.py
# OpenAI Vision 모델을 사용한 OCR 클래스
import logging
from typing import Any, Optional

from libs.ocr.base import BaseOCR

logger = logging.getLogger("ocr-openai")

# 기본 모델
DEFAULT_OPENAI_MODEL = "gpt-4o"


class OpenAIOCR(BaseOCR):
    """
    OpenAI Vision 모델을 사용한 OCR 처리 클래스.

    지원 모델: gpt-4-vision-preview, gpt-4o, gpt-4o-mini 등

    Example:
        ```python
        from libs.ocr.ocr_engine import OpenAIOCR

        # 방법 1: api_key와 model로 초기화
        ocr = OpenAIOCR(api_key="sk-...", model="gpt-4o")

        # 방법 2: 기존 LLM 클라이언트 사용
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key="sk-...")
        ocr = OpenAIOCR(llm_client=llm)

        # 단일 이미지 변환
        result = await ocr.convert_image_to_text("/path/to/image.png")

        # 텍스트 내 이미지 태그 처리
        text = "문서 내용 [Image:/path/to/image.png] 계속..."
        processed = await ocr.process_text(text)
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        """
        OpenAI OCR 초기화.

        Args:
            api_key: OpenAI API 키 (llm_client가 없을 경우 필수)
            model: 사용할 모델명 (기본값: gpt-4o)
            llm_client: 기존 LangChain OpenAI 클라이언트 (있으면 api_key, model 무시)
            prompt: 사용자 정의 프롬프트 (None이면 기본 프롬프트 사용)
            temperature: 생성 온도 (기본값: 0.0)
            max_tokens: 최대 토큰 수 (None이면 모델 기본값 사용)
            base_url: OpenAI API base URL (Azure 등 사용 시)
        """
        if llm_client is None:
            if api_key is None:
                raise ValueError("api_key 또는 llm_client 중 하나는 필수입니다.")

            from langchain_openai import ChatOpenAI

            client_kwargs = {
                "model": model,
                "api_key": api_key,
                "temperature": temperature,
            }

            if max_tokens is not None:
                client_kwargs["max_tokens"] = max_tokens

            if base_url is not None:
                client_kwargs["base_url"] = base_url

            llm_client = ChatOpenAI(**client_kwargs)
            logger.info(f"[OpenAI OCR] 클라이언트 생성 완료: model={model}")

        super().__init__(llm_client=llm_client, prompt=prompt)
        self.model = model
        logger.info("[OpenAI OCR] 초기화 완료")

    @property
    def provider(self) -> str:
        return "openai"

    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        return [
            {"type": "text", "text": self.prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
            }
        ]
