# libs/ocr/ocr_engine/gemini_ocr.py
# Google Gemini Vision 모델을 사용한 OCR 클래스
import logging
from typing import Any, Optional

from libs.ocr.base import BaseOCR

logger = logging.getLogger("ocr-gemini")

# 기본 모델
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class GeminiOCR(BaseOCR):
    """
    Google Gemini Vision 모델을 사용한 OCR 처리 클래스.

    지원 모델: gemini-pro-vision, gemini-1.5-pro, gemini-2.0-flash 등

    Example:
        ```python
        from libs.ocr.ocr_engine import GeminiOCR

        # 방법 1: api_key와 model로 초기화
        ocr = GeminiOCR(api_key="...", model="gemini-2.0-flash")

        # 방법 2: 기존 LLM 클라이언트 사용
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="...")
        ocr = GeminiOCR(llm_client=llm)

        # 단일 이미지 변환
        result = await ocr.convert_image_to_text("/path/to/image.png")
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        """
        Gemini OCR 초기화.

        Args:
            api_key: Google API 키 (llm_client가 없을 경우 필수)
            model: 사용할 모델명 (기본값: gemini-2.0-flash)
            llm_client: 기존 LangChain Gemini 클라이언트 (있으면 api_key, model 무시)
            prompt: 사용자 정의 프롬프트 (None이면 기본 프롬프트 사용)
            temperature: 생성 온도 (기본값: 0.0)
            max_tokens: 최대 토큰 수 (None이면 모델 기본값 사용)
        """
        if llm_client is None:
            if api_key is None:
                raise ValueError("api_key 또는 llm_client 중 하나는 필수입니다.")

            from langchain_google_genai import ChatGoogleGenerativeAI

            client_kwargs = {
                "model": model,
                "google_api_key": api_key,
                "temperature": temperature,
            }

            if max_tokens is not None:
                client_kwargs["max_output_tokens"] = max_tokens

            llm_client = ChatGoogleGenerativeAI(**client_kwargs)
            logger.info(f"[Gemini OCR] 클라이언트 생성 완료: model={model}")

        super().__init__(llm_client=llm_client, prompt=prompt)
        self.model = model
        logger.info("[Gemini OCR] 초기화 완료")

    @property
    def provider(self) -> str:
        return "gemini"

    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        return [
            {"type": "text", "text": self.prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
            }
        ]
