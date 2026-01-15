# libs/ocr/ocr_engine/vllm_ocr.py
# vLLM 기반 Vision 모델을 사용한 OCR 클래스
import logging
from typing import Any, Optional

from libs.ocr.base import BaseOCR

logger = logging.getLogger("ocr-vllm")

# 기본 모델 (사용자 환경에 따라 다름)
DEFAULT_VLLM_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


class VllmOCR(BaseOCR):
    """
    vLLM 기반 Vision 모델을 사용한 OCR 처리 클래스.

    vLLM 서버에서 제공하는 OpenAI 호환 API를 사용합니다.

    Example:
        ```python
        from libs.ocr.ocr_engine import VllmOCR

        # 방법 1: base_url과 model로 초기화
        ocr = VllmOCR(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen2-VL-7B-Instruct"
        )

        # 방법 2: api_key가 필요한 경우
        ocr = VllmOCR(
            base_url="http://your-vllm-server:8000/v1",
            api_key="your-api-key",
            model="Qwen/Qwen2-VL-7B-Instruct"
        )

        # 방법 3: 기존 LLM 클라이언트 사용
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="Qwen/Qwen2-VL-7B-Instruct",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY"
        )
        ocr = VllmOCR(llm_client=llm)

        # 단일 이미지 변환
        result = await ocr.convert_image_to_text("/path/to/image.png")
        ```
    """

    # vLLM은 간단한 프롬프트 사용
    DEFAULT_PROMPT = "Describe the contents of this image."

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = "EMPTY",
        model: str = DEFAULT_VLLM_MODEL,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        """
        vLLM OCR 초기화.

        Args:
            base_url: vLLM 서버 URL (예: "http://localhost:8000/v1")
            api_key: API 키 (기본값: "EMPTY", vLLM 기본 설정)
            model: 사용할 모델명 (기본값: Qwen/Qwen2-VL-7B-Instruct)
            llm_client: 기존 LangChain 클라이언트 (있으면 base_url, api_key, model 무시)
            prompt: 사용자 정의 프롬프트 (None이면 SIMPLE_PROMPT 사용)
            temperature: 생성 온도 (기본값: 0.0)
            max_tokens: 최대 토큰 수 (None이면 모델 기본값 사용)
        """
        # vLLM은 기본적으로 간단한 프롬프트 사용
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        if llm_client is None:
            if base_url is None:
                raise ValueError("base_url 또는 llm_client 중 하나는 필수입니다.")

            from langchain_openai import ChatOpenAI

            client_kwargs = {
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "temperature": temperature,
            }

            if max_tokens is not None:
                client_kwargs["max_tokens"] = max_tokens

            llm_client = ChatOpenAI(**client_kwargs)
            logger.info(f"[vLLM OCR] 클라이언트 생성 완료: base_url={base_url}, model={model}")

        super().__init__(llm_client=llm_client, prompt=prompt)
        self.model = model
        self.base_url = base_url
        logger.info("[vLLM OCR] 초기화 완료")

    @property
    def provider(self) -> str:
        return "vllm"

    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        return [
            {"type": "text", "text": self.prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
            }
        ]
