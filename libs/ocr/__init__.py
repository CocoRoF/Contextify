# libs/ocr/__init__.py
# OCR 모듈 패키지 초기화
"""
OCR 처리 모듈

이 모듈은 다양한 LLM Vision 모델을 사용하여 이미지에서 텍스트를 추출하는
OCR 기능을 제공합니다.

사용 예시:
    ```python
    from libs.ocr.ocr_engine import OpenAIOCR, AnthropicOCR, GeminiOCR, VllmOCR

    # OpenAI Vision 모델로 OCR 처리
    ocr = OpenAIOCR(api_key="sk-...", model="gpt-4o")
    result = await ocr.convert_image_to_text("/path/to/image.png")

    # Anthropic Claude Vision 모델로 OCR 처리
    ocr = AnthropicOCR(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
    result = await ocr.convert_image_to_text("/path/to/image.png")

    # Google Gemini Vision 모델로 OCR 처리
    ocr = GeminiOCR(api_key="...", model="gemini-2.0-flash")
    result = await ocr.convert_image_to_text("/path/to/image.png")

    # vLLM 기반 Vision 모델로 OCR 처리
    ocr = VllmOCR(base_url="http://localhost:8000/v1", model="Qwen/Qwen2-VL-7B-Instruct")
    result = await ocr.convert_image_to_text("/path/to/image.png")
    ```

클래스:
    - BaseOCR: OCR 처리를 위한 추상 기본 클래스
    - OpenAIOCR: OpenAI Vision 모델 기반 OCR (ocr_engine 모듈)
    - AnthropicOCR: Anthropic Claude Vision 모델 기반 OCR (ocr_engine 모듈)
    - GeminiOCR: Google Gemini Vision 모델 기반 OCR (ocr_engine 모듈)
    - VllmOCR: vLLM 기반 Vision 모델 OCR (ocr_engine 모듈)
"""

from libs.ocr.base import BaseOCR
from libs.ocr.ocr_engine import OpenAIOCR, AnthropicOCR, GeminiOCR, VllmOCR
from libs.ocr.ocr_processor import (
    IMAGE_TAG_PATTERN,
    extract_image_tags,
    load_image_from_path,
    convert_image_to_text_with_llm,
    process_text_with_ocr,
    process_text_with_ocr_progress,
    process_batch_texts_with_ocr,
    _b64_from_file,
    _get_mime_type,
)

__all__ = [
    # Base Class
    "BaseOCR",
    # OCR Engines
    "OpenAIOCR",
    "AnthropicOCR",
    "GeminiOCR",
    "VllmOCR",
    # Functions
    "IMAGE_TAG_PATTERN",
    "extract_image_tags",
    "load_image_from_path",
    "convert_image_to_text_with_llm",
    "process_text_with_ocr",
    "process_text_with_ocr_progress",
    "process_batch_texts_with_ocr",
]
