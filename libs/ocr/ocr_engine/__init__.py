# libs/ocr/ocr_engine/__init__.py
# OCR 엔진 모듈 초기화
"""
OCR 엔진 모듈

각 LLM 프로바이더별 OCR 엔진 클래스를 제공합니다.
"""

from libs.ocr.ocr_engine.openai_ocr import OpenAIOCR
from libs.ocr.ocr_engine.anthropic_ocr import AnthropicOCR
from libs.ocr.ocr_engine.gemini_ocr import GeminiOCR
from libs.ocr.ocr_engine.vllm_ocr import VllmOCR

__all__ = [
    "OpenAIOCR",
    "AnthropicOCR",
    "GeminiOCR",
    "VllmOCR",
]
