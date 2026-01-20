# libs/ocr/ocr_engine/__init__.py
# OCR engine module initialization
"""
OCR Engine Module

Provides OCR engine classes for each LLM provider.
"""

from contextifier.ocr.ocr_engine.openai_ocr import OpenAIOCR
from contextifier.ocr.ocr_engine.anthropic_ocr import AnthropicOCR
from contextifier.ocr.ocr_engine.gemini_ocr import GeminiOCR
from contextifier.ocr.ocr_engine.vllm_ocr import VllmOCR
from contextifier.ocr.ocr_engine.bedrock_ocr import BedrockOCR

__all__ = [
    "OpenAIOCR",
    "AnthropicOCR",
    "GeminiOCR",
    "VllmOCR",
    "BedrockOCR",
]
