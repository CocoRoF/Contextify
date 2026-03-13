# contextifier_new/ocr/engines/anthropic_engine.py
"""Anthropic Claude Vision OCR engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from contextifier_new.ocr.base import BaseOCREngine


class AnthropicOCREngine(BaseOCREngine):
    """OCR engine using Anthropic Claude Vision API."""

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        super().__init__(llm_client, prompt=prompt)

    @property
    def provider(self) -> str:
        return "anthropic"

    def build_message_content(
        self,
        b64_image: str,
        mime_type: str,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": b64_image,
                },
            },
            {"type": "text", "text": prompt},
        ]
