# contextifier_new/ocr/engines/gemini_engine.py
"""Google Gemini Vision OCR engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from contextifier_new.ocr.base import BaseOCREngine


class GeminiOCREngine(BaseOCREngine):
    """OCR engine using Google Gemini Vision API."""

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        super().__init__(llm_client, prompt=prompt)

    @property
    def provider(self) -> str:
        return "gemini"

    def build_message_content(
        self,
        b64_image: str,
        mime_type: str,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
            },
        ]
