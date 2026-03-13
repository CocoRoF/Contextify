# contextifier_new/ocr/engines/openai_engine.py
"""OpenAI Vision OCR engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from contextifier_new.ocr.base import BaseOCREngine


class OpenAIOCREngine(BaseOCREngine):
    """OCR engine using OpenAI Vision API (GPT-4V / GPT-4o)."""

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        super().__init__(llm_client, prompt=prompt)

    @property
    def provider(self) -> str:
        return "openai"

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
