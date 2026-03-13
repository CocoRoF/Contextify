# contextifier_new/ocr/engines/vllm_engine.py
"""Self-hosted VLLM Vision OCR engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from contextifier_new.ocr.base import BaseOCREngine, SIMPLE_OCR_PROMPT


class VLLMOCREngine(BaseOCREngine):
    """OCR engine using a self-hosted VLLM Vision model."""

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        # VLLM uses simple prompt by default
        super().__init__(llm_client, prompt=prompt or SIMPLE_OCR_PROMPT)

    @property
    def provider(self) -> str:
        return "vllm"

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
