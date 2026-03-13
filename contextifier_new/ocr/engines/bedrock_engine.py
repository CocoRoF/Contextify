# contextifier_new/ocr/engines/bedrock_engine.py
"""AWS Bedrock (Claude) Vision OCR engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from contextifier_new.ocr.base import BaseOCREngine


class BedrockOCREngine(BaseOCREngine):
    """OCR engine using AWS Bedrock Claude Vision API."""

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        super().__init__(llm_client, prompt=prompt)

    @property
    def provider(self) -> str:
        return "aws_bedrock"

    def build_message_content(
        self,
        b64_image: str,
        mime_type: str,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        # Bedrock Claude uses the same format as direct Anthropic
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
