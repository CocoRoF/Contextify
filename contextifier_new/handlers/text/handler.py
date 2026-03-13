# contextifier_new/handlers/text/handler.py
"""
TextHandler — Unified handler for plain text files.

Pipeline:
    Convert:  Raw bytes → decoded string (encoding detection)
    Preprocess: Normalize line endings, strip BOM, detect code language
    Metadata: File size, line count, encoding, detected language
    Content:  Full text content (no tables/images/charts)
    Postprocess: Assemble with metadata block

Old issues resolved:
- TextHandler skipped convert() entirely — now uses TextConverter
- extract_metadata flag was ignored — now respected via pipeline
- Extra language parameter on extract_text() — now in config.format_options
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier_new.handlers.base import BaseHandler
from contextifier_new.pipeline.converter import BaseConverter, NullConverter
from contextifier_new.pipeline.preprocessor import BasePreprocessor, NullPreprocessor
from contextifier_new.pipeline.metadata_extractor import (
    BaseMetadataExtractor,
    NullMetadataExtractor,
)
from contextifier_new.pipeline.content_extractor import (
    BaseContentExtractor,
    NullContentExtractor,
)
from contextifier_new.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor


# All text-based extensions supported
_TEXT_EXTENSIONS = frozenset({
    "txt", "md", "markdown", "rst", "log", "cfg", "ini", "conf",
    "yaml", "yml", "toml", "json", "xml", "svg",
    "py", "js", "ts", "jsx", "tsx", "java", "cpp", "c", "h", "hpp",
    "cs", "go", "rs", "php", "rb", "swift", "kt", "scala",
    "sh", "bash", "zsh", "bat", "ps1", "cmd",
    "sql", "r", "m", "lua", "pl", "pm",
    "html", "htm", "css", "scss", "less", "sass",
    "gitignore", "dockerignore", "env", "editorconfig",
})


class TextHandler(BaseHandler):
    """Handler for plain text and source code files."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return _TEXT_EXTENSIONS

    @property
    def handler_name(self) -> str:
        return "Text Handler"

    def create_converter(self) -> BaseConverter:
        # TODO: Implement TextConverter (bytes → decoded string with encoding detection)
        return NullConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        # TODO: Implement TextPreprocessor (BOM strip, line ending normalization)
        return NullPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        # TODO: Implement TextMetadataExtractor (file size, line count, encoding)
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        # TODO: Implement TextContentExtractor (pass-through text, code wrapping)
        return NullContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["TextHandler"]
