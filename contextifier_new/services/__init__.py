# contextifier_new/services/__init__.py
"""
Services — Shared Processing Services

Services are stateful, singleton-per-processor objects that provide
cross-cutting functionality to all handlers.

Unlike pipeline components (which are format-specific), services are
format-agnostic and shared:

    ImageService     — Save images, generate image tags, deduplicate
    TagService       — Generate page/slide/sheet structural tags
    ChartService     — Format ChartData into tagged strings
    TableService     — Format TableData into HTML/Markdown/Text
    MetadataService  — Format DocumentMetadata into tagged block
    StorageBackend   — Persist files (local/cloud)

Services are created once by DocumentProcessor and injected into all
handlers through their constructor. This ensures:
- Consistent tag formatting across all handlers
- Shared image dedup state within a processing session
- Single configuration source
"""

from contextifier_new.services.image_service import ImageService
from contextifier_new.services.tag_service import TagService
from contextifier_new.services.chart_service import ChartService
from contextifier_new.services.table_service import TableService
from contextifier_new.services.metadata_service import MetadataService

__all__ = [
    "ImageService",
    "TagService",
    "ChartService",
    "TableService",
    "MetadataService",
]
