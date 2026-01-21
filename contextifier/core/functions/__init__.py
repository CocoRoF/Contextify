# libs/core/functions/__init__.py
"""
Functions - Common Utility Functions Module

Provides common utility functions used in document processing.

Module Components:
- utils: Text cleaning, code cleaning, JSON sanitization utilities
- img_processor: Image processing and storage (ImageProcessor class)
- storage_backend: Storage backend implementations (Local, MinIO, S3)
- metadata_extractor: Document metadata extraction interface

Usage Example:
    from contextifier.core.functions import clean_text, clean_code_text
    from contextifier.core.functions import ImageProcessor, save_image_to_file
    from contextifier.core.functions.storage_backend import LocalStorageBackend
    from contextifier.core.functions.utils import sanitize_text_for_json
"""

from contextifier.core.functions.utils import (
    clean_text,
    clean_code_text,
    sanitize_text_for_json,
)

# Storage backend module
from contextifier.core.functions.storage_backend import (
    StorageType,
    BaseStorageBackend,
    LocalStorageBackend,
    MinIOStorageBackend,
    S3StorageBackend,
    create_storage_backend,
    get_default_backend,
)

# Image processor module
from contextifier.core.functions.img_processor import (
    ImageProcessor,
    ImageProcessorConfig,
    ImageFormat,
    NamingStrategy,
    save_image_to_file,
    create_image_processor,
    DEFAULT_IMAGE_CONFIG,
)

# Metadata extraction module
from contextifier.core.functions.metadata_extractor import (
    MetadataField,
    DocumentMetadata,
    MetadataFormatter,
    BaseMetadataExtractor,
    format_metadata,
)

__all__ = [
    # Text utilities
    "clean_text",
    "clean_code_text",
    "sanitize_text_for_json",
    # Storage backends
    "StorageType",
    "BaseStorageBackend",
    "LocalStorageBackend",
    "MinIOStorageBackend",
    "S3StorageBackend",
    "create_storage_backend",
    "get_default_backend",
    # Image processor (base class for all format-specific processors)
    "ImageProcessor",
    "ImageProcessorConfig",
    "ImageFormat",
    "NamingStrategy",
    "save_image_to_file",
    "create_image_processor",
    "DEFAULT_IMAGE_CONFIG",
    # Metadata extraction
    "MetadataField",
    "DocumentMetadata",
    "MetadataFormatter",
    "BaseMetadataExtractor",
    "format_metadata",
]
