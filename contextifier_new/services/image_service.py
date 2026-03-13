# contextifier_new/services/image_service.py
"""
ImageService — Image Processing, Storage & Tag Generation

Replaces and unifies the old ImageProcessor (concrete class),
format-specific image processors (DOCXImageProcessor, PDFImageProcessor, etc.),
and the image-to-tag logic from ImageProcessor.save_image().

Design changes from old code:
1. Image SAVING (storage) is delegated to StorageBackend
2. Image TAGGING uses TagService for tag format
3. Duplicate detection via content hashing
4. Format-specific image extraction logic stays in ContentExtractor,
   NOT in the service (separation of concerns)
5. The service is format-agnostic — it handles raw bytes

The format-specific image extraction (e.g., extracting images from
PDF pages, DOCX relationships, PPTX slides) is done by each
format's ContentExtractor. The ContentExtractor calls:
    image_service.save_and_tag(image_data) → str (tag)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Set

from contextifier_new.config import ProcessingConfig, ImageConfig
from contextifier_new.types import NamingStrategy
from contextifier_new.errors import ImageServiceError

from contextifier_new.services.storage.base import BaseStorageBackend
from contextifier_new.services.storage.local import LocalStorageBackend


class ImageService:
    """
    Shared service for saving images and generating image tags.

    Thread-safe for concurrent handler use within one processor.
    """

    def __init__(
        self,
        config: ProcessingConfig,
        storage_backend: Optional[BaseStorageBackend] = None,
    ) -> None:
        self._config = config
        self._image_config: ImageConfig = config.images
        self._tag_config = config.tags
        self._storage = storage_backend or LocalStorageBackend(
            base_path=self._image_config.directory_path
        )
        self._logger = logging.getLogger("contextifier.services.image")

        # Deduplication state
        self._processed_hashes: Set[str] = set()
        self._processed_paths: List[str] = []
        self._counter: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    def save_and_tag(
        self,
        image_data: bytes,
        *,
        custom_name: Optional[str] = None,
        skip_duplicate: Optional[bool] = None,
    ) -> Optional[str]:
        """
        Save image data and return the image tag string.

        This is the primary method for handlers. It:
        1. Checks for duplicates (if enabled)
        2. Generates a filename
        3. Saves via storage backend
        4. Returns the formatted tag

        Args:
            image_data: Raw image bytes.
            custom_name: Optional custom filename (overrides naming strategy).
            skip_duplicate: Override duplicate skipping. None = use config.

        Returns:
            Image tag string (e.g., "[Image:path/to/img.png]"),
            or None if duplicate was skipped.
        """
        if not image_data:
            return None

        should_skip = (
            skip_duplicate if skip_duplicate is not None
            else self._image_config.skip_duplicate
        )

        # Duplicate check
        if should_skip:
            content_hash = self._hash(image_data)
            if content_hash in self._processed_hashes:
                return None
            self._processed_hashes.add(content_hash)

        # Generate filename
        filename = custom_name or self._generate_filename(image_data)

        # Build full path
        file_path = f"{self._image_config.directory_path}/{filename}"

        # Save via storage backend
        try:
            self._storage.save(image_data, file_path)
        except Exception as e:
            raise ImageServiceError(
                f"Failed to save image: {e}",
                context={"path": file_path},
                cause=e,
            )

        self._processed_paths.append(file_path)

        # Build and return tag
        tag = f"{self._tag_config.image_prefix}{file_path}{self._tag_config.image_suffix}"
        return tag

    def get_processed_count(self) -> int:
        """Number of images processed in this session."""
        return len(self._processed_paths)

    def get_processed_paths(self) -> List[str]:
        """List of all saved image paths."""
        return list(self._processed_paths)

    def clear_state(self) -> None:
        """Reset deduplication state and counters."""
        self._processed_hashes.clear()
        self._processed_paths.clear()
        self._counter = 0

    def get_image_tag_pattern(self) -> str:
        """Get regex pattern string for matching image tags."""
        import re
        prefix = re.escape(self._tag_config.image_prefix)
        suffix = re.escape(self._tag_config.image_suffix)
        return rf"{prefix}([^{re.escape(self._tag_config.image_suffix[0])}]+){suffix}"

    # ── Private ───────────────────────────────────────────────────────────

    def _generate_filename(self, image_data: bytes) -> str:
        """Generate a filename using the configured naming strategy."""
        ext = self._image_config.default_format
        strategy = self._image_config.naming_strategy

        if strategy == NamingStrategy.HASH:
            name = hashlib.md5(image_data).hexdigest()[:16]
        elif strategy == NamingStrategy.UUID:
            name = uuid.uuid4().hex[:16]
        elif strategy == NamingStrategy.SEQUENTIAL:
            self._counter += 1
            name = f"img_{self._counter:04d}"
        elif strategy == NamingStrategy.TIMESTAMP:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = f"img_{ts}"
        else:
            name = hashlib.md5(image_data).hexdigest()[:16]

        return f"{name}.{ext}"

    @staticmethod
    def _hash(data: bytes) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(data).hexdigest()


__all__ = ["ImageService"]
