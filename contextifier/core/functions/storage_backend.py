# contextifier/core/functions/storage_backend.py
"""
Storage Backend Module

Provides abstract base class and implementations for image storage backends.
ImageProcessor uses these backends to save images to different storage systems.

Storage Backends:
- LocalStorageBackend: Save to local file system
- MinIOStorageBackend: Save to MinIO object storage (stub)
- S3StorageBackend: Save to AWS S3 (stub)

Usage Example:
    from contextifier.core.functions.storage_backend import (
        LocalStorageBackend,
        MinIOStorageBackend,
    )
    from contextifier.core.functions.img_processor import ImageProcessor

    # Use local storage (default)
    processor = ImageProcessor()

    # Use MinIO storage
    minio_backend = MinIOStorageBackend(
        endpoint="localhost:9000",
        bucket="images"
    )
    processor = ImageProcessor(storage_backend=minio_backend)
"""
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("contextify.storage")


class StorageType(Enum):
    """Storage backend types."""
    LOCAL = "local"
    MINIO = "minio"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"  # Google Cloud Storage


class BaseStorageBackend(ABC):
    """
    Abstract base class for storage backends.
    
    Each storage type implements this interface to provide
    storage-specific save/delete logic.
    
    Subclasses must implement:
        - save(): Save data to storage
        - delete(): Delete file from storage
        - exists(): Check if file exists
        - ensure_ready(): Prepare storage (create dirs, validate connection)
    """
    
    def __init__(self, storage_type: StorageType):
        self._storage_type = storage_type
        self._logger = logging.getLogger(
            f"contextify.storage.{self.__class__.__name__}"
        )
    
    @property
    def storage_type(self) -> StorageType:
        """Get storage type."""
        return self._storage_type
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger."""
        return self._logger
    
    @abstractmethod
    def save(self, data: bytes, file_path: str) -> bool:
        """
        Save data to storage.
        
        Args:
            data: Binary data to save
            file_path: Target file path or key
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, file_path: str) -> bool:
        """
        Delete file from storage.
        
        Args:
            file_path: File path or key to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """
        Check if file exists in storage.
        
        Args:
            file_path: File path or key to check
            
        Returns:
            True if file exists
        """
        pass
    
    @abstractmethod
    def ensure_ready(self, directory_path: str) -> None:
        """
        Ensure storage is ready (create directory, validate connection, etc.).
        
        Args:
            directory_path: Base directory or bucket path
        """
        pass
    
    def build_url(self, file_path: str) -> str:
        """
        Build URL or path for the saved file.
        
        Override in subclasses for storage-specific URL formats.
        
        Args:
            file_path: File path or key
            
        Returns:
            URL or path string
        """
        return file_path.replace("\\", "/")


class LocalStorageBackend(BaseStorageBackend):
    """
    Local file system storage backend.
    
    Saves files to the local file system.
    """
    
    def __init__(self):
        super().__init__(StorageType.LOCAL)
    
    def save(self, data: bytes, file_path: str) -> bool:
        """Save data to local file."""
        try:
            with open(file_path, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            self._logger.error(f"Failed to save file {file_path}: {e}")
            return False
    
    def delete(self, file_path: str) -> bool:
        """Delete local file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            self._logger.warning(f"Failed to delete file {file_path}: {e}")
            return False
    
    def exists(self, file_path: str) -> bool:
        """Check if local file exists."""
        return os.path.exists(file_path)
    
    def ensure_ready(self, directory_path: str) -> None:
        """Create directory if it doesn't exist."""
        path = Path(directory_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self._logger.debug(f"Created directory: {path}")


class MinIOStorageBackend(BaseStorageBackend):
    """
    MinIO object storage backend (STUB - Not Implemented).
    
    This is a placeholder for MinIO integration.
    Requires minio package to be installed.
    
    Args:
        endpoint: MinIO server endpoint
        access_key: MinIO access key
        secret_key: MinIO secret key
        bucket: Target bucket name
        secure: Use HTTPS (default: True)
    """
    
    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "",
        secret_key: str = "",
        bucket: str = "images",
        secure: bool = True,
    ):
        super().__init__(StorageType.MINIO)
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._secure = secure
        self._client = None
        
        self._logger.warning(
            "MinIOStorageBackend is a stub implementation. "
            "Full implementation is pending."
        )
    
    @property
    def bucket(self) -> str:
        """Get bucket name."""
        return self._bucket
    
    @property
    def endpoint(self) -> str:
        """Get endpoint."""
        return self._endpoint
    
    def save(self, data: bytes, file_path: str) -> bool:
        """Upload data to MinIO bucket."""
        raise NotImplementedError(
            "MinIOStorageBackend.save() is not yet implemented. "
            "Use LocalStorageBackend for now."
        )
    
    def delete(self, file_path: str) -> bool:
        """Delete object from MinIO bucket."""
        raise NotImplementedError(
            "MinIOStorageBackend.delete() is not yet implemented."
        )
    
    def exists(self, file_path: str) -> bool:
        """Check if object exists in MinIO bucket."""
        raise NotImplementedError(
            "MinIOStorageBackend.exists() is not yet implemented."
        )
    
    def ensure_ready(self, directory_path: str) -> None:
        """Initialize MinIO client and ensure bucket exists."""
        raise NotImplementedError(
            "MinIOStorageBackend.ensure_ready() is not yet implemented."
        )
    
    def build_url(self, file_path: str) -> str:
        """Build MinIO URL for the file."""
        # Would return presigned URL or object path
        protocol = "https" if self._secure else "http"
        return f"{protocol}://{self._endpoint}/{self._bucket}/{file_path}"


class S3StorageBackend(BaseStorageBackend):
    """
    AWS S3 storage backend (STUB - Not Implemented).
    
    This is a placeholder for AWS S3 integration.
    Requires boto3 package to be installed.
    
    Args:
        bucket: S3 bucket name
        region: AWS region (default: "us-east-1")
        prefix: Key prefix for uploaded objects
    """
    
    def __init__(
        self,
        bucket: str = "",
        region: str = "us-east-1",
        prefix: str = "",
    ):
        super().__init__(StorageType.S3)
        self._bucket = bucket
        self._region = region
        self._prefix = prefix
        self._client = None
        
        self._logger.warning(
            "S3StorageBackend is a stub implementation. "
            "Full implementation is pending."
        )
    
    @property
    def bucket(self) -> str:
        """Get bucket name."""
        return self._bucket
    
    @property
    def region(self) -> str:
        """Get region."""
        return self._region
    
    def save(self, data: bytes, file_path: str) -> bool:
        """Upload data to S3 bucket."""
        raise NotImplementedError(
            "S3StorageBackend.save() is not yet implemented. "
            "Use LocalStorageBackend for now."
        )
    
    def delete(self, file_path: str) -> bool:
        """Delete object from S3 bucket."""
        raise NotImplementedError(
            "S3StorageBackend.delete() is not yet implemented."
        )
    
    def exists(self, file_path: str) -> bool:
        """Check if object exists in S3 bucket."""
        raise NotImplementedError(
            "S3StorageBackend.exists() is not yet implemented."
        )
    
    def ensure_ready(self, directory_path: str) -> None:
        """Initialize S3 client and verify bucket access."""
        raise NotImplementedError(
            "S3StorageBackend.ensure_ready() is not yet implemented."
        )
    
    def build_url(self, file_path: str) -> str:
        """Build S3 URL for the file."""
        # Would return S3 URI or presigned URL
        return f"s3://{self._bucket}/{file_path}"


# Default backend instance
_default_backend = LocalStorageBackend()


def get_default_backend() -> BaseStorageBackend:
    """Get the default storage backend (local)."""
    return _default_backend


def create_storage_backend(
    storage_type: StorageType = StorageType.LOCAL,
    **kwargs
) -> BaseStorageBackend:
    """
    Factory function to create a storage backend.
    
    Args:
        storage_type: Type of storage backend
        **kwargs: Storage-specific options
        
    Returns:
        BaseStorageBackend instance
    """
    if storage_type == StorageType.LOCAL:
        return LocalStorageBackend()
    elif storage_type == StorageType.MINIO:
        return MinIOStorageBackend(**kwargs)
    elif storage_type == StorageType.S3:
        return S3StorageBackend(**kwargs)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


__all__ = [
    # Enum
    "StorageType",
    # Base class
    "BaseStorageBackend",
    # Implementations
    "LocalStorageBackend",
    "MinIOStorageBackend",
    "S3StorageBackend",
    # Factory
    "create_storage_backend",
    "get_default_backend",
]
