# libs/core/functions/preprocessor.py
"""
BasePreprocessor - Abstract base class for data preprocessing

Defines the interface for preprocessing data after file conversion.
Used when converted data needs special handling before content extraction.

The preprocessor's job is to:
1. Clean/normalize converted data
2. Extract embedded resources (images, etc.)
3. Detect encoding information
4. Return preprocessed data ready for further processing

Processing Pipeline Position:
    1. FileConverter.convert() → Format-specific object
    2. Preprocessor.preprocess() → Cleaned/processed data (THIS STEP)
    3. MetadataExtractor.extract() → Metadata
    4. Content extraction

Usage:
    class PDFPreprocessor(BasePreprocessor):
        def preprocess(self, converted_data: Any, **kwargs) -> PreprocessedData:
            # Process the fitz.Document, normalize pages, etc.
            return PreprocessedData(
                clean_content=b"",
                encoding="utf-8",
                extracted_resources={"document": converted_data}
            )

        def get_format_name(self) -> str:
            return "PDF Preprocessor"
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PreprocessedData:
    """
    Result of preprocessing operation.

    Contains cleaned content and any extracted resources.

    Attributes:
        raw_content: Original input data (for reference)
        clean_content: Processed content ready for use - THIS IS THE TRUE SOURCE
                      Can be any type: bytes, str, Document, Workbook, OleFileIO, etc.
        encoding: Detected or default encoding (for text-based content)
        extracted_resources: Dict of extracted resources (images, etc.)
        metadata: Any metadata discovered during preprocessing
    """
    raw_content: Any = None
    clean_content: Any = None  # TRUE SOURCE - The processed result
    encoding: str = "utf-8"
    extracted_resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessors.

    Preprocesses converted data after FileConverter.convert().
    Used when converted data needs normalization or special handling
    before content extraction.

    Processing Pipeline:
        1. FileConverter.convert() → Format-specific object
        2. Preprocessor.preprocess() → Cleaned/processed data (THIS STEP)
        3. MetadataExtractor.extract() → Metadata
        4. Content extraction

    Subclasses must implement:
    - preprocess(): Process converted data and return PreprocessedData
    - get_format_name(): Return human-readable format name
    """

    @abstractmethod
    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess converted data.

        Args:
            converted_data: Data from FileConverter.convert()
                           (format-specific object, bytes, or other type)
            **kwargs: Additional format-specific options

        Returns:
            PreprocessedData containing cleaned content and extracted resources

        Raises:
            PreprocessingError: If preprocessing fails
        """
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """
        Return human-readable format name.

        Returns:
            Format name string (e.g., "PDF Preprocessor")
        """
        pass

    def validate(self, data: Any) -> bool:
        """
        Validate if the data can be preprocessed by this preprocessor.

        Override this method to add format-specific validation.
        Default implementation returns True.

        Args:
            data: Data to validate (converted data or raw bytes)

        Returns:
            True if data can be preprocessed, False otherwise
        """
        _ = data  # Suppress unused argument warning
        return True


class NullPreprocessor(BasePreprocessor):
    """
    Null preprocessor that passes data through unchanged.

    Used as default when no preprocessing is needed.
    clean_content always contains the processed result (same as input for pass-through).
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """Pass data through unchanged. clean_content = converted_data."""
        encoding = kwargs.get("encoding", "utf-8")

        # clean_content is ALWAYS the True Source - contains the processed result
        # For pass-through, it's the same as the input
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE
            encoding=encoding,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "Null Preprocessor (pass-through)"


__all__ = [
    'BasePreprocessor',
    'NullPreprocessor',
    'PreprocessedData',
]
