# libs/core/processor/base_handler.py
"""
BaseHandler - Abstract base class for document processing handlers

Defines the base interface for all document handlers.
Manages config, ImageProcessor, PageTagProcessor, ChartProcessor, MetadataExtractor,
Preprocessor, and format-specific ImageProcessor passed from DocumentProcessor at
instance level for reuse by internal methods.

Each handler should override:
- _create_file_converter(): Provide format-specific file converter
- _create_preprocessor(): Provide format-specific preprocessor
- _create_chart_extractor(): Provide format-specific chart extractor
- _create_metadata_extractor(): Provide format-specific metadata extractor
- _create_format_image_processor(): Provide format-specific image processor

Processing Pipeline:
    1. file_converter.convert() - Binary → Format-specific object (e.g., bytes → fitz.Document)
    2. preprocessor.preprocess() - Process/clean the converted data
    3. metadata_extractor.extract() - Extract document metadata
    4. Format-specific content extraction (text, images, charts, tables)

Usage Example:
    class PDFHandler(BaseHandler):
        def _create_file_converter(self):
            return PDFFileConverter()

        def _create_preprocessor(self):
            return PDFPreprocessor()  # Or NullPreprocessor() if no preprocessing needed

        def _create_metadata_extractor(self):
            return PDFMetadataExtractor()

        def _create_format_image_processor(self):
            return PDFImageProcessor(image_processor=self._image_processor)

        def extract_text(self, current_file: CurrentFile, extract_metadata: bool = True) -> str:
            # Step 1: Convert binary to format-specific object
            doc = self.convert_file(current_file)
            # Step 2: Preprocess the converted object
            preprocessed = self.preprocess(doc)
            # Step 3: Extract metadata
            metadata = self.extract_metadata(doc)
            # Step 4: Process content
            ...
"""
import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.page_tag_processor import PageTagProcessor
from contextifier.core.functions.chart_processor import ChartProcessor
from contextifier.core.functions.chart_extractor import BaseChartExtractor, NullChartExtractor
from contextifier.core.functions.metadata_extractor import (
    BaseMetadataExtractor,
    DocumentMetadata,
)
from contextifier.core.functions.file_converter import (
    BaseFileConverter,
    NullFileConverter,
)
from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    NullPreprocessor,
    PreprocessedData,
)

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor")


class NullMetadataExtractor(BaseMetadataExtractor):
    """
    Null implementation of metadata extractor.

    Used as default when no format-specific extractor is provided.
    Always returns empty metadata.
    """

    def extract(self, source: Any) -> DocumentMetadata:
        """Return empty metadata."""
        return DocumentMetadata()


class BaseHandler(ABC):
    """
    Abstract base class for document handlers.

    All handlers inherit from this class.
    config, image_processor, page_tag_processor, chart_processor, metadata_extractor,
    preprocessor, and format_image_processor are passed at creation and stored as
    instance variables.

    Each handler should override:
    - _create_file_converter(): Provide format-specific file converter
    - _create_preprocessor(): Provide format-specific preprocessor
    - _create_chart_extractor(): Provide format-specific chart extractor
    - _create_metadata_extractor(): Provide format-specific metadata extractor
    - _create_format_image_processor(): Provide format-specific image processor

    All are lazy-initialized on first access.

    Processing Pipeline:
        1. file_converter.convert() - Binary → Format-specific object
        2. preprocessor.preprocess() - Process/clean the converted data
        3. metadata_extractor.extract() - Extract document metadata
        4. Format-specific content extraction

    Attributes:
        config: Configuration dictionary passed from DocumentProcessor
        image_processor: Core ImageProcessor instance passed from DocumentProcessor
        format_image_processor: Format-specific image processor (lazy-initialized)
        page_tag_processor: PageTagProcessor instance passed from DocumentProcessor
        chart_processor: ChartProcessor instance passed from DocumentProcessor
        chart_extractor: Format-specific chart extractor instance
        preprocessor: Format-specific preprocessor instance
        metadata_extractor: Format-specific metadata extractor instance
        file_converter: Format-specific file converter instance
        logger: Logging instance
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        image_processor: Optional[ImageProcessor] = None,
        page_tag_processor: Optional[PageTagProcessor] = None,
        chart_processor: Optional[ChartProcessor] = None
    ):
        """
        Initialize BaseHandler.

        Args:
            config: Configuration dictionary (passed from DocumentProcessor)
            image_processor: ImageProcessor instance (passed from DocumentProcessor)
            page_tag_processor: PageTagProcessor instance (passed from DocumentProcessor)
            chart_processor: ChartProcessor instance (passed from DocumentProcessor)
        """
        self._config = config or {}
        self._image_processor = image_processor or ImageProcessor()
        self._page_tag_processor = page_tag_processor or self._get_page_tag_processor_from_config()
        self._chart_processor = chart_processor or self._get_chart_processor_from_config()
        self._chart_extractor: Optional[BaseChartExtractor] = None
        self._metadata_extractor: Optional[BaseMetadataExtractor] = None
        self._file_converter: Optional[BaseFileConverter] = None
        self._preprocessor: Optional[BasePreprocessor] = None
        self._format_image_processor: Optional[ImageProcessor] = None
        self._logger = logging.getLogger(f"document-processor.{self.__class__.__name__}")

    def _get_page_tag_processor_from_config(self) -> PageTagProcessor:
        """Get PageTagProcessor from config or create default."""
        if self._config and "page_tag_processor" in self._config:
            return self._config["page_tag_processor"]
        return PageTagProcessor()

    def _get_chart_processor_from_config(self) -> ChartProcessor:
        """Get ChartProcessor from config or create default."""
        if self._config and "chart_processor" in self._config:
            return self._config["chart_processor"]
        return ChartProcessor()

    def _create_chart_extractor(self) -> BaseChartExtractor:
        """
        Create format-specific chart extractor.

        Override this method in subclasses to provide the appropriate
        chart extractor for the file format.

        Returns:
            BaseChartExtractor subclass instance
        """
        return NullChartExtractor(self._chart_processor)

    def _create_metadata_extractor(self) -> BaseMetadataExtractor:
        """
        Create format-specific metadata extractor.

        Override this method in subclasses to provide the appropriate
        metadata extractor for the file format.

        Returns:
            BaseMetadataExtractor subclass instance
        """
        return NullMetadataExtractor()

    def _create_format_image_processor(self) -> ImageProcessor:
        """
        Create format-specific image processor.

        Override this method in subclasses to provide the appropriate
        image processor for the file format.

        Returns:
            ImageProcessor subclass instance
        """
        return self._image_processor

    def _create_file_converter(self) -> BaseFileConverter:
        """
        Create format-specific file converter.

        Override this method in subclasses to provide the appropriate
        file converter for the file format.

        The file converter transforms raw binary data into a workable
        format-specific object (e.g., Document, Workbook, OLE file).

        Returns:
            BaseFileConverter subclass instance
        """
        return NullFileConverter()

    def _create_preprocessor(self) -> BasePreprocessor:
        """
        Create format-specific preprocessor.

        Override this method in subclasses to provide the appropriate
        preprocessor for the file format.

        The preprocessor processes/cleans the converted data before
        further extraction. This is the SECOND step in the pipeline,
        after file_converter.convert().

        Pipeline:
            1. file_converter.convert() → Format-specific object
            2. preprocessor.preprocess() → Cleaned/processed data
            3. metadata_extractor.extract() → Metadata
            4. Content extraction

        Returns:
            BasePreprocessor subclass instance (NullPreprocessor if no preprocessing needed)
        """
        return NullPreprocessor()

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration dictionary."""
        return self._config

    @property
    def image_processor(self) -> ImageProcessor:
        """ImageProcessor instance."""
        return self._image_processor

    @property
    def page_tag_processor(self) -> PageTagProcessor:
        """PageTagProcessor instance."""
        return self._page_tag_processor

    @property
    def chart_processor(self) -> ChartProcessor:
        """ChartProcessor instance."""
        return self._chart_processor

    @property
    def chart_extractor(self) -> BaseChartExtractor:
        """
        Format-specific chart extractor (lazy-initialized).

        Returns the chart extractor for this handler's file format.
        """
        if self._chart_extractor is None:
            self._chart_extractor = self._create_chart_extractor()
        return self._chart_extractor

    @property
    def metadata_extractor(self) -> BaseMetadataExtractor:
        """
        Format-specific metadata extractor (lazy-initialized).

        Returns the metadata extractor for this handler's file format.
        """
        if self._metadata_extractor is None:
            extractor = self._create_metadata_extractor()
            # If subclass returns None, use NullMetadataExtractor
            self._metadata_extractor = extractor if extractor is not None else NullMetadataExtractor()
        return self._metadata_extractor

    @property
    def format_image_processor(self) -> ImageProcessor:
        """
        Format-specific image processor (lazy-initialized).

        Returns the image processor for this handler's file format.
        Each handler should override _create_format_image_processor() to provide
        format-specific image handling capabilities.
        """
        if self._format_image_processor is None:
            processor = self._create_format_image_processor()
            # If subclass returns None, use default image_processor
            self._format_image_processor = processor if processor is not None else self._image_processor
        return self._format_image_processor

    @property
    def file_converter(self) -> BaseFileConverter:
        """
        Format-specific file converter (lazy-initialized).

        Returns the file converter for this handler's file format.
        Each handler should override _create_file_converter() to provide
        format-specific binary-to-object conversion.
        """
        if self._file_converter is None:
            converter = self._create_file_converter()
            # If subclass returns None, use NullFileConverter
            self._file_converter = converter if converter is not None else NullFileConverter()
        return self._file_converter

    @property
    def preprocessor(self) -> BasePreprocessor:
        """
        Format-specific preprocessor (lazy-initialized).

        Returns the preprocessor for this handler's file format.
        Each handler should override _create_preprocessor() to provide
        format-specific data preprocessing after conversion.

        This is called AFTER file_converter.convert() to process/clean
        the converted data before content extraction.
        """
        if self._preprocessor is None:
            preprocessor = self._create_preprocessor()
            # If subclass returns None, use NullPreprocessor
            self._preprocessor = preprocessor if preprocessor is not None else NullPreprocessor()
        return self._preprocessor

    @property
    def logger(self) -> logging.Logger:
        """Logger instance."""
        return self._logger

    @abstractmethod
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from file.

        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options

        Returns:
            Extracted text
        """
        pass

    def extract_metadata(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from source using format-specific extractor.

        Convenience method that wraps self.metadata_extractor.extract().

        Args:
            source: Format-specific source object

        Returns:
            DocumentMetadata instance
        """
        return self.metadata_extractor.extract(source)

    def format_metadata(self, metadata: DocumentMetadata) -> str:
        """
        Format metadata as string.

        Convenience method that wraps self.metadata_extractor.format().

        Args:
            metadata: DocumentMetadata instance

        Returns:
            Formatted metadata string
        """
        return self.metadata_extractor.format(metadata)

    def extract_and_format_metadata(self, source: Any) -> str:
        """
        Extract and format metadata in one step.

        Convenience method that combines extract and format.

        Args:
            source: Format-specific source object

        Returns:
            Formatted metadata string
        """
        return self.metadata_extractor.extract_and_format(source)

    def convert_file(self, current_file: "CurrentFile", **kwargs) -> Any:
        """
        Convert binary file data to workable format.

        Convenience method that wraps self.file_converter.convert().

        This is the first step in the processing pipeline:
        Binary Data → FileConverter → Workable Object

        Args:
            current_file: CurrentFile dict containing file info and binary data
            **kwargs: Additional format-specific options

        Returns:
            Format-specific workable object (Document, Workbook, OLE file, etc.)
        """
        file_data = current_file.get("file_data", b"")
        file_stream = self.get_file_stream(current_file)
        return self.file_converter.convert(file_data, file_stream, **kwargs)

    def preprocess(self, converted_data: Any, **kwargs) -> PreprocessedData:
        """
        Preprocess the converted data.

        Convenience method that wraps self.preprocessor.preprocess().

        This is the SECOND step in the processing pipeline:
        1. file_converter.convert() → Format-specific object
        2. preprocessor.preprocess() → Cleaned/processed data (THIS STEP)
        3. metadata_extractor.extract() → Metadata
        4. Content extraction

        Args:
            converted_data: The data returned from file_converter.convert()
            **kwargs: Additional format-specific options

        Returns:
            PreprocessedData containing cleaned content and extracted resources
        """
        # If converted_data is bytes, pass it directly
        if isinstance(converted_data, bytes):
            return self.preprocessor.preprocess(converted_data, **kwargs)

        # For other types, the preprocessor should handle it
        # (e.g., Document object preprocessing)
        return self.preprocessor.preprocess(converted_data, **kwargs)

    def get_file_stream(self, current_file: "CurrentFile") -> io.BytesIO:
        """
        Get a fresh BytesIO stream from current_file.

        Resets the stream position to the beginning for reuse.

        Args:
            current_file: CurrentFile dict

        Returns:
            BytesIO stream ready for reading
        """
        stream = current_file.get("file_stream")
        if stream is not None:
            stream.seek(0)
            return stream
        # Fallback: create new stream from file_data
        return io.BytesIO(current_file.get("file_data", b""))

    def save_image(self, image_data: bytes, processed_images: Optional[set] = None) -> Optional[str]:
        """
        Save image and return tag.

        Convenience method that wraps self.image_processor.save_image().

        Args:
            image_data: Image binary data
            processed_images: Set of processed image hashes (for deduplication)

        Returns:
            Image tag string or None
        """
        return self._image_processor.save_image(image_data, processed_images=processed_images)

    def create_page_tag(self, page_number: int) -> str:
        """
        Create a page number tag.

        Convenience method that wraps self.page_tag_processor.create_page_tag().

        Args:
            page_number: Page number

        Returns:
            Page tag string (e.g., "[Page Number: 1]")
        """
        return self._page_tag_processor.create_page_tag(page_number)

    def create_slide_tag(self, slide_number: int) -> str:
        """
        Create a slide number tag.

        Convenience method that wraps self.page_tag_processor.create_slide_tag().

        Args:
            slide_number: Slide number

        Returns:
            Slide tag string (e.g., "[Slide Number: 1]")
        """
        return self._page_tag_processor.create_slide_tag(slide_number)

    def create_sheet_tag(self, sheet_name: str) -> str:
        """
        Create a sheet name tag.

        Convenience method that wraps self.page_tag_processor.create_sheet_tag().

        Args:
            sheet_name: Sheet name

        Returns:
            Sheet tag string (e.g., "[Sheet: Sheet1]")
        """
        return self._page_tag_processor.create_sheet_tag(sheet_name)

    def process_chart(self, chart_element: Any) -> str:
        """
        Process chart element using the format-specific chart extractor.

        This is the main method for chart processing. It uses the chart_extractor
        to extract data from the format-specific chart element and formats it
        using ChartProcessor.

        Args:
            chart_element: Format-specific chart object/element

        Returns:
            Formatted chart text with tags
        """
        return self.chart_extractor.process(chart_element)


__all__ = [
    "BaseHandler",
    "NullMetadataExtractor",
    "BasePreprocessor",
    "NullPreprocessor",
    "PreprocessedData",
]
