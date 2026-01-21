# libs/core/processor/rtf_handler.py
"""
RTF Handler - RTF Document Processor

Class-based handler for RTF files inheriting from BaseHandler.
Provides standard interface for RTF document processing.

This handler can be used directly for .rtf files or called by DOCHandler
when a .doc file is detected to be in RTF format.
"""
import io
import logging
import re
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

from striprtf.striprtf import rtf_to_text

from contextifier.core.processor.base_handler import BaseHandler
from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.chart_extractor import BaseChartExtractor, NullChartExtractor

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile
    from contextifier.core.processor.rtf_helper import RTFDocument

logger = logging.getLogger("document-processor")


class RTFHandler(BaseHandler):
    """RTF Document Processing Handler Class.
    
    Provides standard interface for processing RTF files.
    Can be used directly for .rtf files or called from DOCHandler.
    """
    
    def _create_file_converter(self):
        """Create RTF-specific file converter."""
        from contextifier.core.processor.rtf_helper.rtf_file_converter import RTFFileConverter
        return RTFFileConverter()
    
    def _create_chart_extractor(self) -> BaseChartExtractor:
        """RTF files do not contain charts. Return NullChartExtractor."""
        return NullChartExtractor(self._chart_processor)
    
    def _create_metadata_extractor(self):
        """Create RTF-specific metadata extractor."""
        from contextifier.core.processor.rtf_helper import RTFMetadataExtractor
        return RTFMetadataExtractor()
    
    def _create_format_image_processor(self) -> ImageProcessor:
        """Create RTF-specific image processor."""
        from contextifier.core.processor.rtf_helper.rtf_image_processor import RTFImageProcessor
        return RTFImageProcessor(
            directory_path=self._image_processor.config.directory_path,
            tag_prefix=self._image_processor.config.tag_prefix,
            tag_suffix=self._image_processor.config.tag_suffix,
            storage_backend=self._image_processor.storage_backend,
        )
    
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from RTF file.
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options
            
        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        
        self.logger.info(f"RTF processing: {file_path}")
        
        if not file_data:
            self.logger.error(f"Empty file data: {file_path}")
            return f"[RTF file is empty: {file_path}]"
        
        # Validate RTF format
        if not self.file_converter.validate(file_data):
            self.logger.warning(f"Invalid RTF format: {file_path}")
            return self._extract_fallback(current_file, extract_metadata)
        
        try:
            # Configure converter with image processor
            self.file_converter.configure(
                image_processor=self.format_image_processor,
                processed_images=set()
            )
            
            # Convert to RTFDocument
            rtf_doc = self.file_converter.convert(file_data)
            
            # Extract content
            return self._extract_from_rtf_document(rtf_doc, current_file, extract_metadata)
            
        except Exception as e:
            self.logger.error(f"Error in RTF processing: {e}")
            return self._extract_fallback(current_file, extract_metadata)
    
    def extract_from_rtf_document(
        self,
        rtf_doc: "RTFDocument",
        current_file: "CurrentFile",
        extract_metadata: bool = True
    ) -> str:
        """
        Extract text from pre-converted RTFDocument.
        
        This method is called by DOCHandler when it detects an RTF file.
        
        Args:
            rtf_doc: Pre-converted RTFDocument object
            current_file: CurrentFile dict for context
            extract_metadata: Whether to extract metadata
            
        Returns:
            Extracted text
        """
        return self._extract_from_rtf_document(rtf_doc, current_file, extract_metadata)
    
    def _extract_from_rtf_document(
        self,
        rtf_doc: "RTFDocument",
        current_file: "CurrentFile",
        extract_metadata: bool
    ) -> str:
        """
        Internal method to extract content from RTFDocument.
        
        Args:
            rtf_doc: RTFDocument object
            current_file: CurrentFile dict
            extract_metadata: Whether to extract metadata
            
        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        
        try:
            result_parts = []
            
            # Extract metadata using standard interface
            if extract_metadata and rtf_doc.metadata:
                metadata_str = self.extract_and_format_metadata(rtf_doc.metadata)
                if metadata_str:
                    result_parts.append(metadata_str + "\n\n")
            
            # Add page tag
            page_tag = self.create_page_tag(1)
            result_parts.append(f"{page_tag}\n")
            
            # Extract inline content (preserves table positions)
            inline_content = rtf_doc.get_inline_content()
            if inline_content:
                result_parts.append(inline_content)
            else:
                # Fallback to separate text and tables
                if rtf_doc.text_content:
                    result_parts.append(rtf_doc.text_content)
                
                for table in rtf_doc.tables:
                    if not table.rows:
                        continue
                    if table.is_real_table():
                        result_parts.append("\n" + table.to_html() + "\n")
                    else:
                        result_parts.append("\n" + table.to_text_list() + "\n")
            
            result = "\n".join(result_parts)
            
            # Clean up invalid image tags
            result = re.sub(r'\[image:[^\]]*uploads/\.[^\]]*\]', '', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting from RTFDocument: {e}")
            return self._extract_fallback(current_file, extract_metadata)
    
    def _extract_fallback(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool
    ) -> str:
        """
        Fallback extraction using striprtf library.
        
        Args:
            current_file: CurrentFile dict
            extract_metadata: Whether to extract metadata
            
        Returns:
            Extracted text
        """
        file_data = current_file.get("file_data", b"")
        
        # Try different encodings
        content = None
        for encoding in ['utf-8', 'cp949', 'euc-kr', 'cp1252', 'latin-1']:
            try:
                content = file_data.decode(encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if content is None:
            content = file_data.decode('cp1252', errors='replace')
        
        result_parts = []
        
        # Extract metadata from raw content
        if extract_metadata:
            metadata = self._extract_raw_metadata(content)
            if metadata:
                metadata_str = self.extract_and_format_metadata(metadata)
                if metadata_str:
                    result_parts.append(metadata_str + "\n\n")
        
        # Add page tag
        page_tag = self.create_page_tag(1)
        result_parts.append(f"{page_tag}\n")
        
        # Extract text using striprtf
        try:
            text = rtf_to_text(content)
        except:
            # Manual cleanup
            text = re.sub(r'\\[a-z]+\d*\s?', '', content)
            text = re.sub(r"\\'[0-9a-fA-F]{2}", '', text)
            text = re.sub(r'[{}]', '', text)
        
        if text:
            text = re.sub(r'\n{3,}', '\n\n', text)
            result_parts.append(text.strip())
        
        return "\n".join(result_parts)
    
    def _extract_raw_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from raw RTF content.
        
        Args:
            content: Raw RTF content string
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        patterns = {
            'title': r'\\title\s*\{([^}]*)\}',
            'subject': r'\\subject\s*\{([^}]*)\}',
            'author': r'\\author\s*\{([^}]*)\}',
            'keywords': r'\\keywords\s*\{([^}]*)\}',
            'comments': r'\\doccomm\s*\{([^}]*)\}',
            'last_saved_by': r'\\operator\s*\{([^}]*)\}',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value:
                    metadata[key] = value
        
        return metadata


__all__ = ['RTFHandler']
