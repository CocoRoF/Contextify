# libs/core/processor/csv_handler.py
"""
CSV Handler - CSV/TSV 파일 처리기

Class-based handler for CSV/TSV files inheriting from BaseHandler.
"""
import logging
import os
from typing import Any, Dict, Optional

from libs.core.processor.base_handler import BaseHandler
from libs.core.processor.csv_helper import (
    CSVMetadata,
    extract_csv_metadata,
    format_metadata,
    read_file_with_encoding,
    detect_delimiter,
    parse_csv_content,
    detect_header,
    convert_rows_to_table,
)

logger = logging.getLogger("document-processor")


class CSVHandler(BaseHandler):
    """CSV/TSV 파일 처리 핸들러 클래스"""
    
    def extract_text(
        self,
        file_path: str,
        extract_metadata: bool = True,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        CSV/TSV 파일에서 텍스트를 추출합니다.
        
        Args:
            file_path: CSV/TSV 파일 경로
            extract_metadata: 메타데이터 추출 여부
            encoding: 인코딩 (None이면 자동 감지)
            delimiter: 구분자 (None이면 자동 감지)
            **kwargs: 추가 옵션
            
        Returns:
            추출된 텍스트
        """
        ext = os.path.splitext(file_path)[1].lower()
        self.logger.info(f"CSV processing: {file_path}, ext: {ext}")
        
        if ext == '.tsv' and delimiter is None:
            delimiter = '\t'
        
        try:
            result_parts = []
            
            content, detected_encoding = read_file_with_encoding(file_path, encoding)
            
            if delimiter is None:
                delimiter = detect_delimiter(content)
            
            self.logger.info(f"CSV: encoding={detected_encoding}, delimiter={repr(delimiter)}")
            
            rows = parse_csv_content(content, delimiter)
            
            if not rows:
                return ""
            
            has_header = detect_header(rows)
            
            if extract_metadata:
                metadata = extract_csv_metadata(file_path, detected_encoding, delimiter, rows, has_header)
                metadata_str = format_metadata(metadata)
                if metadata_str:
                    result_parts.append(metadata_str + "\n\n")
            
            table = convert_rows_to_table(rows, has_header)
            if table:
                result_parts.append(table)
            
            result = "".join(result_parts)
            self.logger.info(f"CSV processing completed: {len(rows)} rows")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting text from CSV {file_path}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise
