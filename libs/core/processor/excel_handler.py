# your_package/document_processor/excel_handler.py
"""
Excel Handler - Excel 문서 처리기 (XLSX/XLS)

주요 기능:
- 메타데이터 추출 (제목, 작성자, 주제, 키워드, 작성일, 수정일 등)
- 텍스트 추출 (openpyxl/xlrd를 통한 직접 파싱)
- 테이블 추출 (병합셀 유무에 따라 Markdown 또는 HTML 변환)
- 인라인 이미지 추출 및 로컬 저장
- 차트 처리 (1순위: 테이블로 변환, 2순위: matplotlib 이미지)
- 다중 시트 지원

Class-based Handler:
- ExcelHandler 클래스가 BaseHandler를 상속받아 config/image_processor를 관리
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from libs.core.processor.base_handler import BaseHandler
from libs.core.functions.img_processor import ImageProcessor

if TYPE_CHECKING:
    from openpyxl.workbook import Workbook
    from openpyxl.worksheet.worksheet import Worksheet
from libs.core.processor.excel_helper import (
    # Textbox
    extract_textboxes_from_xlsx,
    # Metadata
    extract_xlsx_metadata,
    extract_xls_metadata,
    format_metadata,
    # Chart
    extract_charts_from_xlsx,
    process_chart,
    extract_chart_info_basic,
    # Image
    extract_images_from_xlsx,
    get_sheet_images,
    # Table
    convert_xlsx_sheet_to_table,
    convert_xls_sheet_to_table,
    # Object Detection (개별 테이블 청킹)
    convert_xlsx_objects_to_tables,
    convert_xls_objects_to_tables,
)

import xlrd
from openpyxl import load_workbook

logger = logging.getLogger("document-processor")


# ============================================================================
# ExcelHandler Class
# ============================================================================

class ExcelHandler(BaseHandler):
    """
    Excel 문서 처리 핸들러 (XLSX/XLS)
    
    BaseHandler를 상속받아 config와 image_processor를 인스턴스 레벨에서 관리합니다.
    
    Usage:
        handler = ExcelHandler(config=config, image_processor=image_processor)
        text = handler.extract_text(file_path)
    """
    
    def extract_text(
        self,
        file_path: str,
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Excel 파일에서 텍스트를 추출합니다.
        
        Args:
            file_path: Excel 파일 경로
            extract_metadata: 메타데이터 추출 여부
            **kwargs: 추가 옵션
            
        Returns:
            추출된 텍스트
        """
        ext = os.path.splitext(file_path)[1].lower()
        self.logger.info(f"Excel processing: {file_path}, ext: {ext}")

        if ext == '.xlsx':
            return self._extract_xlsx(file_path, extract_metadata)
        elif ext == '.xls':
            return self._extract_xls(file_path, extract_metadata)
        else:
            raise ValueError(f"지원하지 않는 Excel 형식입니다: {ext}")
    
    def _extract_xlsx(
        self,
        file_path: str,
        extract_metadata: bool = True
    ) -> str:
        """XLSX 파일 처리."""
        self.logger.info(f"XLSX processing: {file_path}")

        try:
            wb = load_workbook(file_path, data_only=True)
            preload = self._preload_xlsx_data(file_path, wb, extract_metadata)

            result_parts = [preload["metadata_str"]] if preload["metadata_str"] else []
            processed_images: Set[str] = set()
            stats = {"charts": 0, "images": 0, "textboxes": 0}

            for sheet_name in wb.sheetnames:
                sheet_result = self._process_xlsx_sheet(
                    wb[sheet_name], sheet_name, preload, processed_images, stats
                )
                result_parts.append(sheet_result)

            remaining = self._process_remaining_charts(
                preload["charts"], preload["chart_idx"], processed_images, stats
            )
            if remaining:
                result_parts.append(remaining)

            result = "".join(result_parts)
            self.logger.info(
                f"XLSX processing completed: {len(wb.sheetnames)} sheets, "
                f"{stats['charts']} charts, {stats['images']} images"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error in XLSX processing: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

    def _extract_xls(
        self,
        file_path: str,
        extract_metadata: bool = True
    ) -> str:
        """XLS 파일 처리."""
        self.logger.info(f"XLS processing: {file_path}")

        try:
            wb = xlrd.open_workbook(file_path, formatting_info=True)
            result_parts = []

            if extract_metadata:
                metadata = extract_xls_metadata(wb)
                metadata_str = format_metadata(metadata)
                if metadata_str:
                    result_parts.append(metadata_str + "\n\n")

            for sheet_idx in range(wb.nsheets):
                ws = wb.sheet_by_index(sheet_idx)
                result_parts.append(f"\n=== 시트: {ws.name} ===\n")

                table_contents = convert_xls_objects_to_tables(ws, wb)
                if table_contents:
                    for i, table_content in enumerate(table_contents, 1):
                        if len(table_contents) > 1:
                            result_parts.append(f"\n[테이블 {i}]\n{table_content}\n")
                        else:
                            result_parts.append(f"\n{table_content}\n")

            result = "".join(result_parts)
            self.logger.info(f"XLS processing completed: {wb.nsheets} sheets")
            return result

        except Exception as e:
            self.logger.error(f"Error in XLS processing: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

    def _preload_xlsx_data(
        self, file_path: str, wb, extract_metadata: bool
    ) -> Dict[str, Any]:
        """XLSX 파일에서 전처리 데이터를 추출합니다."""
        result = {
            "metadata_str": "",
            "charts": [],
            "images_data": [],
            "textboxes_by_sheet": {},
            "chart_idx": 0,
        }

        if extract_metadata:
            metadata = extract_xlsx_metadata(wb)
            result["metadata_str"] = format_metadata(metadata)
            if result["metadata_str"]:
                result["metadata_str"] += "\n\n"

        result["charts"] = extract_charts_from_xlsx(file_path)
        result["images_data"] = extract_images_from_xlsx(file_path)
        result["textboxes_by_sheet"] = extract_textboxes_from_xlsx(file_path)

        return result

    def _process_xlsx_sheet(
        self, ws, sheet_name: str, preload: Dict[str, Any],
        processed_images: Set[str], stats: Dict[str, int]
    ) -> str:
        """XLSX 시트 하나를 처리합니다."""
        parts = [f"\n=== 시트: {sheet_name} ===\n"]

        table_contents = convert_xlsx_objects_to_tables(ws)
        if table_contents:
            for i, table_content in enumerate(table_contents, 1):
                if len(table_contents) > 1:
                    parts.append(f"\n[테이블 {i}]\n{table_content}\n")
                else:
                    parts.append(f"\n{table_content}\n")

        # 차트 처리
        if hasattr(ws, '_charts') and ws._charts:
            charts = preload["charts"]
            for chart in ws._charts:
                if preload["chart_idx"] < len(charts):
                    chart_data = charts[preload["chart_idx"]]
                    # process_chart 시그니처: (chart_info, processed_images, upload_func)
                    chart_output = process_chart(
                        chart_data,
                        processed_images,
                        self.image_processor.save_image
                    )
                    if chart_output:
                        parts.append(f"\n{chart_output}\n")
                        stats["charts"] += 1
                    preload["chart_idx"] += 1

        # 이미지 처리
        # get_sheet_images 반환값: List[Tuple[bytes, str]] - (이미지 바이트, 앵커)
        sheet_images = get_sheet_images(ws, preload["images_data"], "")
        for image_data, anchor in sheet_images:
            if image_data:
                image_tag = self.image_processor.save_image(image_data)
                if image_tag:
                    parts.append(f"\n{image_tag}\n")
                    stats["images"] += 1

        # 텍스트박스
        textboxes = preload["textboxes_by_sheet"].get(sheet_name, [])
        for tb in textboxes:
            if tb.get("text"):
                parts.append(f"\n[텍스트박스] {tb['text']}\n")
                stats["textboxes"] += 1

        return "".join(parts)

    def _process_remaining_charts(
        self, charts: List, chart_idx: int,
        processed_images: Set[str], stats: Dict[str, int]
    ) -> str:
        """남은 차트를 처리합니다."""
        parts = []
        while chart_idx < len(charts):
            chart_data = charts[chart_idx]
            # process_chart 시그니처: (chart_info, processed_images, upload_func)
            chart_output = process_chart(
                chart_data,
                processed_images,
                self.image_processor.save_image
            )
            if chart_output:
                parts.append(f"\n{chart_output}\n")
                stats["charts"] += 1
            chart_idx += 1
        return "".join(parts)


__all__ = ["ExcelHandler"]
