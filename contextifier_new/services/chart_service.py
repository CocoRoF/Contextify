# contextifier_new/services/chart_service.py
"""
ChartService — Chart Data Formatting

Replaces and unifies:
- Old ChartProcessor (concrete formatting class)
- Duplicated _format_chart_data() in 5 handlers (DOCX, PPT, Excel, HWP, HWPX)
- Old ChartExtractor.process() method

Design:
- ChartService receives ChartData and produces formatted text blocks
- Uses TagService for chart tag wrapping
- Supports HTML table and text-based chart rendering
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from contextifier_new.config import ProcessingConfig, ChartConfig
from contextifier_new.types import ChartData, ChartSeries


# OOXML chart type name mapping
_CHART_TYPE_MAP: Dict[str, str] = {
    "barChart": "Bar Chart",
    "bar3DChart": "3D Bar Chart",
    "lineChart": "Line Chart",
    "line3DChart": "3D Line Chart",
    "pieChart": "Pie Chart",
    "pie3DChart": "3D Pie Chart",
    "doughnutChart": "Doughnut Chart",
    "areaChart": "Area Chart",
    "area3DChart": "3D Area Chart",
    "scatterChart": "Scatter Chart",
    "radarChart": "Radar Chart",
    "surfaceChart": "Surface Chart",
    "surface3DChart": "3D Surface Chart",
    "bubbleChart": "Bubble Chart",
    "stockChart": "Stock Chart",
}


class ChartService:
    """
    Formats ChartData instances into tagged text blocks.

    Example output (HTML table mode):
        [chart]
        Chart Type: Bar Chart
        Title: Sales by Region
        <table>
        <tr><th>Category</th><th>Series 1</th><th>Series 2</th></tr>
        <tr><td>Q1</td><td>100</td><td>200</td></tr>
        ...
        </table>
        [/chart]
    """

    def __init__(self, config: ProcessingConfig) -> None:
        self._config = config
        self._chart_config: ChartConfig = config.charts
        self._tag_config = config.tags
        self._logger = logging.getLogger("contextifier.services.chart")

    def format_chart(self, chart_data: ChartData) -> str:
        """
        Format a ChartData instance into a tagged text block.

        Args:
            chart_data: Standardized chart data.

        Returns:
            Formatted chart text wrapped in chart tags.
        """
        parts: List[str] = []

        # Chart type
        if self._chart_config.include_chart_type and chart_data.chart_type:
            display_type = _CHART_TYPE_MAP.get(
                chart_data.chart_type, chart_data.chart_type
            )
            parts.append(f"Chart Type: {display_type}")

        # Chart title
        if self._chart_config.include_chart_title and chart_data.title:
            parts.append(f"Title: {chart_data.title}")

        # Chart data as table
        if chart_data.categories or chart_data.series:
            if self._chart_config.use_html_table:
                table_str = self._format_as_html_table(chart_data)
            else:
                table_str = self._format_as_text_table(chart_data)
            if table_str:
                parts.append(table_str)
        elif chart_data.raw_content:
            parts.append(chart_data.raw_content)

        if not parts:
            return ""

        content = "\n".join(parts)
        return (
            f"{self._tag_config.chart_prefix}\n"
            f"{content}\n"
            f"{self._tag_config.chart_suffix}"
        )

    def format_chart_fallback(
        self,
        chart_type: Optional[str] = None,
        title: Optional[str] = None,
        message: Optional[str] = None,
    ) -> str:
        """Format a fallback chart block when extraction fails."""
        parts: List[str] = []
        if chart_type:
            parts.append(f"Chart Type: {_CHART_TYPE_MAP.get(chart_type, chart_type)}")
        if title:
            parts.append(f"Title: {title}")
        if message:
            parts.append(message)
        elif not parts:
            parts.append("(Chart data could not be extracted)")

        content = "\n".join(parts)
        return (
            f"{self._tag_config.chart_prefix}\n"
            f"{content}\n"
            f"{self._tag_config.chart_suffix}"
        )

    def get_chart_type_name(self, ooxml_type: str) -> str:
        """Convert OOXML chart type to display name."""
        return _CHART_TYPE_MAP.get(ooxml_type, ooxml_type)

    def get_chart_pattern(self) -> re.Pattern:
        """Get compiled regex for chart blocks."""
        prefix = re.escape(self._tag_config.chart_prefix)
        suffix = re.escape(self._tag_config.chart_suffix)
        return re.compile(rf"{prefix}(.*?){suffix}", re.DOTALL)

    def has_chart_blocks(self, text: str) -> bool:
        """Check if text contains chart blocks."""
        return bool(self.get_chart_pattern().search(text))

    def find_chart_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """Find all chart blocks. Returns [(start, end, content), ...]"""
        return [
            (m.start(), m.end(), m.group(1))
            for m in self.get_chart_pattern().finditer(text)
        ]

    # ── Private formatting ────────────────────────────────────────────────

    def _format_as_html_table(self, data: ChartData) -> str:
        """Format chart data as an HTML table."""
        if not data.series:
            return ""

        rows: List[str] = ["<table>"]

        # Header row
        header_cells = ["<th>Category</th>"]
        for s in data.series:
            name = s.name or "Series"
            header_cells.append(f"<th>{name}</th>")
        rows.append(f"<tr>{''.join(header_cells)}</tr>")

        # Data rows
        num_rows = len(data.categories) if data.categories else (
            max((len(s.values) for s in data.series), default=0)
        )
        for i in range(num_rows):
            cells: List[str] = []
            cat = data.categories[i] if i < len(data.categories) else ""
            cells.append(f"<td>{cat}</td>")
            for s in data.series:
                val = s.values[i] if i < len(s.values) else ""
                cells.append(f"<td>{val}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")

        rows.append("</table>")
        return "\n".join(rows)

    def _format_as_text_table(self, data: ChartData) -> str:
        """Format chart data as plain text."""
        if not data.series:
            return ""

        lines: List[str] = []
        for i, cat in enumerate(data.categories or []):
            values = [
                str(s.values[i]) if i < len(s.values) else ""
                for s in data.series
            ]
            lines.append(f"{cat}: {', '.join(values)}")

        return "\n".join(lines)


__all__ = ["ChartService"]
