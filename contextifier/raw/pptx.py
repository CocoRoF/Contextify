# contextifier/raw/pptx.py
"""
PptxRawDocument — the raw (lossless, writable) model for .pptx decks.

Slides are thin views over their XML parts: reading never dirties, and
edits are surgical, so untouched parts round-trip byte-identical (the
OPC byte-preservation contract). Two operations exist specifically to
fix what template-driven pipelines (edit2docs-style) get wrong today:

* :meth:`RawSlide.replace_content` — swap a slide's XML for freshly
  generated markup while *pulling the original native objects along*:
  chart / table / diagram graphicFrames (and optionally pictures) are
  lifted out of the old tree and re-appended into the new one, ids
  renumbered, relationships untouched. Native charts stay native
  instead of being rasterized or dropped.
* :meth:`PptxRawDocument.remove_slide` — deletes the slide *and*
  reference-counts every part it pulled in (charts, embedded workbooks,
  images, notes), removing the ones no remaining slide uses. No more
  orphan-part bloat in the package.
"""

from __future__ import annotations

import copy
import posixpath
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from contextifier.raw.base import RawDocumentBase
from contextifier.raw.chart import ChartModel, find_chart_parts
from contextifier.raw.opc import make_part_renamer
from contextifier.raw.xmlpart import NS, qn

if TYPE_CHECKING:  # pragma: no cover
    from lxml.etree import _Element

__all__ = [
    "PptxRawDocument",
    "RawSlide",
    "RawShapeInfo",
    "RawTable",
    "RawTableCell",
]

_PRESENTATION = "ppt/presentation.xml"

#: graphicData/@uri tail → shape kind
_GRAPHIC_KIND = {
    "table": "table",
    "chart": "chart",
    "chartex": "chart",
    "diagram": "diagram",
}

#: uri tails of graphicFrames replace_content must always carry over
_NATIVE_FRAME_TAILS = ("table", "chart", "chartex", "diagram")

#: nv*Pr wrappers whose first p:cNvPr identifies the shape
_NV_PR_TAGS = frozenset(
    qn(t)
    for t in (
        "p:nvSpPr",
        "p:nvPicPr",
        "p:nvGraphicFramePr",
        "p:nvGrpSpPr",
        "p:nvCxnSpPr",
    )
)


@dataclass
class RawShapeInfo:
    """Inventory entry for one shape on a slide.

    ``left/top/width/height`` are the shape's own transform in EMU, or ``None``
    when it inherits placement from a placeholder (no explicit ``a:xfrm``).
    """

    id: int
    name: str
    kind: str  # "text" | "picture" | "table" | "chart" | "group" | "diagram" | "other"
    text: str | None
    left: int | None = None
    top: int | None = None
    width: int | None = None
    height: int | None = None


# -- XML helpers ---------------------------------------------------------------


def _para_text(para: "_Element") -> str:
    return "".join(t.text or "" for t in para.iter(qn("a:t")))


def _body_text(tx_body: "_Element") -> str:
    return "\n".join(_para_text(p) for p in tx_body.findall(qn("a:p")))


def _replace_para_text(para: "_Element", new_text: str) -> None:
    """Set a paragraph's text under the format-preserving contract.

    The first ``a:r`` keeps its ``a:rPr`` (formatting) and receives the
    new text; every *other* plain-text run is dropped; anything that is
    not a plain run — ``a:fld``, ``a:br``, ``a:pPr``, ``a:endParaRPr`` —
    stays exactly where it was.
    """
    from lxml import etree

    a_r, a_t = qn("a:r"), qn("a:t")
    runs = [child for child in para if child.tag == a_r]
    if runs:
        first = runs[0]
        t = first.find(a_t)
        if t is None:
            t = etree.SubElement(first, a_t)
        t.text = new_text
        for extra in runs[1:]:
            para.remove(extra)
        return
    run = etree.SubElement(para, a_r)
    etree.SubElement(run, a_t).text = new_text
    end = para.find(qn("a:endParaRPr"))
    if end is not None:
        end.addprevious(run)


#: EMU per inch / point — DrawingML's absolute unit.
EMU_PER_INCH = 914400
EMU_PER_POINT = 12700

#: a:rPr child order (subset) so an inserted solidFill lands schema-valid.
_RPR_FILL_PREDECESSORS = (qn("a:ln"),)

#: a:spPr child order (subset): the fill group sits after geometry, before ln.
_SPPR_FILL_PREDECESSORS = (
    qn("a:xfrm"),
    qn("a:custGeom"),
    qn("a:prstGeom"),
)

#: a:tcPr child order: the fill group sits AFTER the border lines + cell3D.
_TCPR_FILL_PREDECESSORS = (
    qn("a:lnL"),
    qn("a:lnR"),
    qn("a:lnT"),
    qn("a:lnB"),
    qn("a:lnTlToBr"),
    qn("a:lnBlToTr"),
    qn("a:cell3D"),
)
_FILL_TAGS = (
    qn("a:noFill"),
    qn("a:solidFill"),
    qn("a:gradFill"),
    qn("a:blipFill"),
    qn("a:pattFill"),
    qn("a:grpFill"),
)


def _norm_hex(color: str) -> str:
    return color.lstrip("#").upper()


def _solid_fill(color: str) -> "_Element":
    from lxml import etree

    fill = etree.Element(qn("a:solidFill"))
    etree.SubElement(fill, qn("a:srgbClr")).set("val", _norm_hex(color))
    return fill


def _insert_after_last(
    parent: "_Element", child: "_Element", predecessors: tuple
) -> None:
    """Insert *child* right after the last predecessor present, else at front."""
    anchor = None
    for el in parent:
        if el.tag in predecessors:
            anchor = el
    if anchor is not None:
        anchor.addnext(child)
    else:
        parent.insert(0, child)


def _set_fill(prop_el: "_Element", color: str, predecessors: tuple) -> None:
    """Replace *prop_el*'s solid fill (drop any existing fill-group element)."""
    for el in list(prop_el):
        if el.tag in _FILL_TAGS:
            prop_el.remove(el)
    _insert_after_last(prop_el, _solid_fill(color), predecessors)


def _apply_run_props(
    rpr: "_Element",
    *,
    color,
    size_pt,
    bold,
    italic,
) -> None:
    if size_pt is not None:
        rpr.set("sz", str(int(round(size_pt * 100))))
    if bold is not None:
        rpr.set("b", "1" if bold else "0")
    if italic is not None:
        rpr.set("i", "1" if italic else "0")
    if color is not None:
        _set_fill(rpr, color, _RPR_FILL_PREDECESSORS)


def _shape_cnvpr(shape_el: "_Element") -> "_Element | None":
    """The shape's own ``p:cNvPr`` (never a nested child's)."""
    for child in shape_el:
        if child.tag in _NV_PR_TAGS:
            return child.find(qn("p:cNvPr"))
    return None


def _shape_geometry(
    shape_el: "_Element",
) -> tuple[int | None, int | None, int | None, int | None]:
    """The shape's OWN transform (left, top, width, height) in EMU, or Nones.

    graphicFrames carry ``p:xfrm`` directly; sp/pic/cxnSp keep it under
    ``spPr/a:xfrm``; groups under ``grpSpPr/a:xfrm``. We read only the shape's
    own xfrm, never a descendant's.
    """
    tag = shape_el.tag
    xfrm = None
    if tag == qn("p:graphicFrame"):
        xfrm = shape_el.find(qn("p:xfrm"))
    elif tag == qn("p:grpSp"):
        pr = shape_el.find(qn("p:grpSpPr"))
        xfrm = pr.find(qn("a:xfrm")) if pr is not None else None
    else:
        pr = shape_el.find(qn("p:spPr"))
        xfrm = pr.find(qn("a:xfrm")) if pr is not None else None
    if xfrm is None:
        return (None, None, None, None)
    off = xfrm.find(qn("a:off"))
    ext = xfrm.find(qn("a:ext"))

    def _int(el, attr):
        if el is None or el.get(attr) is None:
            return None
        try:
            return int(el.get(attr))
        except ValueError:
            return None

    return (_int(off, "x"), _int(off, "y"), _int(ext, "cx"), _int(ext, "cy"))


def _graphic_kind(frame_el: "_Element") -> str | None:
    data = frame_el.find("a:graphic/a:graphicData", NS)
    if data is None:
        return None
    uri = data.get("uri") or ""
    return _GRAPHIC_KIND.get(uri.rsplit("/", 1)[-1])


def _walk_shapes(container: "_Element") -> Iterator["_Element"]:
    """Shape elements in document order, descending into p:grpSp."""
    shape_tags = (
        qn("p:sp"),
        qn("p:pic"),
        qn("p:graphicFrame"),
        qn("p:grpSp"),
        qn("p:cxnSp"),
    )
    for child in container:
        if child.tag in shape_tags:
            yield child
            if child.tag == qn("p:grpSp"):
                yield from _walk_shapes(child)


def _first_rid(el: "_Element", attr: str) -> str | None:
    """First ``attr`` (a qualified name) found on *el* or a descendant."""
    for node in el.iter():
        rid = node.get(attr)
        if rid:
            return rid
    return None


# -- tables --------------------------------------------------------------------


class RawTableCell:
    """One ``a:tc`` — read/format-preserving-write of its text."""

    def __init__(self, table: "RawTable", tc_el: "_Element"):
        self._table = table
        self._tc = tc_el

    @property
    def text(self) -> str:
        tx = self._tc.find(qn("a:txBody"))
        return _body_text(tx) if tx is not None else ""

    def set_text(self, text: str) -> None:
        """Replace the first paragraph's text (first run's ``a:rPr`` and
        any non-text elements are preserved — same rules as
        :meth:`RawSlide.set_text`)."""
        from lxml import etree

        tx = self._tc.find(qn("a:txBody"))
        if tx is None:
            tx = etree.SubElement(self._tc, qn("a:txBody"))
            etree.SubElement(tx, qn("a:bodyPr"))
            etree.SubElement(tx, qn("a:lstStyle"))
        para = tx.find(qn("a:p"))
        if para is None:
            para = etree.SubElement(tx, qn("a:p"))
        _replace_para_text(para, text)
        self._table._slide._mark_dirty()

    def set_style(
        self,
        *,
        fill: str | None = None,
        color: str | None = None,
        size_pt: float | None = None,
        bold: bool | None = None,
        italic: bool | None = None,
    ) -> None:
        """Restyle the cell: ``fill`` ("RRGGBB") sets the cell background;
        ``color``/``size_pt``/``bold``/``italic`` restyle its text runs. Only
        the attributes given change; the cell text is untouched."""
        from lxml import etree

        if any(v is not None for v in (color, size_pt, bold, italic)):
            tx = self._tc.find(qn("a:txBody"))
            if tx is not None:
                for p in tx.findall(qn("a:p")):
                    for run in p.findall(qn("a:r")):
                        rpr = run.find(qn("a:rPr"))
                        if rpr is None:
                            rpr = etree.Element(qn("a:rPr"))
                            run.insert(0, rpr)
                        _apply_run_props(
                            rpr, color=color, size_pt=size_pt, bold=bold, italic=italic
                        )
        if fill is not None:
            # a:tcPr holds the cell fill; it must be the LAST child of a:tc.
            tc_pr = self._tc.find(qn("a:tcPr"))
            if tc_pr is None:
                # a:tcPr must be the LAST child of a:tc (after a:txBody).
                tc_pr = etree.SubElement(self._tc, qn("a:tcPr"))
            # The fill group sits after the border lines / cell3D in tcPr.
            _set_fill(tc_pr, fill, _TCPR_FILL_PREDECESSORS)
        self._table._slide._mark_dirty()


class RawTable:
    """A native DrawingML table (``a:tbl``) hosted in a graphicFrame.

    v0.4 scope: cell text + row insert/delete. Column operations would
    require ``a:gridCol`` surgery and are deliberately out of scope.
    """

    def __init__(self, slide: "RawSlide", frame_el: "_Element"):
        self._slide = slide
        self._frame = frame_el
        cnvpr = _shape_cnvpr(frame_el)
        self.shape_id: int = int(cnvpr.get("id")) if cnvpr is not None else -1
        tbl = frame_el.find("a:graphic/a:graphicData/a:tbl", NS)
        if tbl is None:  # pragma: no cover - guarded by caller's uri check
            raise ValueError("graphicFrame does not contain an a:tbl")
        self._tbl = tbl

    # -- geometry ---------------------------------------------------------------

    @property
    def _rows(self) -> list["_Element"]:
        return self._tbl.findall(qn("a:tr"))

    @property
    def n_rows(self) -> int:
        return len(self._rows)

    @property
    def n_cols(self) -> int:
        return len(self._tbl.findall("a:tblGrid/a:gridCol", NS))

    # -- cells ------------------------------------------------------------------

    def cell(self, r: int, c: int) -> RawTableCell:
        rows = self._rows
        if not 0 <= r < len(rows):
            raise IndexError(f"row {r} out of range (table has {len(rows)} rows)")
        cells = rows[r].findall(qn("a:tc"))
        if not 0 <= c < len(cells):
            raise IndexError(f"col {c} out of range (row has {len(cells)} cells)")
        return RawTableCell(self, cells[c])

    # -- rows -------------------------------------------------------------------

    def insert_row(self, idx: int) -> None:
        """Insert an empty row at *idx*, cloning the row above (or the
        first row) as the style template. Column count stays consistent
        with ``a:tblGrid`` because the template row already matches it."""
        from lxml import etree

        rows = self._rows
        if not rows:
            raise ValueError("cannot insert into a table with no template row")
        if not 0 <= idx <= len(rows):
            raise IndexError(f"insert index {idx} out of range (0..{len(rows)})")
        template = rows[idx - 1] if idx > 0 else rows[0]
        new_row = copy.deepcopy(template)
        for tc in new_row.findall(qn("a:tc")):
            tx = tc.find(qn("a:txBody"))
            if tx is not None:
                for para in tx.findall(qn("a:p")):
                    tx.remove(para)
                etree.SubElement(tx, qn("a:p"))
        if idx == len(rows):
            rows[-1].addnext(new_row)
        else:
            rows[idx].addprevious(new_row)
        self._slide._mark_dirty()

    def delete_row(self, idx: int) -> None:
        rows = self._rows
        if not 0 <= idx < len(rows):
            raise IndexError(f"row {idx} out of range (table has {len(rows)} rows)")
        self._tbl.remove(rows[idx])
        self._slide._mark_dirty()

    # -- columns ----------------------------------------------------------------

    @property
    def _grid_cols(self) -> list["_Element"]:
        grid = self._tbl.find(qn("a:tblGrid"))
        return grid.findall(qn("a:gridCol")) if grid is not None else []

    def insert_column(self, idx: int) -> None:
        """Insert an empty column at *idx*, cloning the column to its left (or
        the first) as the width/style template — one ``a:gridCol`` plus one
        ``a:tc`` in every row."""
        from lxml import etree

        cols = self._grid_cols
        n = len(cols)
        if not 0 <= idx <= n:
            raise IndexError(f"insert index {idx} out of range (0..{n})")
        grid = self._tbl.find(qn("a:tblGrid"))
        template_col = cols[idx - 1] if idx > 0 else (cols[0] if cols else None)
        new_col = (
            copy.deepcopy(template_col)
            if template_col is not None
            else etree.Element(qn("a:gridCol"))
        )
        if idx == n:
            (cols[-1].addnext(new_col) if cols else grid.append(new_col))
        else:
            cols[idx].addprevious(new_col)
        for row in self._rows:
            cells = row.findall(qn("a:tc"))
            tmpl = cells[idx - 1] if idx > 0 else (cells[0] if cells else None)
            new_tc = (
                copy.deepcopy(tmpl) if tmpl is not None else etree.Element(qn("a:tc"))
            )
            body = new_tc.find(qn("a:txBody"))
            if body is not None:
                for para in body.findall(qn("a:p")):
                    body.remove(para)
                etree.SubElement(body, qn("a:p"))
            # a fresh cell carries no merge span
            new_tc.attrib.pop("gridSpan", None)
            new_tc.attrib.pop("hMerge", None)
            if idx == len(cells):
                (cells[-1].addnext(new_tc) if cells else row.append(new_tc))
            else:
                cells[idx].addprevious(new_tc)
        self._slide._mark_dirty()

    def delete_column(self, idx: int) -> None:
        cols = self._grid_cols
        if not 0 <= idx < len(cols):
            raise IndexError(f"col {idx} out of range (table has {len(cols)} cols)")
        cols[idx].getparent().remove(cols[idx])
        for row in self._rows:
            cells = row.findall(qn("a:tc"))
            if idx < len(cells):
                row.remove(cells[idx])
        self._slide._mark_dirty()

    def merge_cells(self, r1: int, c1: int, r2: int, c2: int) -> None:
        """Merge the rectangular block (r1,c1)-(r2,c2) into one cell.

        The top-left cell gets ``gridSpan``/``rowSpan``; covered cells are
        marked ``hMerge``/``vMerge`` (kept, per the OOXML table model). The
        top-left cell's text is preserved; covered cells' text is cleared."""
        rows = self._rows
        r1, r2 = sorted((r1, r2))
        c1, c2 = sorted((c1, c2))
        if not (0 <= r1 and r2 < len(rows)):
            raise IndexError("merge row range out of range")
        span_cols, span_rows = c2 - c1 + 1, r2 - r1 + 1
        if span_cols == 1 and span_rows == 1:
            return
        for ri in range(r1, r2 + 1):
            cells = rows[ri].findall(qn("a:tc"))
            if c2 >= len(cells):
                raise IndexError("merge col range out of range")
            for ci in range(c1, c2 + 1):
                tc = cells[ci]
                if ri == r1 and ci == c1:
                    if span_cols > 1:
                        tc.set("gridSpan", str(span_cols))
                    if span_rows > 1:
                        tc.set("rowSpan", str(span_rows))
                else:
                    if ci > c1:
                        tc.set("hMerge", "1")
                    if ri > r1:
                        tc.set("vMerge", "1")
                    body = tc.find(qn("a:txBody"))  # clear covered cell text
                    if body is not None:
                        for para in body.findall(qn("a:p")):
                            body.remove(para)
                        from lxml import etree

                        etree.SubElement(body, qn("a:p"))
        self._slide._mark_dirty()


# -- slides --------------------------------------------------------------------


class RawSlide:
    """One slide part, addressed by shape id."""

    def __init__(self, doc: "PptxRawDocument", part_name: str, index: int):
        self._doc = doc
        self.part_name = part_name
        self.index = index

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<RawSlide #{self.index} {self.part_name!r}>"

    @property
    def _xp(self):
        return self._doc.xml_part(self.part_name)

    def _mark_dirty(self) -> None:
        self._xp.mark_dirty()

    @property
    def _sp_tree(self) -> "_Element":
        tree = self._xp.find("p:cSld/p:spTree")
        if tree is None:
            raise ValueError(f"{self.part_name} has no p:cSld/p:spTree")
        return tree

    # -- inventory ---------------------------------------------------------------

    @property
    def shapes(self) -> list[RawShapeInfo]:
        """All shapes (document order, groups flattened in place)."""
        out: list[RawShapeInfo] = []
        for el in _walk_shapes(self._sp_tree):
            cnvpr = _shape_cnvpr(el)
            shape_id = int(cnvpr.get("id")) if cnvpr is not None else -1
            name = (cnvpr.get("name") or "") if cnvpr is not None else ""
            kind, text = "other", None
            if el.tag == qn("p:sp"):
                tx = el.find(qn("p:txBody"))
                if tx is not None:
                    kind, text = "text", _body_text(tx)
            elif el.tag == qn("p:pic"):
                kind = "picture"
            elif el.tag == qn("p:grpSp"):
                kind = "group"
            elif el.tag == qn("p:graphicFrame"):
                kind = _graphic_kind(el) or "other"
            left, top, width, height = _shape_geometry(el)
            out.append(
                RawShapeInfo(
                    id=shape_id,
                    name=name,
                    kind=kind,
                    text=text,
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                )
            )
        return out

    def _find_shape(self, shape_id: int) -> "_Element":
        for el in _walk_shapes(self._sp_tree):
            cnvpr = _shape_cnvpr(el)
            if cnvpr is not None and cnvpr.get("id") == str(shape_id):
                return el
        raise KeyError(f"No shape with id={shape_id} on {self.part_name}")

    # -- text --------------------------------------------------------------------

    def get_text(self, shape_id: int) -> str:
        el = self._find_shape(shape_id)
        tx = el.find(qn("p:txBody"))
        if tx is None:
            raise ValueError(f"Shape id={shape_id} has no text body")
        return _body_text(tx)

    def get_paragraphs(self, shape_id: int) -> list[str]:
        """Per-``a:p`` text of a shape (one entry per paragraph), so callers
        address paragraphs by the SAME index :meth:`set_text` mutates — even
        when a run's text contains a literal newline (which would make a
        ``get_text().split("\\n")`` index diverge from the real ``a:p`` list)."""
        el = self._find_shape(shape_id)
        tx = el.find(qn("p:txBody"))
        if tx is None:
            raise ValueError(f"Shape id={shape_id} has no text body")
        return [_para_text(p) for p in tx.findall(qn("a:p"))]

    def paragraphs_by_shape(self) -> dict[int, list[str]]:
        """``{shape_id: [para text, …]}`` for every text shape, in ONE walk of
        the slide — so callers building an outline don't pay an O(shapes²) cost
        re-finding each shape with :meth:`get_paragraphs`."""
        out: dict[int, list[str]] = {}
        for el in _walk_shapes(self._sp_tree):
            if el.tag != qn("p:sp"):
                continue
            tx = el.find(qn("p:txBody"))
            if tx is None:
                continue
            cnvpr = _shape_cnvpr(el)
            if cnvpr is None or cnvpr.get("id") is None:
                continue
            out[int(cnvpr.get("id"))] = [_para_text(p) for p in tx.findall(qn("a:p"))]
        return out

    def set_text(self, shape_id: int, new_text: str, para: int = 0) -> None:
        """Replace paragraph *para*'s text, preserving the first run's
        formatting (``a:rPr``) and any non-text elements (``a:fld``,
        ``a:br``); other plain-text runs in the paragraph are removed."""
        from lxml import etree

        el = self._find_shape(shape_id)
        tx = el.find(qn("p:txBody"))
        if tx is None:
            raise ValueError(f"Shape id={shape_id} has no text body")
        paras = tx.findall(qn("a:p"))
        if para == 0 and not paras:
            # An empty text placeholder (txBody with no a:p) is advertised as
            # para 0 by callers — materialize it rather than reject the edit.
            paras = [etree.SubElement(tx, qn("a:p"))]
        if not 0 <= para < len(paras):
            raise IndexError(
                f"paragraph {para} out of range (shape has {len(paras)} paragraphs)"
            )
        _replace_para_text(paras[para], new_text)
        self._mark_dirty()

    # -- style / geometry (format-surgical, in place) -----------------------

    def set_shape_font(
        self,
        shape_id: int,
        *,
        color: str | None = None,
        size_pt: float | None = None,
        bold: bool | None = None,
        italic: bool | None = None,
        para: int | None = None,
    ) -> None:
        """Restyle a text shape's runs in place.

        Sets only the attributes given (``color`` as ``RRGGBB``, ``size_pt`` in
        points, ``bold``/``italic``). Applied to every run of every paragraph,
        or of one ``para`` when given. Each paragraph's ``a:endParaRPr`` is kept
        in sync so an empty trailing line inherits the same look. Runs keep
        their text and every other property.
        """
        from lxml import etree

        el = self._find_shape(shape_id)
        tx = el.find(qn("p:txBody"))
        if tx is None:
            raise ValueError(f"Shape id={shape_id} has no text body")
        paras = tx.findall(qn("a:p"))
        if para is not None:
            if not 0 <= para < len(paras):
                raise IndexError(
                    f"paragraph {para} out of range (shape has {len(paras)} paragraphs)"
                )
            paras = [paras[para]]
        for p in paras:
            for run in p.findall(qn("a:r")):
                rpr = run.find(qn("a:rPr"))
                if rpr is None:
                    rpr = etree.Element(qn("a:rPr"))
                    run.insert(0, rpr)
                _apply_run_props(
                    rpr, color=color, size_pt=size_pt, bold=bold, italic=italic
                )
            end = p.find(qn("a:endParaRPr"))
            if end is not None:
                _apply_run_props(
                    end, color=color, size_pt=size_pt, bold=bold, italic=italic
                )
        self._mark_dirty()

    def _ensure_sp_pr(self, el: "_Element") -> "_Element":
        """The shape's ``p:spPr`` (created after ``nvSpPr`` if absent).

        Only ``p:sp`` / ``p:pic`` / ``p:cxnSp`` carry a ``p:spPr``; a
        ``graphicFrame`` (table/chart) has none, so styling it is rejected."""
        from lxml import etree

        if el.tag not in (qn("p:sp"), qn("p:pic"), qn("p:cxnSp")):
            raise ValueError(
                "shape has no spPr (a table/chart/other frame cannot be "
                "styled this way)"
            )
        sp_pr = el.find(qn("p:spPr"))
        if sp_pr is None:
            sp_pr = etree.Element(qn("p:spPr"))
            nv = next((c for c in el if c.tag in _NV_PR_TAGS), None)
            if nv is not None:  # spPr follows nvSpPr in CT_Shape
                nv.addnext(sp_pr)
            else:
                el.insert(0, sp_pr)
        return sp_pr

    def set_shape_fill(self, shape_id: int, color: str) -> None:
        """Set a shape's solid fill to ``color`` (``RRGGBB``), in place.

        Replaces any existing fill; ``a:xfrm`` / geometry / line stay put.
        """
        sp_pr = self._ensure_sp_pr(self._find_shape(shape_id))
        _set_fill(sp_pr, color, _SPPR_FILL_PREDECESSORS)
        self._mark_dirty()

    def set_shape_position(
        self,
        shape_id: int,
        *,
        left: int | None = None,
        top: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Move/resize a shape (values in EMU; only the given ones change).

        Writes ``p:spPr/a:xfrm/a:off`` (position) and ``a:ext`` (size),
        creating the ``a:xfrm`` if the shape inherited its placement from a
        placeholder. Everything else on the shape is untouched.
        """
        from lxml import etree

        el = self._find_shape(shape_id)
        if el.tag == qn("p:graphicFrame"):
            # graphicFrames (tables/charts) carry p:xfrm directly, and it is
            # REQUIRED — so it always exists; edit it in place.
            xfrm = el.find(qn("p:xfrm"))
            if xfrm is None:
                xfrm = etree.Element(qn("p:xfrm"))
                nv = next((c for c in el if c.tag in _NV_PR_TAGS), None)
                (nv.addnext(xfrm) if nv is not None else el.insert(0, xfrm))
        else:
            if el.tag == qn("p:grpSp"):
                pr = el.find(qn("p:grpSpPr"))
                if pr is None:
                    pr = etree.SubElement(el, qn("p:grpSpPr"))
            else:
                pr = self._ensure_sp_pr(el)
            xfrm = pr.find(qn("a:xfrm"))
            if xfrm is None:
                xfrm = etree.Element(qn("a:xfrm"))
                pr.insert(0, xfrm)  # xfrm is the first child of spPr/grpSpPr
        off = xfrm.find(qn("a:off"))
        if off is None:
            off = etree.SubElement(xfrm, qn("a:off"))
        ext = xfrm.find(qn("a:ext"))
        if ext is None:
            ext = etree.SubElement(xfrm, qn("a:ext"))
        if left is not None:
            off.set("x", str(int(left)))
        if top is not None:
            off.set("y", str(int(top)))
        if width is not None:
            ext.set("cx", str(int(width)))
        if height is not None:
            ext.set("cy", str(int(height)))
        # off/ext must carry both coordinates — backfill from current geometry.
        off.set("x", off.get("x") or "0")
        off.set("y", off.get("y") or "0")
        ext.set("cx", ext.get("cx") or "0")
        ext.set("cy", ext.get("cy") or "0")
        self._mark_dirty()

    # -- shape lifecycle (add / delete / duplicate) -------------------------

    def _next_shape_id(self) -> int:
        used = {
            int(c.get("id"))
            for c in self._sp_tree.iter(qn("p:cNvPr"))
            if c.get("id") and c.get("id").isdigit()
        }
        return (max(used) + 1) if used else 2

    def add_textbox(
        self,
        text: str,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
        color: str | None = None,
        size_pt: float | None = None,
        bold: bool | None = None,
        italic: bool | None = None,
    ) -> int:
        """Append a new text box (a real ``p:sp`` with ``txBox="1"``) at the
        given EMU rectangle, carrying *text* and optional run formatting. Returns
        the new shape's id."""
        from lxml import etree

        sid = self._next_shape_id()
        sp = etree.SubElement(self._sp_tree, qn("p:sp"))
        nv = etree.SubElement(sp, qn("p:nvSpPr"))
        etree.SubElement(nv, qn("p:cNvPr")).attrib.update(
            {"id": str(sid), "name": f"TextBox {sid}"}
        )
        etree.SubElement(nv, qn("p:cNvSpPr")).set("txBox", "1")
        etree.SubElement(nv, qn("p:nvPr"))
        sp_pr = etree.SubElement(sp, qn("p:spPr"))
        xfrm = etree.SubElement(sp_pr, qn("a:xfrm"))
        etree.SubElement(xfrm, qn("a:off")).attrib.update(
            {"x": str(int(left)), "y": str(int(top))}
        )
        etree.SubElement(xfrm, qn("a:ext")).attrib.update(
            {"cx": str(int(width)), "cy": str(int(height))}
        )
        geom = etree.SubElement(sp_pr, qn("a:prstGeom"))
        geom.set("prst", "rect")
        etree.SubElement(geom, qn("a:avLst"))
        tx = etree.SubElement(sp, qn("p:txBody"))
        etree.SubElement(tx, qn("a:bodyPr")).set("wrap", "square")
        etree.SubElement(tx, qn("a:lstStyle"))
        para = etree.SubElement(tx, qn("a:p"))
        run = etree.SubElement(para, qn("a:r"))
        rpr = etree.SubElement(run, qn("a:rPr"))
        rpr.set("lang", "en-US")
        _apply_run_props(rpr, color=color, size_pt=size_pt, bold=bold, italic=italic)
        etree.SubElement(run, qn("a:t")).text = text
        self._mark_dirty()
        return sid

    def delete_shape(self, shape_id: int) -> None:
        """Remove a shape from the slide. Any media/chart parts it solely
        referenced are swept as orphans on the next ``to_bytes``."""
        el = self._find_shape(shape_id)
        el.getparent().remove(el)
        self._mark_dirty()
        # let the document sweep now-unreferenced parts (mirrors remove_slide)
        sweep = getattr(self._doc, "_sweep_orphan_parts", None) or getattr(
            self._doc, "_sweep_orphans", None
        )
        if sweep is not None:
            try:
                sweep()
            except Exception:
                pass

    def duplicate_shape(
        self, shape_id: int, *, left: int | None = None, top: int | None = None
    ) -> int:
        """Deep-copy a shape (new ``cNvPr`` id); optionally place the copy at an
        absolute EMU position. Returns the new shape's id.

        Note: a duplicate of a chart/picture shares the original's relationship
        id, so it references the SAME chart/image part — fine for a visual
        clone, but editing one chart would affect both."""
        src = self._find_shape(shape_id)
        clone = copy.deepcopy(src)
        sid = self._next_shape_id()
        cnvpr = _shape_cnvpr(clone)
        if cnvpr is not None:
            cnvpr.set("id", str(sid))
            if cnvpr.get("name"):
                cnvpr.set("name", f"{cnvpr.get('name')} copy")
        src.addnext(clone)
        if left is not None or top is not None:
            self._mark_dirty()
            self.set_shape_position(sid, left=left, top=top)
        else:
            self._mark_dirty()
        return sid

    def set_runs(self, shape_id: int, para: int, runs: list[dict]) -> None:
        """Replace paragraph *para* with a sequence of independently-styled runs.

        Each run: ``{"text": str, "color"?, "size_pt"?, "bold"?, "italic"?}``.
        Enables mixed formatting in one line ("bold **just** one word"). The
        paragraph's ``a:pPr`` / ``a:endParaRPr`` are preserved."""
        from lxml import etree

        el = self._find_shape(shape_id)
        tx = el.find(qn("p:txBody"))
        if tx is None:
            raise ValueError(f"Shape id={shape_id} has no text body")
        paras = tx.findall(qn("a:p"))
        if not 0 <= para < len(paras):
            raise IndexError(
                f"paragraph {para} out of range (shape has {len(paras)} paragraphs)"
            )
        p = paras[para]
        # drop existing runs/fields/breaks; keep a:pPr + a:endParaRPr
        for child in list(p):
            if child.tag in (qn("a:r"), qn("a:br"), qn("a:fld")):
                p.remove(child)
        end = p.find(qn("a:endParaRPr"))
        for spec in runs:
            run = etree.Element(qn("a:r"))
            rpr = etree.SubElement(run, qn("a:rPr"))
            rpr.set("lang", "en-US")
            _apply_run_props(
                rpr,
                color=spec.get("color"),
                size_pt=spec.get("size_pt"),
                bold=spec.get("bold"),
                italic=spec.get("italic"),
            )
            etree.SubElement(run, qn("a:t")).text = str(spec.get("text", ""))
            if end is not None:
                end.addprevious(run)
            else:
                p.append(run)
        self._mark_dirty()

    # -- tables / charts / notes ---------------------------------------------------

    @property
    def tables(self) -> list[RawTable]:
        return [
            RawTable(self, el)
            for el in _walk_shapes(self._sp_tree)
            if el.tag == qn("p:graphicFrame") and _graphic_kind(el) == "table"
        ]

    @property
    def chart_part_names(self) -> list[str]:
        """Chart parts referenced from this slide's relationships."""
        return find_chart_parts(self._doc.package, self.part_name)

    @property
    def charts(self) -> list[ChartModel]:
        """Lazy :class:`ChartModel` views (implemented in milestone C3)."""
        return [
            ChartModel(self._doc.xml_part(name), self._doc.package)
            for name in self.chart_part_names
        ]

    @property
    def notes_text(self) -> str | None:
        """Text of the notes slide's body placeholder, or ``None``."""
        rels = self._doc.package.rels_for(self.part_name)
        if rels is None:
            return None
        notes_rels = rels.by_type("/notesSlide")
        if not notes_rels:
            return None
        notes_part = rels.resolve(self.part_name, notes_rels[0]["target"])
        xp = self._doc.xml_part(notes_part)
        for sp in xp.findall(".//p:sp"):
            ph = sp.find("p:nvSpPr/p:nvPr/p:ph", NS)
            tx = sp.find(qn("p:txBody"))
            if tx is not None and ph is not None and ph.get("type") == "body":
                return _body_text(tx)
        texts = [_body_text(tx) for tx in xp.root.iter(qn("p:txBody"))]
        return "\n".join(t for t in texts if t)

    # -- content replacement --------------------------------------------------------

    def replace_content(
        self,
        new_slide_xml: bytes,
        *,
        preserve_native: bool = True,
        preserve_pictures: bool = True,
    ) -> list[str]:
        """Replace this slide's XML with *new_slide_xml*, carrying the
        original native objects over.

        With ``preserve_native=True`` every chart / table / diagram
        graphicFrame of the ORIGINAL slide (and, when
        ``preserve_pictures``, every ``p:pic`` that references embedded
        media) is deep-copied into the new tree's ``p:spTree``, with
        their ``p:cNvPr/@id`` renumbered past any id used by the new
        XML. The slide keeps its part name, so its relationships part is
        untouched and every carried-over ``r:id`` / ``r:embed`` keeps
        resolving.

        *new_slide_xml* must be a complete ``<p:sld>`` document and may
        only reference relationship ids that already exist in this
        slide's rels (or none at all) — this method never edits the rels
        part, so unknown ``r:id`` / ``r:embed`` values in the new XML
        would dangle.

        Returns descriptions of the preserved elements, e.g.
        ``["table", "chart:chart1.xml", "picture:image1.png"]``.
        """
        from lxml import etree

        new_root = etree.fromstring(new_slide_xml)
        if new_root.tag != qn("p:sld"):
            raise ValueError("new_slide_xml must be a complete <p:sld> document")
        new_tree = new_root.find("p:cSld/p:spTree", NS)
        if new_tree is None:
            raise ValueError("new_slide_xml has no p:cSld/p:spTree")

        preserved: list[str] = []
        if preserve_native:
            rels = self._doc.package.rels_for(self.part_name)
            keep: list[tuple["_Element", str]] = []
            for el in _walk_shapes(self._sp_tree):
                if el.tag == qn("p:graphicFrame"):
                    data = el.find("a:graphic/a:graphicData", NS)
                    uri_tail = (
                        (data.get("uri") or "").rsplit("/", 1)[-1]
                        if data is not None
                        else ""
                    )
                    if uri_tail not in _NATIVE_FRAME_TAILS:
                        continue
                    desc = _GRAPHIC_KIND[uri_tail]
                    if desc == "chart":
                        rid = _first_rid(data, qn("r:id"))
                        target = rels.target_of(rid) if (rels and rid) else None
                        if target:
                            desc = f"chart:{posixpath.basename(target)}"
                    keep.append((el, desc))
                elif el.tag == qn("p:pic") and preserve_pictures:
                    blip = el.find("p:blipFill/a:blip", NS)
                    rid = blip.get(qn("r:embed")) if blip is not None else None
                    if not rid:
                        continue
                    target = rels.target_of(rid) if rels else None
                    desc = (
                        f"picture:{posixpath.basename(target)}" if target else "picture"
                    )
                    keep.append((el, desc))

            used_ids = {
                int(c.get("id"))
                for c in new_root.iter(qn("p:cNvPr"))
                if (c.get("id") or "").isdigit()
            }
            next_id = max(used_ids, default=1) + 1
            for el, desc in keep:
                clone = copy.deepcopy(el)
                for cnvpr in clone.iter(qn("p:cNvPr")):
                    cnvpr.set("id", str(next_id))
                    next_id += 1
                new_tree.append(clone)
                preserved.append(desc)

        xp = self._xp
        xp._root = new_root  # swap the facade's tree in place
        xp.mark_dirty()
        return preserved


# -- document ------------------------------------------------------------------


class PptxRawDocument(RawDocumentBase):
    """Raw model for a .pptx package."""

    format = "pptx"

    @property
    def slides(self) -> list[RawSlide]:
        """Slides in presentation order (``p:sldIdLst``)."""
        pres = self.xml_part(_PRESENTATION)
        rels = self.package.rels_for(_PRESENTATION)
        sld_id_lst = pres.find("p:sldIdLst")
        if sld_id_lst is None or rels is None:
            return []
        out: list[RawSlide] = []
        for sld_id in sld_id_lst.findall(qn("p:sldId")):
            rid = sld_id.get(qn("r:id"))
            target = rels.target_of(rid) if rid else None
            if target is None:
                continue
            out.append(RawSlide(self, rels.resolve(_PRESENTATION, target), len(out)))
        return out

    # -- slide removal ------------------------------------------------------------

    def remove_slide(self, index: int) -> None:
        """Remove the slide at *index* and everything only it used.

        Beyond dropping the ``p:sldId`` entry and the slide part itself
        (plus its rels and notes slide), this reference-counts every
        part transitively reachable from the removed slide — charts,
        embedded chart workbooks, images, chart colors/style parts —
        against the relationships of everything still in the package,
        and deletes the now-orphaned ones, including their content-type
        overrides. Parts shared with surviving slides (or anchored by
        the presentation / masters, like layouts and the notes master)
        are untouched, and surviving slide parts stay byte-identical.
        """
        slides = self.slides
        if not 0 <= index < len(slides):
            raise IndexError(f"slide index {index} out of range (0..{len(slides) - 1})")
        part_name = slides[index].part_name

        pres = self.xml_part(_PRESENTATION)
        pres_rels = self.package.rels_for(_PRESENTATION)
        if pres_rels is None:  # pragma: no cover - malformed package
            raise ValueError("presentation has no relationships part")
        rid = next(
            (
                rel["id"]
                for rel in pres_rels.by_type("/slide")
                if pres_rels.resolve(_PRESENTATION, rel["target"]) == part_name
            ),
            None,
        )
        sld_id_lst = pres.find("p:sldIdLst")
        if sld_id_lst is not None and rid is not None:
            for sld_id in list(sld_id_lst):
                if sld_id.get(qn("r:id")) == rid:
                    sld_id_lst.remove(sld_id)
            pres.mark_dirty()
        if rid is not None:
            pres_rels.remove(rid)

        # The slide dies; so does its notes slide.
        doomed = [part_name]
        slide_rels = self.package.rels_for(part_name)
        if slide_rels is not None:
            doomed += [
                slide_rels.resolve(part_name, rel["target"])
                for rel in slide_rels.by_type("/notesSlide")
            ]

        # Delete the doomed parts and sweep every part only they anchored
        # (charts, embedded workbooks, images, notes) — shared base logic.
        self._sweep_orphans(doomed)

    # -- reordering & duplication ---------------------------------------------

    #: relationship Types a duplicated slide REFERENCES (shares) instead of
    #: copying: read-only or globally-anchored assets. Everything else
    #: (charts + their embedded workbooks/colors/styles, and the notes
    #: slide) is cloned, so editing the copy never mutates the original.
    _SHARE_ON_DUPLICATE = (
        "/slideLayout",
        "/notesMaster",
        "/image",
        "/audio",
        "/video",
        "/tags",
    )

    def move_slide(self, index: int, to: int) -> None:
        """Move the slide at *index* to position *to*.

        A pure reorder of ``p:sldIdLst`` in ``ppt/presentation.xml`` — no
        part is copied, renamed or deleted, so every slide part (and
        everything else in the package) stays byte-identical."""
        pres = self.xml_part(_PRESENTATION)
        sld_id_lst = pres.find("p:sldIdLst")
        if sld_id_lst is None:
            raise ValueError("presentation has no p:sldIdLst")
        ids = sld_id_lst.findall(qn("p:sldId"))
        n = len(ids)
        if not 0 <= index < n:
            raise IndexError(f"slide index {index} out of range (0..{n - 1})")
        if not 0 <= to < n:
            raise IndexError(f"destination {to} out of range (0..{n - 1})")
        if index == to:
            return
        el = ids[index]
        sld_id_lst.remove(el)
        remaining = sld_id_lst.findall(qn("p:sldId"))
        if to >= len(remaining):
            sld_id_lst.append(el)
        else:
            remaining[to].addprevious(el)
        pres.mark_dirty()

    def duplicate_slide(self, index: int, *, at: int | None = None) -> int:
        """Insert an independent copy of slide *index* at position *at*
        (default: right after the source). Returns the new slide's index.

        The slide's XML and its non-shared referenced parts (charts,
        embedded workbooks, chart colors/styles, notes slide) are deep-
        copied under fresh names via ``clone_part_graph``; images, layouts
        and the notes master are shared (see ``_SHARE_ON_DUPLICATE``). The
        notes slide's back-reference is retargeted to the copy. A new
        ``p:sldId`` (id = max + 1, ≥ 256) and presentation relationship are
        added; existing slide parts stay byte-identical."""
        slides = self.slides
        if not 0 <= index < len(slides):
            raise IndexError(f"slide index {index} out of range (0..{len(slides) - 1})")
        src_part = slides[index].part_name

        renamer = make_part_renamer(self.package)
        new_part, _ = self.package.clone_part_graph(
            src_part, rename=renamer, share_types=self._SHARE_ON_DUPLICATE
        )

        pres = self.xml_part(_PRESENTATION)
        pres_rels = self.package.rels_for(_PRESENTATION)
        if pres_rels is None:  # pragma: no cover - malformed package
            raise ValueError("presentation has no relationships part")
        slide_rel_type = next(
            (rel["type"] for rel in pres_rels.by_type("/slide")),
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
        )
        new_rid = pres_rels.next_id()
        target = posixpath.relpath(new_part, posixpath.dirname(_PRESENTATION))
        pres_rels.add(new_rid, slide_rel_type, target)

        sld_id_lst = pres.find("p:sldIdLst")
        if sld_id_lst is None:  # pragma: no cover - malformed package
            raise ValueError("presentation has no p:sldIdLst")
        existing = sld_id_lst.findall(qn("p:sldId"))
        used_ids = [int(s.get("id")) for s in existing if (s.get("id") or "").isdigit()]
        new_el = sld_id_lst.makeelement(
            qn("p:sldId"),
            {"id": str(max(used_ids, default=255) + 1), qn("r:id"): new_rid},
        )
        pos = index + 1 if at is None else max(0, min(at, len(existing)))
        if pos >= len(existing):
            sld_id_lst.append(new_el)
        else:
            existing[pos].addprevious(new_el)
        pres.mark_dirty()
        return pos
