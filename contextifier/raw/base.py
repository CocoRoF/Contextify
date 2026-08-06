# contextifier/raw/base.py
"""
Shared base for format raw-document models (xlsx / docx / pptx).

A format model owns a set of :class:`~contextifier.raw.xmlpart.XmlPart`
facades over the parts it understands. ``save()`` flushes every dirty
facade into the package, then serializes the package under the
byte-preservation contract. Parts the model does NOT understand are
never touched at all.
"""

from __future__ import annotations

import posixpath
from pathlib import Path
from typing import BinaryIO

from contextifier.raw.opc import OpcPackage, OpcPart
from contextifier.raw.xmlpart import XmlPart, qn

__all__ = ["RawDocumentBase"]


class RawDocumentBase:
    """Common plumbing: part registry, flush-on-save, byte export."""

    #: subclasses set this ("xlsx" / "docx" / "pptx")
    format: str = ""

    def __init__(self, package: OpcPackage):
        self.package = package
        self._xml_parts: dict[str, XmlPart] = {}

    # -- part facades ------------------------------------------------------------

    def xml_part(self, name: str) -> XmlPart:
        """The (cached) XmlPart facade for a package part."""
        if name not in self._xml_parts:
            self._xml_parts[name] = XmlPart(self.package.get_part(name))
        return self._xml_parts[name]

    def raw_part(self, name: str) -> OpcPart:
        """Direct part access — the escape hatch for anything the model
        doesn't cover."""
        return self.package.get_part(name)

    # -- persistence ----------------------------------------------------------

    def flush(self) -> None:
        """Serialize every dirty XML facade into its package part."""
        for xp in self._xml_parts.values():
            xp.flush()

    def to_bytes(self) -> bytes:
        self.flush()
        return self.package.to_bytes()

    def save(self, target: str | Path | BinaryIO | None = None) -> bytes:
        """Write the package; untouched parts stay byte-identical."""
        self.flush()
        return self.package.save(target)

    def close(self) -> None:
        self.package.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- part removal & orphan sweep -------------------------------------------
    # Shared by PptxRawDocument.remove_slide and XlsxRawDocument.delete_sheet:
    # a structural delete drops one anchor part, then reference-counts every
    # part it transitively pulled in against everything still in the package
    # and deletes the now-orphaned ones (charts, embedded workbooks, images,
    # notes …), leaving surviving parts byte-identical.

    def _delete_part(self, name: str) -> list[str]:
        """Remove *name* and its ``.rels`` part; returns what was removed."""
        removed: list[str] = []
        if self.package.has_part(name):
            self.package.remove_part(name)
            removed.append(name)
        rels_name = OpcPackage._rels_name_for(name)
        if self.package.has_part(rels_name):
            self.package.remove_part(rels_name)
            removed.append(rels_name)
        self._xml_parts.pop(name, None)
        self._xml_parts.pop(rels_name, None)
        return removed

    def _referenced_parts(self) -> set[str]:
        """Internal targets of every relationships part still present."""
        referenced: set[str] = set()
        for rels_name in list(self.package.part_names):
            if not rels_name.endswith(".rels"):
                continue
            directory, base = posixpath.split(rels_name)
            owner_dir = posixpath.dirname(directory)
            owner_base = base[: -len(".rels")]
            owner = posixpath.join(owner_dir, owner_base) if owner_base else ""
            rels = self.package.rels_for(owner)
            if rels is None:
                continue
            for rel in rels:
                if rel["mode"] == "External":
                    continue
                referenced.add(rels.resolve(owner, rel["target"]))
        return referenced

    def _drop_content_type_overrides(self, part_names: list[str]) -> None:
        from lxml import etree

        ct_part = self.package.get_part("[Content_Types].xml")
        root = etree.fromstring(ct_part.read())
        doomed = {f"/{name}" for name in part_names}
        changed = False
        for el in list(root):
            if el.tag == qn("ct:Override") and el.get("PartName") in doomed:
                root.remove(el)
                changed = True
        if changed:
            ct_part.write(
                etree.tostring(
                    root, xml_declaration=True, encoding="UTF-8", standalone=True
                )
            )

    def _sweep_orphans(self, doomed: list[str]) -> list[str]:
        """Delete *doomed* parts, then every part only they referenced.

        Candidates (everything transitively reachable from *doomed*) are
        collected while their rels still exist; after the doomed parts go,
        any candidate no surviving relationship points at is deleted, to a
        fixpoint (removing a chart un-anchors its embedded workbook, …).
        Returns every part name removed (for content-type cleanup)."""
        candidates: set[str] = set()
        stack, visited = list(doomed), set(doomed)
        while stack:
            src = stack.pop()
            rels = self.package.rels_for(src)
            if rels is None:
                continue
            for rel in rels:
                if rel["mode"] == "External":
                    continue
                target = rels.resolve(src, rel["target"])
                if target not in visited:
                    visited.add(target)
                    candidates.add(target)
                    stack.append(target)

        removed: list[str] = []
        for name in doomed:
            removed += self._delete_part(name)

        while True:
            referenced = self._referenced_parts()
            orphans = [
                c
                for c in sorted(candidates)
                if self.package.has_part(c) and c not in referenced
            ]
            if not orphans:
                break
            for name in orphans:
                removed += self._delete_part(name)
                candidates.discard(name)

        self._drop_content_type_overrides(removed)
        return removed
