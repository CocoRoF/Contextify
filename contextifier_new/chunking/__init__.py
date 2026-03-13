# contextifier_new/chunking/__init__.py
"""
Chunking — Text Chunking Subsystem

Replaces the old monolithic create_chunks() function with a class-based
TextChunker that uses pluggable strategies.

Architecture:
    TextChunker (facade)
      └── ChunkingStrategy (abstract)
            ├── PageChunkingStrategy    — split by page markers
            ├── TableChunkingStrategy   — table-aware splitting
            ├── ProtectedChunkingStrategy — protected region splitting
            └── PlainChunkingStrategy   — recursive text splitting

Design improvements:
- Strategy pattern replaces mega-function branching
- TextChunker is injectable (services passed to constructor)
- Each strategy is independently testable
- Constants/patterns centralized in one module
"""

from contextifier_new.chunking.chunker import TextChunker

__all__ = [
    "TextChunker",
]
