"""Pure smart-chunking helpers.

Depends only on ``domain/``. Produces :class:`ChunkRecord` objects with rich
metadata (``doc_id``, ``source_name``, ``page_number``, ``section_heading``,
``chunk_index``).
"""

import re
from typing import List, Optional, Tuple

from domain.models import ChunkRecord


def _detect_heading(block: str) -> Optional[str]:
    """Detect a section heading from a text block.

    Recognises Markdown headings and short standalone heading-like lines.
    Returns the heading text or ``None``.
    """
    stripped = block.strip()
    if not stripped:
        return None
    m = re.match(r"^#{1,6}\s+(.+)$", stripped, re.MULTILINE)
    if m:
        return m.group(1).strip()
    single_line = "\n" not in stripped
    if single_line and len(stripped) <= 70 and not stripped.endswith((".", "!", "?")):
        if stripped.isupper() or (stripped[:1].isupper() and any(c.isalpha() for c in stripped)):
            return stripped
    return None


def chunk_document(
    text_by_page: List[Tuple[Optional[int], str]],
    source_name: str,
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[ChunkRecord]:
    """Smart-chunk a document into overlapping chunks with rich metadata.

    Args:
        text_by_page: list of ``(page_number, page_text)``. ``page_number`` is
            ``None`` for formats without pages (.txt, .md).
        source_name: original file name.
        doc_id: parent document id.
        chunk_size: target number of words per chunk.
        overlap: number of overlapping words between consecutive chunks.

    Returns:
        List of :class:`ChunkRecord` objects.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: List[ChunkRecord] = []
    chunk_index = 0
    current_section: Optional[str] = None

    for page_number, page_text in text_by_page:
        if not page_text or not page_text.strip():
            continue

        blocks = [b for b in page_text.split("\n\n") if b.strip()]

        buffer_words: List[str] = []
        chunk_section = current_section

        def flush() -> None:
            nonlocal chunk_index, buffer_words, chunk_section
            if not buffer_words:
                return
            text = " ".join(buffer_words).strip()
            if text:
                chunks.append(
                    ChunkRecord(
                        doc_id=doc_id,
                        source_name=source_name,
                        chunk_index=chunk_index,
                        text=text,
                        page_number=page_number,
                        section_heading=chunk_section,
                    )
                )
                chunk_index += 1
            buffer_words = buffer_words[-overlap:] if overlap else []
            chunk_section = current_section

        for block in blocks:
            heading = _detect_heading(block)
            if heading:
                # Start a new section: flush pending chunk under the old section.
                flush()
                current_section = heading
                chunk_section = heading
                words = heading.split()
            else:
                words = block.split()

            for word in words:
                buffer_words.append(word)
                if len(buffer_words) >= chunk_size:
                    flush()

        flush()

    return chunks
