"""Document parsers (.txt, .md, .pdf).

Returns ``List[Tuple[Optional[int], str]]`` — ``(page_number, page_text)``.
``page_number`` is ``None`` for page-less formats (.txt, .md).

Raises :class:`UnsupportedFileFormatError` for unsupported formats and
:class:`EmptyDocumentError` for corrupted PDFs or zero extractable text
(scanned/image-only PDFs).
"""

from __future__ import annotations

import io
from typing import List, Optional, Tuple

from domain.exceptions import DomainError, EmptyDocumentError, UnsupportedFileFormatError

SUPPORTED_EXTENSIONS = {"txt", "md", "pdf"}


def get_content_type(filename: str) -> str:
    ext = _extension(filename)
    return {
        "txt": "text/plain",
        "md": "text/markdown",
        "pdf": "application/pdf",
    }.get(ext, "application/octet-stream")


def is_supported(filename: str) -> bool:
    return _extension(filename) in SUPPORTED_EXTENSIONS


def _extension(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def parse_document(filename: str, data: bytes) -> List[Tuple[Optional[int], str]]:
    ext = _extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileFormatError(filename)
    if ext == "pdf":
        return _parse_pdf(filename, data)
    return _parse_text(filename, data, ext)


def _parse_text(
    filename: str, data: bytes, ext: str
) -> List[Tuple[Optional[int], str]]:
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:
        raise DomainError(f"Could not decode {filename}: {exc}", status_code=422)
    return [(None, text)]


def _parse_pdf(filename: str, data: bytes) -> List[Tuple[Optional[int], str]]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise DomainError("PDF parsing library not available", status_code=500)

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise EmptyDocumentError(
            filename, reason=f"corrupted PDF ({exc}); nothing to index"
        )

    text_by_page: List[Tuple[Optional[int], str]] = []
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        text_by_page.append((i + 1, page_text))

    full_text = "\n".join(t for _, t in text_by_page if t and t.strip())
    if not full_text.strip():
        raise EmptyDocumentError(
            filename,
            reason=(
                f"PDF has zero extractable text (possible scanned/image-only file). "
                f"pages={len(reader.pages)}"
            ),
        )
    return text_by_page
