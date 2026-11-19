"""Documents router: upload, list, delete."""

from __future__ import annotations

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse

from domain.exceptions import DocumentNotFoundError
from domain.models import DocumentRecord
from infrastructure.parsing import get_content_type, parse_document

router = APIRouter(tags=["documents"])


def _container(request: Request):
    return request.app.state.container


@router.post("/api/documents", response_model=DocumentRecord, status_code=201)
async def upload_document(file: UploadFile, request: Request) -> DocumentRecord:
    data = await file.read()
    filename = file.filename or "upload"

    # Parse (.txt/.md/.pdf). Raises UnsupportedFileFormatError / EmptyDocumentError.
    text_by_page = parse_document(filename, data)
    container = _container(request)
    document = container.ingest.execute(
        source_name=filename,
        text_by_page=text_by_page,
        content_type=get_content_type(filename),
        size_bytes=len(data),
        file_path=None,
    )
    return document


@router.get("/api/documents", response_model=list[DocumentRecord])
async def list_documents(request: Request) -> list[DocumentRecord]:
    container = _container(request)
    return container.document_repo.list_documents()


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request) -> JSONResponse:
    container = _container(request)
    deleted = container.delete_document.execute(doc_id)
    if not deleted:
        raise DocumentNotFoundError(doc_id)
    return JSONResponse(status_code=200, content={"deleted": True, "doc_id": doc_id})
