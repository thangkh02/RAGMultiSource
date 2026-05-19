from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.deps import get_document_service
from app.schemas.convert_schema import DocumentConvertResponse
from app.schemas.ingestion_job_schema import IngestionJobItem
from app.schemas.document_schema import DocumentItem, DocumentUploadResponse
from app.services.document_service import DocumentService
from app.services.markitdown_converter import MarkItDownConversionService

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    owner_user_id: str = Form(...),
    session_id: str | None = Form(default=None),
    service: DocumentService = Depends(get_document_service),
):
    try:
        payload = await service.upload_user_document(file=file, owner_user_id=owner_user_id, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    document = payload["document"]
    return DocumentUploadResponse(
        document_id=document.id,
        job_id=payload.get("job_id"),
        filename=document.filename,
        status=document.status,
        raw_storage_path=document.raw_storage_path,
        message="Document uploaded and queued for processing",
    )


@router.get("", response_model=list[DocumentItem])
async def list_documents(
    owner_user_id: str = Query(...),
    service: DocumentService = Depends(get_document_service),
):
    return await service.list_documents(owner_user_id)


@router.get("/{document_id}", response_model=DocumentItem)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
):
    document = await service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{document_id}/ingestion-jobs", response_model=list[IngestionJobItem])
async def list_ingestion_jobs(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
):
    return await service.list_ingestion_jobs(document_id)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
):
    deleted = await service.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True, "document_id": document_id}


@router.post("/{document_id}/convert", response_model=DocumentConvertResponse)
async def convert_document(
    document_id: str,
    service: MarkItDownConversionService = Depends(MarkItDownConversionService),
):
    try:
        document = await service.convert_document(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return DocumentConvertResponse(
        document_id=document["_id"],
        status=document["status"],
        markdown_storage_path=document["markdown_storage_path"],
    )
