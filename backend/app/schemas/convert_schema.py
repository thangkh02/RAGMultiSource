from pydantic import BaseModel


class DocumentConvertResponse(BaseModel):
    document_id: str
    status: str
    markdown_storage_path: str
