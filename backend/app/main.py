from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.sessions import router as sessions_router
from app.api.routes.system_documents import router as system_documents_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.init_db import init_mongodb


configure_logging()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Chatbot API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(documents_router, prefix="/documents", tags=["documents"])
    app.include_router(system_documents_router, prefix="/system-documents", tags=["system-documents"])
    app.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
    app.include_router(chat_router, prefix="/chat", tags=["chat"])

    @app.on_event("startup")
    async def startup_event() -> None:
        await init_mongodb()

    return app


app = create_app()
