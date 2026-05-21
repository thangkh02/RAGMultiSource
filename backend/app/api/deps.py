from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.models.user import UserModel

bearer_scheme = HTTPBearer(auto_error=False)


def get_document_service():
    from app.services.document_service import DocumentService

    return DocumentService()


def get_system_document_service():
    from app.services.system_document_service import SystemDocumentService

    return SystemDocumentService()


def get_session_service():
    from app.services.session_service import SessionService

    return SessionService()


def get_chat_service():
    from app.services.chat_service import ChatService

    return ChatService()


def get_user_service():
    from app.services.user_service import UserService

    return UserService()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    service=Depends(get_user_service),
) -> UserModel:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    user = await service.get_user_by_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return user


async def get_current_user_id(current_user: UserModel = Depends(get_current_user)) -> str:
    return current_user.id
