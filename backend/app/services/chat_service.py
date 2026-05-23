from app.models.message import MessageModel
from app.models.retrieval_log import RetrievalLogModel
from app.repositories.message_repository import MessageRepository
from app.repositories.retrieval_log_repository import RetrievalLogRepository
from app.rag.pipeline.qa_pipeline import QAPipeline
from app.rag.retrieval.resolvers import ConversationStateManager
from app.services.session_service import SessionService
from app.utils.id_utils import generate_id


class ChatService:
    def __init__(self) -> None:
        self.qa_pipeline = QAPipeline()
        self.conversation_state_manager = ConversationStateManager()
        self.message_repository = MessageRepository()
        self.retrieval_log_repository = RetrievalLogRepository()
        self.session_service = SessionService()

    async def ask_question(self, request, user_id: str):
        session = None
        if request.session_id:
            session = await self.session_service.get_session(request.session_id, user_id)
            if session is None:
                raise ValueError("Session not found.")
        conversation_state = self.conversation_state_manager.load(session, user_id, request.session_id)
        if request.session_id:
            recent_messages = await self.message_repository.list_session_messages(request.session_id)
            conversation_state["recent_chat_history"] = [
                {
                    "role": message.get("role"),
                    "content": message.get("content", ""),
                }
                for message in recent_messages[-8:]
            ]
        result = await self.qa_pipeline.run(
            question=request.question,
            user_id=user_id,
            session_id=request.session_id,
            scope=request.scope,
            selected_document_ids=request.selected_document_ids,
            conversation_state=conversation_state,
        )
        next_conversation_state = self.conversation_state_manager.update_after_answer(
            state=conversation_state,
            intent=result["intent_resolution"].get("intent"),
            scope=result["scope"],
            sources=result["sources"],
            selected_document_ids=result["document_resolution"].get("selected_document_ids", []),
            rewritten_question=result["query_rewrite"].get("rewritten_question"),
            detected_procedure_title=result["scope_resolution"].get("detected_procedure_title"),
            detected_filename=result["scope_resolution"].get("detected_filename"),
        )
        user_message = MessageModel(
            id=generate_id("msg"),
            session_id=request.session_id or "no_session",
            owner_user_id=user_id,
            role="user",
            content=request.question,
            metadata={
                "intent_resolution": result["intent_resolution"],
                "scope_resolution": result["scope_resolution"],
                "document_resolution": result["document_resolution"],
                "query_rewrite": result["query_rewrite"],
                "retrieval_plan": result["retrieval_plan"],
                "context_validation": result["context_validation"],
                "retrieval_filter": result["retrieval_filter"],
            },
        )
        assistant_message = MessageModel(
            id=generate_id("msg"),
            session_id=request.session_id or "no_session",
            owner_user_id=user_id,
            role="assistant",
            content=result["answer"],
            sources=result["sources"],
            metadata={"conversation_state": next_conversation_state},
        )
        await self.message_repository.create_message(user_message)
        await self.message_repository.create_message(assistant_message)
        await self.retrieval_log_repository.create_log(
            RetrievalLogModel(
                id=generate_id("rlog"),
                user_id=user_id,
                session_id=request.session_id,
                question=request.question,
                resolved_scope=result["scope"],
                selected_document_ids=result["document_resolution"].get("selected_document_ids", []),
                retrieval_filter=result["retrieval_filter"],
                top_k=max(
                    [branch.get("top_k", 0) for branch in result["retrieval_plan"].get("branches", [])],
                    default=0,
                ),
                retrieved_chunk_ids=[
                    item.get("metadata", {}).get("chunk_id", item.get("id", ""))
                    for item in result["raw_contexts"]
                ],
                response_metadata={
                    "intent_resolution": result["intent_resolution"],
                    "scope_resolution": result["scope_resolution"],
                    "document_resolution": result["document_resolution"],
                    "query_rewrite": result["query_rewrite"],
                    "retrieval_plan": result["retrieval_plan"],
                    "context_validation": result["context_validation"],
                    "sources": result["sources"],
                    "answer_preview": result["answer"][:500],
                },
            )
        )
        if request.session_id:
            await self.session_service.update_conversation_state(request.session_id, next_conversation_state)
            await self.session_service.touch_session(request.session_id)
        return result
