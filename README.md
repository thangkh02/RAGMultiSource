# RAG Chatbot

RAG chatbot hoi dap van ban hanh chinh, ho tro hai nguon tai lieu:

- Tai lieu he thong: van ban/thu tuc hanh chinh co san.
- Tai lieu user upload: file Word/PDF trong session hien tai hoac cac session truoc.

Backend dung FastAPI, MongoDB, Chroma, LangGraph, LangChain/OpenRouter/OpenAI va LangSmith tracing.
Frontend dung Next.js.

## Kien Truc Tong Quan

```text
frontend/
  Next.js UI cho chat, session, upload tai lieu

backend/
  app/api/              FastAPI routes
  app/services/         business logic: chat, session, document
  app/repositories/     MongoDB repositories
  app/models/           data models
  app/schemas/          API schemas
  app/rag/              RAG pipeline
    graph/              LangGraph workflow
    rewrite/            rewrite gate + query rewriter
    query/              intent router
    retrieval/          filters, resolvers, retriever, strategy, validation
    generation/         LLM answer + source formatter
    vectorstore/        Chroma adapter
```

## RAG Pipeline Hien Tai

`QAPipeline.run()` van la facade cho API cu, nhung ben trong goi `RAGGraphRunner`.

```text
User Query
-> ChatService
-> QAPipeline.run
-> RAGGraphRunner
-> load_context_node
-> rewrite_detector_node
-> rewrite_query_node neu can
-> use_original_query_node neu khong can rewrite
-> intent_router_node
-> scope_resolver_node
-> neu should_reuse_last_filter=true: build_filter_node
-> neu should_reuse_last_filter=false: document_resolver_node
-> candidate_selector_node
-> build_filter_node
-> retrieval_strategy_node
-> retrieval_node
-> evidence_validation_node
-> answer_node / no_context_node / direct_answer_node / clarification_node / unsupported_node
-> update_state_node
-> Return API Response
```

Chi tiet flow va thiet ke node nam trong [pipeline.md](./pipeline.md).

## Vai Tro Cac Node Chinh

- `rewrite_detector_node`: quyet dinh query co can rewrite khong.
- `rewrite_query_node`: rewrite follow-up/mo ho thanh standalone query.
- `intent_router_node`: phan loai user muon lam gi, vi du `ask_question`, `compare_documents`, `general_query`.
- `scope_resolver_node`: state-aware + LLM structured output de xac nhan scope cuoi.
- `document_resolver_node`: resolve document/procedure bang MongoDB metadata.
- `candidate_selector_node`: chon document candidate neu du chac, hoi lai neu mo ho.
- `build_filter_node`: build metadata filter cuoi cung bang code.
- `retrieval_node`: search Chroma trong metadata filter.
- `evidence_validation_node`: chan chunk sai metadata/score thap, fallback neu khong du bang chung.
- `answer_node`: sinh cau tra loi dua tren context.

## Nguyen Tac Bao Mat Retrieval

LLM khong duoc tu build metadata filter va khong duoc quyet dinh quyen truy cap.

Filter cuoi cung duoc build deterministic trong `build_filter_node`.

Quy tac:

- User upload bat buoc co `owner_user_id`.
- Current session upload bat buoc co `owner_user_id + session_id`.
- User hoi filename bat buoc co `owner_user_id + filename`.
- System docs bat buoc co `source_type=system`, voi system public co `visibility=global`.
- `selected_document_ids` tu UI/request phai duoc check quyen tung document truoc khi dua vao filter.
- Mixed retrieval phai tach branch `system_chunks` va `user_upload_chunks`.
- Khong search toan bo Chroma khi chua co scope/filter hop le.

## Scope Ho Tro

- `system_docs`: tai lieu he thong chung.
- `system_procedure`: mot thu tuc he thong cu the theo `procedure_title`.
- `current_session_uploads`: file user upload trong session hien tai.
- `user_all_uploads`: file user da upload o cac session truoc.
- `user_file_name`: file user upload theo filename.
- `hybrid_system_and_user`: so sanh/doi chieu system docs va user upload.
- `general_query`: cau hoi khong can retrieval.
- `need_clarification`: can hoi lai user.

## LLM Usage

He thong theo huong controlled agentic RAG:

- LLM duoc dung cho:
  - rewrite gate
  - query rewrite
  - intent router neu bat
  - scope analyzer structured output neu bat
  - answer generation
- Rule/code/metadata duoc dung cho:
  - retrieval planner fast path
  - document resolver
  - metadata filter
  - permission check
  - evidence validation

Prompt techniques dang dung:

- Structured JSON output.
- Enum constraint cho intent/scope/resolution mode.
- Negative instruction: khong build filter, khong quyet quyen.
- Few-shot ngan cho query rewrite.
- Temperature `0`.
- Fallback deterministic khi LLM loi.
- Security guard bang code sau LLM.

## LangSmith Tracing

Da them trace cho:

- `rag_qa_pipeline`
- `rag_langgraph_run`
- cac node dieu phoi nhu rewrite, intent, planner, scope, document resolver, candidate selector, build filter, retrieval strategy.

Khong trace truc tiep full retrieved chunks de tranh day context/log qua lon.

Bat tracing bang env:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=RAGMultiDocs
```

## Environment Variables

Backend can cac bien chinh:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=rag_chatbot

CHROMA_PERSIST_DIR=chroma
CHROMA_COLLECTION_NAME=rag_chunks

OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini

LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4.1-nano
OPENROUTER_QUERY_REWRITE_MODEL=google/gemini-3.1-flash-lite-preview
OPENROUTER_REWRITE_GATE_MODEL=google/gemini-3.1-flash-lite-preview
OPENROUTER_INTENT_MODEL=google/gemini-3.1-flash-lite-preview
OPENROUTER_SCOPE_MODEL=google/gemini-3.1-flash-lite-preview

INTENT_ROUTER_USE_LLM=true
SCOPE_RESOLVER_USE_LLM=true

UPLOAD_DIR=storage/raw
MARKDOWN_DIR=storage/markdown
CORS_ORIGINS=http://localhost:3000
```

Frontend:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Chay Local

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend mac dinh chay tai:

```text
http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend mac dinh chay tai:

```text
http://localhost:3000
```

### Docker

```bash
docker compose up --build
```

## API Chinh

- `POST /auth/...`: dang nhap/dang ky tuy implementation hien tai.
- `POST /sessions`: tao session.
- `GET /sessions`: list session cua user.
- `POST /documents/upload`: upload PDF/DOC/DOCX.
- `POST /chat`: hoi dap RAG.

Frontend hien goi:

```text
POST /chat
```

Payload chat gom:

```json
{
  "question": "...",
  "session_id": "...",
  "scope": "auto",
  "selected_document_ids": []
}
```

## Document Ingestion

Flow tai lieu:

```text
Upload PDF/DOC/DOCX
-> convert sang Markdown
-> chunking
-> tao metadata
-> embedding
-> luu vector vao Chroma
-> luu document/chunk metadata vao MongoDB
```

Metadata quan trong:

- `document_id`
- `chunk_id`
- `source_type`
- `owner_user_id`
- `session_id`
- `filename`
- `procedure_title`
- `visibility`
- `page_number`
- `section_title`

## Testing

Chay toan bo backend tests:

```bash
cd backend
$env:PYTHONPATH='.'
$env:INTENT_ROUTER_USE_LLM='false'
$env:SCOPE_RESOLVER_USE_LLM='false'
pytest -q
```

Ket qua gan nhat:

```text
56 passed, 17 warnings
```

Warnings hien tai chu yeu la:

- FastAPI `on_event` deprecated.
- `datetime.utcnow()` deprecated.

## Trang Thai Hien Tai

Da co:

- LangGraph RAG pipeline.
- Rewrite gate + query rewriter.
- Intent router.
- Scope analyzer co LLM structured output va fallback.
- Deterministic metadata filter.
- Permission check theo user/session.
- Candidate selector.
- Evidence validation.
- Source formatter.
- LangSmith trace.
- API chat tu frontend toi backend.

Can cai thien tiep:

- Chuyen logic persist state that vao `update_state_node`, de `ChatService` chi persist ket qua.
- Bo sung semantic document catalog fields: `auto_summary`, `keywords`, `detected_entities`, `document_topic`.
- Cai thien candidate selector voi confidence scoring/LLM selector top 3-5 metadata.
- Them BM25/hybrid retrieval/reranking neu can.
- Doi FastAPI startup sang lifespan.
- Thay `datetime.utcnow()` bang timezone-aware datetime.
