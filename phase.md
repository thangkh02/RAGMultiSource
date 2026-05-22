## Các đầu việc đã làm được

- **1. Tách xử lý chunking cho cả 2 nguồn tài liệu**  
  Đã xử lý chunking riêng cho tài liệu hệ thống và tài liệu người dùng upload, đảm bảo mỗi nguồn có pipeline tiền xử lý phù hợp.

- **2. Chuẩn hóa metadata cho document/chunk**  
  Đã lưu metadata đúng và đủ để phục vụ truy xuất, phân biệt được nguồn tài liệu và hỗ trợ filtering trong pipeline RAG.

- **3. Lưu embedding vào vector database**  
  Đã tạo và lưu vector embedding cho các chunk tài liệu để phục vụ semantic retrieval.

- **4. Có cơ chế retrieval cosine similarity**  
  Đã triển khai truy xuất đơn giản bằng cosine similarity để lấy các chunk liên quan từ embedding.

- **5. Hỗ trợ cả tài liệu hệ thống và tài liệu upload**  
  Đã có nền tảng xử lý cho hai nhóm tài liệu chính của project: document hệ thống và document người dùng upload.

---

## Các chức năng cần phải làm hiện tại

Hiện tại chưa xong công việc nào.

## Kế hoạch các đầu việc còn lại

### Giai đoạn 1: Hoàn thiện truy xuất đúng tài liệu

- **1. Làm Scope Resolver**  
  Xác định query này cần tìm trong: file vừa upload, file trong session hiện tại, file cũ của user, tài liệu hệ thống, hoặc mixed.

- **2. Làm Document Resolver**  
  Xác định đúng document cụ thể trước khi retrieve chunk, ví dụ: "file này", "file hôm qua", "tài liệu lần trước", "file vừa upload".

- **3. Bổ sung Conversation State**  
  Lưu session hiện tại, danh sách file đã upload, document vừa dùng, scope vừa dùng, `last_referenced_doc`, `last_scope`, `last_sources`, `current_session_docs` để hỗ trợ hỏi tiếp.

- **4. Bắt buộc metadata filter khi retrieve**  
  Không search toàn bộ vector DB. Cần filter theo `document_id`, `source_type`, `owner_user_id`, `session_id`, `visibility`, `uploaded_at` để đảm bảo đúng ngữ cảnh và đúng quyền.

- **5. Lưu `last_referenced_doc` cho câu hỏi follow-up**  
  Sau mỗi lượt trả lời, lưu lại document vừa dùng để hỗ trợ các câu hỏi nối tiếp như "thế thời hạn bao lâu?" hoặc "còn lệ phí thì sao?".

### Giai đoạn 2: Hoàn thiện truy vấn và câu trả lời

- **6. Làm Intent Router**  
  Nhận diện user đang hỏi file hiện tại, tài liệu hệ thống, file cũ, câu hỏi tiếp theo hay câu hỏi so sánh.

- **7. Rewrite câu hỏi mơ hồ sau khi đã biết scope/document**  
  Biến câu hỏi thiếu ngữ cảnh thành câu hỏi đầy đủ hơn để tăng độ chính xác khi truy xuất.

- **8. Thiết kế prompt trả lời bám sát context**  
  Bắt LLM chỉ trả lời dựa trên chunk retrieve được, không tự suy diễn nếu tài liệu không có thông tin.

- **9. Xử lý khi không tìm thấy thông tin**  
  Nếu retrieval không có chunk đủ liên quan, hệ thống phải trả lời rõ là không tìm thấy trong tài liệu, không được bịa.

- **10. Trả lời rõ nguồn tài liệu**  
  Phân biệt rõ câu trả lời theo tài liệu upload hay theo tài liệu hệ thống. Nếu có nhiều nguồn thì tách phần trả lời tương ứng.

### Giai đoạn 3: Logging và kiểm thử

- **11. Logging toàn bộ pipeline**  
  Ghi lại query gốc, query rewrite, intent, scope, document được chọn, filter retrieval, chunk retrieved và câu trả lời cuối cùng.

- **12. Tạo test case theo từng nhóm tình huống**  
  Chuẩn bị test cho file mới upload, file cũ, system docs, follow-up, mixed source và câu hỏi mơ hồ.

- **13. Kiểm tra retrieval có lấy đúng chunk hay không**  
  Xác nhận pipeline truy xuất ra đúng chunk cần thiết trước khi đưa vào LLM.

- **14. Đánh giá answer theo context**  
  Kiểm tra câu trả lời có đúng nguồn, đúng nội dung và không suy diễn ngoài tài liệu.

### Giai đoạn 4: Tối ưu sau

- **15. Hybrid retrieval**  
  Kết hợp vector search với keyword/BM25 nếu cần.

- **16. Reranking**  
  Chọn lại chunk liên quan hơn sau bước retrieve ban đầu.

- **17. Dedupe chunk**  
  Loại bỏ chunk trùng hoặc nhiễu để tránh lặp context.

- **18. Context packing**  
  Sắp xếp context theo document, page, section trước khi đưa vào LLM.

- **19. Tối ưu top-k và threshold**  
  Tune số lượng chunk retrieve, ngưỡng similarity và số lượng context đưa vào LLM để cân bằng giữa đúng và gọn.

---

## Pipeline nên triển khai

```text
User query
→ Load conversation state / last_referenced_doc
→ Intent Router
→ Scope Resolver
→ Document Resolver
→ Query Rewrite
→ Retrieval Planner
→ Metadata Filter
→ Vector / Hybrid Search
→ Rerank
→ Context Packing
→ Answer Generator
→ Save last document + log trace
```
