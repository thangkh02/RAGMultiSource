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

## Các đầu việc cần làm

- **1. Xác định rõ các scope truy xuất**  
  Chia rõ hệ thống sẽ retrieve trong: file vừa upload, file trong session hiện tại, file cũ của user, tài liệu hệ thống, hoặc cả hai nguồn.

- **2. Làm Intent Router**  
  Nhận diện user đang hỏi loại nào: hỏi file hiện tại, hỏi tài liệu hệ thống, hỏi file cũ, hỏi tiếp câu trước, hay hỏi so sánh.

- **3. Làm Scope Resolver**  
  Từ intent, xác định chính xác nên tìm trong nguồn nào: `current_upload`, `current_session`, `user_old_docs`, `system_docs`, `mixed`.

- **4. Làm Document Resolver**  
  Tìm đúng document cụ thể trước khi retrieve chunk. Ví dụ: "file hôm qua", "file này", "tài liệu lần trước", "file vừa upload".

- **5. Xử lý câu hỏi follow-up**  
  Với câu như "thế thời hạn bao lâu?", "còn lệ phí thì sao?", hệ thống phải dùng lại document vừa hỏi gần nhất.

- **6. Rewrite câu hỏi trước khi retrieve**  
  Biến câu hỏi mơ hồ thành câu hỏi đầy đủ hơn. Ví dụ: "thế thời hạn bao lâu?" -> "Trong tài liệu đang hỏi, thời hạn giải quyết là bao lâu?"

- **7. Tạo Retrieval Planner**  
  Quyết định mỗi câu hỏi cần search ở đâu, search mấy nguồn, top-k bao nhiêu, filter metadata nào.

- **8. Bắt buộc dùng metadata filter khi retrieve**  
  Không search toàn bộ vector DB. Phải filter theo `document_id`, `source_type`, `owner_user_id`, `session_id`, `visibility`, `uploaded_at`.

- **9. Tách retrieval cho từng nguồn**  
  Nếu hỏi file user thì chỉ search file user. Nếu hỏi tài liệu hệ thống thì chỉ search system docs. Nếu hỏi so sánh thì search riêng 2 nguồn.

- **10. Thêm hybrid retrieval nếu cần**  
  Kết hợp vector search với keyword/BM25 để tăng độ chính xác, nhất là với câu hỏi có tên thủ tục, mã hồ sơ, ngày tháng, số hiệu văn bản.

- **11. Thêm reranking**  
  Sau khi lấy top-k chunk ban đầu, dùng reranker để chọn lại những chunk liên quan nhất trước khi đưa vào LLM.

- **12. Xử lý chunk gần nhau**  
  Nếu retrieve trúng một chunk, có thể lấy thêm chunk trước/sau để tránh mất ngữ cảnh.

- **13. Loại bỏ chunk trùng hoặc nhiễu**  
  Dedupe các chunk gần giống nhau, tránh đưa quá nhiều context lặp vào prompt.

- **14. Context packing**  
  Sắp xếp context đưa vào LLM theo document, page, section để câu trả lời mạch lạc và dễ dẫn nguồn.

- **15. Thiết kế prompt trả lời bám sát context**  
  Bắt LLM chỉ trả lời dựa trên chunk retrieve được, không tự suy diễn nếu tài liệu không có thông tin.

- **16. Xử lý khi không tìm thấy thông tin**  
  Nếu retrieval không có chunk đủ liên quan, hệ thống phải nói "không tìm thấy trong tài liệu", không được bịa.

- **17. Trả lời rõ nguồn**  
  Phân biệt rõ: "Theo tài liệu bạn upload..." hoặc "Theo tài liệu hệ thống...". Nếu mixed thì tách hai phần.

- **18. Lưu lại document vừa dùng**  
  Sau mỗi câu trả lời, lưu `last_referenced_doc` để hỗ trợ câu hỏi tiếp theo.

- **19. Logging toàn bộ pipeline**  
  Log lại: query gốc, query rewrite, intent, scope, document chọn, filter retrieval, chunk lấy ra, câu trả lời.

- **20. Tạo bộ test retrieval**  
  Chuẩn bị các câu hỏi test cho: file vừa upload, system docs, file cũ, hỏi tiếp, so sánh, câu mơ hồ.

- **21. Đánh giá retrieval**  
  Kiểm tra top-k có lấy đúng chunk không bằng các metric như Hit@k, Recall@k, MRR hoặc kiểm tra thủ công theo test case.

- **22. Đánh giá answer**  
  Kiểm tra câu trả lời có đúng context không, có bịa không, có dẫn đúng nguồn không.

- **23. Xử lý ambiguity**  
  Nếu user nói "file cũ" nhưng có nhiều file phù hợp, hệ thống nên hỏi lại thay vì tự đoán.

- **24. Kiểm tra phân quyền dữ liệu**  
  Đảm bảo user không bao giờ retrieve được file của user khác.

- **25. Tối ưu top-k và threshold**  
  Tune số lượng chunk retrieve, ngưỡng similarity, số chunk đưa vào LLM để cân bằng đúng và không nhiễu.

---

## Pipeline nên triển khai

```text
User query
→ Rewrite / hiểu câu hỏi
→ Intent Router
→ Scope Resolver
→ Document Resolver
→ Retrieval Planner
→ Metadata Filter
→ Vector / Hybrid Search
→ Rerank
→ Context Packing
→ Answer Generator
→ Save last document + log trace
```
