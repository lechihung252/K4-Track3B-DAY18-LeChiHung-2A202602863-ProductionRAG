# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Lê Chí Hùng — 2A202602863  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

Kết quả `python main.py` (20 câu hỏi, judge `gpt-4o-mini`, embeddings `text-embedding-3-small`, cùng một hàm `evaluate_ragas()` cho cả 2 pipeline):

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8375 | **0.8927** | +0.0552 |
| Answer Relevancy | 0.7483 | **0.8556** | +0.1073 |
| Context Precision | 0.9250 | **0.8605** | −0.0645 |
| Context Recall | 0.9000 | **0.9167** | +0.0167 |

- **Naive:** paragraph chunking (500 chars) → dense-only (bge-m3) top-3 → gpt-4o-mini.
- **Production:** hierarchical chunking (parent 2048 / child 256, không cắt qua header) → M5 enrichment (1 call/chunk, context line ghi rõ phiên bản hiện hành/đã thay thế) → query decomposition → hybrid BM25 + dense + RRF (top-20) → bge-reranker-v2-m3 (top-5, xen kẽ theo câu hỏi con) → gpt-4o-mini (temperature 0, prompt grounded).

**Quá trình cải thiện Production** (cùng corpus, cùng test set):

| Lần chạy | Thay đổi | Faith. | Ans. Rel. | Ctx. Prec. | Ctx. Rec. |
|---|---|---|---|---|---|
| 1 | M1–M5 + prompt mặc định của scaffold, top-3 | 0.748 | 0.564 | 0.908 | 0.892 |
| 2 | + temperature 0, prompt grounded, query decomposition, top-5 | 0.855 | 0.600 | 0.884 | 0.933 |
| 3 | + sửa evaluator: answer_relevancy sinh câu hỏi cùng ngôn ngữ (áp dụng cho cả baseline) | 0.893 | 0.856 | 0.861 | 0.917 |

> **Lưu ý về Answer Relevancy:** RAGAS 0.1 sinh câu hỏi ngược từ câu trả lời bằng prompt tiếng Anh → câu hỏi tiếng Anh, so cosine với câu hỏi tiếng Việt chỉ được ~0.48 (cùng ngôn ngữ ~0.87) dù câu trả lời đúng. Đã thêm instruction *"same language as the answer"* vào `answer_relevancy.question_generation` trong `evaluate_ragas()` → baseline 0.47 → 0.75, production 0.60 → 0.86. Đây là lỗi đo lường, không phải lỗi pipeline.

> **Context Precision giảm −0.06** là trade-off có chủ đích: production trả về 5 contexts (multi-hop cần nhiều nguồn) thay vì 3, và các câu hỏi về phiên bản kéo theo cả chunk phiên bản cũ (v2023) — RAGAS coi chunk cũ là "không liên quan" khi nó xếp trên chunk hiện hành.

## Bottom-5 Failures

Lấy từ `reports/ragas_report.json` → `failures` (sort theo điểm trung bình 4 metrics).

### #1 — Tính phạt tạm ứng sai (avg 0.63)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày, phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (pro-rata ~50.000 VNĐ cho 5 ngày).
- **Got:** Suy luận đúng "20 − 15 = 5 ngày quá hạn", "2%/tháng ≈ 0,0667%/ngày", nhưng nhân sai: 15.000.000 × 0,0667% × 5 = **5.000 VNĐ** (đúng phải là ~50.000 VNĐ).
- **Worst metric:** faithfulness = 0.15 (precision 0.87, recall 0.67, relevancy 0.84)
- **Error Tree:** Output sai → Context đúng? **Có** (chunk "Phạt quá hạn 2%/tháng sau 15 ngày" và "Thời hạn thanh toán 15 ngày" ở rank 1–2) → Query OK? **Có** (không cần decompose) → **Lỗi ở bước Generation: LLM tính toán số học sai**.
- **Root cause:** gpt-4o-mini làm phép tính nhiều bước với số thập phân trong văn bản tự do → sai bậc độ lớn. Faithfulness thấp vì claim "5.000 VNĐ" không suy ra được từ context. (Lần chạy 1, khi chưa có prompt "tính từng bước", model còn tính sai cả số ngày quá hạn: 15 ngày thay vì 5.)
- **Suggested fix:** Không để LLM tự nhân/chia: dùng tool calling với hàm `calculate()` (hoặc sinh biểu thức rồi `eval` an toàn), hoặc dùng model mạnh hơn cho các câu có số liệu; thêm vào test set các câu numeric để regression.

### #2 — Thiếu ý "tự đóng bảo hiểm" khi nghỉ không lương (avg 0.73)
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16–30 ngày cần CEO phê duyệt. Lưu ý: nghỉ trên 14 ngày, nhân viên tự đóng phần bảo hiểm.
- **Got:** "Cần phê duyệt của Giám đốc điều hành (CEO) theo quy định 'Nghỉ từ 16-30 ngày…'" — đúng nhưng thiếu ý bảo hiểm.
- **Worst metric:** context_precision = 0.5, context_recall = 0.5 (faithfulness 1.0)
- **Error Tree:** Output đúng một phần → Context đúng? **Thiếu** — chunk "Nghỉ trên 14 ngày, nhân viên cần tự đóng phần bảo hiểm" (section *Quyền lợi*) không có trong top-5; thay vào đó là 2 chunk về nghỉ phép năm v2023/v2024 và thử việc → Query OK? Query chỉ hỏi "ai phê duyệt", không nhắc bảo hiểm → **Lỗi ở Retrieval: ground truth cần thông tin liên quan mà query không gợi ra**.
- **Root cause:** Child chunk 256 ký tự tách "phê duyệt" và "quyền lợi/bảo hiểm" thành 2 section riêng; pipeline trả về child chứ không trả parent, nên thông tin liền kề trong cùng tài liệu bị mất. Các chunk "nghỉ phép năm" chen vào top-5 làm precision giảm.
- **Suggested fix:** Parent-document retrieval thực sự: retrieve child → trả về **parent** (cả tài liệu `nghi_phep_khong_luong.md` chỉ ~1 parent) cho LLM; hoặc metadata filter theo `source` của top-1 để lấy thêm chunk cùng tài liệu.

### #3 — Không nêu được "chính sách cũ không yêu cầu MFA" (avg 0.77)
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo v2.0 hiện hành bắt buộc MFA cho email, VPN, hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** "Có, tất cả nhân viên bắt buộc phải kích hoạt MFA cho email, VPN và các hệ thống nội bộ."
- **Worst metric:** context_recall = 0.5 (faithfulness 1.0, precision 1.0)
- **Error Tree:** Output đúng nhưng thiếu so sánh phiên bản → Context đúng? Chunk MFA v2.0 ở rank 1 (đúng), nhưng không có chunk nào của `mat_khau_v1.md` → Query OK? Có → **Lỗi ở Retrieval/Data: thông tin "v1.0 không yêu cầu MFA" là thông tin dạng vắng mặt (absence) — v1.0 không có câu nào nhắc đến MFA nên không chunk nào match**.
- **Root cause:** RAG chỉ retrieve được điều *được viết ra*; câu "không yêu cầu" phải suy ra từ việc v1.0 không có mục MFA. Ground truth đòi hỏi so sánh 2 phiên bản ở mức tài liệu.
- **Suggested fix:** Khi chunk top-1 có `version_status = current` (metadata từ M5), tự động kéo thêm chunk cùng section của phiên bản `superseded` (hoặc tạo trước "changelog chunk" so sánh v1 vs v2 lúc ingest) để LLM nói được "phiên bản cũ không có quy định này".

### #4 — Hoàn chi đào tạo: đúng nhưng faithfulness thấp (avg 0.77)
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết làm việc ≥ 1 năm sau khóa học; nghỉ sau 8 tháng là trước hạn → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "…phải hoàn trả 100% chi phí đào tạo nếu nghỉ việc trước thời hạn cam kết 1 năm. Do đó phải hoàn trả 25.000.000 VNĐ." — **đúng**.
- **Worst metric:** faithfulness = 0.40 (precision 1.0, recall 1.0)
- **Error Tree:** Output đúng → Context đúng? **Có** (chunk "Cam kết hoàn chi" rank 1) → Query OK? Có → **Lỗi ở Generation style + Evaluation**: câu trả lời mở đầu bằng việc nhắc lại giả thiết của câu hỏi ("được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng"), các claim này đến từ *câu hỏi*, không có trong context → RAGAS judge đánh là "không suy ra từ context".
- **Root cause:** Prompt quy tắc 1 ("nhắc lại chủ thể câu hỏi") giúp answer_relevancy nhưng sinh thêm claim không có trong context → giảm faithfulness. Đây là trade-off giữa 2 metric.
- **Suggested fix:** Sửa prompt: nhắc lại *chủ thể* nhưng không lặp lại *số liệu giả định* của người hỏi như một sự thật; hoặc tách phần "Giả thiết:" khỏi phần trả lời. Có thể thêm ví dụ few-shot.

### #5 — Lương thử việc Junior: đúng nhưng faithfulness 0.5 (avg 0.83)
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất 20.000.000 VNĐ/tháng; thử việc 85% → 17.000.000 VNĐ/tháng.
- **Got:** "17.000.000 VNĐ… 85% mức lương… mức lương gross cao nhất cho Junior là 20.000.000 VNĐ. Tính toán: 20.000.000 × 85% = 17.000.000" — **đúng**.
- **Worst metric:** faithfulness = 0.50 (relevancy 0.87, precision 0.95, recall 1.0)
- **Error Tree:** Output đúng → Context đúng? **Có** (chunk "Lương thử việc 85%" từ 2 tài liệu + bảng khung lương rank 3) → Query OK? Có → **Lỗi ở Evaluation (judge)**: claim "17.000.000" là kết quả *phép tính* gộp 2 context (85% × 20 triệu), không có nguyên văn trong context → judge `gpt-4o-mini` không chấp nhận suy luận nhiều bước.
- **Root cause:** Faithfulness của RAGAS chấm từng statement có "được context trực tiếp hỗ trợ" hay không; câu multi-hop/numeric luôn bị thiệt. Thêm nữa, bảng markdown bị chunk thành text 1 dòng (`| Junior (P1-P2) | 12.000.000 - 20.000.000 |`) khó đọc cho judge.
- **Suggested fix:** Dùng judge mạnh hơn (gpt-4o) cho faithfulness hoặc chấm lặp nhiều lần lấy trung bình; ở M1 giữ nguyên bảng markdown (header + rows) trong cùng một chunk.

## Case Study (cho presentation)

**Question chọn phân tích:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?" (multi-hop) — câu đã được **sửa thành công** trong lab.

**Error Tree walkthrough:**
1. **Output đúng?** → Lần chạy 1: **Sai một nửa** — trả lời đúng 18 ngày phép nhưng "Về lương, thông tin không có đề cập trong context". (faithfulness 0.6, recall 0.5)
2. **Context đúng?** → **Không**: top-20 của hybrid search và top-6 sau rerank toàn là tài liệu nghỉ phép (`nghi_phep_nam_v2024`, `v2023`, `nghi_phep_khong_luong`…); `bang_luong_2024.md` **không có trong 20 ứng viên**. Câu hỏi có 2 ý, ý "nghỉ phép" lấn át embedding và BM25.
3. **Query rewrite OK?** → **Không có query rewrite** → đây là root cause. Kiểm chứng: tách riêng "Lương của nhân viên Senior trong khoảng nào?" thì `bang_luong_2024.md` lên **rank 0**.
4. **Fix ở bước:** **Query transformation** — thêm `decompose_query()` (1 LLM call, JSON mode) tách câu hỏi thành câu hỏi con giữ nguyên con số; search + rerank cho câu gốc và từng câu con, gộp xen kẽ theo rank. Sau fix: context có cả chunk thâm niên v2024 và bảng lương → trả lời "15 + 3 = 18 ngày, lương Senior 20–35 triệu", context_recall toàn bộ test set 0.892 → 0.917–0.933.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Parent-document retrieval** (trả parent thay vì child cho LLM) → sửa #2, tăng recall cho câu cần ý liền kề.
- **Calculator tool** cho câu numeric → sửa #1.
- **Version-aware retrieval** dùng `version_status` metadata: lọc/hạ điểm chunk `superseded` khi câu hỏi không hỏi về lịch sử, kéo chunk cũ khi cần so sánh → tăng context precision (đang −0.06 so với baseline) và sửa #3.
- **Giảm latency rerank** (8.0 s/query trên CPU, chiếm 68% thời gian query): chạy reranker trên GPU có đủ VRAM, giảm candidates 20 → 10, hoặc dùng model nhẹ hơn.
- **OCR 2 file PDF scan** (`BCTC.pdf`, Nghị định 13/2023) đang bị bỏ qua hoàn toàn.
