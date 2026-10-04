# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Lê Chí Hùng — 2A202602863  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 tạo **208 chunks (avg 99 chars, min 6)** vs basic **51 chunks (avg 410)**. Bị vụn vì scaffold dùng `all-MiniLM-L6-v2` (chủ yếu tiếng Anh) → similarity giữa các câu tiếng Việt thấp, cắt quá nhiều. Muốn dùng thật cho tiếng Việt phải đổi sang encoder đa ngôn ngữ (bge-m3). |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Pipeline dùng strategy này: **26 parents / 125 children (avg 166 chars)**. Bug phát hiện khi đọc kết quả search: child vắt qua 2 section (`"...nhập sai liên tiếp. ## Chính sác..."`) → sửa bằng cách tách theo header trước khi cắt child. `parent_id` gắn `source` để không trùng giữa các file. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunks (avg 196, max 788 — section có bảng dài). Giữ header trong text + `metadata["section"]`; header không có nội dung (H1 ngay trước H2) được gộp vào chunk con để giữ ngữ cảnh. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | underthesea nối từ ghép bằng `_` ("nghỉ_phép") → phải replace thành space, thêm lowercase + bỏ token dấu câu để query "nghỉ phép" khớp. Với "nghỉ phép năm bao nhiêu ngày", BM25 top-1 là `nghi_phep_dac_biet` còn dense top-1 là `nghi_phep_nam_v2023`; RRF (k=60) đưa các tài liệu cả 2 đều xếp cao (v2023, v2024) lên đầu. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3: **~1.1 s/20 docs trên MPS, ~3 s/20 docs trên CPU**; trong pipeline (1–3 câu hỏi con/query) rerank chiếm **8.0 s/query = 68% latency**. Reranker chấm *relevance* chứ không biết *phiên bản*: v2023 (0.995) vẫn > v2024 (0.992) → phải xử lý version ở M5 + prompt. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Final: Faithfulness **0.893**, Answer Relevancy **0.856**, Context Precision **0.861**, Context Recall **0.917**. Metric thấp nhất ban đầu là Answer Relevancy (0.47 baseline / 0.60 production) — **do evaluator**: prompt sinh câu hỏi tiếng Anh, cosine với câu hỏi tiếng Việt chỉ 0.48 (cùng ngôn ngữ 0.87). Sửa instruction "same language" → 0.75 / 0.86. Bài học: phải kiểm tra metric trước khi tối ưu theo metric. |
| Contextual embeddings | M5 | `_enrich_single_call()` | 1 call/chunk (JSON mode) → summary, 3 HyQA questions, context line, metadata. Đưa header tài liệu (phiên bản, ngày hiệu lực) + `version_status` (suy từ `*_v1/_v2.md`) vào prompt → context line ghi rõ *"phiên bản cũ đã bị thay thế"* / *"phiên bản hiện hành"* → LLM trả lời theo bản 2024 (15 ngày) thay vì 2023. 125 chunks cache ở `.cache/enrichment.json`. |
| Query transformation | Pipeline | `decompose_query()`, `retrieve()` | Câu multi-hop "9 năm thâm niên… nghỉ bao nhiêu ngày **và** lương bao nhiêu": `bang_luong_2024.md` không nằm trong top-20; tách câu hỏi con thì lên rank 0. Search câu gốc + câu con, xen kẽ kết quả → context recall 0.892 → 0.933. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

**1. transformers 5.x không chạy với torch 2.2.2 (Mac Intel)**
- **Error:** `NameError: name 'nn' is not defined` (trong `transformers/integrations/accelerate.py`) khi `from sentence_transformers import CrossEncoder`.
- **Debug:** `uname -m` → `x86_64`; torch cho macOS Intel dừng ở 2.2.2, `pip install` kéo transformers 5.18 yêu cầu torch mới hơn.
- **Fix:** Pin `transformers>=4.41,<4.50`, `sentence-transformers>=3.0,<4`, `numpy<2` (torch 2.2.2 build với numpy 1.x) trong `requirements.txt`. Cũng dùng venv Python 3.11 (máy mặc định 3.14, nhiều package chưa hỗ trợ).

**2. MPS out of memory khi chạy `main.py`**
- **Error 1:** `RuntimeError: MPS backend out of memory (MPS allocated: 6.67 GB, other allocations: 100.92 MB, max allowed: 6.77 GB)` tại `m3_rerank.py: model.predict(pairs)`.
- **Debug:** `M3` chạy riêng thì OK → khác biệt là `main.py` chạy naive + production **cùng process**, mỗi `DenseSearch()` load một bản bge-m3 (~2.2 GB) + reranker 2.2 GB trên GPU AMD 4 GB.
- **Fix 1:** Cache encoder theo tên model ở module level (`_ENCODER_CACHE`) → chỉ 1 bản bge-m3.
- **Error 2:** `MPS allocated: 4.57 GB, other allocations: 2.18 GB` — 2 model vẫn không vừa.
- **Fix 2:** Benchmark reranker trên CPU (~3 s/20 docs, chấp nhận được) → `RERANK_DEVICE` trong `config.py`, mặc định `cpu`.

**3. Mạng chậm / download treo**
- Docker: `failed to resolve reference "docker.io/qdrant/qdrant:latest": failed to authorize: failed to fetch anonymous token: ... EOF` → Docker Desktop vừa khởi động, retry sau 10 s là được.
- Flashrank model treo ở 2.1 MB; `curl` đo được **~6 KB/s** tới HuggingFace (model 103 MB) → bỏ qua (optional, pipeline không dùng), xoá file zip dở dang để không đọc phải file hỏng.

**4. RAGAS chấm sai Answer Relevancy cho tiếng Việt**
- **Triệu chứng:** câu "Nhân viên được nghỉ bao nhiêu ngày phép năm?" trả lời hoàn hảo nhưng answer_relevancy = 0.41.
- **Debug:** Gọi thẳng prompt `answer_relevancy.question_generation` → sinh ra `"How many paid leave days does an employee receive according to the 2024 leave policy?"`; đo cosine `text-embedding-3-small`: EN↔VI 0.48, VI↔VI 0.87.
- **Fix:** Thêm instruction *"The generated question MUST be written in the same language as the answer"* (idempotent) trong `evaluate_ragas()`, áp dụng cho cả baseline lẫn production để so sánh công bằng.

**5. Các bẫy khác**
- RAGAS mặc định `max_retries=10, max_wait=60` → với key placeholder `sk-...` test bị treo rất lâu → check key sớm + `RunConfig(max_retries=3)`.
- `check_lab.py` chạy pytest với `timeout=120`; test M5 gọi OpenAI thật mất ~140 s → cache response LLM ra đĩa, lần sau 0.02 s.
- Gọi API tuần tự quá chậm (5–60 s/call) → `ThreadPoolExecutor` 16 workers trong `enrich_chunks()`.

**Kiến thức còn thiếu & cách bổ sung:**
- Cách RAGAS tính từng metric (đặc biệt answer_relevancy dùng embedding + câu hỏi sinh ngược, context_precision là rank-weighted) → đọc source `ragas/metrics/*.py` thay vì chỉ đọc docs; hiểu được vì sao tăng top-k 3→5 gần như không làm giảm precision.
- Quản lý bộ nhớ GPU với nhiều model trong một process → đọc lỗi MPS, đo bằng thực nghiệm (CPU vs MPS) trước khi quyết định.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Chatbot hỏi đáp chính sách nội bộ (HR / IT / Tài chính)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Kết quả lab này — 26 tài liệu markdown/PDF → hierarchical chunking → enrichment 1 call/chunk → hybrid BM25 + bge-m3 (Qdrant) + RRF → query decomposition → bge-reranker-v2-m3 → gpt-4o-mini. RAGAS: F 0.89 / AR 0.86 / CP 0.86 / CR 0.92 trên 20 câu.
- **Vấn đề / Bottlenecks:**
  - Latency **~11.8 s/query** (rerank CPU 8.0 s, search 1.8 s, LLM 1.2 s, decompose 0.9 s) — quá chậm cho chatbot.
  - 2 PDF scan (BCTC, Nghị định 13/2023) bị bỏ qua vì không có text layer.
  - LLM tính toán sai (phạt tạm ứng 5.000 thay vì 50.000 VNĐ).
  - Xung đột phiên bản: chunk v2023 vẫn lọt vào context → context precision thấp hơn baseline.
  - Test set chỉ 20 câu; RAGAS judge dao động ±0.02–0.05 giữa các lần chạy.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Giữ **Hierarchical** (child 256 để retrieve chính xác) nhưng **trả parent cho LLM** (parent-document retrieval) để không mất ý liền kề (lỗi #2 "tự đóng bảo hiểm"). Giữ nguyên bảng markdown trong 1 chunk. OCR PDF scan (Tesseract `vie` hoặc OCR API) trước khi chunk.
2. **Search retrieval:** **Hybrid BM25 + dense + RRF** — BM25 bắt mã/số hiệu chính xác ("Nghị định 13", "P3-P4"), dense bắt diễn đạt khác. Thêm **metadata filter theo `version_status`**: mặc định chỉ lấy `current`, chỉ lấy `superseded` khi câu hỏi có từ khoá so sánh/lịch sử. Chỉ gọi decomposition khi câu hỏi có dấu hiệu multi-hop ("và", nhiều dấu hỏi) để tiết kiệm ~0.9 s.
3. **Reranking:** Có — **bge-reranker-v2-m3** (tốt cho tiếng Việt), chạy trên GPU server; giảm candidates 20 → 10; cân nhắc ONNX/quantization. Mục tiêu rerank < 500 ms.
4. **Evaluation:** RAGAS 4 metrics (giữ fix "same language" cho answer_relevancy) + metric riêng: **exact-match cho câu numeric** và **"version correctness"** (câu trả lời có dùng phiên bản hiện hành không). Mở rộng test set 20 → 100 câu từ log câu hỏi thật của nhân viên; chạy 3 lần lấy trung bình; chạy trong CI mỗi khi đổi prompt/chunking.
5. **Enrichment:** Giữ **combined single-call** (contextual prepend + metadata + version status) vì là yếu tố giúp model chọn đúng phiên bản; dùng HyQA questions làm thêm 1 vector/chunk để match câu hỏi tự nhiên. Thêm **calculator tool** (function calling) cho câu hỏi tính phạt/lương/hoàn trả.

#### 3. Timeline triển khai
- **Tuần 1:** OCR 2 PDF scan; parent-document retrieval; giữ bảng markdown nguyên khối; mở rộng test set lên 50 câu (thêm numeric + version + negation). Đo lại RAGAS làm baseline mới.
- **Tuần 2:** Version-aware filter bằng `version_status`; calculator tool cho câu numeric; metric exact-match + version correctness. Mục tiêu: Faithfulness ≥ 0.92, Context Precision ≥ 0.90.
- **Tuần 3:** Tối ưu latency: reranker lên GPU / ONNX, candidates 10, decomposition có điều kiện, cache câu hỏi lặp lại. Mục tiêu p95 < 3 s/query.
- **Tuần 4:** Đóng gói API (FastAPI) + UI chat nội bộ, logging câu hỏi/câu trả lời/contexts, RAGAS chạy hằng tuần trên mẫu log thật; pilot với 1 phòng ban, thu feedback 👍/👎 để bổ sung test set.
