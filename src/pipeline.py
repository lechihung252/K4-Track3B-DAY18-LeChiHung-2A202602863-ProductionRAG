from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import json, os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import OPENAI_API_KEY, RERANK_TOP_K

LLM_MODEL = "gpt-4o-mini"

ANSWER_PROMPT = """Bạn là trợ lý trả lời câu hỏi về chính sách nội bộ công ty. Chỉ dùng thông tin trong CONTEXT.
Quy tắc:
1. Trả lời trực tiếp vào câu hỏi bằng câu hoàn chỉnh, nhắc lại chủ thể của câu hỏi. Với câu hỏi có/không, bắt đầu bằng "Có" hoặc "Không".
2. Mọi con số, mức tiền, ngưỡng, người phê duyệt phải lấy đúng từ CONTEXT và nêu quy định làm căn cứ (vd: "đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt").
3. Nếu cần so sánh với ngưỡng hoặc tính toán: trình bày ngắn gọn từng bước (quy định → số liệu → phép tính → kết quả).
4. Nếu CONTEXT có nhiều phiên bản chính sách: trả lời theo phiên bản hiện hành và nói rõ phiên bản cũ đã bị thay thế.
5. Không suy đoán, không thêm thông tin ngoài CONTEXT. Chỉ khi CONTEXT hoàn toàn không đề cập thì nói "Không tìm thấy thông tin trong tài liệu." cho phần đó."""

DECOMPOSE_PROMPT = """Phân tích câu hỏi của người dùng về chính sách công ty.
Chỉ tách khi câu hỏi hỏi NHIỀU thông tin thuộc các chủ đề/chính sách khác nhau
(vd: số ngày phép VÀ mức lương; người phê duyệt VÀ yêu cầu của phòng CNTT).
Câu hỏi về 1 chủ đề (kể cả cần tính toán) thì KHÔNG tách.
Mỗi câu hỏi con phải tự đầy đủ ngữ cảnh và GIỮ NGUYÊN mọi con số, số tiền, thời gian, chức danh trong câu hỏi gốc.
Tối đa 3 câu hỏi con. Trả về JSON: {"sub_queries": ["...", "..."]} (không tách → mảng chỉ chứa câu hỏi gốc)."""

# Latency breakdown (bonus): thời gian từng bước build + trung bình mỗi query
LATENCY: dict = {"build_s": {}, "query_ms": {}}

_CLIENT = None


def _llm(system: str, user: str, json_mode: bool = False, max_tokens: int = 500) -> str:
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        _CLIENT = OpenAI(timeout=60, max_retries=3)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _CLIENT.chat.completions.create(
        model=LLM_MODEL, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    LATENCY["build_s"]["chunking"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    LATENCY["build_s"]["enrichment"] = time.time() - t0

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    LATENCY["build_s"]["indexing"] = time.time() - t0
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3) — load ngay để latency query không tính thời gian load model
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    LATENCY["build_s"]["load_reranker"] = time.time() - t0
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def decompose_query(query: str) -> list[str]:
    """Tách câu hỏi multi-hop thành câu hỏi con (vd: '9 năm thâm niên được nghỉ bao nhiêu ngày và lương bao nhiêu')."""
    if not OPENAI_API_KEY:
        return [query]
    try:
        subs = json.loads(_llm(DECOMPOSE_PROMPT, query, json_mode=True, max_tokens=300)).get("sub_queries", [])
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()][:3]
        # Luôn search cả câu hỏi gốc trước, rồi đến các câu hỏi con
        return [query] + [s for s in subs if s != query]
    except Exception as e:
        print(f"  ⚠️  Query decomposition failed: {e}", flush=True)
        return [query]


def retrieve(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
             timings: dict | None = None) -> list[str]:
    """Decompose → hybrid search + rerank cho từng câu hỏi con → gộp xen kẽ theo rank."""
    timings = timings if timings is not None else {}

    t0 = time.perf_counter()
    sub_queries = decompose_query(query)
    timings["decompose"] = timings.get("decompose", 0) + (time.perf_counter() - t0) * 1000

    ranked_lists = []
    for sq in sub_queries:
        t0 = time.perf_counter()
        results = search.search(sq)
        timings["search"] = timings.get("search", 0) + (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
        reranked = reranker.rerank(sq, docs, top_k=RERANK_TOP_K)
        timings["rerank"] = timings.get("rerank", 0) + (time.perf_counter() - t0) * 1000
        ranked_lists.append([r.text for r in reranked] if reranked else [r.text for r in results[:RERANK_TOP_K]])

    # Xen kẽ: top-1 của mỗi câu hỏi con trước, rồi top-2... → mỗi khía cạnh đều có context ở đầu
    max_contexts = max(RERANK_TOP_K, 2 * (len(sub_queries) - 1))
    contexts = []
    for rank in range(RERANK_TOP_K):
        for lst in ranked_lists:
            if rank < len(lst) and lst[rank] not in contexts:
                contexts.append(lst[rank])
    return contexts[:max_contexts]


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              timings: dict | None = None) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    timings = timings if timings is not None else {}
    contexts = retrieve(query, search, reranker, timings)

    t0 = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            context_str = "\n\n---\n\n".join(contexts)
            answer = _llm(ANSWER_PROMPT, f"CONTEXT:\n{context_str}\n\nCâu hỏi: {query}")
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    timings["generate"] = timings.get("generate", 0) + (time.perf_counter() - t0) * 1000
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    per_query_timings = []

    for i, item in enumerate(test_set):
        timings: dict = {}
        answer, contexts = run_query(item["question"], search, reranker, timings)
        per_query_timings.append(timings)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    LATENCY["build_s"]["ragas_eval"] = time.time() - t0
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    # Latency breakdown
    steps = ["decompose", "search", "rerank", "generate"]
    n = max(len(per_query_timings), 1)
    LATENCY["query_ms"] = {s: sum(t.get(s, 0) for t in per_query_timings) / n for s in steps}
    LATENCY["query_ms"]["total"] = sum(LATENCY["query_ms"].values())
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    for step, sec in LATENCY["build_s"].items():
        print(f"  build  {step:<14} {sec:>9.1f} s")
    for step, ms in LATENCY["query_ms"].items():
        print(f"  query  {step:<14} {ms:>9.0f} ms (avg/query)")

    failures = failure_analysis(results.get("per_question", []))
    latency = {"build_s": {k: round(v, 2) for k, v in LATENCY["build_s"].items()},
               "query_ms_avg": {k: round(v, 1) for k, v in LATENCY["query_ms"].items()}}
    save_report(results, failures, extra={
        "latency": latency,
        "per_question": [
            {"question": q, "answer": a, "contexts": c, "ground_truth": g}
            for q, a, c, g in zip(questions, answers, all_contexts, ground_truths)
        ],
    })
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
