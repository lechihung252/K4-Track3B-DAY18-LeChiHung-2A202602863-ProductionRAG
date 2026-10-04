from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import functools, glob, hashlib, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Helpers ─────────────────────────────────────────────

LLM_MODEL = "gpt-4o-mini"
MAX_WORKERS = 16  # Gọi API song song — mỗi call 5-60s tuỳ mạng nên tuần tự rất chậm
# Cache kết quả combined enrichment theo (source, text, prompt version) → chạy lại pipeline không tốn API
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "enrichment.json")
CACHE_VERSION = "v1"
_cache_lock = threading.Lock()

_CLIENT = None


def _has_api_key() -> bool:
    return bool(OPENAI_API_KEY) and not OPENAI_API_KEY.startswith("sk-...")


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
    """Gọi gpt-4o-mini (client dùng chung giữa các thread)."""
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


@functools.lru_cache(maxsize=None)
def _doc_header(source: str) -> str:
    """Tiêu đề + dòng phiên bản/hiệu lực của document (các dòng trước section '## ' đầu tiên)."""
    path = os.path.join(DATA_DIR, source)
    if not source.endswith(".md") or not os.path.exists(path):
        return ""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("## "):
                break
            if line.strip():
                lines.append(line.strip().lstrip("#> ").strip())
    return " | ".join(lines)


@functools.lru_cache(maxsize=None)
def _version_status(source: str) -> str:
    """'superseded' / 'current' nếu document có nhiều phiên bản (vd: mat_khau_v1.md, mat_khau_v2.md), else ''."""
    m = re.match(r"^(.*)_v(\d+)\.md$", source)
    if not m:
        return ""
    versions = [int(re.search(r"_v(\d+)\.md$", f).group(1))
                for f in glob.glob(os.path.join(DATA_DIR, f"{m.group(1)}_v*.md"))]
    if len(versions) < 2:
        return ""
    return "current" if int(m.group(2)) == max(versions) else "superseded"


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


_ENRICH_CACHE: dict = _load_cache()


def _cache_key(text: str, source: str) -> str:
    return hashlib.sha256(f"{CACHE_VERSION}|{source}|{text}".encode()).hexdigest()


def _version_note(source: str) -> str:
    status = _version_status(source)
    if status == "superseded":
        return "PHIÊN BẢN CŨ — đã bị thay thế bởi phiên bản mới hơn, không còn hiệu lực."
    if status == "current":
        return "PHIÊN BẢN HIỆN HÀNH — thay thế các phiên bản trước."
    return ""


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if _has_api_key():
        try:
            return _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. "
                         "Giữ nguyên mọi con số, mức tiền, thời hạn.", text, max_tokens=150)
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    # Extractive fallback (không cần API)
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if _has_api_key():
        try:
            content = _chat(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. "
                            "Trả về mỗi câu hỏi trên 1 dòng, không đánh số.", text, max_tokens=200)
            questions = [q.strip().lstrip("0123456789.-) ").strip() for q in content.split("\n")]
            return [q for q in questions if q][:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    # Extractive fallback
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if _has_api_key():
        try:
            context = _chat("Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
                            "Chỉ trả về 1 câu.",
                            f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80)
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    # Simple fallback
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    default = {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
    if _has_api_key():
        try:
            content = _chat('Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], '
                            '"category": "policy|hr|it|finance", "language": "vi|en"}',
                            text, max_tokens=150, json_mode=True)
            return {**default, **json.loads(content)}
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")
    return default


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    header = _doc_header(source)
    version_note = _version_note(source)
    version_meta = {"version_status": _version_status(source)} if version_note else {}

    key = _cache_key(text, source)
    cached = _ENRICH_CACHE.get(key)
    if cached is not None:
        return cached

    if _has_api_key():
        try:
            system = """Bạn chuẩn bị chunk tài liệu nội bộ cho hệ thống tìm kiếm (RAG). Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu, giữ nguyên con số",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn thuộc tài liệu nào (tên chính sách, phiên bản, ngày hiệu lực) và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}
Nếu tài liệu được ghi chú là PHIÊN BẢN CŨ thì câu "context" BẮT BUỘC nêu rõ đây là phiên bản cũ đã bị thay thế.
Nếu là PHIÊN BẢN HIỆN HÀNH thì câu "context" nêu rõ đây là phiên bản hiện hành."""
            user = f"Tài liệu: {source}\nThông tin tài liệu: {header}\n"
            if version_note:
                user += f"Ghi chú phiên bản: {version_note}\n"
            user += f"\nĐoạn văn:\n{text}"
            result = json.loads(_chat(system, user, max_tokens=500, json_mode=True))
            result["metadata"] = {**result.get("metadata", {}), **version_meta}
            with _cache_lock:
                _ENRICH_CACHE[key] = result
            return result
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    # Fallback không cần API: context từ header + ghi chú phiên bản
    context = " ".join(x for x in [f"Trích từ: {header}." if header else "", version_note] if x)
    return {"context": context, "metadata": version_meta} if context else {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    def _enrich_one(chunk: dict) -> EnrichedChunk:
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        return EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        )

    # Thread pool giữ nguyên thứ tự output (executor.map)
    n_cached = sum(_cache_key(c["text"], c.get("metadata", {}).get("source", "")) in _ENRICH_CACHE for c in chunks)
    if use_combined and n_cached:
        print(f"  ({n_cached}/{len(chunks)} chunks lấy từ cache {os.path.relpath(CACHE_PATH)})", flush=True)
    enriched = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, ec in enumerate(executor.map(_enrich_one, chunks)):
            enriched.append(ec)
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    if use_combined:
        with _cache_lock:
            _save_cache(_ENRICH_CACHE)
    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
