from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
RAGAS_LLM_MODEL = "gpt-4o-mini"
RAGAS_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {m: 0.0 for m in METRICS}
    zeros["per_question"] = []

    # Key rỗng / placeholder → RAGAS sẽ retry 10 lần × tối đa 60s mỗi call → bỏ qua sớm
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-..."):
        print("  ⚠️  RAGAS evaluation skipped: OPENAI_API_KEY chưa được cấu hình trong .env")
        return zeros

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ChatOpenAI(model=RAGAS_LLM_MODEL, temperature=0),
            embeddings=OpenAIEmbeddings(model=RAGAS_EMBEDDING_MODEL),
            run_config=RunConfig(timeout=120, max_retries=3, max_wait=20, max_workers=8),
        )
        df = result.to_pandas()

        def _score(row, metric) -> float:
            v = row.get(metric, 0.0)
            return 0.0 if v is None or math.isnan(v) else float(v)

        per_question = [
            EvalResult(question=row["question"], answer=row["answer"],
                       contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                       **{m: _score(row, m) for m in METRICS})
            for _, row in df.iterrows()
        ]
        # Aggregate bỏ qua NaN (câu RAGAS không chấm được) thay vì kéo về 0
        aggregate = {m: float(df[m].mean()) if df[m].notna().any() else 0.0 for m in METRICS}
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    analyzed = []
    for r in eval_results:
        scores = {m: getattr(r, m) for m in METRICS}
        worst_metric = min(scores, key=scores.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": round(sum(scores.values()) / len(scores), 4),
            "scores": {m: round(v, 4) for m, v in scores.items()},
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
