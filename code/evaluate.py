"""
Computes a few retrieval metrics (precision@k, MRR, context coverage, latency)
"""

from __future__ import annotations

import time
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple

from document_loader import SanskritDocumentLoader
from retriever import HybridRetriever


# Test queries (question, list-of-keywords that should appear in context)


EVAL_QUERIES: List[Dict] = [
    {
        "query": "कालीदासः कः?",
        "relevant_keywords": ["कालीदास", "kAlIdAsa", "कवि", "poet", "भोज"],
        "description": "Who is Kalidasa?",
    },
    {
        "query": "Who is Shankhanaad?",
        "relevant_keywords": ["शंखनाद", "shankh", "भृत्य", "servant", "गोवर्धन"],
        "description": "About the foolish servant Shankhanaad",
    },
    {
        "query": "घण्टाकर्णः कः?",
        "relevant_keywords": ["घण्टा", "राक्षस", "bell", "demon", "वानर"],
        "description": "Who is Ghantakarna?",
    },
    {
        "query": "What did King Bhoj offer to poets?",
        "relevant_keywords": ["लक्ष", "रुप्यक", "bhoj", "भोज", "poet", "कवि"],
        "description": "Reward offered by King Bhoj",
    },
    {
        "query": "भक्तस्य कथा",
        "relevant_keywords": ["भक्त", "देव", "प्रार्थना", "शकट", "वृष्टि"],
        "description": "Story of the devotee",
    },
    {
        "query": "What trick did Kalidasa use to help the poet?",
        "relevant_keywords": ["बाध", "पालखी", "पण्डित", "काव्य", "shloka"],
        "description": "Kalidasa's clever trick",
    },
]



# Relevance judge


def is_relevant(chunk_text: str, keywords: List[str]) -> bool:
    """Return True if any keyword appears in the chunk text."""
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


# Metrics


def precision_at_k(retrieved: List[str], keywords: List[str]) -> float:
    relevant = [is_relevant(t, keywords) for t in retrieved]
    return sum(relevant) / len(relevant) if relevant else 0.0


def mean_reciprocal_rank(retrieved: List[str], keywords: List[str]) -> float:
    for rank, text in enumerate(retrieved, 1):
        if is_relevant(text, keywords):
            return 1.0 / rank
    return 0.0


def context_coverage(query: str, retrieved_texts: List[str]) -> float:
    """Share of query tokens that appear in the retrieved text."""
    query_tokens = set(query.lower().split())
    context = " ".join(retrieved_texts).lower()
    found = sum(1 for tok in query_tokens if tok in context)
    return found / len(query_tokens) if query_tokens else 0.0


# Main evaluation loop

def evaluate(
    retriever: HybridRetriever,
    queries: List[Dict],
    top_k: int = 5,
) -> Dict:
    results = []

    for q in queries:
        t0 = time.time()
        retrieved = retriever.retrieve(q["query"], top_k=top_k)
        latency_ms = int((time.time() - t0) * 1000)

        texts = [chunk.text for chunk, _ in retrieved]
        scores = [score for _, score in retrieved]

        p_at_k = precision_at_k(texts, q["relevant_keywords"])
        mrr = mean_reciprocal_rank(texts, q["relevant_keywords"])
        coverage = context_coverage(q["query"], texts)

        results.append({
            "query": q["query"],
            "description": q["description"],
            "precision_at_k": round(p_at_k, 4),
            "mrr": round(mrr, 4),
            "context_coverage": round(coverage, 4),
            "latency_ms": latency_ms,
            "top_scores": [round(s, 4) for s in scores[:3]],
        })

        print(f"\n  Query   : {q['description']}")
        print(f"  P@{top_k}    : {p_at_k:.2f}  |  MRR: {mrr:.2f}  |  Coverage: {coverage:.2f}  |  {latency_ms} ms")

    # Aggregate
    avg_p = sum(r["precision_at_k"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_cov = sum(r["context_coverage"] for r in results) / len(results)
    avg_lat = sum(r["latency_ms"] for r in results) / len(results)

    summary = {
        "n_queries": len(results),
        "top_k": top_k,
        "avg_precision_at_k": round(avg_p, 4),
        "avg_mrr": round(avg_mrr, 4),
        "avg_context_coverage": round(avg_cov, 4),
        "avg_latency_ms": round(avg_lat, 1),
        "per_query": results,
    }
    return summary



# CLI


if __name__ == "__main__":
    import sys, argparse

    parser = argparse.ArgumentParser(description="Evaluate the Sanskrit RAG retriever")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_json", default="eval_results.json")
    args = parser.parse_args()

    print("[Eval] Loading and indexing documents …")
    loader = SanskritDocumentLoader(chunk_size=300, chunk_overlap=50)
    chunks = loader.load_directory(args.data_dir)

    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    print(f"\n[Eval] Running {len(EVAL_QUERIES)} queries (top_k={args.top_k}) …")
    summary = evaluate(retriever, EVAL_QUERIES, top_k=args.top_k)

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"  Queries evaluated : {summary['n_queries']}")
    print(f"  Avg Precision@{args.top_k}  : {summary['avg_precision_at_k']:.4f}")
    print(f"  Avg MRR           : {summary['avg_mrr']:.4f}")
    print(f"  Avg Coverage      : {summary['avg_context_coverage']:.4f}")
    print(f"  Avg Latency       : {summary['avg_latency_ms']} ms")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {args.output_json}")
