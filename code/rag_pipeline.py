"""
rag_pipeline.py
---------------
End-to-end Sanskrit RAG pipeline orchestrator.

Ties together:
  - SanskritDocumentLoader  (document_loader.py)
  - HybridRetriever         (retriever.py)
  - LlamaCppGenerator / GPT4AllGenerator / ExtractiveGenerator (generator.py)

Usage:
    # Extractive (zero install):
    python rag_pipeline.py --data_dir ../data --query "कालीदासः कः?" --generator extractive

    # GPT4All (auto-downloads model):
    python rag_pipeline.py --data_dir ../data --query "..." --generator gpt4all

    # llama.cpp with local GGUF:
    python rag_pipeline.py --data_dir ../data --query "..." --generator llama_cpp \
        --model_path /path/to/model.gguf

    # Interactive REPL:
    python rag_pipeline.py --data_dir ../data --interactive --generator extractive
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

from document_loader import SanskritDocumentLoader, DocumentChunk
from retriever import HybridRetriever
from generator import get_generator

INDEX_CACHE = "sanskrit_index.pkl"


class SanskritRAGPipeline:
    """
    Full RAG pipeline for Sanskrit documents.

    Steps:
        1. Load & chunk documents
        2. Build / load TF-IDF index
        3. Retrieve relevant chunks for a query
        4. Generate answer via local LLM or extractive fallback
    """

    def __init__(
        self,
        data_dir: str = "../data",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        top_k: int = 5,
        score_threshold: float = 0.0,
        generator_type: str = "auto",
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        index_path: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.index_path = index_path or INDEX_CACHE

        self.loader = SanskritDocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.retriever = HybridRetriever()
        self.generator = get_generator(
            generator_type=generator_type,
            model_path=model_path,
            model_name=model_name,
        )

    def initialize(self, force_reindex: bool = False) -> None:
        """Load documents and build (or restore) the retrieval index."""
        if not force_reindex and Path(self.index_path).exists():
            print(f"[Pipeline] Loading cached index from '{self.index_path}' …")
            self.retriever.load(self.index_path)
        else:
            print(f"[Pipeline] Loading documents from '{self.data_dir}' …")
            t0 = time.time()
            chunks = self.loader.load_directory(self.data_dir)
            print(f"[Pipeline] {len(chunks)} chunks loaded in {time.time()-t0:.2f}s")

            print("[Pipeline] Building TF-IDF index …")
            t0 = time.time()
            self.retriever.index_chunks(chunks)
            print(f"[Pipeline] Index built in {time.time()-t0:.2f}s")
            self.retriever.save(self.index_path)

    def query(self, question: str, verbose: bool = True) -> dict:
        """Execute a full RAG query. Returns dict with question/chunks/answer/latency."""
        t0 = time.time()

        results = self.retriever.retrieve(
            question, top_k=self.top_k, score_threshold=self.score_threshold
        )
        chunks = [c for c, _ in results]
        scores = [s for _, s in results]

        if verbose and chunks:
            print(f"\n[Retriever] Top-{len(chunks)} chunks:")
            for i, (c, s) in enumerate(zip(chunks, scores), 1):
                print(f"  {i}. {c.chunk_id} (score={s:.4f})")
                print(f"     {c.text[:120].replace(chr(10),' ')}…")

        answer = self.generator.generate(question, chunks)
        latency_ms = int((time.time() - t0) * 1000)

        return {
            "question": question,
            "retrieved_chunks": [
                {"chunk_id": c.chunk_id, "source": c.source_file, "score": s, "text": c.text}
                for c, s in zip(chunks, scores)
            ],
            "answer": answer,
            "latency_ms": latency_ms,
        }

    def interactive(self) -> None:
        """Run an interactive query loop in the terminal."""
        print("\n" + "="*60)
        print("Sanskrit RAG System — Interactive Mode")
        print("Type your query (Sanskrit or English). Enter 'quit' to exit.")
        print("="*60 + "\n")

        while True:
            try:
                question = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye.")
                break

            result = self.query(question)
            print("\n" + "-"*60)
            print("ANSWER:")
            print(result["answer"])
            print(f"\n(latency: {result['latency_ms']} ms)")
            print("-"*60 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sanskrit RAG Pipeline (fully local, CPU-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default="../data",
                   help="Directory containing Sanskrit documents (.docx / .txt)")
    p.add_argument("--query", default=None,
                   help="Single query (non-interactive mode)")
    p.add_argument("--interactive", action="store_true",
                   help="Launch interactive REPL")
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of chunks to retrieve")
    p.add_argument("--chunk_size", type=int, default=300)
    p.add_argument("--chunk_overlap", type=int, default=50)
    p.add_argument("--force_reindex", action="store_true",
                   help="Rebuild the index even if a cache exists")
    p.add_argument("--generator", default="auto",
                   choices=["auto", "llama_cpp", "gpt4all", "extractive"],
                   help="Generator backend: llama_cpp | gpt4all | extractive | auto")
    p.add_argument("--model_path", default=None,
                   help="Path to GGUF model file (required for llama_cpp)")
    p.add_argument("--model_name", default=None,
                   help="GPT4All model name (optional)")
    p.add_argument("--output_json", default=None,
                   help="Write query result to this JSON file")
    return p


def main():
    args = build_arg_parser().parse_args()

    pipeline = SanskritRAGPipeline(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        generator_type=args.generator,
        model_path=args.model_path,
        model_name=args.model_name,
    )
    pipeline.initialize(force_reindex=args.force_reindex)

    if args.interactive:
        pipeline.interactive()
    elif args.query:
        result = pipeline.query(args.query)
        print("\n" + "="*60)
        print("ANSWER:")
        print(result["answer"])
        print(f"\n(latency: {result['latency_ms']} ms)")
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Result saved to {args.output_json}")
    else:
        print("No query provided. Use --query '...' or --interactive.")
        build_arg_parser().print_help()


if __name__ == "__main__":
    main()
