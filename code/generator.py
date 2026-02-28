# Local CPU-only generators:
# - LlamaCppGenerator:   uses llama-cpp-python with a GGUF model
# - GPT4AllGenerator:    uses the gpt4all package
# - ExtractiveGenerator: simple fallback that pulls sentences from context
#
# get_generator() picks an available option.

from __future__ import annotations

import re
from typing import List, Optional

from document_loader import DocumentChunk

# Prompt builder  (shared by all generators)


SYSTEM_PROMPT = (
    "You are a knowledgeable assistant specializing in Sanskrit literature, "
    "grammar (vyakarana), and traditional Indian texts. "
    "Answer using only the provided context. "
    "If the answer is not in the context, say so clearly."
)


def build_prompt(query: str, context_chunks: List[DocumentChunk]) -> str:
#Construct the RAG prompt from retrieved context chunks.
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        parts.append(f"[Context {i} — {chunk.source_file}]\n{chunk.text}")
    context_str = "\n\n---\n\n".join(parts)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== Relevant context ===\n{context_str}\n\n"
        f"=== Question ===\n{query}\n\n"
        f"=== Answer ===\n"
    )



# 1. llama-cpp-python  (recommended — works with any GGUF model)


class LlamaCppGenerator:
    """
LlamaCppGenerator: local CPU-only generator using llama-cpp-python.

Requires:
    pip install llama-cpp-python

The generator loads any compatible GGUF model provided by the user.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        max_tokens: int = 512,
        temperature: float = 0.2,
        n_threads: int = 4,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "Install llama-cpp-python:\n"
                "  pip install llama-cpp-python\n"
                "Then download a GGUF model file (see class docstring)."
            )
        print(f"[LlamaCppGenerator] Loading model from '{model_path}' …")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        print("[LlamaCppGenerator] Model loaded.")

    def generate(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        prompt = build_prompt(query, context_chunks)
        output = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["===", "\n\n\n"],
        )
        return output["choices"][0]["text"].strip()



# 2. GPT4All  (easy install, good model selection)


class GPT4AllGenerator:
    """
    Requires:
        pip install gpt4all
    """

    def __init__(
        self,
        model_name: str = "Phi-3-mini-4k-instruct.Q4_0.gguf",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ):
        try:
            from gpt4all import GPT4All
        except ImportError:
            raise ImportError(
                "Install gpt4all:\n"
                "  pip install gpt4all\n"
                "The model will be auto-downloaded on first run."
            )
        print(f"[GPT4AllGenerator] Loading '{model_name}' …")
        self.model = GPT4All(model_name, allow_download=True)
        self.max_tokens = max_tokens
        self.temperature = temperature
        print("[GPT4AllGenerator] Model ready.")

    def generate(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        prompt = build_prompt(query, context_chunks)
        with self.model.chat_session():
            response = self.model.generate(
                prompt,
                max_tokens=self.max_tokens,
                temp=self.temperature,
            )
        return response.strip()


# 3. Extractive Generator  (zero-dependency fallback)


class ExtractiveGenerator:
    """
    Zero-dependency extractive generator.

    Scores every sentence in the retrieved chunks by counting how many
    query tokens appear in that sentence, then returns the top sentences.
    Works entirely offline with no model downloads.
    """

    def __init__(self, max_sentences: int = 5):
        self.max_sentences = max_sentences

    def _tokenize(self, text: str) -> set:
        tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z0-9]+", text.lower())
        return set(tokens)

    def generate(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        if not context_chunks:
            return "No relevant context found for your query."

        query_tokens = self._tokenize(query)

        # Collect all sentences from all chunks
        all_sentences = []
        for chunk in context_chunks:
            sentences = re.split(r"(?<=[।\.\?!])\s+", chunk.text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20:
                    score = len(query_tokens & self._tokenize(sent))
                    all_sentences.append((score, sent, chunk.source_file))

        # Sort by relevance score descending
        all_sentences.sort(key=lambda x: x[0], reverse=True)
        top = all_sentences[: self.max_sentences]

        if not top:
            return f"[Extractive] No relevant sentences found.\nTop chunk: {context_chunks[0].text[:300]}"

        answer_lines = [f"[Extractive Answer — source: {top[0][2]}]\n"]
        for _, sent, _ in top:
            answer_lines.append(sent)

        return "\n".join(answer_lines)



# Factory helper — auto-selects best available generator

def get_generator(
    generator_type: str = "auto",
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """
    Return the best available local generator.

    Args:
        generator_type: "llama_cpp" | "gpt4all" | "extractive" | "auto"
        model_path:     Path to GGUF file (for llama_cpp)
        model_name:     Model name string (for gpt4all)

    Auto-selection order:
        llama_cpp (if model_path given) → gpt4all (if installed) → extractive
    """
    if generator_type == "llama_cpp" or (generator_type == "auto" and model_path):
        if not model_path:
            raise ValueError("model_path is required for llama_cpp generator.")
        return LlamaCppGenerator(model_path=model_path)

    if generator_type == "gpt4all" or generator_type == "auto":
        try:
            import gpt4all  # noqa: F401
            name = model_name or "Phi-3-mini-4k-instruct.Q4_0.gguf"
            return GPT4AllGenerator(model_name=name)
        except ImportError:
            if generator_type == "gpt4all":
                raise
            print("[generator] gpt4all not installed, falling back to extractive.")

    print("[generator] Using ExtractiveGenerator (zero-dependency fallback).")
    return ExtractiveGenerator()



# Quick smoke test

if __name__ == "__main__":
    from document_loader import DocumentChunk

    fake_chunk = DocumentChunk(
        chunk_id="test_0000",
        text=(
            "कालीदासः भोजराज्ञः दरबारे प्रसिद्धः कविः आसीत् । "
            "kAlIdAsa was a famous poet in King Bhoj's court. "
            "He was known for his cleverness and mastery of Sanskrit grammar."
        ),
        source_file="test.docx",
        chunk_index=0,
    )

    gen = get_generator()   # will use extractive since no model installed
    answer = gen.generate("Who was Kalidasa?", [fake_chunk])
    print(answer)
