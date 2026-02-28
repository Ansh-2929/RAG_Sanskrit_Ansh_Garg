
#Loads and preprocesses Sanskrit documents from .docx or .txt files.

import os
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:80].replace('\n', ' ')
        return f"DocumentChunk(id={self.chunk_id}, source={self.source_file}, text='{preview}...')"


class SanskritDocumentLoader:
  

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
       
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # Public API


    def load_directory(self, directory: str) -> List[DocumentChunk]:
        """Load all .txt/.docx files from a directory."""
        chunks = []
        for path in sorted(Path(directory).rglob("*")):
            if path.suffix.lower() in {".docx", ".txt"}:
                chunks.extend(self.load_file(str(path)))
        return chunks

    def load_file(self, filepath: str) -> List[DocumentChunk]:
        """Load a file and return its text chunks."""
        suffix = Path(filepath).suffix.lower()
        if suffix == ".docx":
            raw_text = self._load_docx(filepath)
        elif suffix == ".txt":
            raw_text = self._load_txt(filepath)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        cleaned = self._preprocess(raw_text)
        return self._split_into_chunks(cleaned, source_file=os.path.basename(filepath))

    
    # Loaders
    
    def _load_docx(self, filepath: str) -> str:
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx is required: pip install python-docx")
        doc = Document(filepath)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n\n".join(paragraphs)

    def _load_txt(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    
    # Preprocessing
    

    def _preprocess(self, text: str) -> str:
        """Basic cleanup: remove invisible chars, trim spaces, collapse blank lines."""
        # Remove zero-width / invisible Unicode characters
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        # Collapse multiple spaces on the same line
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ consecutive blank lines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    
    # Chunking
   

    def _split_into_chunks(self, text: str, source_file: str) -> List[DocumentChunk]:

        """Split text into overlapping chunks, keeping paragraph breaks."""
        
        # Split on double newlines (paragraph boundaries) first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: List[DocumentChunk] = []
        buffer = ""
        chunk_idx = 0

        for para in paragraphs:
            if len(buffer) + len(para) + 2 <= self.chunk_size:
                buffer = f"{buffer}\n\n{para}".strip() if buffer else para
            else:
                if buffer:
                    chunks.append(self._make_chunk(buffer, source_file, chunk_idx))
                    chunk_idx += 1
                    # Carry over overlap
                    buffer = buffer[-self.chunk_overlap:].strip() + "\n\n" + para
                else:
                    # Single paragraph larger than chunk_size — hard-split
                    for sub in self._hard_split(para):
                        chunks.append(self._make_chunk(sub, source_file, chunk_idx))
                        chunk_idx += 1
                    buffer = ""

        if buffer:
            chunks.append(self._make_chunk(buffer, source_file, chunk_idx))

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        """Split a long paragraph into fixed-size pieces with overlap."""
        pieces = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            pieces.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return pieces

    def _make_chunk(self, text: str, source_file: str, idx: int) -> DocumentChunk:
        chunk_id = f"{Path(source_file).stem}_{idx:04d}"
        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            source_file=source_file,
            chunk_index=idx,
            metadata={"char_count": len(text)},
        )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "../data"
    loader = SanskritDocumentLoader(chunk_size=300, chunk_overlap=50)
    chunks = loader.load_directory(path)
    print(f"Loaded {len(chunks)} chunks from '{path}'")
    for c in chunks[:3]:
        print("-" * 60)
        print(c)
        print(c.text[:200])
