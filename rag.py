"""
rag.py — Retrieval-Augmented Generation pipeline

    Document → Load & clean → Chunk → Embed → Vector store → Retrieve → Prompt → LLM → Answer

The vector store is a small NumPy implementation (cosine similarity), so it runs anywhere for free.
The Colab notebook shows the same idea with ChromaDB, a real vector database.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- 1. load
def load_document(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return a list of (page_number, text). Supports PDF, TXT and Markdown."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, clean_text(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    return [(None, clean_text(data.decode("utf-8", errors="ignore")))]


def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-\n(?=\w)", "", text)        # join words hyphenated across PDF lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- 2. chunk
@dataclass
class Chunk:
    id: int
    text: str
    source: str
    page: int | None = None

    @property
    def label(self) -> str:
        return f"{self.source}" + (f", page {self.page}" if self.page else "")


def _units(text: str) -> list[str]:
    """Split into paragraphs, then sentences: natural boundaries keep meaning together."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(*#])", para) if s.strip()]
            merged: list[str] = []
            for s in parts:                        # keep "## 3." together with its heading text
                if merged and re.fullmatch(r"[#>*\s]*\d+(\.\d+)*\.", merged[-1]):
                    merged[-1] += " " + s
                else:
                    merged.append(s)
            units.extend(merged)
            units[-1] = units[-1] + "\n"           # remember the paragraph break
    return units


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Sentence-aware chunking: pack sentences into chunks of ≤ chunk_size characters.
    Each new chunk starts with the last ~overlap characters of the previous one, so facts
    that sit on a boundary are not lost."""
    overlap = max(0, min(overlap, chunk_size // 2))   # keep overlap smaller than a chunk
    pieces: list[str] = []
    for u in _units(text):                         # hard-split very long sentences
        while len(u) > chunk_size:
            pieces.append(u[:chunk_size])
            u = u[chunk_size - overlap:]
        pieces.append(u)

    chunks, current = [], []
    for p in pieces:
        size_now = sum(len(x) + 1 for x in current)
        if current and p.lstrip().startswith("#") and size_now > chunk_size * 0.25:
            chunks.append(" ".join(current).strip())  # structure-aware: a heading starts a new chunk
            current = []
        elif current and size_now + len(p) > chunk_size:
            chunks.append(" ".join(current).strip())
            tail, size = [], 0                     # carry the overlap forward
            for x in reversed(current):
                if size + len(x) > overlap:
                    break
                tail.insert(0, x)
                size += len(x) + 1
            current = tail
        current.append(p)
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if len(c) > 20]


def chunk_documents(docs: dict[str, list[tuple[int | None, str]]],
                    chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    out: list[Chunk] = []
    for source, pages in docs.items():
        for page, text in pages:
            for c in chunk_text(text, chunk_size, overlap):
                out.append(Chunk(len(out), c, source, page))
    return out


# --------------------------------------------------------------------------- 3. vector store
def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


class VectorStore:
    """Minimal in-memory vector database: stores normalized embeddings + chunks,
    searches by cosine similarity (dot product of unit vectors)."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def __len__(self):
        return len(self.chunks)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1] if len(self) else 0

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = normalize(vectors)
        self.vectors = vectors if not len(self) else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 4, min_score: float = 0.0,
               mmr: bool = False, fetch_k: int = 20, lambda_: float = 0.7) -> list[tuple[Chunk, float]]:
        if not len(self):
            return []
        q = normalize(query_vec).ravel()
        scores = self.vectors @ q
        order = np.argsort(-scores)
        if mmr:
            order = self._mmr(scores, order[:fetch_k], k, lambda_)
        results = [(self.chunks[i], float(scores[i])) for i in order[:k]]
        return [(c, s) for c, s in results if s >= min_score]

    def _mmr(self, scores, candidates, k, lambda_):
        """Maximal Marginal Relevance: prefer chunks that are relevant AND different
        from chunks already chosen. Reduces near-duplicate results."""
        selected, candidates = [], list(candidates)
        while candidates and len(selected) < k:
            def mmr_score(i):
                redundancy = max((float(self.vectors[i] @ self.vectors[j]) for j in selected), default=0.0)
                return lambda_ * scores[i] - (1 - lambda_) * redundancy
            best = max(candidates, key=mmr_score)
            selected.append(best)
            candidates.remove(best)
        return selected


# --------------------------------------------------------------------------- 4. prompt
SYSTEM_PROMPT = """You are a careful assistant that answers questions about the user's documents.
Rules:
1. Use ONLY the information in the CONTEXT. Do not use outside knowledge.
2. If the answer is not in the context, reply exactly: "I couldn't find this in the uploaded documents."
3. Cite the sources you used with their numbers, like [1] or [2][3], right after the sentence.
4. Be concise and use bullet points for lists."""


def build_prompt(question: str, retrieved: list[tuple[Chunk, float]],
                 history: list[dict] | None = None) -> str:
    context = "\n\n".join(f"[{i}] (source: {c.label})\n{c.text}"
                          for i, (c, _) in enumerate(retrieved, 1)) or "(no relevant context found)"
    convo = ""
    if history:
        recent = history[-4:]  # short memory so follow-up questions make sense
        convo = "RECENT CONVERSATION:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content'][:500]}" for m in recent) + "\n\n"
    return f"CONTEXT:\n{context}\n\n{convo}QUESTION: {question}\n\nANSWER:"


def retrieval_query(question: str, history: list[dict] | None = None) -> str:
    """Short follow-ups like 'and abroad?' lack keywords, so add the previous user question."""
    if history and len(question.split()) < 6:
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev:
            return prev[-1] + " " + question
    return question
