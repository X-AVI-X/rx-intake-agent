"""Retrieval over the pharmacy guidance knowledge base.

Three retrievers behind one interface, so the eval can compare them:
  * BM25Retriever    - keyword scoring. Exact terms ("BMDC", "eGFR") rank well;
                       synonyms and brand names ("Napa", "blood thinner") do not.
  * VectorRetriever  - embeddings + cosine similarity. Finds passages with the
                       same meaning even when the words differ.
  * HybridRetriever  - runs both and merges the rankings with Reciprocal Rank
                       Fusion, so each covers the other's blind spots.

The knowledge base is 30 short passages, so a plain Python list is the index.
At thousands of passages this would move to pgvector or a vector database;
the interfaces here would stay the same.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_KB = DATA_DIR / "guidance.jsonl"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


@dataclass(frozen=True)
class Passage:
    id: str
    drug: str | None
    topic: str
    title: str
    text: str

    @property
    def content(self) -> str:
        """What gets indexed: drug, title and body together."""
        return f"{self.drug or 'general'}. {self.title}. {self.text}"


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


def load_passages(path: Path = DEFAULT_KB) -> list[Passage]:
    return [Passage(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 3) -> list[Hit]: ...


# --------------------------------------------------------------------------- BM25

_STOPWORDS = frozenset("a an and are as at be by for from has have in is it its of on or should such that the this to was with".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    """Okapi BM25, the standard keyword-ranking formula used by search engines.

    score(doc) = sum over query terms of
        idf(term) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))

    idf: rare terms count more than common ones.
    k1:  how quickly repeated terms stop adding score (saturation).
    b:   how much long documents are penalised.
    """

    name = "bm25"

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(p.content)) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        n = len(passages)
        df = Counter(term for d in self.docs for term in d)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = tokenize(query)
        scored = []
        for p, doc, length in zip(self.passages, self.docs, self.lengths):
            s = 0.0
            for t in terms:
                if tf := doc.get(t):
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if s > 0:
                scored.append(Hit(p, round(s, 4)))
        return sorted(scored, key=lambda h: h.score, reverse=True)[:k]


# ---------------------------------------------------------------------- embeddings

class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], kind: str) -> list[list[float]]: ...


class OllamaEmbedder:
    """Embeddings from a self-hosted model, so guidance and queries stay on our network.

    nomic-embed-text was trained with task prefixes: documents and queries are
    embedded differently ("search_document:" / "search_query:"). Leaving the
    prefixes out measurably lowers retrieval quality.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434", timeout_s: float = 60.0):
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        prefix = "search_query: " if kind == "query" else "search_document: "
        r = httpx.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": [prefix + t for t in texts]},
                       timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["embeddings"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorRetriever:
    """Embeds every passage once (cached on disk), then ranks by cosine similarity to the query."""

    def __init__(self, passages: list[Passage], embedder: Embedder, cache_dir: Path | None = CACHE_DIR):
        self.passages = passages
        self.embedder = embedder
        self.name = f"vector:{embedder.name}"
        self.cache_dir = cache_dir
        self._vectors: list[list[float]] | None = None

    @property
    def vectors(self) -> list[list[float]]:
        # Embedded on first use, not at construction, so the service starts even if the embedding model is down.
        if self._vectors is None:
            self._vectors = self._load_or_embed(self.cache_dir)
        return self._vectors

    def _load_or_embed(self, cache_dir: Path | None) -> list[list[float]]:
        contents = [p.content for p in self.passages]
        # The cache key covers the model and every passage, so editing the knowledge base re-embeds it.
        key = hashlib.sha256(json.dumps([self.embedder.name, contents]).encode()).hexdigest()[:16]
        path = cache_dir / f"embeddings-{key}.json" if cache_dir else None
        if path and path.exists():
            return json.loads(path.read_text())
        vectors = self.embedder.embed(contents, kind="document")
        if path:
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(vectors))
        return vectors

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = self.embedder.embed([query], kind="query")[0]
        scored = [Hit(p, round(cosine(q, v), 4)) for p, v in zip(self.passages, self.vectors)]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:k]


# ---------------------------------------------------------------------------- hybrid

class HybridRetriever:
    """Reciprocal Rank Fusion: score = sum of 1 / (rrf_k + rank) across retrievers.

    RRF uses ranks, not raw scores, because BM25 scores (0 to ~15) and cosine
    similarities (0 to 1) are on different scales and can't be added directly.
    rrf_k = 60 is the value from the original paper; it stops the top rank of
    one retriever from dominating.
    """

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 10):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth
        self.name = "hybrid(" + "+".join(r.name for r in retrievers) + ")"

    def search(self, query: str, k: int = 3) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id: dict[str, Passage] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, self.depth), start=1):
                fused[hit.passage.id] = fused.get(hit.passage.id, 0.0) + 1 / (self.rrf_k + rank)
                by_id[hit.passage.id] = hit.passage
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(by_id[pid], round(score, 5)) for pid, score in ranked]


def build_retriever(mode: str, passages: list[Passage] | None = None, embedder: Embedder | None = None) -> Retriever:
    passages = passages if passages is not None else load_passages()
    if mode == "bm25":
        return BM25Retriever(passages)
    embedder = embedder or OllamaEmbedder()
    if mode == "vector":
        return VectorRetriever(passages, embedder)
    if mode == "hybrid":
        return HybridRetriever([BM25Retriever(passages), VectorRetriever(passages, embedder)])
    raise ValueError(f"Unknown retrieval mode {mode!r}; use bm25, vector or hybrid")


def expand_query(query: str, formulary) -> str:
    """Add the generic name for any brand or alias in the query ("Napa" -> "paracetamol").

    This is cheap, deterministic query rewriting using data we already trust.
    Embedding models trained mostly on English text often don't know local
    brands such as Napa or Seclo, but the formulary does.
    """
    words = set(tokenize(query))
    extra = []
    for drug in formulary.drugs:
        if drug.generic_name not in query.lower() and any(set(tokenize(a)) <= words for a in drug.aliases):
            extra.append(drug.generic_name)
    return f"{query} ({', '.join(extra)})" if extra else query
