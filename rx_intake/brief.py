"""Reviewer brief: retrieval-augmented guidance for a pharmacist reviewing a flagged intake.

For each issue on an intake, retrieve the guidance passages that explain it,
then (optionally) ask a model to write a short summary that cites them.

The brief only informs the human. It never changes the intake's status, and the
pipeline does not depend on it, so a failure here costs convenience, not safety.

Grounding checks on the model's summary (any failure drops the summary and the
pharmacist sees the raw passages instead):
  * every sentence cites a passage, e.g. [G01]
  * every cited id was actually retrieved for this intake
  * every number in the summary appears in the passages or the issue messages,
    which catches the most dangerous hallucination here: an invented dose
"""

from __future__ import annotations

import re
from typing import Protocol

import httpx
from pydantic import BaseModel

from . import observability as obs
from .formulary import Formulary
from .retrieval import Hit, Retriever, expand_query
from .schemas import IntakeResult, Severity

MAX_PASSAGES = 4
PER_ISSUE = 2

SUMMARY_PROMPT = """You help a pharmacist review a flagged prescription.
Write at most 3 short sentences explaining what to check, using ONLY the passages provided.
End every sentence with the id of the passage it comes from in square brackets, e.g. [G01].
Do not add doses, numbers or advice that are not in the passages. Plain text only."""


class BriefPassage(BaseModel):
    id: str
    title: str
    text: str
    score: float


class ReviewBrief(BaseModel):
    intake_id: str
    retriever: str
    retrieval_fallback: bool
    passages: list[BriefPassage]
    summary: str | None
    summary_status: str  # "ok", "skipped", or "rejected: <reason>"


class Summarizer(Protocol):
    name: str

    def summarize(self, issues: list[str], passages: list[BriefPassage]) -> str: ...


class OllamaSummarizer:
    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434", timeout_s: float = 120.0):
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def summarize(self, issues: list[str], passages: list[BriefPassage]) -> str:
        docs = "\n".join(f'<passage id="{p.id}">{p.title}. {p.text}</passage>' for p in passages)
        user = "Issues found:\n- " + "\n- ".join(issues) + f"\n\nPassages:\n{docs}"
        r = httpx.post(f"{self.base_url}/api/chat", timeout=self.timeout_s, json={
            "model": self.model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": user}],
        })
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class BriefBuilder:
    def __init__(self, primary: Retriever, fallback: Retriever, formulary: Formulary, summarizer: Summarizer | None = None):
        self.primary = primary
        self.fallback = fallback
        self.formulary = formulary
        self.summarizer = summarizer

    def build(self, result: IntakeResult) -> ReviewBrief:
        flagged = [i for i in result.issues if i.severity in (Severity.REVIEW, Severity.BLOCK)]
        drug = result.formulary_match.generic_name if result.formulary_match else ""
        queries = [expand_query(f"{i.message} {drug}".strip(), self.formulary) for i in flagged]

        hits, retriever_name, used_fallback = self._retrieve(queries, result.trace_id)
        passages = [BriefPassage(id=h.passage.id, title=h.passage.title, text=h.passage.text, score=h.score) for h in hits]

        summary, status = None, "skipped"
        if self.summarizer and passages:
            summary, status = self._summarize([i.message for i in flagged], passages, result.trace_id)

        obs.event("brief.built", result.trace_id, retriever=retriever_name, fallback=used_fallback,
                  passages=[p.id for p in passages], summary_status=status)
        return ReviewBrief(intake_id=result.id, retriever=retriever_name, retrieval_fallback=used_fallback,
                           passages=passages, summary=summary, summary_status=status)

    def _retrieve(self, queries: list[str], trace_id: str) -> tuple[list[Hit], str, bool]:
        try:
            return _merge([self.primary.search(q, PER_ISSUE) for q in queries]), self.primary.name, False
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            # Embedding model down: keyword search still works and needs no model.
            obs.event("retriever.failed", trace_id, retriever=self.primary.name, error=str(exc)[:200])
            return _merge([self.fallback.search(q, PER_ISSUE) for q in queries]), self.fallback.name, True

    def _summarize(self, issues: list[str], passages: list[BriefPassage], trace_id: str) -> tuple[str | None, str]:
        try:
            text = self.summarizer.summarize(issues, passages)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            obs.event("summarizer.failed", trace_id, error=str(exc)[:200])
            return None, "rejected: summarizer unavailable"
        problem = check_grounding(text, passages, issues)
        if problem:
            obs.event("summary.rejected", trace_id, reason=problem)
            return None, f"rejected: {problem}"
        return text, "ok"


def _merge(per_query: list[list[Hit]]) -> list[Hit]:
    """Interleave results so each issue gets its best passage before any issue gets a second one."""
    seen, merged = set(), []
    for rank in range(PER_ISSUE):
        for hits in per_query:
            if rank < len(hits) and hits[rank].passage.id not in seen:
                seen.add(hits[rank].passage.id)
                merged.append(hits[rank])
    return merged[:MAX_PASSAGES]


_CITATION = re.compile(r"\[(G\d+)\]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def check_grounding(summary: str, passages: list[BriefPassage], issues: list[str]) -> str | None:
    """Return why the summary is not grounded in the sources, or None if it is."""
    if not summary.strip():
        return "empty summary"
    allowed = {p.id for p in passages}
    cited = set(_CITATION.findall(summary))
    if not cited:
        return "no citations"
    if unknown := cited - allowed:
        return f"cites passages that were not retrieved: {sorted(unknown)}"
    # A citation written after the full stop ("... [G01]. [G02]") belongs to the sentence before it.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?!\[)", summary.strip()) if s.strip()]
    if any(not _CITATION.search(s) for s in sentences):
        return "a sentence has no citation"
    source_numbers = set(_NUMBER.findall(" ".join([p.text for p in passages] + issues)))
    claimed = set(_NUMBER.findall(_CITATION.sub("", summary)))
    if invented := claimed - source_numbers:
        return f"numbers not found in sources: {sorted(invented)}"
    return None
