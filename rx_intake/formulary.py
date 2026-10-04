"""Grounding: map the drug name the model extracted onto a known formulary entry.

The model is never trusted to "know" a drug. Whatever it extracts is looked up
here, and every dose rule runs against the formulary record, not model output.
Fuzzy matching (stdlib difflib) is enough for a few hundred names; a larger
catalogue would move to an indexed search (Postgres trigram or embeddings).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .schemas import FormularyMatch

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "formulary.json"
MATCH_THRESHOLD = 0.82


@dataclass(frozen=True)
class Drug:
    generic_name: str
    aliases: tuple[str, ...]
    strengths_mg: tuple[float, ...]
    max_daily_mg: float | None
    routes: tuple[str, ...]
    max_duration_days: int
    high_alert: bool = False
    controlled: bool = False
    names: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", (self.generic_name, *self.aliases))


class Formulary:
    def __init__(self, drugs: list[Drug]):
        self.drugs = drugs

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Formulary":
        raw = json.loads(path.read_text())
        drugs = [
            Drug(
                generic_name=d["generic_name"],
                aliases=tuple(d.get("aliases", [])),
                strengths_mg=tuple(d.get("strengths_mg", [])),
                max_daily_mg=d.get("max_daily_mg"),
                routes=tuple(d.get("routes", [])),
                max_duration_days=d["max_duration_days"],
                high_alert=d.get("high_alert", False),
                controlled=d.get("controlled", False),
            )
            for d in raw
        ]
        return cls(drugs)

    def lookup(self, name: str | None) -> tuple[Drug, FormularyMatch] | None:
        """Return the best-matching drug, or None if nothing is close enough."""
        if not name:
            return None
        query = _normalise(name)
        best: tuple[float, Drug, str] | None = None
        for drug in self.drugs:
            for candidate in drug.names:
                score = _similarity(query, candidate)
                if best is None or score > best[0]:
                    best = (score, drug, candidate)
        if best is None or best[0] < MATCH_THRESHOLD:
            return None
        score, drug, matched = best
        return drug, FormularyMatch(generic_name=drug.generic_name, matched_text=matched, score=round(score, 3))

    def all_names(self) -> list[str]:
        return [n for d in self.drugs for n in d.names]


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def _similarity(query: str, candidate: str) -> float:
    if query == candidate:
        return 1.0
    # "Napa Extra" should still match "napa": score each query token too.
    token_best = max((SequenceMatcher(None, tok, candidate).ratio() for tok in query.split()), default=0.0)
    return max(SequenceMatcher(None, query, candidate).ratio(), token_best)
