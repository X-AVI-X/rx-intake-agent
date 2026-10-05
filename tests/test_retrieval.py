import httpx
import pytest

from rx_intake.brief import BriefBuilder, BriefPassage, check_grounding
from rx_intake.formulary import Formulary
from rx_intake.pipeline import IntakePipeline
from rx_intake.extractors import RulesExtractor
from rx_intake.retrieval import BM25Retriever, HybridRetriever, Passage, VectorRetriever, expand_query, load_passages

PASSAGES = load_passages()
FORMULARY = Formulary.load()


class FakeEmbedder:
    """Bag-of-letters vectors: deterministic and model-free, good enough to test the plumbing."""

    name = "fake"

    def embed(self, texts, kind):
        return [[t.lower().count(c) for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]


class DownEmbedder:
    name = "down"

    def embed(self, texts, kind):
        raise httpx.ConnectError("embedding model unavailable")


def test_bm25_ranks_exact_keyword_first():
    hits = BM25Retriever(PASSAGES).search("eGFR below 30", k=3)
    assert hits[0].passage.id == "G09"


def test_bm25_returns_nothing_for_unknown_words():
    assert BM25Retriever(PASSAGES).search("zzqx wobble", k=3) == []


def test_expand_query_adds_generic_for_local_brand():
    assert "paracetamol" in expand_query("Napa 500 two tabs", FORMULARY)
    assert expand_query("paracetamol 500", FORMULARY) == "paracetamol 500"


def test_vector_retriever_embeds_lazily_and_caches(tmp_path):
    r = VectorRetriever(PASSAGES, FakeEmbedder(), cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing embedded at construction
    r.search("warfarin", k=1)
    assert len(list(tmp_path.iterdir())) == 1


def test_rrf_rewards_agreement_between_retrievers():
    p = [Passage(f"P{i}", None, "t", f"title {i}", "x") for i in range(3)]

    class Fixed:
        def __init__(self, name, order):
            self.name, self.order = name, order

        def search(self, query, k=3):
            from rx_intake.retrieval import Hit
            return [Hit(p[i], 1.0) for i in self.order][:k]

    # P1 is second in both lists; P0 and P2 are each first in only one. Agreement wins.
    hybrid = HybridRetriever([Fixed("a", [0, 1]), Fixed("b", [2, 1])])
    assert hybrid.search("q", k=1)[0].passage.id == "P1"


def _flagged_result(text):
    pipeline = IntakePipeline(primary=RulesExtractor(), fallback=None, formulary=FORMULARY)
    return pipeline.process(text)


OVERDOSE = "Patient: Selina Parvin\nAge: 44\nRx: Paracetamol 500mg tablet, 3 tabs QID x 5 days, oral\nDr. Sharmin Sultana, BMDC Reg No: A-20981"


def test_brief_falls_back_to_keyword_search_when_embeddings_are_down():
    builder = BriefBuilder(VectorRetriever(PASSAGES, DownEmbedder(), cache_dir=None), BM25Retriever(PASSAGES), FORMULARY)
    brief = builder.build(_flagged_result(OVERDOSE))
    assert brief.retrieval_fallback is True
    assert brief.passages[0].id == "G01"


class FakeSummarizer:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def summarize(self, issues, passages):
        return self.text


@pytest.mark.parametrize("summary, ok", [
    ("Adults must not exceed 4 g a day [G01]. Ask about other paracetamol products [G01].", True),
    ("Adults must not exceed 4 g a day [G01]. [G02]", True),
    ("Adults must not exceed 4 g a day.", False),                        # no citation
    ("Adults must not exceed 4 g a day [G01]. Lower it for liver disease.", False),  # uncited sentence
    ("See the warfarin guidance [G20].", False),                         # not retrieved
    ("Adults must not exceed 6 g a day [G01].", False),                  # invented number
])
def test_grounding_check(summary, ok):
    passages = [BriefPassage(id="G01", title="Paracetamol daily limit", text=PASSAGES[0].text, score=1.0),
                BriefPassage(id="G02", title="Liver", text=PASSAGES[1].text, score=0.9)]
    assert (check_grounding(summary, passages, ["dose above max"]) is None) is ok


def test_ungrounded_summary_is_dropped_but_passages_remain():
    builder = BriefBuilder(BM25Retriever(PASSAGES), BM25Retriever(PASSAGES), FORMULARY,
                           summarizer=FakeSummarizer("Give 8000 mg daily [G01]."))
    brief = builder.build(_flagged_result(OVERDOSE))
    assert brief.summary is None
    assert brief.summary_status.startswith("rejected: numbers not found")
    assert brief.passages
