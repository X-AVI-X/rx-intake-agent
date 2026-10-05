"""Score retrieval quality on labelled queries.

    python -m evals.retrieval                 # bm25, vector, hybrid (vector needs Ollama)
    python -m evals.retrieval --modes bm25    # no model, runs in CI

Metrics:
  recall@3  share of the relevant passages that appear in the top 3
  hit@1     the first result is relevant
  MRR       mean of 1 / rank of the first relevant result (1.0 = always first)

Each mode is run twice: on the raw query and after brand -> generic expansion
from the formulary, to measure what that cheap step is worth.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from rx_intake.formulary import Formulary
from rx_intake.retrieval import build_retriever, expand_query, load_passages

HERE = Path(__file__).resolve().parent
K = 3


def score(retriever, cases: list[dict], formulary: Formulary | None) -> dict:
    recalls, hits1, rrs, latencies = [], [], [], []
    by_tag: dict[str, list[float]] = defaultdict(list)
    misses = []
    for case in cases:
        query = expand_query(case["query"], formulary) if formulary else case["query"]
        start = time.perf_counter()
        ids = [h.passage.id for h in retriever.search(query, k=10)]
        latencies.append((time.perf_counter() - start) * 1000)
        relevant = set(case["relevant"])
        recall = len(relevant & set(ids[:K])) / len(relevant)
        first = next((i for i, pid in enumerate(ids, 1) if pid in relevant), None)
        recalls.append(recall)
        hits1.append(float(bool(ids) and ids[0] in relevant))
        rrs.append(1 / first if first else 0.0)
        for tag in case["tags"]:
            by_tag[tag].append(recall)
        if recall < 1:
            misses.append({"id": case["id"], "query": case["query"], "want": case["relevant"], "got": ids[:K]})
    return {
        "recall@3": round(statistics.mean(recalls), 3),
        "hit@1": round(statistics.mean(hits1), 3),
        "mrr": round(statistics.mean(rrs), 3),
        "recall@3_by_tag": {t: round(statistics.mean(v), 3) for t, v in sorted(by_tag.items())},
        "latency_ms_p50": round(statistics.median(latencies), 1),
        "misses": misses,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["bm25", "vector", "hybrid"])
    parser.add_argument("--dataset", type=Path, default=HERE / "retrieval_dataset.jsonl")
    parser.add_argument("--min-recall", type=float, default=None, help="exit 1 if any mode scores below this")
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    passages, formulary = load_passages(), Formulary.load()
    report = {}
    for mode in args.modes:
        retriever = build_retriever(mode, passages)
        report[mode] = score(retriever, cases, None)
        report[f"{mode}+expand"] = score(retriever, cases, formulary)

    tags = sorted({t for c in cases for t in c["tags"]})
    lines = [f"## Retrieval eval ({len(cases)} queries, {len(passages)} passages)", "",
             "| Retriever | recall@3 | hit@1 | MRR | " + " | ".join(f"recall@3 {t}" for t in tags) + " | p50 latency |",
             "|---" * (5 + len(tags)) + "|"]
    for name, r in report.items():
        per_tag = " | ".join(f"{r['recall@3_by_tag'].get(t, 0):.0%}" for t in tags)
        lines.append(f"| {name} | {r['recall@3']:.0%} | {r['hit@1']:.0%} | {r['mrr']:.2f} | {per_tag} | {r['latency_ms_p50']} ms |")
    lines += ["", "### Misses (relevant passage not in the top 3)", ""]
    for name, r in report.items():
        for m in r["misses"]:
            lines.append(f"- `{name}` {m['id']} \"{m['query']}\": wanted {m['want']}, got {m['got']}")
    md = "\n".join(lines) + "\n"

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    suffix = "-".join(args.modes)
    (out / f"retrieval-{suffix}.json").write_text(json.dumps(report, indent=2))
    (out / f"retrieval-{suffix}.md").write_text(md)
    print(md)

    if args.min_recall is not None and any(r["recall@3"] < args.min_recall for r in report.values()):
        print(f"FAIL: a retriever scored below recall@3 {args.min_recall}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
