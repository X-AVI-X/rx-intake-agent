"""Score the pipeline against the labelled dataset.

    python -m evals.run --provider rules
    python -m evals.run --provider ollama
    python -m evals.run --provider claude --fail-on-unsafe

Headline safety metric: unsafe auto-accepts, meaning a case that needs a human
was auto-accepted. It must be 0; CI fails the build otherwise. Field accuracy
and over-review rate measure how much work the system saves.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from rx_intake import observability as obs
from rx_intake.config import build_pipeline
from rx_intake.formulary import Formulary

HERE = Path(__file__).resolve().parent


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def field_ok(name: str, expected, actual, formulary: Formulary) -> bool:
    if expected is None:
        return actual in (None, "")
    if actual is None:
        return False
    if name == "drug_name":
        # The model should copy the name as written; grounding maps it later.
        exp, act = formulary.lookup(str(expected)), formulary.lookup(str(actual))
        if exp and act:
            return exp[0].generic_name == act[0].generic_name
        return str(expected).lower() == str(actual).lower()
    if name == "route":
        return str(expected) == getattr(actual, "value", str(actual))
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) < 1e-6
        except (TypeError, ValueError):
            return False
    return str(expected).strip().lower() == str(actual).strip().lower()


def run(provider: str, dataset: Path) -> dict:
    pipeline = build_pipeline(provider)
    formulary = pipeline.formulary
    cases = load_cases(dataset)

    per_field: dict[str, list[bool]] = defaultdict(list)
    rows, latencies = [], []
    unsafe, over_review, routed_ok, flags_hit, flags_total, fallbacks, cost = 0, 0, 0, 0, 0, 0, 0.0

    for case in cases:
        result = pipeline.process(case["text"])
        latencies.append(result.latency_ms)
        cost += result.est_cost_usd
        fallbacks += int(result.used_fallback)
        rx = result.prescription

        wrong = []
        for name, expected in case["expected"].items():
            ok = rx is not None and field_ok(name, expected, getattr(rx, name), formulary)
            per_field[name].append(ok)
            if not ok:
                wrong.append(f"{name}: expected {expected!r}, got {getattr(rx, name, None)!r}")

        got, want = result.status.value, case["expected_status"]
        routed_ok += got == want
        if want == "needs_review" and got == "auto_accepted":
            unsafe += 1
        if want == "auto_accepted" and got != "auto_accepted":
            over_review += 1

        codes = {i.code for i in result.issues}
        if flag := case.get("must_flag"):
            flags_total += 1
            flags_hit += flag in codes

        rows.append({"id": case["id"], "expected_status": want, "status": got, "issues": sorted(codes),
                     "wrong_fields": wrong, "latency_ms": result.latency_ms, "fallback": result.used_fallback})

    all_checks = [ok for oks in per_field.values() for ok in oks]
    n = len(cases)
    return {
        "provider": pipeline.primary.name,
        "cases": n,
        "field_accuracy": round(sum(all_checks) / len(all_checks), 3),
        "per_field_accuracy": {k: round(sum(v) / len(v), 3) for k, v in sorted(per_field.items())},
        "routing_accuracy": round(routed_ok / n, 3),
        "unsafe_auto_accepts": unsafe,
        "over_review": over_review,
        "must_flag_recall": round(flags_hit / flags_total, 3) if flags_total else None,
        "fallbacks_used": fallbacks,
        "latency_ms_p50": round(statistics.median(latencies), 1),
        "latency_ms_p95": round(sorted(latencies)[max(0, int(0.95 * n) - 1)], 1),
        "est_cost_usd_total": round(cost, 4),
        "rows": rows,
    }


def to_markdown(r: dict) -> str:
    lines = [
        f"## Eval: `{r['provider']}` ({r['cases']} cases)", "",
        "| Metric | Value |", "|---|---|",
        f"| Unsafe auto-accepts (must be 0) | **{r['unsafe_auto_accepts']}** |",
        f"| Routing accuracy | {r['routing_accuracy']:.0%} |",
        f"| Field accuracy | {r['field_accuracy']:.0%} |",
        f"| Must-flag recall | {r['must_flag_recall']:.0%} |",
        f"| Over-review (safe cases sent to a human) | {r['over_review']} |",
        f"| Fallbacks used | {r['fallbacks_used']} |",
        f"| Latency p50 / p95 | {r['latency_ms_p50']} ms / {r['latency_ms_p95']} ms |",
        f"| Est. cost | ${r['est_cost_usd_total']} |", "",
        "Per-field accuracy: " + ", ".join(f"{k} {v:.0%}" for k, v in r["per_field_accuracy"].items()), "",
        "| Case | Expected | Got | Wrong fields |", "|---|---|---|---|",
    ]
    for row in r["rows"]:
        mark = "" if row["status"] == row["expected_status"] else " ⚠"
        lines.append(f"| {row['id']} | {row['expected_status']} | {row['status']}{mark} | {'; '.join(row['wrong_fields']) or '-'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="rules", choices=["rules", "ollama", "claude"])
    parser.add_argument("--dataset", type=Path, default=HERE / "dataset.jsonl")
    parser.add_argument("--fail-on-unsafe", action="store_true")
    args = parser.parse_args()

    obs.configure("WARNING")
    report = run(args.provider, args.dataset)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.provider}.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (out / f"{args.provider}.md").write_text(md)
    print(md)

    if args.fail_on_unsafe and report["unsafe_auto_accepts"] > 0:
        print(f"FAIL: {report['unsafe_auto_accepts']} unsafe auto-accept(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
