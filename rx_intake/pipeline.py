"""The intake workflow: screen -> extract -> ground -> rules -> route.

This is a fixed workflow, not an open-ended agent. The steps are known in
advance, so letting a model choose them would add cost and failure modes
without adding value. The model does the one thing code can't: read messy text.
"""

from __future__ import annotations

import uuid

from . import guard, observability as obs, rules
from .extractors import Extraction, ExtractionError, Extractor
from .formulary import Formulary
from .schemas import IntakeResult, Issue, Severity, Status


class IntakePipeline:
    def __init__(self, primary: Extractor, fallback: Extractor | None, formulary: Formulary):
        self.primary = primary
        self.fallback = fallback
        self.formulary = formulary

    def process(self, text: str, trace_id: str | None = None) -> IntakeResult:
        trace_id = trace_id or obs.new_trace_id()
        obs.event("intake.received", trace_id, doc_hash=obs.fingerprint(text), chars=len(text))

        issues: list[Issue] = guard.screen(text)
        if issues:
            obs.event("guard.flagged", trace_id, codes=[i.code for i in issues])

        with obs.timed() as t:
            extraction, extractor_name, used_fallback, error = self._extract(text, trace_id)

        if extraction is None:
            issues.append(Issue(code="extraction_failed", severity=Severity.BLOCK, message=error or "extraction failed"))
            result = self._result(trace_id, Status.NEEDS_REVIEW, None, None, issues, extractor_name, used_fallback, t["ms"])
            obs.event("intake.routed", trace_id, status=result.status.value, reason="extraction_failed")
            return result

        rule_issues, match = rules.check(extraction.prescription, self.formulary)
        issues.extend(rule_issues)
        if used_fallback:
            issues.append(Issue(code="fallback_extractor_used", severity=Severity.REVIEW,
                                message="Primary model unavailable; parsed by rules baseline"))

        status = _route(issues)
        result = self._result(trace_id, status, extraction, match, issues, extractor_name, used_fallback, t["ms"])
        obs.event("intake.routed", trace_id, status=status.value, extractor=extractor_name, fallback=used_fallback,
                  latency_ms=t["ms"], input_tokens=extraction.input_tokens, output_tokens=extraction.output_tokens,
                  cost_usd=extraction.cost_usd, issues=[i.code for i in issues])
        return result

    def _extract(self, text: str, trace_id: str) -> tuple[Extraction | None, str, bool, str | None]:
        try:
            return self.primary.extract(text), self.primary.name, False, None
        except ExtractionError as exc:
            obs.event("extractor.failed", trace_id, extractor=self.primary.name, error=str(exc)[:300])
            if self.fallback is None:
                return None, self.primary.name, False, str(exc)
        try:
            return self.fallback.extract(text), self.fallback.name, True, None
        except ExtractionError as exc:
            obs.event("extractor.failed", trace_id, extractor=self.fallback.name, error=str(exc)[:300])
            return None, self.fallback.name, True, str(exc)

    @staticmethod
    def _result(trace_id, status, extraction, match, issues, extractor_name, used_fallback, ms) -> IntakeResult:
        return IntakeResult(
            id=uuid.uuid4().hex[:12],
            trace_id=trace_id,
            status=status,
            prescription=extraction.prescription if extraction else None,
            formulary_match=match,
            issues=issues,
            extractor=extractor_name,
            used_fallback=used_fallback,
            latency_ms=ms,
            input_tokens=extraction.input_tokens if extraction else 0,
            output_tokens=extraction.output_tokens if extraction else 0,
            est_cost_usd=extraction.cost_usd if extraction else 0.0,
        )


def _route(issues: list[Issue]) -> Status:
    """Anything a human should see goes to review; nothing is auto-rejected.

    BLOCK issues still go to a human (with the block reason shown) rather than
    being silently rejected: a pharmacist can call the prescriber and fix it.
    """
    if any(i.severity in (Severity.BLOCK, Severity.REVIEW) for i in issues):
        return Status.NEEDS_REVIEW
    return Status.AUTO_ACCEPTED
