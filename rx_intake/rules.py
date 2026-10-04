"""Deterministic safety rules.

The LLM only reads text. Every decision that matters (is the dose safe, is the
drug known, is anything missing) is plain code here, so it is testable,
explainable to a pharmacist, and cannot be talked out of by the input text.
"""

from __future__ import annotations

from .formulary import Drug, Formulary
from .schemas import FormularyMatch, Issue, Prescription, Severity

REQUIRED_FIELDS = (
    "patient_name",
    "drug_name",
    "strength_mg",
    "units_per_dose",
    "doses_per_day",
    "duration_days",
    "prescriber_name",
)


def check(rx: Prescription, formulary: Formulary) -> tuple[list[Issue], FormularyMatch | None]:
    issues: list[Issue] = []

    for name in REQUIRED_FIELDS:
        if getattr(rx, name) in (None, ""):
            issues.append(Issue(code=f"missing_{name}", severity=Severity.REVIEW, message=f"{name} was not found"))

    if not rx.prescriber_license:
        issues.append(Issue(code="missing_prescriber_license", severity=Severity.REVIEW, message="No prescriber registration number"))

    found = formulary.lookup(rx.drug_name)
    if rx.drug_name and found is None:
        issues.append(Issue(code="unknown_drug", severity=Severity.BLOCK, message=f"'{rx.drug_name}' is not in the formulary"))
        return issues, None
    if found is None:
        return issues, None

    drug, match = found
    issues.extend(_drug_checks(rx, drug, match))
    return issues, match


def _drug_checks(rx: Prescription, drug: Drug, match: FormularyMatch) -> list[Issue]:
    issues: list[Issue] = []

    if match.score < 1.0:
        issues.append(Issue(code="fuzzy_drug_match", severity=Severity.REVIEW,
                            message=f"'{rx.drug_name}' matched '{drug.generic_name}' with score {match.score}"))
    if drug.high_alert:
        issues.append(Issue(code="high_alert_drug", severity=Severity.REVIEW, message=f"{drug.generic_name} is a high-alert medication"))
    if drug.controlled:
        issues.append(Issue(code="controlled_drug", severity=Severity.REVIEW, message=f"{drug.generic_name} is a controlled substance"))

    if rx.route.value != "unknown" and rx.route.value not in drug.routes:
        issues.append(Issue(code="route_mismatch", severity=Severity.BLOCK,
                            message=f"Route {rx.route.value} not allowed for {drug.generic_name} ({', '.join(drug.routes)})"))

    if rx.strength_mg is not None and drug.strengths_mg and rx.strength_mg not in drug.strengths_mg:
        issues.append(Issue(code="unavailable_strength", severity=Severity.REVIEW,
                            message=f"{rx.strength_mg}mg is not a stocked strength of {drug.generic_name}"))

    daily = _daily_mg(rx)
    if daily is not None and drug.max_daily_mg is not None and daily > drug.max_daily_mg:
        issues.append(Issue(code="dose_above_max", severity=Severity.BLOCK,
                            message=f"{daily:g}mg/day exceeds max {drug.max_daily_mg:g}mg/day for {drug.generic_name}"))

    if rx.duration_days is not None:
        if rx.duration_days <= 0:
            issues.append(Issue(code="invalid_duration", severity=Severity.BLOCK, message="Duration must be positive"))
        elif rx.duration_days > drug.max_duration_days:
            issues.append(Issue(code="duration_above_max", severity=Severity.REVIEW,
                                message=f"{rx.duration_days} days exceeds usual max {drug.max_duration_days} for {drug.generic_name}"))

    if rx.patient_age is not None and rx.patient_age < 12:
        issues.append(Issue(code="paediatric_patient", severity=Severity.REVIEW, message="Paediatric dosing needs pharmacist review"))

    return issues


def _daily_mg(rx: Prescription) -> float | None:
    if None in (rx.strength_mg, rx.units_per_dose, rx.doses_per_day):
        return None
    return rx.strength_mg * rx.units_per_dose * rx.doses_per_day
