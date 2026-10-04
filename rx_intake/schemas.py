"""Data contracts shared by the extractor, rules engine, review queue and API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Route(str, Enum):
    ORAL = "oral"
    TOPICAL = "topical"
    INHALED = "inhaled"
    INJECTION = "injection"
    UNKNOWN = "unknown"


class Prescription(BaseModel):
    """Fields extracted from free-text prescription notes.

    Every field is optional: the extractor must return null instead of guessing,
    and the rules engine decides what a missing field means.
    """

    patient_name: str | None = Field(None, description="Patient full name as written")
    patient_age: int | None = Field(None, description="Patient age in years")
    drug_name: str | None = Field(None, description="Medication name exactly as written")
    strength_mg: float | None = Field(None, description="Strength of one unit in mg, e.g. 500 for '500mg tab'")
    units_per_dose: float | None = Field(None, description="Tablets/capsules/puffs per dose, e.g. 2 for '2 tabs'")
    doses_per_day: float | None = Field(None, description="Times per day: OD=1, BID=2, TID=3, QID=4, q8h=3, q6h=4")
    duration_days: int | None = Field(None, description="Course length in days")
    route: Route = Field(Route.UNKNOWN, description="Route of administration")
    prescriber_name: str | None = Field(None, description="Prescribing doctor's name")
    prescriber_license: str | None = Field(None, description="Prescriber registration/license number")


class Status(str, Enum):
    AUTO_ACCEPTED = "auto_accepted"  # passed every check, no human needed
    ACCEPTED = "accepted"  # approved by a human reviewer
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class Severity(str, Enum):
    INFO = "info"
    REVIEW = "review"  # a human must look before this is accepted
    BLOCK = "block"  # never accept automatically; a human must override


class Issue(BaseModel):
    code: str
    severity: Severity
    message: str


class FormularyMatch(BaseModel):
    generic_name: str
    matched_text: str
    score: float


class IntakeResult(BaseModel):
    id: str
    trace_id: str
    status: Status
    prescription: Prescription | None
    formulary_match: FormularyMatch | None
    issues: list[Issue]
    extractor: str
    used_fallback: bool
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0
