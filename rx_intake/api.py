"""HTTP API: submit notes, work the review queue, read the audit trail."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import observability as obs
from .config import build_pipeline
from .schemas import IntakeResult
from .store import ReviewError, Store

obs.configure(os.getenv("RX_LOG_LEVEL", "INFO"))

app = FastAPI(title="rx-intake-agent", version="0.1.0")
app.state.pipeline = build_pipeline()
app.state.store = Store(os.getenv("RX_DB_PATH", ":memory:"))


class IntakeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=20_000)


class Decision(BaseModel):
    reviewer: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "extractor": app.state.pipeline.primary.name}


@app.post("/intakes", response_model=IntakeResult)
def create_intake(req: IntakeRequest) -> IntakeResult:
    result = app.state.pipeline.process(req.text)
    app.state.store.save(result)
    return result


@app.get("/intakes/{intake_id}", response_model=IntakeResult)
def get_intake(intake_id: str) -> IntakeResult:
    result = app.state.store.get(intake_id)
    if result is None:
        raise HTTPException(404, "not found")
    return result


@app.get("/reviews", response_model=list[IntakeResult])
def review_queue() -> list[IntakeResult]:
    return app.state.store.pending_reviews()


@app.post("/reviews/{intake_id}/approve", response_model=IntakeResult)
def approve(intake_id: str, decision: Decision) -> IntakeResult:
    return _decide(intake_id, True, decision)


@app.post("/reviews/{intake_id}/reject", response_model=IntakeResult)
def reject(intake_id: str, decision: Decision) -> IntakeResult:
    return _decide(intake_id, False, decision)


@app.get("/intakes/{intake_id}/audit")
def audit(intake_id: str) -> list[dict]:
    return app.state.store.audit_trail(intake_id)


def _decide(intake_id: str, approve_: bool, decision: Decision) -> IntakeResult:
    try:
        return app.state.store.decide(intake_id, approve_, decision.reviewer, decision.reason)
    except ReviewError as exc:
        code = 404 if str(exc) == "not found" else 409
        raise HTTPException(code, str(exc)) from exc
