import os

os.environ["RX_PROVIDER"] = "rules"
os.environ["RX_RETRIEVER"] = "bm25"  # no embedding model in CI
os.environ["RX_BRIEF_SUMMARY"] = "off"

from fastapi.testclient import TestClient  # noqa: E402

from rx_intake.api import app  # noqa: E402

client = TestClient(app)

CLEAN = "Patient: Rahim Uddin\nAge: 34\nRx: Amoxicillin 500mg capsule, 1 cap TID x 7 days, oral\nDr. Farhana Islam, BMDC Reg No: A-45122"
OVERDOSE = "Patient: Selina Parvin\nAge: 44\nRx: Paracetamol 500mg tablet, 3 tabs QID x 5 days, oral\nDr. Sharmin Sultana, BMDC Reg No: A-20981"


def test_clean_note_is_auto_accepted():
    r = client.post("/intakes", json={"text": CLEAN})
    assert r.status_code == 200
    assert r.json()["status"] == "auto_accepted"


def test_review_flow_with_audit_trail():
    intake = client.post("/intakes", json={"text": OVERDOSE}).json()
    assert intake["status"] == "needs_review"
    assert intake["id"] in [i["id"] for i in client.get("/reviews").json()]

    r = client.post(f"/reviews/{intake['id']}/reject", json={"reviewer": "pharmacist.rina", "reason": "Exceeds 4g/day"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"

    # A decided item cannot be decided again.
    again = client.post(f"/reviews/{intake['id']}/approve", json={"reviewer": "someone", "reason": "override"})
    assert again.status_code == 409

    trail = client.get(f"/intakes/{intake['id']}/audit").json()
    assert [e["action"] for e in trail] == ["created", "rejected"]
    assert trail[1]["actor"] == "human:pharmacist.rina"


def test_decision_requires_reviewer_and_reason():
    intake = client.post("/intakes", json={"text": OVERDOSE}).json()
    r = client.post(f"/reviews/{intake['id']}/approve", json={"reviewer": "", "reason": ""})
    assert r.status_code == 422


def test_unknown_intake_is_404():
    assert client.get("/intakes/doesnotexist").status_code == 404


def test_brief_returns_guidance_for_flagged_intake():
    intake = client.post("/intakes", json={"text": OVERDOSE}).json()
    r = client.get(f"/reviews/{intake['id']}/brief")
    assert r.status_code == 200
    brief = r.json()
    assert brief["passages"][0]["id"] == "G01"  # paracetamol daily limit
    assert brief["summary_status"] == "skipped"
    # Reading a brief never changes the decision state.
    assert client.get(f"/intakes/{intake['id']}").json()["status"] == "needs_review"


def test_brief_unknown_intake_is_404():
    assert client.get("/reviews/nope/brief").status_code == 404
