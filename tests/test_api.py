"""
tests/test_api.py
==================
Tests for the FastAPI endpoints using TestClient (no live server needed).
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_payload():
    return {
        "application": {
            "SK_ID_CURR": 999999,
            "AMT_INCOME_TOTAL": 135000,
            "AMT_CREDIT": 450000,
            "AMT_ANNUITY": 20250,
            "AMT_GOODS_PRICE": 450000,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -2000,
            "DAYS_REGISTRATION": -3000,
            "DAYS_ID_PUBLISH": -1500,
            "EXT_SOURCE_1": 0.54,
            "EXT_SOURCE_2": 0.71,
            "EXT_SOURCE_3": 0.62,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "F",
            "CNT_FAM_MEMBERS": 2.0,
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": "Y",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House / apartment"
        },
        "bureau_summary": {
            "bur_count": 3,
            "bur_active_count": 1,
            "bur_credit_sum_total": 200000,
            "bur_credit_sum_overdue": 0,
            "bur_overdue_ratio": 0.0
        },
        "previous_loans_summary": {
            "prev_count": 2,
            "prev_approval_rate": 1.0,
            "inst_pct_late": 0.05,
            "inst_days_late_mean": 1.2,
            "pos_dpd_max": 0,
            "cc_utilization_mean": 0.45
        }
    }


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_has_artifacts_loaded_field(self):
        response = client.get("/health")
        data = response.json()
        assert "artifacts_loaded" in data
        assert isinstance(data["artifacts_loaded"], bool)


# ── Model-info endpoint ───────────────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_model_info_returns_200_or_404(self):
        """Returns 200 if model_meta.json exists, 404 if training hasn't run yet."""
        response = client.get("/model-info")
        assert response.status_code in (200, 404)

    def test_model_info_200_has_metadata(self):
        response = client.get("/model-info")
        if response.status_code == 200:
            data = response.json()
            assert "model_metadata" in data


# ── Predict endpoint ──────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(self, valid_payload):
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self, valid_payload):
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert "default_probability" in data
        assert "decision" in data
        assert "risk_level" in data
        assert "threshold_used" in data

    def test_predict_probability_in_range(self, valid_payload):
        response = client.post("/predict", json=valid_payload)
        prob = response.json()["default_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_decision_is_valid(self, valid_payload):
        response = client.post("/predict", json=valid_payload)
        decision = response.json()["decision"]
        assert decision in ("APPROVE", "REJECT")

    def test_predict_risk_level_is_valid(self, valid_payload):
        response = client.post("/predict", json=valid_payload)
        risk = response.json()["risk_level"]
        assert risk in ("LOW", "MEDIUM", "HIGH")

    def test_predict_missing_application_returns_422(self):
        """Sending an empty body should return validation error."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_custom_threshold_forces_reject(self, valid_payload):
        """threshold=0.0 should always produce REJECT."""
        payload = {**valid_payload, "threshold": 0.0}
        response = client.post("/predict", json=payload)
        assert response.json()["decision"] == "REJECT"

    def test_predict_custom_threshold_forces_approve(self, valid_payload):
        """threshold=1.0 should always produce APPROVE."""
        payload = {**valid_payload, "threshold": 1.0}
        response = client.post("/predict", json=payload)
        assert response.json()["decision"] == "APPROVE"

    def test_predict_invalid_threshold_returns_422(self, valid_payload):
        """threshold outside [0, 1] should fail validation."""
        payload = {**valid_payload, "threshold": 1.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ── Batch predict endpoint ────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, valid_payload):
        response = client.post("/predict/batch", json={"applicants": [valid_payload]})
        assert response.status_code == 200

    def test_batch_predict_summary_fields(self, valid_payload):
        response = client.post("/predict/batch", json={"applicants": [valid_payload, valid_payload]})
        data = response.json()
        assert data["total"] == 2
        assert "approved" in data
        assert "rejected" in data
        assert "avg_default_probability" in data

    def test_batch_approved_plus_rejected_equals_total(self, valid_payload):
        response = client.post("/predict/batch", json={"applicants": [valid_payload, valid_payload]})
        data = response.json()
        assert data["approved"] + data["rejected"] == data["total"]

    def test_batch_empty_list_returns_422(self):
        response = client.post("/predict/batch", json={"applicants": []})
        assert response.status_code == 422
