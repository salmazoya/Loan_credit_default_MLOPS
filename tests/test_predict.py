"""
tests/test_predict.py
=====================
Tests for the prediction pipeline — artifact loading, preprocessing, and inference.
"""

import pytest
import numpy as np
import pandas as pd


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_applicant():
    """Minimal valid applicant dict matching the expected predict() input format."""
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
            "NAME_HOUSING_TYPE": "House / apartment",
        },
        "bureau_summary": {
            "bur_count": 3,
            "bur_active_count": 1,
            "bur_credit_sum_total": 200000,
            "bur_credit_sum_overdue": 0,
            "bur_overdue_ratio": 0.0,
        },
        "previous_loans_summary": {
            "prev_count": 2,
            "prev_approval_rate": 1.0,
            "inst_pct_late": 0.05,
            "inst_days_late_mean": 1.2,
            "pos_dpd_max": 0,
            "cc_utilization_mean": 0.45,
        },
    }


@pytest.fixture
def minimal_applicant():
    """Applicant with only the bare minimum fields — tests robustness to missing columns."""
    return {
        "application": {
            "SK_ID_CURR": 111111,
            "AMT_INCOME_TOTAL": 90000,
            "AMT_CREDIT": 270000,
            "AMT_ANNUITY": 13500,
            "DAYS_BIRTH": -15000,
            "DAYS_EMPLOYED": -3000,
            "EXT_SOURCE_1": 0.3,
            "EXT_SOURCE_2": 0.4,
            "EXT_SOURCE_3": 0.35,
        }
    }


# ── Artifact loading ──────────────────────────────────────────────────────────

class TestLoadArtifacts:
    def test_artifacts_load_successfully(self):
        """All 4 pkl files must load without error."""
        from src.predict import load_artifacts
        model, medians, encoders, feature_columns = load_artifacts()

        assert model is not None
        assert isinstance(medians, dict)
        assert isinstance(encoders, dict)
        assert isinstance(feature_columns, list)
        assert len(feature_columns) > 0

    def test_feature_columns_are_strings(self):
        """Every feature column name must be a plain string."""
        from src.predict import load_artifacts
        _, _, _, feature_columns = load_artifacts()
        assert all(isinstance(c, str) for c in feature_columns)

    def test_model_is_list_of_estimators(self):
        """Model should be a list of fold estimators (saved by CV training)."""
        from src.predict import load_artifacts
        model, _, _, _ = load_artifacts()
        assert isinstance(model, list)
        assert len(model) > 0
        assert all(hasattr(m, "predict_proba") for m in model)


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocessSingleApplicant:
    def test_output_shape(self, sample_applicant):
        """Preprocessed DataFrame must have exactly one row and correct column count."""
        from src.predict import load_artifacts, preprocess_single_applicant
        _, medians, encoders, feature_columns = load_artifacts()

        X = preprocess_single_applicant(sample_applicant, medians, encoders, feature_columns)

        assert X.shape[0] == 1
        assert X.shape[1] == len(feature_columns)

    def test_output_columns_match_feature_columns(self, sample_applicant):
        """Column names and order must exactly match saved feature_columns."""
        from src.predict import load_artifacts, preprocess_single_applicant
        _, medians, encoders, feature_columns = load_artifacts()

        X = preprocess_single_applicant(sample_applicant, medians, encoders, feature_columns)

        assert list(X.columns) == feature_columns

    def test_no_nulls_in_output(self, sample_applicant):
        """After imputation, no NaN values should remain."""
        from src.predict import load_artifacts, preprocess_single_applicant
        _, medians, encoders, feature_columns = load_artifacts()

        X = preprocess_single_applicant(sample_applicant, medians, encoders, feature_columns)

        assert not X.isnull().any().any(), "NaN values found after preprocessing"

    def test_minimal_applicant_does_not_crash(self, minimal_applicant):
        """Prediction must succeed even when most optional fields are absent."""
        from src.predict import load_artifacts, preprocess_single_applicant
        _, medians, encoders, feature_columns = load_artifacts()

        X = preprocess_single_applicant(minimal_applicant, medians, encoders, feature_columns)

        assert X.shape[0] == 1
        assert X.shape[1] == len(feature_columns)


# ── Prediction output ─────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_all_keys(self, sample_applicant):
        """Result dict must contain all expected keys."""
        from src.predict import predict
        result = predict(sample_applicant)

        expected_keys = {
            "applicant_id", "default_probability",
            "decision", "risk_level", "threshold_used"
        }
        assert expected_keys.issubset(result.keys())

    def test_probability_in_range(self, sample_applicant):
        """Default probability must be between 0 and 1."""
        from src.predict import predict
        result = predict(sample_applicant)

        assert 0.0 <= result["default_probability"] <= 1.0

    def test_decision_is_valid(self, sample_applicant):
        """Decision must be APPROVE or REJECT."""
        from src.predict import predict
        result = predict(sample_applicant)

        assert result["decision"] in ("APPROVE", "REJECT")

    def test_risk_level_is_valid(self, sample_applicant):
        """Risk level must be LOW, MEDIUM, or HIGH."""
        from src.predict import predict
        result = predict(sample_applicant)

        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_applicant_id_preserved(self, sample_applicant):
        """SK_ID_CURR from the input should appear in the result."""
        from src.predict import predict
        result = predict(sample_applicant)

        assert result["applicant_id"] == 999999

    def test_custom_threshold_applied(self, sample_applicant):
        """Passing threshold=0.0 should force REJECT; threshold=1.0 should force APPROVE."""
        from src.predict import predict

        result_reject = predict(sample_applicant, threshold=0.0)
        assert result_reject["decision"] == "REJECT"

        result_approve = predict(sample_applicant, threshold=1.0)
        assert result_approve["decision"] == "APPROVE"

    def test_high_risk_applicant(self):
        """An applicant with very weak external scores should get a higher probability."""
        from src.predict import predict

        risky = {
            "application": {
                "SK_ID_CURR": 777777,
                "AMT_INCOME_TOTAL": 27000,
                "AMT_CREDIT": 900000,
                "AMT_ANNUITY": 45000,
                "DAYS_BIRTH": -8000,
                "DAYS_EMPLOYED": 365243,   # unemployed anomaly
                "EXT_SOURCE_1": 0.05,
                "EXT_SOURCE_2": 0.08,
                "EXT_SOURCE_3": 0.06,
            }
        }
        safe = {
            "application": {
                "SK_ID_CURR": 888888,
                "AMT_INCOME_TOTAL": 200000,
                "AMT_CREDIT": 200000,
                "AMT_ANNUITY": 10000,
                "DAYS_BIRTH": -18000,
                "DAYS_EMPLOYED": -5000,
                "EXT_SOURCE_1": 0.85,
                "EXT_SOURCE_2": 0.90,
                "EXT_SOURCE_3": 0.88,
            }
        }

        risky_prob = predict(risky)["default_probability"]
        safe_prob  = predict(safe)["default_probability"]

        assert risky_prob > safe_prob, (
            f"Expected risky ({risky_prob:.4f}) > safe ({safe_prob:.4f})"
        )
