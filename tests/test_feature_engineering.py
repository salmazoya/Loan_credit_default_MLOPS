"""
tests/test_feature_engineering.py
===================================
Tests for engineer_application_features() in data_preprocessing.py.
These run fast with no pkl files needed — pure DataFrame logic.
"""

import pytest
import numpy as np
import pandas as pd

from src.data_preprocessing import engineer_application_features


@pytest.fixture
def base_row():
    """A minimal but complete application row."""
    return pd.DataFrame([{
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
        "CNT_FAM_MEMBERS": 2.0,
        "DEF_30_CNT_SOCIAL_CIRCLE": 1,
        "OBS_30_CNT_SOCIAL_CIRCLE": 10,
    }])


class TestEngineeredFeatures:
    def test_age_years_computed(self, base_row):
        df = engineer_application_features(base_row)
        expected_age = int(12000 / 365)
        assert df["AGE_YEARS"].iloc[0] == expected_age

    def test_employed_years_computed(self, base_row):
        df = engineer_application_features(base_row)
        assert abs(df["EMPLOYED_YEARS"].iloc[0] - (2000 / 365)) < 0.01

    def test_employed_anomaly_flagged(self):
        """DAYS_EMPLOYED = 365243 is the 'unemployed' sentinel — must be flagged."""
        row = pd.DataFrame([{
            "AMT_INCOME_TOTAL": 50000, "AMT_CREDIT": 100000, "AMT_ANNUITY": 5000,
            "DAYS_BIRTH": -10000, "DAYS_EMPLOYED": 365243,
            "EXT_SOURCE_1": 0.4, "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": 0.45,
        }])
        df = engineer_application_features(row)
        assert df["EMPLOYED_ANOMALY"].iloc[0] == 1
        assert pd.isna(df["EMPLOYED_YEARS"].iloc[0])

    def test_credit_term_computed(self, base_row):
        df = engineer_application_features(base_row)
        expected = 20250 / 450000
        assert abs(df["CREDIT_TERM"].iloc[0] - expected) < 1e-6

    def test_ext_source_mean(self, base_row):
        df = engineer_application_features(base_row)
        expected = np.mean([0.54, 0.71, 0.62])
        assert abs(df["EXT_SOURCE_MEAN"].iloc[0] - expected) < 1e-6

    def test_ext_source_prod(self, base_row):
        df = engineer_application_features(base_row)
        expected = 0.54 * 0.71 * 0.62
        assert abs(df["EXT_SOURCE_PROD"].iloc[0] - expected) < 1e-6

    def test_social_circle_default_rate(self, base_row):
        df = engineer_application_features(base_row)
        expected = 1 / 10
        assert abs(df["SOCIAL_CIRCLE_DEFAULT_RATE"].iloc[0] - expected) < 1e-6

    def test_social_circle_zero_obs_gives_nan(self):
        """Zero observations should result in NaN, not division error."""
        row = pd.DataFrame([{
            "AMT_INCOME_TOTAL": 50000, "AMT_CREDIT": 100000, "AMT_ANNUITY": 5000,
            "DAYS_BIRTH": -10000, "DAYS_EMPLOYED": -2000,
            "EXT_SOURCE_1": 0.4, "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": 0.45,
            "DEF_30_CNT_SOCIAL_CIRCLE": 0,
            "OBS_30_CNT_SOCIAL_CIRCLE": 0,
        }])
        df = engineer_application_features(row)
        assert pd.isna(df["SOCIAL_CIRCLE_DEFAULT_RATE"].iloc[0])

    def test_missing_optional_columns_do_not_crash(self):
        """Function must not crash when optional columns are absent."""
        row = pd.DataFrame([{
            "AMT_INCOME_TOTAL": 50000,
            "AMT_CREDIT": 100000,
            "AMT_ANNUITY": 5000,
            "DAYS_BIRTH": -10000,
            "DAYS_EMPLOYED": -2000,
            "EXT_SOURCE_1": 0.4,
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": 0.45,
            # no DAYS_REGISTRATION, no AMT_GOODS_PRICE, no social circle cols
        }])
        df = engineer_application_features(row)
        assert df is not None
        assert "EXT_SOURCE_MEAN" in df.columns

    def test_income_per_person_with_zero_family(self):
        """Zero CNT_FAM_MEMBERS should give NaN, not division by zero."""
        row = pd.DataFrame([{
            "AMT_INCOME_TOTAL": 50000, "AMT_CREDIT": 100000, "AMT_ANNUITY": 5000,
            "DAYS_BIRTH": -10000, "DAYS_EMPLOYED": -2000,
            "EXT_SOURCE_1": 0.4, "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": 0.45,
            "CNT_FAM_MEMBERS": 0,
        }])
        df = engineer_application_features(row)
        assert pd.isna(df["INCOME_PER_PERSON"].iloc[0])

    def test_output_has_more_columns_than_input(self, base_row):
        """Engineered output must have strictly more columns than the raw input."""
        df = engineer_application_features(base_row)
        assert df.shape[1] > base_row.shape[1]
