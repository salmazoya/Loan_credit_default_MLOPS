"""
predict.py — Single Applicant Inference
========================================
Loads all .pkl files from artifacts/models/ and predicts
the default probability for one loan applicant.

Called by:
  - FastAPI /predict endpoint  (real-time, one applicant at a time)
  - CLI (for quick manual testing)

Expected input format (dict):
  {
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
        "CNT_FAM_MEMBERS": 2,
        ...
    },
    "bureau_summary": {           # optional — pre-aggregated bureau features
        "bur_count": 3,
        "bur_active_count": 1,
        "bur_credit_sum_total": 200000,
        "bur_credit_sum_overdue": 0,
        "bur_overdue_ratio": 0.0
    },
    "previous_loans_summary": {   # optional — pre-aggregated loan history
        "prev_count": 2,
        "prev_approval_rate": 1.0,
        "inst_pct_late": 0.05,
        "inst_days_late_mean": 1.2,
        "pos_dpd_max": 0,
        "cc_utilization_mean": 0.45
    }
  }
"""

import os
import re
import joblib
import numpy as np
import pandas as pd

from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_preprocessing import engineer_application_features, clean_column_names
from config.paths_config import (
    MODELS_DIR,
    MODEL_FILE_PATH,
    MEDIANS_FILE_PATH,
    ENCODERS_FILE_PATH,
    FEATURE_COLUMNS_PATH
)

logger = get_logger(__name__)

# Default classification threshold (set from modeling notebook output)
DEFAULT_THRESHOLD = 0.66


# ─────────────────────────────────────────────────────────────────────────────
# Load Artifacts
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts() -> tuple:
    """
    Load all 4 .pkl files from artifacts/models/:
      - lgbm_model.pkl
      - medians.pkl
      - encoders.pkl
      - feature_columns.pkl
    """
    try:
        required = {
            "model"           : MODEL_FILE_PATH,
            "medians"         : MEDIANS_FILE_PATH,
            "encoders"        : ENCODERS_FILE_PATH,
            "feature_columns" : FEATURE_COLUMNS_PATH,
        }

        for name, path in required.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing artifact '{name}': {path}\n"
                    f"Run data_preprocessing.py and train.py first."
                )

        model           = joblib.load(MODEL_FILE_PATH)
        medians         = joblib.load(MEDIANS_FILE_PATH)
        encoders        = joblib.load(ENCODERS_FILE_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

        logger.info(f"All artifacts loaded from '{MODELS_DIR}/'")
        logger.info(f"  Model           : {MODEL_FILE_PATH}")
        logger.info(f"  Medians         : {MEDIANS_FILE_PATH}")
        logger.info(f"  Encoders        : {ENCODERS_FILE_PATH}")
        logger.info(f"  Feature columns : {FEATURE_COLUMNS_PATH} ({len(feature_columns)} features)")

        return model, medians, encoders, feature_columns

    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        raise CustomException("Failed to load model artifacts", e)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocess Single Applicant
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_single_applicant(
    applicant_dict: dict,
    medians: dict,
    encoders: dict,
    feature_columns: list
) -> pd.DataFrame:
    """
    Transform a raw applicant dict into a single-row DataFrame
    that matches exactly what the model was trained on.

    Steps:
      1. Flatten application + bureau_summary + previous_loans_summary
      2. Engineer application features
      3. Impute missing values using saved medians
      4. Encode categoricals using saved encoders
      5. Clean column names
      6. Align to saved feature_columns (add missing as 0, drop extras)
    """
    try:
        logger.info("Preprocessing single applicant...")

        # Step 1 — Flatten all sections into one row
        flat = {}
        flat.update(applicant_dict.get('application', {}))
        flat.update(applicant_dict.get('bureau_summary', {}))
        flat.update(applicant_dict.get('previous_loans_summary', {}))
        df = pd.DataFrame([flat])

        # Step 2 — Engineer application features (same logic as training)
        df = engineer_application_features(df)

        # Step 3 — Impute using saved medians
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(medians.get(col, 0))

        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')

        # Step 4 — Encode categoricals using saved encoders
        le_cols  = encoders.get('__le_cols__', [])
        hc_cols  = encoders.get('__hc_cols__', [])
        ohe_cols = encoders.get('__ohe_cols__', [])

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()

        for col in le_cols + hc_cols:
            if col in df.columns:
                known_classes = encoders[col]['classes']
                le.classes_   = np.array(known_classes)
                df[col] = df[col].astype(str).map(
                    lambda x, c=col: x if x in encoders[c]['classes'] else encoders[c]['classes'][0]
                )
                df[col] = le.transform(df[col])

        if ohe_cols:
            present_ohe = [c for c in ohe_cols if c in df.columns]
            df = pd.get_dummies(df, columns=present_ohe, dummy_na=False)

        # Step 5 — Clean column names
        df = clean_column_names(df)

        # Step 6 — Align to training feature columns
        # Add all missing columns at once (avoids DataFrame fragmentation warning)
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            df = pd.concat(
                [df, pd.DataFrame(0, index=df.index, columns=missing_cols)],
                axis=1
            )
        df = df[feature_columns]

        logger.info(f"Single applicant preprocessed. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error preprocessing single applicant: {e}")
        raise CustomException("Failed to preprocess single applicant", e)


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict(applicant_dict: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    End-to-end prediction for a single loan applicant.

    Args:
        applicant_dict : raw applicant data (see module docstring for format)
        threshold      : classification cutoff (default 0.66 from model tuning)

    Returns:
        {
            "applicant_id"        : int,
            "default_probability" : float,   # 0.0 – 1.0
            "decision"            : str,     # "APPROVE" or "REJECT"
            "risk_level"          : str,     # "LOW" / "MEDIUM" / "HIGH"
            "threshold_used"      : float
        }
    """
    try:
        logger.info("Starting prediction pipeline...")

        # Load all .pkl files
        model, medians, encoders, feature_columns = load_artifacts()

        # Preprocess
        X = preprocess_single_applicant(
            applicant_dict, medians, encoders, feature_columns
        )

        # Predict probability
        # model may be a list of fold models (saved by train.py CV) — average them
        if isinstance(model, list):
            prob = float(np.mean([m.predict_proba(X)[0][1] for m in model]))
        else:
            prob = float(model.predict_proba(X)[0][1])

        # Decision
        decision = "REJECT" if prob >= threshold else "APPROVE"

        # Risk level
        if prob < 0.20:
            risk_level = "LOW"
        elif prob < 0.50:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        applicant_id = applicant_dict.get('application', {}).get('SK_ID_CURR', None)

        result = {
            "applicant_id"        : applicant_id,
            "default_probability" : round(prob, 4),
            "decision"            : decision,
            "risk_level"          : risk_level,
            "threshold_used"      : threshold
        }

        logger.info(
            f"Prediction complete — ID: {applicant_id} | "
            f"Prob: {prob:.4f} | Decision: {decision} | Risk: {risk_level}"
        )
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise CustomException("Prediction failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — for manual testing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Example applicant — replace with real values
    sample_applicant = {
        "application": {
            "SK_ID_CURR"        : 999999,
            "AMT_INCOME_TOTAL"  : 135000,
            "AMT_CREDIT"        : 450000,
            "AMT_ANNUITY"       : 20250,
            "AMT_GOODS_PRICE"   : 450000,
            "DAYS_BIRTH"        : -12000,
            "DAYS_EMPLOYED"     : -2000,
            "DAYS_REGISTRATION" : -3000,
            "DAYS_ID_PUBLISH"   : -1500,
            "EXT_SOURCE_1"      : 0.54,
            "EXT_SOURCE_2"      : 0.71,
            "EXT_SOURCE_3"      : 0.62,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER"       : "F",
            "CNT_FAM_MEMBERS"   : 2.0,
            "FLAG_OWN_CAR"      : "N",
            "FLAG_OWN_REALTY"   : "Y",
            "NAME_INCOME_TYPE"  : "Working",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE" : "House / apartment",
        },
        "bureau_summary": {
            "bur_count"            : 3,
            "bur_active_count"     : 1,
            "bur_credit_sum_total" : 200000,
            "bur_credit_sum_overdue": 0,
            "bur_overdue_ratio"    : 0.0,
        },
        "previous_loans_summary": {
            "prev_count"         : 2,
            "prev_approval_rate" : 1.0,
            "inst_pct_late"      : 0.05,
            "inst_days_late_mean": 1.2,
            "pos_dpd_max"        : 0,
            "cc_utilization_mean": 0.45,
        }
    }

    result = predict(sample_applicant)

    print("\n" + "=" * 45)
    print("       LOAN DEFAULT PREDICTION RESULT")
    print("=" * 45)
    for k, v in result.items():
        print(f"  {k:<25}: {v}")
    print("=" * 45)
