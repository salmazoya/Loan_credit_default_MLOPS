"""
monitor.py — Evidently AI Drift & Performance Monitoring
=========================================================
Compares a reference dataset (training data) against a current dataset
(new incoming data / test set) and generates three reports:

  1. Data Drift Report      — are feature distributions shifting?
  2. Target Drift Report    — is the default rate (TARGET) shifting?
  3. Model Performance Report — is AUC / F1 degrading? (requires labels)

Reports are saved as HTML to artifacts/monitoring/ and optionally
logged as MLflow artifacts.

Usage:
  # Standalone — compare train vs test
  python -m src.monitor

  # From train.py after each retraining run
  from src.monitor import run_monitoring
  run_monitoring(reference_df, current_df, mlflow_run_id)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import mlflow

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    ClassificationPreset,
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import (
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
    MODEL_FILE_PATH,
    FEATURE_COLUMNS_PATH,
    MEDIANS_FILE_PATH,
    ENCODERS_FILE_PATH,
    CONFIG_PATH,
)
from utils.common_fnctions import read_yaml

logger = get_logger(__name__)

MONITORING_DIR = "artifacts/monitoring"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_and_features():
    """Load all pkl artifacts needed for inference."""
    model           = joblib.load(MODEL_FILE_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    encoders        = joblib.load(ENCODERS_FILE_PATH)
    medians         = joblib.load(MEDIANS_FILE_PATH)
    return model, feature_columns, encoders, medians


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    Average predict_proba across all fold models.
    model is either a list (CV ensemble) or a single estimator.
    """
    if isinstance(model, list):
        return np.mean([m.predict_proba(X)[:, 1] for m in model], axis=0)
    return model.predict_proba(X)[:, 1]


def _encode_categoricals(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply saved label encoders to categorical columns — same logic as predict.py."""
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    le = LabelEncoder()

    le_cols  = encoders.get('__le_cols__', [])
    hc_cols  = encoders.get('__hc_cols__', [])
    ohe_cols = encoders.get('__ohe_cols__', [])

    for col in le_cols + hc_cols:
        if col in df.columns:
            known_classes  = encoders[col]['classes']
            le.classes_    = np.array(known_classes)
            df[col] = df[col].astype(str).map(
                lambda x, c=col: x if x in encoders[c]['classes'] else encoders[c]['classes'][0]
            )
            df[col] = le.transform(df[col])

    if ohe_cols:
        present_ohe = [c for c in ohe_cols if c in df.columns]
        df = pd.get_dummies(df, columns=present_ohe, dummy_na=False)

    return df


def _prepare_dataset(
    df: pd.DataFrame,
    feature_columns: list,
    model,
    encoders: dict,
    medians: dict,
    threshold: float,
    label_col: str = "TARGET",
) -> pd.DataFrame:
    """
    Build a dataset that Evidently can consume:
      - encode categoricals using saved encoders
      - impute missing values using saved medians
      - align to model feature columns
      - add 'prediction', 'prediction_label', 'target' columns
    """
    X = df.copy()

    # Impute numeric missing values
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(medians.get(col, 0))

    # Fill categorical missing values
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = X[col].fillna('Unknown')

    # Encode categoricals using saved encoders
    X = _encode_categoricals(X, encoders)

    # Align to training feature columns
    available = [c for c in feature_columns if c in X.columns]
    X = X[available].copy()
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        X = pd.concat(
            [X, pd.DataFrame(0, index=X.index, columns=missing)],
            axis=1,
        )
    X = X[feature_columns]

    # Generate predictions
    probs  = _predict_proba(model, X)
    labels = (probs >= threshold).astype(int)

    out = X.copy()
    out["prediction"]       = probs
    out["prediction_label"] = labels

    if label_col in df.columns:
        out["target"] = df[label_col].values

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Individual Reports
# ─────────────────────────────────────────────────────────────────────────────

def run_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    save_path: str,
    n_features_to_check: int = 20,
) -> dict:
    """
    Data Drift Report — checks whether feature distributions have shifted.

    Uses the top N most important features so the report stays readable.
    Returns a summary dict with drift share and whether drift was detected.
    """
    try:
        logger.info("Running Data Drift Report...")

        # Use top N features by column order (importance-ordered if from feature_columns)
        cols_to_check = [
            c for c in reference.columns
            if c not in ("prediction", "prediction_label", "target")
        ][:n_features_to_check]

        ref = reference[cols_to_check].copy()
        cur = current[cols_to_check].copy()

        report = Report(metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            DataDriftPreset(),
        ])
        report.run(reference_data=ref, current_data=cur)
        report.save_html(save_path)

        # Extract drift summary by walking the snapshot metrics
        drift_flag  = False
        drift_share = 0.0
        try:
            snap = report.as_dict()
            for metric in snap.get("metrics", []):
                r = metric.get("result", {})
                if "share_of_drifted_columns" in r:
                    drift_share = r["share_of_drifted_columns"]
                    drift_flag  = r.get("dataset_drift", drift_share > 0.5)
                    break
        except Exception:
            pass  # summary extraction is best-effort; report HTML is still saved

        summary = {
            "drift_detected"  : drift_flag,
            "drift_share"     : round(drift_share, 4),
            "features_checked": len(cols_to_check),
            "report_path"     : save_path,
        }

        logger.info(
            f"Data Drift — detected: {drift_flag} | "
            f"share: {drift_share:.2%} | report: {save_path}"
        )
        return summary

    except Exception as e:
        logger.error(f"Data drift report failed: {e}")
        raise CustomException("Data drift report failed", e)


def run_target_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    save_path: str,
) -> dict:
    """
    Target Drift Report — checks if the default rate (TARGET distribution) is shifting.
    """
    try:
        logger.info("Running Target Drift Report...")

        if "target" not in reference.columns or "target" not in current.columns:
            logger.warning("No 'target' column found — skipping target drift report.")
            return {"skipped": True, "reason": "no target column"}

        ref = reference[["target", "prediction"]].copy()
        cur = current[["target", "prediction"]].copy()

        # Use DataDriftPreset scoped to the target + prediction columns
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        report.save_html(save_path)

        ref_default_rate = reference["target"].mean()
        cur_default_rate = current["target"].mean()
        shift            = cur_default_rate - ref_default_rate

        summary = {
            "reference_default_rate": round(float(ref_default_rate), 4),
            "current_default_rate"  : round(float(cur_default_rate), 4),
            "default_rate_shift"    : round(float(shift), 4),
            "report_path"           : save_path,
        }

        logger.info(
            f"Target Drift — ref rate: {ref_default_rate:.2%} | "
            f"current rate: {cur_default_rate:.2%} | shift: {shift:+.2%}"
        )
        return summary

    except Exception as e:
        logger.error(f"Target drift report failed: {e}")
        raise CustomException("Target drift report failed", e)


def run_model_performance_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    save_path: str,
) -> dict:
    """
    Model Performance Report — compares AUC, precision, recall, F1
    between reference and current data.
    Requires 'target' column in both datasets.
    """
    try:
        logger.info("Running Model Performance Report...")

        if "target" not in reference.columns or "target" not in current.columns:
            logger.warning("No 'target' column — skipping performance report.")
            return {"skipped": True, "reason": "no target column"}

        ref = reference[["target", "prediction", "prediction_label"]].copy()
        cur = current[["target", "prediction", "prediction_label"]].copy()

        # Evidently expects columns named 'target' and 'prediction'
        ref = ref.rename(columns={"prediction_label": "prediction_label"})
        cur = cur.rename(columns={"prediction_label": "prediction_label"})

        report = Report(metrics=[ClassificationPreset()])
        report.run(
            reference_data=ref.rename(columns={"prediction": "score", "prediction_label": "prediction"}),
            current_data=cur.rename(columns={"prediction": "score", "prediction_label": "prediction"}),
        )
        report.save_html(save_path)

        summary = {"report_path": save_path}
        logger.info(f"Model Performance report saved → {save_path}")
        return summary

    except Exception as e:
        logger.error(f"Model performance report failed: {e}")
        # Non-fatal — return empty summary instead of crashing
        logger.warning("Continuing without model performance report.")
        return {"skipped": True, "reason": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_monitoring(
    reference_df: pd.DataFrame = None,
    current_df: pd.DataFrame   = None,
    mlflow_run_id: str         = None,
    log_to_mlflow: bool        = True,
) -> dict:
    """
    Run all three Evidently reports and optionally log them to MLflow.

    Args:
        reference_df   : Training data (loaded from TRAIN_FILE_PATH if None)
        current_df     : New/test data  (loaded from TEST_FILE_PATH if None)
        mlflow_run_id  : Active MLflow run to attach reports to
        log_to_mlflow  : Whether to upload HTML reports as MLflow artifacts

    Returns:
        Summary dict with drift flags and report paths.
    """
    try:
        logger.info("=" * 55)
        logger.info("Starting Evidently Drift Monitoring")
        logger.info("=" * 55)

        config    = read_yaml(CONFIG_PATH)
        threshold = config["model"]["threshold"]

        # ── Load data ─────────────────────────────────────────────────────────
        if reference_df is None:
            logger.info(f"Loading reference data from {TRAIN_FILE_PATH}")
            reference_df = pd.read_csv(TRAIN_FILE_PATH)

        if current_df is None:
            logger.info(f"Loading current data from {TEST_FILE_PATH}")
            current_df = pd.read_csv(TEST_FILE_PATH)

        logger.info(f"Reference : {reference_df.shape[0]:,} rows")
        logger.info(f"Current   : {current_df.shape[0]:,} rows")

        # ── Load model ────────────────────────────────────────────────────────
        model, feature_columns, encoders, medians = _load_model_and_features()

        # ── Sample for speed (Evidently can be slow on 200k+ rows) ───────────
        ref_sample = reference_df.sample(
            n=min(5000, len(reference_df)), random_state=42
        )
        cur_sample = current_df.sample(
            n=min(5000, len(current_df)), random_state=42
        )

        # ── Prepare datasets with predictions ─────────────────────────────────
        logger.info("Generating predictions for monitoring datasets...")
        ref_prepared = _prepare_dataset(ref_sample, feature_columns, model, encoders, medians, threshold)
        cur_prepared = _prepare_dataset(cur_sample, feature_columns, model, encoders, medians, threshold)

        # ── Output folder ─────────────────────────────────────────────────────
        os.makedirs(MONITORING_DIR, exist_ok=True)

        drift_path   = os.path.join(MONITORING_DIR, "data_drift_report.html")
        target_path  = os.path.join(MONITORING_DIR, "target_drift_report.html")
        perf_path    = os.path.join(MONITORING_DIR, "model_performance_report.html")

        # ── Run reports ───────────────────────────────────────────────────────
        drift_summary  = run_data_drift_report(ref_prepared, cur_prepared, drift_path)
        target_summary = run_target_drift_report(ref_prepared, cur_prepared, target_path)
        perf_summary   = run_model_performance_report(ref_prepared, cur_prepared, perf_path)

        # ── Log to MLflow ─────────────────────────────────────────────────────
        if log_to_mlflow:
            try:
                tracking_uri = config["mlflow"]["tracking_uri"]
                mlflow.set_tracking_uri(tracking_uri)

                # Use active run if inside one, otherwise start a new one
                active_run = mlflow.active_run()
                ctx = (
                    mlflow.start_run(run_id=mlflow_run_id)
                    if (mlflow_run_id and not active_run)
                    else mlflow.start_run(run_name="drift_monitoring")
                    if not active_run
                    else None
                )

                def _log():
                    mlflow.log_metric("drift_detected",    int(drift_summary.get("drift_detected", 0)))
                    mlflow.log_metric("drift_share",       drift_summary.get("drift_share", 0))
                    mlflow.log_metric("ref_default_rate",  target_summary.get("reference_default_rate", 0))
                    mlflow.log_metric("cur_default_rate",  target_summary.get("current_default_rate", 0))
                    mlflow.log_metric("default_rate_shift",target_summary.get("default_rate_shift", 0))

                    for path in [drift_path, target_path, perf_path]:
                        if os.path.exists(path):
                            mlflow.log_artifact(path, artifact_path="monitoring")

                    mlflow.set_tag("monitoring_run", "true")
                    logger.info("Monitoring metrics and reports logged to MLflow.")

                if ctx:
                    with ctx:
                        _log()
                else:
                    _log()

            except Exception as e:
                logger.warning(f"MLflow logging skipped: {e}")

        # ── Final summary ─────────────────────────────────────────────────────
        summary = {
            "data_drift"  : drift_summary,
            "target_drift": target_summary,
            "performance" : perf_summary,
        }

        logger.info("=" * 55)
        logger.info("Monitoring Complete")
        logger.info(f"  Drift detected   : {drift_summary.get('drift_detected')}")
        logger.info(f"  Drift share      : {drift_summary.get('drift_share', 0):.2%}")
        logger.info(f"  Default rate shift: {target_summary.get('default_rate_shift', 0):+.2%}")
        logger.info(f"  Reports saved to : {MONITORING_DIR}/")
        logger.info("=" * 55)

        return summary

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise CustomException("Monitoring failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    summary = run_monitoring()

    print("\n" + "=" * 50)
    print("         MONITORING SUMMARY")
    print("=" * 50)

    drift = summary["data_drift"]
    print(f"  Data Drift Detected : {drift.get('drift_detected')}")
    print(f"  Drift Share         : {drift.get('drift_share', 0):.2%}")

    target = summary["target_drift"]
    if not target.get("skipped"):
        print(f"  Ref Default Rate    : {target.get('reference_default_rate', 0):.2%}")
        print(f"  Cur Default Rate    : {target.get('current_default_rate', 0):.2%}")
        print(f"  Default Rate Shift  : {target.get('default_rate_shift', 0):+.2%}")

    print(f"\n  Reports saved to    : {MONITORING_DIR}/")
    print("=" * 50)
    print("\nOpen reports in browser:")
    print(f"  open {MONITORING_DIR}/data_drift_report.html")
    print(f"  open {MONITORING_DIR}/target_drift_report.html")
    print(f"  open {MONITORING_DIR}/model_performance_report.html")
