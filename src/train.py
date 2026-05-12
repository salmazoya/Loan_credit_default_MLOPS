"""
train.py — Full Training Pipeline with MLflow Tracking
=======================================================
Runs end-to-end:
  1. Data Ingestion      → loads & splits raw files
  2. Data Preprocessing  → feature engineering, saves .pkl artifacts
  3. LightGBM Training   → 5-fold stratified CV
  4. Evaluation          → AUC, F1, confusion matrix on held-out test set
  5. MLflow Logging      → params, metrics, artifacts, model
  6. Model Promotion     → saves new model only if AUC improves

Run:
  cd ~/Desktop/MLOPS_NEW/loan_credit_default_mlops
  python -m src.train

View MLflow UI:
  mlflow ui --port 5000
  → open http://localhost:5000
"""

import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    precision_recall_curve, f1_score
)

from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from config.paths_config import (
    MODELS_DIR, MODEL_FILE_PATH,
    MEDIANS_FILE_PATH, ENCODERS_FILE_PATH, FEATURE_COLUMNS_PATH,
    CONFIG_PATH
)
from utils.common_fnctions import read_yaml

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_mlflow(config: dict) -> str:
    """Configure MLflow tracking URI and experiment."""
    try:
        tracking_uri     = config["mlflow"]["tracking_uri"]
        experiment_name  = config["mlflow"]["experiment_name"]

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracking URI  : {tracking_uri}")
        logger.info(f"MLflow experiment    : {experiment_name}")
        logger.info(f"MLflow UI command    : mlflow ui --port 5000")
        return experiment_name

    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        raise CustomException("Failed to setup MLflow", e)


# ─────────────────────────────────────────────────────────────────────────────
# Build LightGBM Params from Config
# ─────────────────────────────────────────────────────────────────────────────

def build_lgbm_params(config: dict, scale_pos_weight: float) -> dict:
    """Build LightGBM parameter dict from config.yaml + computed class weight."""
    m = config["model"]
    return {
        "objective"         : m["objective"],
        "metric"            : m["metric"],
        "boosting_type"     : m["boosting_type"],
        "learning_rate"     : m["learning_rate"],
        "n_estimators"      : m["n_estimators"],
        "num_leaves"        : m["num_leaves"],
        "max_depth"         : m["max_depth"],
        "min_child_samples" : m["min_child_samples"],
        "subsample"         : m["subsample"],
        "subsample_freq"    : m["subsample_freq"],
        "colsample_bytree"  : m["colsample_bytree"],
        "reg_alpha"         : m["reg_alpha"],
        "reg_lambda"        : m["reg_lambda"],
        "scale_pos_weight"  : round(scale_pos_weight, 4),
        "random_state"      : m["random_state"],
        "n_jobs"            : m["n_jobs"],
        "verbose"           : m["verbose"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation Training
# ─────────────────────────────────────────────────────────────────────────────

def train_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    params: dict,
    n_folds: int,
    early_stopping_rounds: int
) -> tuple:
    """
    Train LightGBM with stratified K-fold CV.
    Returns (oof_preds, test_preds, fold_scores, feature_importances, models).
    """
    try:
        skf          = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=params["random_state"])
        oof_preds    = np.zeros(len(X_train))
        test_preds   = np.zeros(len(X_test))
        fold_scores  = []
        feature_imps = pd.DataFrame()
        models       = []

        logger.info(f"Starting {n_folds}-fold Stratified CV...")
        logger.info("=" * 55)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds,
                                       verbose=False),
                    lgb.log_evaluation(period=200)
                ]
            )

            val_pred = model.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, val_pred)

            oof_preds[val_idx]  = val_pred
            test_preds         += model.predict_proba(X_test)[:, 1] / n_folds
            fold_scores.append(fold_auc)
            models.append(model)

            fi = pd.DataFrame({
                "feature"   : X_train.columns,
                "importance": model.feature_importances_,
                "fold"      : fold
            })
            feature_imps = pd.concat([feature_imps, fi], ignore_index=True)

            logger.info(
                f"Fold {fold}/{n_folds} | "
                f"Best iter: {model.best_iteration_:>4} | "
                f"Val AUC: {fold_auc:.5f}"
            )

        oof_auc = roc_auc_score(y_train, oof_preds)
        logger.info("=" * 55)
        logger.info(f"Fold scores : {[round(s, 5) for s in fold_scores]}")
        logger.info(f"Mean CV AUC : {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
        logger.info(f"OOF AUC     : {oof_auc:.5f}")

        return oof_preds, test_preds, fold_scores, feature_imps, models

    except Exception as e:
        logger.error(f"CV training failed: {e}")
        raise CustomException("CV training failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float
) -> dict:
    """Compute full evaluation metrics at a given threshold."""
    try:
        y_pred = (y_prob >= threshold).astype(int)
        auc    = roc_auc_score(y_true, y_prob)
        f1     = f1_score(y_true, y_pred)
        cm     = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics = {
            "auc"           : round(auc, 5),
            "f1_score"      : round(f1, 5),
            "precision"     : round(precision, 5),
            "recall"        : round(recall, 5),
            "true_positives" : int(tp),
            "true_negatives" : int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "threshold"     : threshold,
        }

        logger.info("Evaluation metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k:<20}: {v}")

        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise CustomException("Model evaluation failed", e)


def find_optimal_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Find the threshold that maximises F1 score on OOF predictions."""
    best_threshold, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f1     = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1        = f1
            best_threshold = t
    logger.info(f"Optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return round(float(best_threshold), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Plot Helpers (saved as artifacts to MLflow)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, auc: float, save_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='#2196F3', linewidth=2,
            label=f'LightGBM (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (OOF)', fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_feature_importance(feature_imps: pd.DataFrame, save_path: str,
                            top_n: int = 30) -> None:
    mean_imp = (
        feature_imps.groupby('feature')['importance']
        .mean().sort_values(ascending=False).head(top_n)
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(mean_imp.index[::-1], mean_imp.values[::-1],
            color='#2196F3', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Mean Importance')
    ax.set_title(f'Top {top_n} Feature Importances', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    axes[0].set_title('Confusion Matrix (counts)', fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    axes[1].set_title('Confusion Matrix (normalised)', fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_fold_scores(fold_scores: list, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([f'Fold {i}' for i in range(1, len(fold_scores)+1)],
                  fold_scores, color='#2196F3', edgecolor='white', width=0.6)
    ax.axhline(np.mean(fold_scores), color='red', linestyle='--',
               label=f'Mean: {np.mean(fold_scores):.4f}')
    ax.set_ylim(min(fold_scores) - 0.005, max(fold_scores) + 0.005)
    ax.set_ylabel('AUC')
    ax.set_title('AUC per Fold', fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, score in zip(bars, fold_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{score:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Model Promotion — compare vs existing model
# ─────────────────────────────────────────────────────────────────────────────

def should_promote_model(new_auc: float, min_improvement: float) -> bool:
    """
    Compare new model AUC against the currently saved model.
    Returns True if new model should replace the old one.
    """
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")

    if not os.path.exists(meta_path):
        logger.info("No existing model found. Promoting new model by default.")
        return True

    with open(meta_path, "r") as f:
        meta = json.load(f)

    current_auc = meta.get("oof_auc", 0.0)
    improvement = new_auc - current_auc

    logger.info(f"Current model AUC : {current_auc:.5f}")
    logger.info(f"New model AUC     : {new_auc:.5f}")
    logger.info(f"Improvement       : {improvement:+.5f} (min required: {min_improvement})")

    if improvement >= min_improvement:
        logger.info("✅ New model is better — promoting.")
        return True
    else:
        logger.info("⚠️  New model does not improve enough — keeping existing model.")
        return False


def save_model_meta(metrics: dict, params: dict, run_id: str) -> None:
    """Save a JSON summary of the promoted model for future comparisons."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    meta = {
        "run_id"        : run_id,
        "oof_auc"       : metrics["auc"],
        "f1_score"      : metrics["f1_score"],
        "threshold"     : metrics["threshold"],
        "learning_rate" : params["learning_rate"],
        "num_leaves"    : params["num_leaves"],
        "n_estimators"  : params["n_estimators"],
    }
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Model metadata saved → {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Orchestrates the full training pipeline:
      1. Data ingestion
      2. Data preprocessing
      3. LightGBM CV training
      4. Evaluation
      5. MLflow logging
      6. Model promotion
    """

    def __init__(self):
        self.config = read_yaml(CONFIG_PATH)
        logger.info("ModelTrainer initialised.")

    def run(self) -> dict:
        try:
            logger.info("=" * 60)
            logger.info("Starting Full Training Pipeline")
            logger.info("=" * 60)

            # ── Step 1: Data Ingestion ────────────────────────────────────────
            logger.info("STEP 1 — Data Ingestion")
            ingestion = DataIngestion()
            ingestion.run()

            # ── Step 2: Data Preprocessing ────────────────────────────────────
            logger.info("STEP 2 — Data Preprocessing")
            preprocessing = DataPreprocessing()
            X_train, y_train, X_test, y_test = preprocessing.run()

            # ── Step 3: Setup MLflow ──────────────────────────────────────────
            logger.info("STEP 3 — MLflow Setup")
            setup_mlflow(self.config)

            # ── Step 4: Build params ──────────────────────────────────────────
            neg  = (y_train == 0).sum()
            pos  = (y_train == 1).sum()
            spw  = neg / pos
            params = build_lgbm_params(self.config, spw)

            n_folds   = self.config["model"]["n_folds"]
            es_rounds = self.config["model"]["early_stopping_rounds"]
            threshold = self.config["model"]["threshold"]
            min_impr  = self.config["model"]["min_auc_improvement"]
            run_prefix = self.config["mlflow"]["run_name_prefix"]

            logger.info(f"Training on {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
            logger.info(f"Class imbalance ratio (scale_pos_weight): {spw:.2f}")

            # ── Step 5: MLflow run ────────────────────────────────────────────
            with mlflow.start_run(run_name=f"{run_prefix}_cv_{n_folds}fold") as run:
                run_id = run.info.run_id
                logger.info(f"MLflow run ID: {run_id}")

                # ── Log all parameters ────────────────────────────────────────
                mlflow.log_params({
                    "model_type"            : "LightGBM",
                    "n_folds"               : n_folds,
                    "learning_rate"         : params["learning_rate"],
                    "n_estimators"          : params["n_estimators"],
                    "num_leaves"            : params["num_leaves"],
                    "max_depth"             : params["max_depth"],
                    "min_child_samples"     : params["min_child_samples"],
                    "subsample"             : params["subsample"],
                    "colsample_bytree"      : params["colsample_bytree"],
                    "reg_alpha"             : params["reg_alpha"],
                    "reg_lambda"            : params["reg_lambda"],
                    "scale_pos_weight"      : round(spw, 4),
                    "early_stopping_rounds" : es_rounds,
                    "threshold"             : threshold,
                    "train_samples"         : len(X_train),
                    "test_samples"          : len(X_test),
                    "n_features"            : X_train.shape[1],
                    "train_default_rate"    : round(float(y_train.mean()), 4),
                })

                # ── CV Training ───────────────────────────────────────────────
                logger.info("STEP 4 — CV Training")
                oof_preds, test_preds, fold_scores, feature_imps, models = train_with_cv(
                    X_train, y_train, X_test, params, n_folds, es_rounds
                )

                # ── Log per-fold AUC ──────────────────────────────────────────
                for i, score in enumerate(fold_scores, 1):
                    mlflow.log_metric(f"fold_{i}_auc", round(score, 5))

                mlflow.log_metric("cv_auc_mean", round(np.mean(fold_scores), 5))
                mlflow.log_metric("cv_auc_std",  round(np.std(fold_scores), 5))

                # ── Find optimal threshold on OOF ─────────────────────────────
                optimal_threshold = find_optimal_threshold(y_train, oof_preds)
                mlflow.log_metric("optimal_threshold", optimal_threshold)

                # ── OOF evaluation ────────────────────────────────────────────
                logger.info("STEP 5 — OOF Evaluation")
                oof_metrics = evaluate_model(y_train, oof_preds, optimal_threshold)
                for k, v in oof_metrics.items():
                    mlflow.log_metric(f"oof_{k}", v)

                # ── Test set evaluation (if labels available) ─────────────────
                if y_test is not None:
                    logger.info("STEP 5b — Test Set Evaluation")
                    test_metrics = evaluate_model(y_test, test_preds, optimal_threshold)
                    for k, v in test_metrics.items():
                        mlflow.log_metric(f"test_{k}", v)
                    logger.info(f"Test AUC: {test_metrics['auc']:.5f}")

                # ── Save plots as MLflow artifacts ────────────────────────────
                logger.info("STEP 6 — Saving Plots")
                os.makedirs("artifacts/plots", exist_ok=True)

                roc_path = "artifacts/plots/roc_curve.png"
                imp_path = "artifacts/plots/feature_importance.png"
                cm_path  = "artifacts/plots/confusion_matrix.png"
                fld_path = "artifacts/plots/fold_scores.png"

                plot_roc_curve(y_train, oof_preds, oof_metrics["auc"], roc_path)
                plot_feature_importance(feature_imps, imp_path)
                plot_confusion_matrix(
                    y_train,
                    (oof_preds >= optimal_threshold).astype(int),
                    cm_path
                )
                plot_fold_scores(fold_scores, fld_path)

                mlflow.log_artifact(roc_path,  artifact_path="plots")
                mlflow.log_artifact(imp_path,  artifact_path="plots")
                mlflow.log_artifact(cm_path,   artifact_path="plots")
                mlflow.log_artifact(fld_path,  artifact_path="plots")

                # ── Feature importance CSV ────────────────────────────────────
                mean_imp = (
                    feature_imps.groupby('feature')['importance']
                    .mean().sort_values(ascending=False).reset_index()
                )
                imp_csv = "artifacts/plots/feature_importance.csv"
                mean_imp.to_csv(imp_csv, index=False)
                mlflow.log_artifact(imp_csv, artifact_path="plots")

                # ── Model Promotion ───────────────────────────────────────────
                logger.info("STEP 7 — Model Promotion Check")
                promote = should_promote_model(oof_metrics["auc"], min_impr)

                mlflow.log_param("model_promoted", promote)

                if promote:
                    os.makedirs(MODELS_DIR, exist_ok=True)

                    # Save all fold models as a list
                    joblib.dump(models, MODEL_FILE_PATH)
                    logger.info(f"Model saved → {MODEL_FILE_PATH}")

                    # Log .pkl files as MLflow artifacts
                    mlflow.log_artifact(MODEL_FILE_PATH,      artifact_path="models")
                    mlflow.log_artifact(MEDIANS_FILE_PATH,    artifact_path="models")
                    mlflow.log_artifact(ENCODERS_FILE_PATH,   artifact_path="models")
                    mlflow.log_artifact(FEATURE_COLUMNS_PATH, artifact_path="models")

                    # Log model with signature for MLflow Model Registry
                    signature = infer_signature(X_train, oof_preds)
                    mlflow.lightgbm.log_model(
                        lgb_model=models[-1].booster_,
                        artifact_path="lgbm_booster",
                        signature=signature,
                        input_example=X_train.iloc[:3]
                    )

                    # Save metadata for future comparison
                    save_model_meta(oof_metrics, params, run_id)

                    mlflow.set_tag("model_status", "promoted")
                    logger.info("✅ Model promoted and saved.")
                else:
                    mlflow.set_tag("model_status", "not_promoted")

                # ── Step 8: Drift Monitoring ──────────────────────────────────
                logger.info("STEP 8 — Evidently Drift Monitoring")
                try:
                    from src.monitor import run_monitoring
                    from config.paths_config import TRAIN_FILE_PATH, TEST_FILE_PATH

                    ref_df = pd.read_csv(TRAIN_FILE_PATH)
                    cur_df = pd.read_csv(TEST_FILE_PATH)

                    monitoring_summary = run_monitoring(
                        reference_df   = ref_df,
                        current_df     = cur_df,
                        mlflow_run_id  = run_id,
                        log_to_mlflow  = True,
                    )
                    drift_detected = monitoring_summary["data_drift"].get("drift_detected", False)
                    drift_share    = monitoring_summary["data_drift"].get("drift_share", 0)

                    mlflow.log_metric("drift_detected", int(drift_detected))
                    mlflow.log_metric("drift_share",    drift_share)

                    if drift_detected:
                        logger.warning(
                            f"⚠️  Data drift detected! {drift_share:.2%} of features drifted. "
                            f"Consider retraining sooner."
                        )
                        mlflow.set_tag("drift_status", "drift_detected")
                    else:
                        logger.info("✅ No significant data drift detected.")
                        mlflow.set_tag("drift_status", "no_drift")

                except Exception as monitor_err:
                    logger.warning(f"Monitoring step failed (non-fatal): {monitor_err}")
                    monitoring_summary = {}
                    drift_detected     = None

                # ── Final summary ─────────────────────────────────────────────
                mlflow.set_tag("run_type", "full_pipeline")
                mlflow.set_tag("framework", "lightgbm")

                logger.info("=" * 60)
                logger.info("Training Pipeline Complete")
                logger.info(f"  OOF AUC       : {oof_metrics['auc']:.5f}")
                logger.info(f"  OOF F1        : {oof_metrics['f1_score']:.5f}")
                logger.info(f"  Threshold     : {optimal_threshold}")
                logger.info(f"  Model promoted: {promote}")
                logger.info(f"  Drift detected: {drift_detected}")
                logger.info(f"  MLflow run ID : {run_id}")
                logger.info(f"  View results  : mlflow ui --port 5000")
                logger.info("=" * 60)

                return {
                    "run_id"        : run_id,
                    "oof_auc"       : oof_metrics["auc"],
                    "cv_auc_mean"   : round(np.mean(fold_scores), 5),
                    "cv_auc_std"    : round(np.std(fold_scores), 5),
                    "threshold"     : optimal_threshold,
                    "model_promoted": promote,
                    "drift_detected": drift_detected,
                }

        except Exception as e:
            logger.error(f"Training Pipeline failed: {e}")
            raise CustomException("Training Pipeline failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run()

    print("\n" + "=" * 50)
    print("        TRAINING RESULTS SUMMARY")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:<20}: {v}")
    print("=" * 50)
    print("\nTo view MLflow UI:")
    print("  mlflow ui --port 5000")
    print("  → http://localhost:5000")
