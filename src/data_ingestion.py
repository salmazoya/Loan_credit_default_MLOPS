import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import (
    RAW_DIR, PROCESSED_DIR,
    RAW_APPLICATION_TRAIN, RAW_APPLICATION_TEST,
    RAW_BUREAU, RAW_BUREAU_BALANCE,
    RAW_PREVIOUS_APPLICATION, RAW_POS_CASH_BALANCE,
    RAW_CREDIT_CARD_BALANCE, RAW_INSTALLMENTS_PAYMENTS,
    TRAIN_FILE_PATH, TEST_FILE_PATH, UNLABELLED_FILE_PATH,
    PROCESSED_BUREAU, PROCESSED_BUREAU_BALANCE,
    PROCESSED_PREVIOUS_APP, PROCESSED_POS_CASH,
    PROCESSED_CREDIT_CARD, PROCESSED_INSTALLMENTS,
    CONFIG_PATH
)
from utils.common_fnctions import read_yaml

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schema Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, required_columns: list, file_name: str) -> None:
    """
    Check that all required columns are present in the dataframe.
    Raises CustomException if any are missing.
    """
    try:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Schema validation failed for '{file_name}'. "
                f"Missing columns: {missing_cols}"
            )
        logger.info(f"Schema validation passed for '{file_name}'.")
    except Exception as e:
        logger.error(f"Schema validation error for '{file_name}': {e}")
        raise CustomException(f"Schema validation failed for '{file_name}'", e)


def validate_not_empty(df: pd.DataFrame, file_name: str) -> None:
    """Raise error if dataframe has no rows."""
    try:
        if df.empty:
            raise ValueError(f"'{file_name}' is empty — no rows found.")
        logger.info(f"'{file_name}' has {len(df):,} rows and {df.shape[1]} columns.")
    except Exception as e:
        logger.error(f"Empty file check failed for '{file_name}': {e}")
        raise CustomException(f"Empty file: '{file_name}'", e)


# ─────────────────────────────────────────────────────────────────────────────
# File Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_file(file_path: str, file_name: str) -> pd.DataFrame:
    """Load a single CSV file with error handling."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Raw file not found: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded '{file_name}': {df.shape[0]:,} rows x {df.shape[1]} cols")
        return df
    except Exception as e:
        logger.error(f"Failed to load '{file_name}': {e}")
        raise CustomException(f"Failed to load '{file_name}'", e)


def load_all_raw_files(config: dict) -> dict:
    """
    Load all 8 raw CSV files from artifacts/raw/.
    Returns a dict of {name: DataFrame}.

    In real life, this function would be replaced / extended to
    pull files from GCS, a database, or an API.
    """
    try:
        logger.info("Loading all raw files from artifacts/raw/")
        required = config["data_ingestion"]["required_columns"]

        # ── Application (main table) ──────────────────────────────────────────
        app_train = load_raw_file(RAW_APPLICATION_TRAIN, "application_train")
        validate_not_empty(app_train, "application_train")
        validate_schema(app_train, required["application"], "application_train")

        app_test = load_raw_file(RAW_APPLICATION_TEST, "application_test")
        validate_not_empty(app_test, "application_test")
        validate_schema(app_test, required["application"], "application_test")

        # ── Secondary tables ──────────────────────────────────────────────────
        bureau = load_raw_file(RAW_BUREAU, "bureau")
        validate_not_empty(bureau, "bureau")
        validate_schema(bureau, required["bureau"], "bureau")

        bureau_bal = load_raw_file(RAW_BUREAU_BALANCE, "bureau_balance")
        validate_not_empty(bureau_bal, "bureau_balance")
        validate_schema(bureau_bal, required["bureau_balance"], "bureau_balance")

        prev_app = load_raw_file(RAW_PREVIOUS_APPLICATION, "previous_application")
        validate_not_empty(prev_app, "previous_application")
        validate_schema(prev_app, required["previous_application"], "previous_application")

        pos_cash = load_raw_file(RAW_POS_CASH_BALANCE, "pos_cash_balance")
        validate_not_empty(pos_cash, "pos_cash_balance")
        validate_schema(pos_cash, required["pos_cash_balance"], "pos_cash_balance")

        cc_bal = load_raw_file(RAW_CREDIT_CARD_BALANCE, "credit_card_balance")
        validate_not_empty(cc_bal, "credit_card_balance")
        validate_schema(cc_bal, required["credit_card_balance"], "credit_card_balance")

        inst = load_raw_file(RAW_INSTALLMENTS_PAYMENTS, "installments_payments")
        validate_not_empty(inst, "installments_payments")
        validate_schema(inst, required["installments_payments"], "installments_payments")

        logger.info("All raw files loaded and validated successfully.")

        return {
            "application_train":    app_train,
            "application_test":     app_test,
            "bureau":               bureau,
            "bureau_balance":       bureau_bal,
            "previous_application": prev_app,
            "pos_cash_balance":     pos_cash,
            "credit_card_balance":  cc_bal,
            "installments_payments": inst
        }

    except Exception as e:
        logger.error(f"Error loading raw files: {e}")
        raise CustomException("Failed to load all raw files", e)


# ─────────────────────────────────────────────────────────────────────────────
# Labelled / Unlabelled Split
# ─────────────────────────────────────────────────────────────────────────────

def separate_labelled_unlabelled(
    app_train: pd.DataFrame,
    app_test: pd.DataFrame,
    target_col: str
) -> tuple:
    """
    In real life we receive ONE dataset without a pre-made train/test split.
    This function:
      - Combines application_train + application_test into one dataset
      - Separates rows WITH a TARGET (labelled) from those WITHOUT (unlabelled)

    Labelled rows   → will be split into train / test for model training
    Unlabelled rows → rows where outcome is not yet known (predict on these)
    """
    try:
        logger.info("Combining application_train and application_test into one dataset.")

        # application_test has no TARGET column — add it as NaN
        if target_col not in app_test.columns:
            app_test = app_test.copy()
            app_test[target_col] = np.nan

        combined = pd.concat([app_train, app_test], ignore_index=True)
        logger.info(f"Combined dataset: {combined.shape[0]:,} rows")

        labelled   = combined[combined[target_col].notna()].copy()
        unlabelled = combined[combined[target_col].isna()].copy()

        labelled[target_col] = labelled[target_col].astype(int)

        logger.info(f"Labelled rows   : {len(labelled):,} "
                    f"(default rate: {labelled[target_col].mean()*100:.1f}%)")
        logger.info(f"Unlabelled rows : {len(unlabelled):,} (no outcome yet)")

        return labelled, unlabelled

    except Exception as e:
        logger.error(f"Error separating labelled/unlabelled: {e}")
        raise CustomException("Failed to separate labelled/unlabelled data", e)


# ─────────────────────────────────────────────────────────────────────────────
# Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────

def split_train_test(
    labelled: pd.DataFrame,
    train_ratio: float,
    target_col: str,
    random_state: int
) -> tuple:
    """
    Stratified train/test split on labelled data.

    Stratified ensures both splits preserve the same ~8% default rate.

    In real life with a date column, replace this with a time-based split:
        split_date = df['APPLICATION_DATE'].quantile(train_ratio)
        train = df[df['APPLICATION_DATE'] <= split_date]
        test  = df[df['APPLICATION_DATE'] >  split_date]
    """
    try:
        logger.info(
            f"Splitting labelled data: {train_ratio*100:.0f}% train / "
            f"{(1-train_ratio)*100:.0f}% test (stratified)"
        )

        train, test = train_test_split(
            labelled,
            test_size=1 - train_ratio,
            stratify=labelled[target_col],
            random_state=random_state
        )

        logger.info(f"Train set : {len(train):,} rows | "
                    f"default rate: {train[target_col].mean()*100:.1f}%")
        logger.info(f"Test set  : {len(test):,}  rows | "
                    f"default rate: {test[target_col].mean()*100:.1f}%")

        return train, test

    except Exception as e:
        logger.error(f"Error splitting train/test: {e}")
        raise CustomException("Failed to split train/test data", e)


# ─────────────────────────────────────────────────────────────────────────────
# Save Processed Files
# ─────────────────────────────────────────────────────────────────────────────

def save_processed_files(data_dict: dict) -> None:
    """
    Save all processed dataframes to artifacts/processed/.
    Keys in data_dict map to their output paths.
    """
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        path_map = {
            "train":                TRAIN_FILE_PATH,
            "test":                 TEST_FILE_PATH,
            "unlabelled":           UNLABELLED_FILE_PATH,
            "bureau":               PROCESSED_BUREAU,
            "bureau_balance":       PROCESSED_BUREAU_BALANCE,
            "previous_application": PROCESSED_PREVIOUS_APP,
            "pos_cash_balance":     PROCESSED_POS_CASH,
            "credit_card_balance":  PROCESSED_CREDIT_CARD,
            "installments_payments": PROCESSED_INSTALLMENTS,
        }

        for name, df in data_dict.items():
            if name in path_map:
                df.to_csv(path_map[name], index=False)
                logger.info(f"Saved '{name}' → {path_map[name]}  ({len(df):,} rows)")

        logger.info("All processed files saved successfully.")

    except Exception as e:
        logger.error(f"Error saving processed files: {e}")
        raise CustomException("Failed to save processed files", e)


# ─────────────────────────────────────────────────────────────────────────────
# Main DataIngestion Class
# ─────────────────────────────────────────────────────────────────────────────

class DataIngestion:
    """
    Orchestrates the full data ingestion pipeline:
      1. Load all 8 raw files
      2. Validate schemas
      3. Separate labelled vs unlabelled rows
      4. Stratified train/test split
      5. Save all processed files to artifacts/processed/
    """

    def __init__(self):
        self.config = read_yaml(CONFIG_PATH)
        self.ingestion_config = self.config["data_ingestion"]
        logger.info("DataIngestion initialised.")

    def run(self) -> dict:
        """
        Run the full ingestion pipeline.
        Returns a dict of processed DataFrames.
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting Data Ingestion Pipeline")
            logger.info("=" * 60)

            # Step 1 — Load all raw files
            raw = load_all_raw_files(self.config)

            # Step 2 — Separate labelled vs unlabelled
            target_col   = self.ingestion_config["target_column"]
            labelled, unlabelled = separate_labelled_unlabelled(
                raw["application_train"],
                raw["application_test"],
                target_col
            )

            # Step 3 — Train/test split on labelled data
            train_ratio  = self.ingestion_config["train_ratio"]
            random_state = self.ingestion_config["random_state"]
            train, test  = split_train_test(
                labelled, train_ratio, target_col, random_state
            )

            # Step 4 — Save all processed files
            processed = {
                "train":                 train,
                "test":                  test,
                "unlabelled":            unlabelled,
                "bureau":                raw["bureau"],
                "bureau_balance":        raw["bureau_balance"],
                "previous_application":  raw["previous_application"],
                "pos_cash_balance":      raw["pos_cash_balance"],
                "credit_card_balance":   raw["credit_card_balance"],
                "installments_payments": raw["installments_payments"],
            }
            save_processed_files(processed)

            logger.info("=" * 60)
            logger.info("Data Ingestion Pipeline completed successfully.")
            logger.info(f"  Train      : {len(train):,} rows")
            logger.info(f"  Test       : {len(test):,} rows")
            logger.info(f"  Unlabelled : {len(unlabelled):,} rows")
            logger.info("=" * 60)

            return processed

        except Exception as e:
            logger.error(f"Data Ingestion Pipeline failed: {e}")
            raise CustomException("Data Ingestion Pipeline failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
