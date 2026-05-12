import os

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "config/config.yaml"

# ── Raw data (8 input files) ──────────────────────────────────────────────────
RAW_DIR                     = "artifacts/raw"
RAW_APPLICATION_TRAIN       = os.path.join(RAW_DIR, "application_train.csv")
RAW_APPLICATION_TEST        = os.path.join(RAW_DIR, "application_test.csv")
RAW_BUREAU                  = os.path.join(RAW_DIR, "bureau.csv")
RAW_BUREAU_BALANCE          = os.path.join(RAW_DIR, "bureau_balance.csv")
RAW_PREVIOUS_APPLICATION    = os.path.join(RAW_DIR, "previous_application.csv")
RAW_POS_CASH_BALANCE        = os.path.join(RAW_DIR, "POS_CASH_balance.csv")
RAW_CREDIT_CARD_BALANCE     = os.path.join(RAW_DIR, "credit_card_balance.csv")
RAW_INSTALLMENTS_PAYMENTS   = os.path.join(RAW_DIR, "installments_payments.csv")
RAW_COLUMNS_DESCRIPTION     = os.path.join(RAW_DIR, "HomeCredit_columns_description.csv")

# ── Processed data (output of data_ingestion) ─────────────────────────────────
PROCESSED_DIR               = "artifacts/processed"
TRAIN_FILE_PATH             = os.path.join(PROCESSED_DIR, "train.csv")
TEST_FILE_PATH              = os.path.join(PROCESSED_DIR, "test.csv")
UNLABELLED_FILE_PATH        = os.path.join(PROCESSED_DIR, "unlabelled.csv")

# Secondary processed files
PROCESSED_BUREAU            = os.path.join(PROCESSED_DIR, "bureau.csv")
PROCESSED_BUREAU_BALANCE    = os.path.join(PROCESSED_DIR, "bureau_balance.csv")
PROCESSED_PREVIOUS_APP      = os.path.join(PROCESSED_DIR, "previous_application.csv")
PROCESSED_POS_CASH          = os.path.join(PROCESSED_DIR, "pos_cash_balance.csv")
PROCESSED_CREDIT_CARD       = os.path.join(PROCESSED_DIR, "credit_card_balance.csv")
PROCESSED_INSTALLMENTS      = os.path.join(PROCESSED_DIR, "installments_payments.csv")

# ── All model & preprocessor artifacts (.pkl) → one folder ───────────────────
#
#   artifacts/models/
#   ├── lgbm_model.pkl          ← trained LightGBM model
#   ├── medians.pkl             ← numeric imputation values (from training)
#   ├── encoders.pkl            ← label / OHE encoder mappings
#   └── feature_columns.pkl    ← exact feature list & order the model expects
#
MODELS_DIR              = "artifacts/models"
MODEL_FILE_PATH         = os.path.join(MODELS_DIR, "lgbm_model.pkl")
MEDIANS_FILE_PATH       = os.path.join(MODELS_DIR, "medians.pkl")
ENCODERS_FILE_PATH      = os.path.join(MODELS_DIR, "encoders.pkl")
FEATURE_COLUMNS_PATH    = os.path.join(MODELS_DIR, "feature_columns.pkl")

# ── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_DIR              = "mlruns"

# ── Logs ──────────────────────────────────────────────────────────────────────
LOGS_DIR                = "logs"
