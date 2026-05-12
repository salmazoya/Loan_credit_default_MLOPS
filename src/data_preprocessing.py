import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import (
    PROCESSED_DIR,
    TRAIN_FILE_PATH, TEST_FILE_PATH,
    PROCESSED_BUREAU, PROCESSED_BUREAU_BALANCE,
    PROCESSED_PREVIOUS_APP, PROCESSED_POS_CASH,
    PROCESSED_CREDIT_CARD, PROCESSED_INSTALLMENTS,
    MODELS_DIR,
    MEDIANS_FILE_PATH, ENCODERS_FILE_PATH, FEATURE_COLUMNS_PATH,
    CONFIG_PATH
)
from utils.common_fnctions import read_yaml

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Application Table Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from the main application table.
    Works identically on train, test, and single-applicant input.
    """
    try:
        logger.info("Engineering application features...")
        df = df.copy()

        # Age & Employment
        df['AGE_YEARS']       = (-df['DAYS_BIRTH'] / 365).astype(int)
        df['EMPLOYED_YEARS']  = (-df['DAYS_EMPLOYED'] / 365)

        # DAYS_EMPLOYED = 365243 is a known anomaly meaning unemployed
        df['EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
        df.loc[df['DAYS_EMPLOYED'] == 365243, 'EMPLOYED_YEARS'] = np.nan
        df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED']  = np.nan

        df['REGISTRATION_YEARS'] = (-df['DAYS_REGISTRATION'] / 365) if 'DAYS_REGISTRATION' in df.columns else np.nan
        df['ID_PUBLISH_YEARS']   = (-df['DAYS_ID_PUBLISH'] / 365) if 'DAYS_ID_PUBLISH' in df.columns else np.nan

        # Income & Credit Ratios
        df['CREDIT_INCOME_RATIO']  = df['AMT_CREDIT']  / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
        df['CREDIT_TERM']          = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)
        df['GOODS_CREDIT_RATIO']   = (df['AMT_GOODS_PRICE'] / df['AMT_CREDIT'].replace(0, np.nan)) if 'AMT_GOODS_PRICE' in df.columns else np.nan
        df['INCOME_PER_PERSON']    = (df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)) if 'CNT_FAM_MEMBERS' in df.columns else np.nan

        # EXT_SOURCE combinations
        ext = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        df['EXT_SOURCE_MEAN'] = df[ext].mean(axis=1)
        df['EXT_SOURCE_STD']  = df[ext].std(axis=1)
        df['EXT_SOURCE_MIN']  = df[ext].min(axis=1)
        df['EXT_SOURCE_MAX']  = df[ext].max(axis=1)
        df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

        # Document flags (columns may be absent for single-applicant input)
        doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
        df['TOTAL_DOCS_SUBMITTED'] = df[doc_cols].sum(axis=1) if doc_cols else 0

        # Social circle default rate (optional columns — default to NaN if missing)
        if 'DEF_30_CNT_SOCIAL_CIRCLE' not in df.columns:
            df['DEF_30_CNT_SOCIAL_CIRCLE'] = np.nan
        if 'OBS_30_CNT_SOCIAL_CIRCLE' not in df.columns:
            df['OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
        df['SOCIAL_CIRCLE_DEFAULT_RATE'] = (
            df['DEF_30_CNT_SOCIAL_CIRCLE'] /
            df['OBS_30_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)
        )

        # Credit enquiries total (columns may be absent for single-applicant input)
        enquiry_cols = [c for c in df.columns if c.startswith('AMT_REQ_CREDIT_BUREAU_')]
        df['TOTAL_CREDIT_ENQUIRIES'] = df[enquiry_cols].sum(axis=1) if enquiry_cols else 0

        logger.info(f"Application features engineered. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error engineering application features: {e}")
        raise CustomException("Failed to engineer application features", e)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Secondary Table Aggregations
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_bureau(bureau: pd.DataFrame, bureau_bal: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau + bureau_balance → one row per SK_ID_CURR."""
    try:
        logger.info("Aggregating bureau data...")

        bureau_bal = bureau_bal.copy()
        bureau_bal['STATUS_NUM'] = bureau_bal['STATUS'].map(
            {'C': 0, 'X': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        )
        bb_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(
            bb_months_count  = ('MONTHS_BALANCE', 'count'),
            bb_status_mean   = ('STATUS_NUM', 'mean'),
            bb_status_max    = ('STATUS_NUM', 'max'),
            bb_dpd_months    = ('STATUS_NUM', lambda x: (x > 0).sum())
        ).reset_index()

        bureau_full = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

        agg = bureau_full.groupby('SK_ID_CURR').agg(
            bur_count              = ('SK_ID_BUREAU', 'count'),
            bur_active_count       = ('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
            bur_closed_count       = ('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
            bur_days_credit_mean   = ('DAYS_CREDIT', 'mean'),
            bur_days_credit_max    = ('DAYS_CREDIT', 'max'),
            bur_credit_sum_mean    = ('AMT_CREDIT_SUM', 'mean'),
            bur_credit_sum_total   = ('AMT_CREDIT_SUM', 'sum'),
            bur_credit_sum_debt    = ('AMT_CREDIT_SUM_DEBT', 'sum'),
            bur_credit_sum_overdue = ('AMT_CREDIT_SUM_OVERDUE', 'sum'),
            bur_overdue_count      = ('AMT_CREDIT_SUM_OVERDUE', lambda x: (x > 0).sum()),
            bur_prolong_sum        = ('CNT_CREDIT_PROLONG', 'sum'),
            bur_bb_status_mean     = ('bb_status_mean', 'mean'),
            bur_bb_status_max      = ('bb_status_max', 'max'),
            bur_bb_dpd_months_sum  = ('bb_dpd_months', 'sum'),
        ).reset_index()

        agg['bur_active_ratio']      = agg['bur_active_count'] / agg['bur_count'].replace(0, np.nan)
        agg['bur_overdue_ratio']     = agg['bur_overdue_count'] / agg['bur_count'].replace(0, np.nan)
        agg['bur_debt_credit_ratio'] = agg['bur_credit_sum_debt'] / agg['bur_credit_sum_total'].replace(0, np.nan)

        logger.info(f"Bureau aggregated: {agg.shape}")
        return agg

    except Exception as e:
        logger.error(f"Error aggregating bureau: {e}")
        raise CustomException("Failed to aggregate bureau data", e)


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous_application → one row per SK_ID_CURR."""
    try:
        logger.info("Aggregating previous application data...")
        prev = prev.copy()

        for col in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
                    'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
            if col in prev.columns:
                prev[col].replace(365243, np.nan, inplace=True)

        prev['APP_CREDIT_RATIO']   = prev['AMT_APPLICATION'] / prev['AMT_CREDIT'].replace(0, np.nan)
        prev['DOWN_PAYMENT_RATIO'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_APPLICATION'].replace(0, np.nan)
        prev['IS_APPROVED']        = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
        prev['IS_REFUSED']         = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)

        agg = prev.groupby('SK_ID_CURR').agg(
            prev_count                 = ('SK_ID_PREV', 'count'),
            prev_approved_count        = ('IS_APPROVED', 'sum'),
            prev_refused_count         = ('IS_REFUSED', 'sum'),
            prev_amt_credit_mean       = ('AMT_CREDIT', 'mean'),
            prev_amt_credit_max        = ('AMT_CREDIT', 'max'),
            prev_amt_annuity_mean      = ('AMT_ANNUITY', 'mean'),
            prev_app_credit_ratio_mean = ('APP_CREDIT_RATIO', 'mean'),
            prev_down_payment_mean     = ('DOWN_PAYMENT_RATIO', 'mean'),
            prev_days_decision_mean    = ('DAYS_DECISION', 'mean'),
            prev_days_decision_min     = ('DAYS_DECISION', 'min'),
            prev_rate_interest_mean    = ('RATE_INTEREST_PRIMARY', 'mean'),
            prev_rate_interest_max     = ('RATE_INTEREST_PRIMARY', 'max'),
            prev_consumer_count        = ('NAME_CONTRACT_TYPE', lambda x: (x == 'Consumer loans').sum()),
            prev_cash_count            = ('NAME_CONTRACT_TYPE', lambda x: (x == 'Cash loans').sum()),
            prev_revolving_count       = ('NAME_CONTRACT_TYPE', lambda x: (x == 'Revolving loans').sum()),
        ).reset_index()

        agg['prev_approval_rate'] = agg['prev_approved_count'] / agg['prev_count'].replace(0, np.nan)
        agg['prev_refusal_rate']  = agg['prev_refused_count']  / agg['prev_count'].replace(0, np.nan)

        logger.info(f"Previous application aggregated: {agg.shape}")
        return agg

    except Exception as e:
        logger.error(f"Error aggregating previous application: {e}")
        raise CustomException("Failed to aggregate previous application data", e)


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance → one row per SK_ID_CURR."""
    try:
        logger.info("Aggregating POS CASH data...")

        agg = pos.groupby('SK_ID_CURR').agg(
            pos_count                      = ('SK_ID_PREV', 'count'),
            pos_months_balance_mean        = ('MONTHS_BALANCE', 'mean'),
            pos_months_balance_min         = ('MONTHS_BALANCE', 'min'),
            pos_cnt_instalment_mean        = ('CNT_INSTALMENT', 'mean'),
            pos_cnt_instalment_future_mean = ('CNT_INSTALMENT_FUTURE', 'mean'),
            pos_dpd_mean                   = ('SK_DPD', 'mean'),
            pos_dpd_max                    = ('SK_DPD', 'max'),
            pos_dpd_def_mean               = ('SK_DPD_DEF', 'mean'),
            pos_dpd_def_max                = ('SK_DPD_DEF', 'max'),
            pos_dpd_count                  = ('SK_DPD', lambda x: (x > 0).sum()),
            pos_active_count               = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Active').sum()),
            pos_completed_count            = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Completed').sum()),
        ).reset_index()

        agg['pos_dpd_rate']    = agg['pos_dpd_count']    / agg['pos_count'].replace(0, np.nan)
        agg['pos_active_rate'] = agg['pos_active_count'] / agg['pos_count'].replace(0, np.nan)

        logger.info(f"POS CASH aggregated: {agg.shape}")
        return agg

    except Exception as e:
        logger.error(f"Error aggregating POS CASH: {e}")
        raise CustomException("Failed to aggregate POS CASH data", e)


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance → one row per SK_ID_CURR."""
    try:
        logger.info("Aggregating credit card data...")
        cc = cc.copy()

        cc['CC_UTILIZATION']   = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
        cc['CC_PAYMENT_RATIO'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY'].replace(0, np.nan)

        agg = cc.groupby('SK_ID_CURR').agg(
            cc_count               = ('SK_ID_PREV', 'count'),
            cc_balance_mean        = ('AMT_BALANCE', 'mean'),
            cc_balance_max         = ('AMT_BALANCE', 'max'),
            cc_credit_limit_mean   = ('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
            cc_utilization_mean    = ('CC_UTILIZATION', 'mean'),
            cc_utilization_max     = ('CC_UTILIZATION', 'max'),
            cc_payment_ratio_mean  = ('CC_PAYMENT_RATIO', 'mean'),
            cc_dpd_mean            = ('SK_DPD', 'mean'),
            cc_dpd_max             = ('SK_DPD', 'max'),
            cc_dpd_def_mean        = ('SK_DPD_DEF', 'mean'),
            cc_dpd_count           = ('SK_DPD', lambda x: (x > 0).sum()),
            cc_drawings_count_mean = ('CNT_DRAWINGS_CURRENT', 'mean'),
            cc_atm_drawings_mean   = ('AMT_DRAWINGS_ATM_CURRENT', 'mean'),
        ).reset_index()

        agg['cc_dpd_rate'] = agg['cc_dpd_count'] / agg['cc_count'].replace(0, np.nan)

        logger.info(f"Credit card aggregated: {agg.shape}")
        return agg

    except Exception as e:
        logger.error(f"Error aggregating credit card: {e}")
        raise CustomException("Failed to aggregate credit card data", e)


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments → one row per SK_ID_CURR."""
    try:
        logger.info("Aggregating installments data...")
        inst = inst.copy()

        inst['DAYS_LATE']     = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
        inst['PAYMENT_RATIO'] = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT'].replace(0, np.nan)
        inst['PAYMENT_DIFF']  = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
        inst['IS_LATE']       = (inst['DAYS_LATE'] > 0).astype(int)
        inst['IS_UNDERPAID']  = (inst['AMT_PAYMENT'] < inst['AMT_INSTALMENT']).astype(int)

        agg = inst.groupby('SK_ID_CURR').agg(
            inst_count               = ('NUM_INSTALMENT_NUMBER', 'count'),
            inst_num_loans           = ('SK_ID_PREV', 'nunique'),
            inst_days_late_mean      = ('DAYS_LATE', 'mean'),
            inst_days_late_max       = ('DAYS_LATE', 'max'),
            inst_days_late_sum       = ('DAYS_LATE', lambda x: x[x > 0].sum()),
            inst_pct_late            = ('IS_LATE', 'mean'),
            inst_pct_underpaid       = ('IS_UNDERPAID', 'mean'),
            inst_payment_ratio_mean  = ('PAYMENT_RATIO', 'mean'),
            inst_payment_ratio_min   = ('PAYMENT_RATIO', 'min'),
            inst_payment_diff_mean   = ('PAYMENT_DIFF', 'mean'),
            inst_payment_diff_max    = ('PAYMENT_DIFF', 'max'),
            inst_amt_payment_sum     = ('AMT_PAYMENT', 'sum'),
            inst_amt_payment_mean    = ('AMT_PAYMENT', 'mean'),
            inst_amt_instalment_mean = ('AMT_INSTALMENT', 'mean'),
        ).reset_index()

        logger.info(f"Installments aggregated: {agg.shape}")
        return agg

    except Exception as e:
        logger.error(f"Error aggregating installments: {e}")
        raise CustomException("Failed to aggregate installments data", e)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Merge All Tables
# ─────────────────────────────────────────────────────────────────────────────

def merge_all_tables(
    app_df: pd.DataFrame,
    bureau_agg: pd.DataFrame,
    prev_agg: pd.DataFrame,
    pos_agg: pd.DataFrame,
    cc_agg: pd.DataFrame,
    inst_agg: pd.DataFrame
) -> pd.DataFrame:
    """Left-join all aggregated secondary tables onto the application table."""
    try:
        logger.info("Merging all tables into master feature matrix...")
        n_before = len(app_df)

        df = app_df.copy()
        df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
        df = df.merge(prev_agg,   on='SK_ID_CURR', how='left')
        df = df.merge(pos_agg,    on='SK_ID_CURR', how='left')
        df = df.merge(cc_agg,     on='SK_ID_CURR', how='left')
        df = df.merge(inst_agg,   on='SK_ID_CURR', how='left')

        assert len(df) == n_before, "Row count changed after merge — check for duplicates!"
        logger.info(f"Master table shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error merging tables: {e}")
        raise CustomException("Failed to merge all tables", e)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Missing Values, Encoding, Column Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def drop_high_missing_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float = 0.80
) -> tuple:
    """Drop columns where more than `threshold` % of train values are missing."""
    try:
        missing_pct  = train.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        cols_to_drop = [c for c in cols_to_drop if c not in ['TARGET', 'SK_ID_CURR']]
        logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing.")
        train = train.drop(columns=cols_to_drop, errors='ignore')
        test  = test.drop(columns=[c for c in cols_to_drop if c in test.columns], errors='ignore')
        return train, test, cols_to_drop

    except Exception as e:
        logger.error(f"Error dropping high-missing columns: {e}")
        raise CustomException("Failed to drop high-missing columns", e)


def impute_missing_values(
    train: pd.DataFrame,
    test: pd.DataFrame,
    saved_medians: dict = None
) -> tuple:
    """
    Impute numeric columns with median (fit on train only).
    Impute categorical columns with 'Unknown'.
    saved_medians: pass pre-saved values during inference to avoid recomputing.
    """
    try:
        num_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                    if c not in ['TARGET', 'SK_ID_CURR']]
        cat_cols = train.select_dtypes(include=['object']).columns.tolist()

        if saved_medians is None:
            medians = train[num_cols].median().to_dict()
            logger.info(f"Computed medians from training data ({len(medians)} columns).")
        else:
            medians = saved_medians
            logger.info("Using saved medians for imputation.")

        train[num_cols] = train[num_cols].fillna(pd.Series(medians))
        test_num_cols   = [c for c in num_cols if c in test.columns]
        test[test_num_cols] = test[test_num_cols].fillna(
            pd.Series({k: v for k, v in medians.items() if k in test_num_cols})
        )

        for col in cat_cols:
            train[col] = train[col].fillna('Unknown')
            if col in test.columns:
                test[col] = test[col].fillna('Unknown')

        logger.info("Missing value imputation complete.")
        return train, test, medians

    except Exception as e:
        logger.error(f"Error imputing missing values: {e}")
        raise CustomException("Failed to impute missing values", e)


def encode_categorical_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    saved_encoders: dict = None
) -> tuple:
    """
    Encode categorical columns:
      <=2 unique  → LabelEncoder
      3–15 unique → One-Hot
      >15 unique  → LabelEncoder (high cardinality)
    saved_encoders: pass pre-saved encoders during inference.
    """
    try:
        cat_cols = train.select_dtypes(include=['object']).columns.tolist()

        if saved_encoders is None:
            encoders = {}
            le_cols, ohe_cols, hc_cols = [], [], []

            for col in cat_cols:
                n = train[col].nunique()
                if n <= 2:
                    le_cols.append(col)
                elif n <= 15:
                    ohe_cols.append(col)
                else:
                    hc_cols.append(col)

            logger.info(f"Encoding — Binary: {len(le_cols)} | OHE: {len(ohe_cols)} | High-card: {len(hc_cols)}")

            le = LabelEncoder()
            for col in le_cols + hc_cols:
                combined = pd.concat([train[col], test[col]]).astype(str)
                le.fit(combined)
                train[col] = le.transform(train[col].astype(str))
                test[col]  = le.transform(test[col].astype(str))
                encoders[col] = {'type': 'label', 'classes': le.classes_.tolist()}

            if ohe_cols:
                train = pd.get_dummies(train, columns=ohe_cols, dummy_na=False)
                test  = pd.get_dummies(test,  columns=ohe_cols, dummy_na=False)
                encoders['__ohe_cols__'] = ohe_cols

            encoders['__le_cols__'] = le_cols
            encoders['__hc_cols__'] = hc_cols

        else:
            logger.info("Using saved encoders for encoding.")
            encoders = saved_encoders
            le_cols  = encoders.get('__le_cols__', [])
            hc_cols  = encoders.get('__hc_cols__', [])
            ohe_cols = encoders.get('__ohe_cols__', [])

            le = LabelEncoder()
            for col in le_cols + hc_cols:
                if col in train.columns:
                    le.classes_ = np.array(encoders[col]['classes'])
                    train[col]  = train[col].astype(str).map(
                        lambda x, c=col: x if x in encoders[c]['classes'] else encoders[c]['classes'][0]
                    )
                    train[col] = le.transform(train[col])

            if ohe_cols:
                train = pd.get_dummies(train, columns=ohe_cols, dummy_na=False)

        # Align test columns to train
        train_cols = [c for c in train.columns if c not in ['TARGET', 'SK_ID_CURR']]
        for col in train_cols:
            if col not in test.columns:
                test[col] = 0

        logger.info("Categorical encoding complete.")
        return train, test, encoders

    except Exception as e:
        logger.error(f"Error encoding categorical columns: {e}")
        raise CustomException("Failed to encode categorical columns", e)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace special characters in column names (LightGBM requirement)."""
    try:
        new_cols = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
        seen, final_cols = {}, []
        for col in new_cols:
            if col in seen:
                seen[col] += 1
                final_cols.append(f'{col}_{seen[col]}')
            else:
                seen[col] = 0
                final_cols.append(col)
        df.columns = final_cols
        return df
    except Exception as e:
        logger.error(f"Error cleaning column names: {e}")
        raise CustomException("Failed to clean column names", e)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Save / Load Preprocessor Artifacts
# ─────────────────────────────────────────────────────────────────────────────

def save_preprocessor_artifacts(
    medians: dict,
    encoders: dict,
    feature_columns: list
) -> None:
    """Save medians, encoders, and feature column list to artifacts/models/."""
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(medians,         MEDIANS_FILE_PATH)
        joblib.dump(encoders,        ENCODERS_FILE_PATH)
        joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
        logger.info(f"Preprocessor artifacts saved to '{MODELS_DIR}/'")
        logger.info(f"  {MEDIANS_FILE_PATH}")
        logger.info(f"  {ENCODERS_FILE_PATH}")
        logger.info(f"  {FEATURE_COLUMNS_PATH}")
    except Exception as e:
        logger.error(f"Error saving preprocessor artifacts: {e}")
        raise CustomException("Failed to save preprocessor artifacts", e)


def load_preprocessor_artifacts() -> tuple:
    """Load medians, encoders, and feature columns from artifacts/models/."""
    try:
        for path in [MEDIANS_FILE_PATH, ENCODERS_FILE_PATH, FEATURE_COLUMNS_PATH]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Artifact not found: {path}")
        medians         = joblib.load(MEDIANS_FILE_PATH)
        encoders        = joblib.load(ENCODERS_FILE_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        logger.info("Preprocessor artifacts loaded successfully.")
        return medians, encoders, feature_columns
    except Exception as e:
        logger.error(f"Error loading preprocessor artifacts: {e}")
        raise CustomException("Failed to load preprocessor artifacts", e)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Main DataPreprocessing Class (Batch Mode)
# ─────────────────────────────────────────────────────────────────────────────

class DataPreprocessing:
    """
    Batch preprocessing pipeline — used during model training and retraining.

    Flow:
      1. Load processed files from artifacts/processed/
      2. Engineer features on application table
      3. Aggregate all secondary tables
      4. Merge into one master feature matrix
      5. Drop high-missing columns
      6. Impute missing values  (fit on train, apply to test)
      7. Encode categoricals    (fit on train+test, apply to both)
      8. Clean column names
      9. Save all .pkl artifacts to artifacts/models/
         and final feature CSVs to artifacts/processed/
    """

    def __init__(self):
        self.config = read_yaml(CONFIG_PATH)
        logger.info("DataPreprocessing initialised.")

    def run(self) -> tuple:
        """Run full batch preprocessing. Returns (X_train, y_train, X_test, y_test)."""
        try:
            logger.info("=" * 60)
            logger.info("Starting Data Preprocessing Pipeline")
            logger.info("=" * 60)

            target_col = self.config["data_ingestion"]["target_column"]
            id_col     = self.config["data_ingestion"]["id_column"]

            # Step 1 — Load processed files
            logger.info("Loading processed files...")
            train   = pd.read_csv(TRAIN_FILE_PATH)
            test    = pd.read_csv(TEST_FILE_PATH)
            bureau  = pd.read_csv(PROCESSED_BUREAU)
            bur_bal = pd.read_csv(PROCESSED_BUREAU_BALANCE)
            prev    = pd.read_csv(PROCESSED_PREVIOUS_APP)
            pos     = pd.read_csv(PROCESSED_POS_CASH)
            cc      = pd.read_csv(PROCESSED_CREDIT_CARD)
            inst    = pd.read_csv(PROCESSED_INSTALLMENTS)
            logger.info(f"Train: {train.shape} | Test: {test.shape}")

            # Step 2 — Application feature engineering
            train = engineer_application_features(train)
            test  = engineer_application_features(test)

            # Step 3 — Aggregate secondary tables
            bureau_agg = aggregate_bureau(bureau, bur_bal)
            prev_agg   = aggregate_previous_application(prev)
            pos_agg    = aggregate_pos_cash(pos)
            cc_agg     = aggregate_credit_card(cc)
            inst_agg   = aggregate_installments(inst)

            # Step 4 — Merge into master table
            train = merge_all_tables(train, bureau_agg, prev_agg, pos_agg, cc_agg, inst_agg)
            test  = merge_all_tables(test,  bureau_agg, prev_agg, pos_agg, cc_agg, inst_agg)

            # Step 5 — Drop high-missing columns
            train, test, _ = drop_high_missing_columns(train, test, threshold=0.80)

            # Step 6 — Impute missing values
            train, test, medians = impute_missing_values(train, test)

            # Step 7 — Encode categoricals
            train, test, encoders = encode_categorical_columns(train, test)

            # Step 8 — Clean column names
            train = clean_column_names(train)
            test  = clean_column_names(test)

            # Step 9 — Align columns
            feature_cols = [c for c in train.columns if c not in [target_col, id_col]]
            for col in feature_cols:
                if col not in test.columns:
                    test[col] = 0
            test = test[[id_col] + feature_cols]

            # Step 10 — Split X / y
            X_train = train[feature_cols]
            y_train = train[target_col]
            X_test  = test[feature_cols]
            y_test  = test[target_col] if target_col in test.columns else None

            # Step 11 — Save all .pkl artifacts to artifacts/models/
            save_preprocessor_artifacts(medians, encoders, feature_cols)

            # Save final feature CSVs
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            train[feature_cols + [target_col, id_col]].to_csv(
                os.path.join(PROCESSED_DIR, "train_features.csv"), index=False
            )
            test[[id_col] + feature_cols].to_csv(
                os.path.join(PROCESSED_DIR, "test_features.csv"), index=False
            )

            logger.info("=" * 60)
            logger.info("Data Preprocessing Pipeline completed successfully.")
            logger.info(f"  X_train  : {X_train.shape}")
            logger.info(f"  X_test   : {X_test.shape}")
            logger.info(f"  Features : {len(feature_cols)}")
            logger.info(f"  PKL files: {MODELS_DIR}/")
            logger.info("=" * 60)

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Data Preprocessing Pipeline failed: {e}")
            raise CustomException("Data Preprocessing Pipeline failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocessing = DataPreprocessing()
    X_train, y_train, X_test, y_test = preprocessing.run()
    print(f"\nX_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
