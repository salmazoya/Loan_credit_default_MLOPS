# Loan Credit Default Risk — MLOps Pipeline

An end-to-end MLOps pipeline for predicting loan default risk using the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. A LightGBM model is trained on 8 data sources, served via a FastAPI REST API, monitored with Evidently AI, and tracked with MLflow.

---

## Results

| Metric | Value |
|--------|-------|
| OOF AUC | 0.785 |
| CV AUC | 0.785 ± 0.003 |
| Classification Threshold | 0.66 |
| Features Engineered | 256 |
| Training Samples | ~246,000 |

**Top features:** `CREDIT_TERM`, `DAYS_BIRTH`, `EXT_SOURCE_MEAN`, `pos_cnt_instalment_future_mean`, `AMT_ANNUITY`

---

## Project Structure

```
loan_credit_default_mlops/
│
├── config/
│   ├── config.yaml              # All hyperparameters, GCP settings, MLflow config
│   └── paths_config.py          # Centralised file paths
│
├── src/
│   ├── data_ingestion.py        # Loads & validates 8 raw CSVs, splits train/test
│   ├── data_preprocessing.py    # Feature engineering, encoding, saves .pkl artifacts
│   ├── train.py                 # LightGBM CV training + MLflow logging + model promotion
│   ├── predict.py               # Single applicant inference (loads pkl files)
│   ├── monitor.py               # Evidently AI drift & performance reports
│   ├── gcp_sync.py              # Push/pull artifacts to GCP Cloud Storage
│   ├── logger.py                # Centralised logging
│   └── custom_exception.py      # Custom exception with file/line info
│
├── tests/
│   ├── conftest.py
│   ├── test_predict.py          # Artifact loading + inference tests
│   ├── test_feature_engineering.py  # Feature logic tests (no pkl needed)
│   └── test_api.py              # FastAPI endpoint tests
│
├── notebook/
│   ├── eda_home_credit.ipynb    # Exploratory data analysis
│   ├── feature_engineering.ipynb
│   └── modeling_lgbm.ipynb      # Model development & threshold tuning
│
├── artifacts/                   # Generated at runtime — stored in GCP
│   ├── raw/                     # Raw CSVs from Kaggle
│   ├── processed/               # train.csv, test.csv
│   ├── models/                  # lgbm_model.pkl, encoders.pkl, medians.pkl,
│   │                            # feature_columns.pkl, model_meta.json
│   ├── plots/                   # roc_curve, confusion_matrix, feature_importance
│   └── monitoring/              # Evidently HTML reports
│
├── main.py                      # FastAPI app
├── Dockerfile
├── docker-entrypoint.sh
├── requirements.txt
└── .gitignore
```

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │         Data Sources         │
                        │  8 CSV files from Kaggle /   │
                        │     GCP Cloud Storage        │
                        └──────────────┬──────────────┘
                                       │
                              data_ingestion.py
                          (validate, split 80/20)
                                       │
                          data_preprocessing.py
                     (feature engineering, encode, impute)
                        saves: medians.pkl, encoders.pkl
                               feature_columns.pkl
                                       │
                                   train.py
                          (5-fold LightGBM CV training)
                          saves: lgbm_model.pkl
                          logs everything to MLflow
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
               monitor.py                          gcp_sync.py
          (Evidently drift reports)            (push pkl files to
           logged to MLflow                      GCP bucket)
                                                      │
                                            ┌─────────┴──────────┐
                                            │    FastAPI (main.py) │
                                            │  POST /predict       │
                                            │  POST /predict/batch │
                                            │  GET  /health        │
                                            │  GET  /model-info    │
                                            └────────────────────-─┘
```

---

## Pipeline Flow

### Batch Retraining (every 3 months)
```bash
python -m src.train
```
1. Runs `DataIngestion` — loads and validates all 8 raw files
2. Runs `DataPreprocessing` — engineers 256 features, saves `.pkl` artifacts
3. Trains LightGBM with 5-fold stratified CV
4. Logs all params, metrics, and plots to MLflow
5. Promotes new model only if AUC improves by ≥ 0.001 over current
6. Runs Evidently drift monitoring automatically
7. Saves model artifacts to `artifacts/models/`

### Real-time Inference
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```
Send a POST request to `/predict` with applicant JSON → get back default probability, APPROVE/REJECT decision, and risk level.

---

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/loan_credit_default_mlops.git
cd loan_credit_default_mlops
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull model artifacts from GCP
```bash
python -m src.gcp_sync --pull --group models
```

### 3. Run the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```
Open `http://127.0.0.1:8080/docs` for the interactive Swagger UI.

### 4. Test a prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "application": {
      "SK_ID_CURR": 999999,
      "AMT_INCOME_TOTAL": 135000,
      "AMT_CREDIT": 450000,
      "AMT_ANNUITY": 20250,
      "DAYS_BIRTH": -12000,
      "DAYS_EMPLOYED": -2000,
      "EXT_SOURCE_1": 0.54,
      "EXT_SOURCE_2": 0.71,
      "EXT_SOURCE_3": 0.62
    }
  }'
```

**Response:**
```json
{
  "applicant_id": 999999,
  "default_probability": 0.1718,
  "decision": "APPROVE",
  "risk_level": "LOW",
  "threshold_used": 0.66,
  "latency_ms": 45.2
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check — confirms artifacts are loaded |
| `GET` | `/model-info` | Current model AUC, threshold, MLflow run ID |
| `POST` | `/predict` | Single applicant prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 100 applicants) |
| `GET` | `/docs` | Swagger UI |

---

## MLflow Tracking

```bash
mlflow ui --port 5001
```

Open `http://127.0.0.1:5001` to view:
- Per-fold AUC scores
- OOF and test set metrics
- ROC curve, confusion matrix, feature importance plots
- Drift monitoring reports
- Model promotion history

---

## Drift Monitoring

```bash
python -m src.monitor
```

Generates three Evidently AI reports comparing training data (reference) against current data:

| Report | What it checks |
|--------|---------------|
| `data_drift_report.html` | Feature distribution shifts |
| `target_drift_report.html` | Default rate shift |
| `model_performance_report.html` | AUC / F1 degradation |

```bash
open artifacts/monitoring/data_drift_report.html
```

Reports are also logged as MLflow artifacts automatically.

---

## GCP Sync

```bash
# Push artifacts to GCP after training
python -m src.gcp_sync --push

# Pull models on a new machine
python -m src.gcp_sync --pull --group models

# Groups: raw | processed | models | all
```

Bucket: `mlops_b1` (configured in `config/config.yaml`)

---

## Docker

```bash
# Build
docker build -t loan-default-api .

# Run (mounts GCP credentials)
docker run -p 8080:8080 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/gcp_key.json \
  -v $(pwd)/credentials:/app/credentials \
  loan-default-api
```

The container pulls the latest model from GCP on startup — no rebuild needed when the model is updated.

---

## Tests

```bash
pip install pytest pytest-cov httpx
pytest tests/ -v --cov=src --cov=main --cov-report=term-missing
```

| Test File | What it covers |
|-----------|---------------|
| `test_feature_engineering.py` | Derived feature logic, edge cases |
| `test_predict.py` | Artifact loading, preprocessing, inference |
| `test_api.py` | All FastAPI endpoints |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | LightGBM |
| API | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Drift Monitoring | Evidently AI |
| Artifact Storage | GCP Cloud Storage |
| Containerisation | Docker |
| Testing | Pytest |
| Language | Python 3.11 |

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — 8 relational tables covering application data, bureau credit history, previous loans, POS cash balance, credit card balance, and installment payments. Target variable: `1` = client defaulted, `0` = loan repaid.
