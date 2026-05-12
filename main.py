"""
main.py — FastAPI Loan Default Risk API
=========================================
Serves real-time loan default predictions via a REST API.

Endpoints:
  GET  /health        → liveness check + artifact status
  GET  /model-info    → current model metadata (AUC, threshold, run_id, etc.)
  POST /predict       → single applicant prediction
  POST /predict/batch → list of applicants (up to 100)

Run locally:
  uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Example curl:
  curl -X POST http://localhost:8080/predict \\
       -H "Content-Type: application/json" \\
       -d @sample_applicant.json
"""

import os
import json
import time
import traceback
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import predict, load_artifacts
from src.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Loan Default Risk API",
    description=(
        "Real-time loan default prediction powered by a LightGBM model "
        "trained on the Home Credit Default Risk dataset. "
        "Returns default probability, decision (APPROVE/REJECT), and risk level."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ApplicationData(BaseModel):
    """Core applicant fields from the application table."""
    SK_ID_CURR: Optional[int] = Field(None, description="Applicant ID")

    # Required financial fields
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None

    # Required date fields (negative integers, days before application)
    DAYS_BIRTH: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None

    # External scores (0–1 range)
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None

    # Categorical fields
    NAME_CONTRACT_TYPE: Optional[str] = None
    CODE_GENDER: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None

    # Other common fields
    CNT_FAM_MEMBERS: Optional[float] = None
    CNT_CHILDREN: Optional[float] = None
    REGION_RATING_CLIENT: Optional[float] = None
    REGION_RATING_CLIENT_W_CITY: Optional[float] = None
    HOUR_APPR_PROCESS_START: Optional[float] = None
    REG_REGION_NOT_WORK_REGION: Optional[float] = None
    LIVE_REGION_NOT_WORK_REGION: Optional[float] = None
    REG_CITY_NOT_WORK_CITY: Optional[float] = None
    LIVE_CITY_NOT_WORK_CITY: Optional[float] = None

    # Flags (0/1)
    FLAG_MOBIL: Optional[float] = None
    FLAG_EMP_PHONE: Optional[float] = None
    FLAG_WORK_PHONE: Optional[float] = None
    FLAG_CONT_MOBILE: Optional[float] = None
    FLAG_PHONE: Optional[float] = None
    FLAG_EMAIL: Optional[float] = None

    # Document flags
    FLAG_DOCUMENT_3: Optional[float] = None
    FLAG_DOCUMENT_6: Optional[float] = None
    FLAG_DOCUMENT_8: Optional[float] = None

    # Social circle
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = None

    # Enquiries
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = None

    class Config:
        extra = "allow"   # accept any additional columns without error


class BureauSummary(BaseModel):
    """Pre-aggregated bureau credit history features."""
    bur_count: Optional[float] = None
    bur_active_count: Optional[float] = None
    bur_credit_sum_total: Optional[float] = None
    bur_credit_sum_debt_total: Optional[float] = None
    bur_credit_sum_overdue: Optional[float] = None
    bur_overdue_ratio: Optional[float] = None
    bur_days_credit_mean: Optional[float] = None
    bur_days_credit_enddate_mean: Optional[float] = None

    class Config:
        extra = "allow"


class PreviousLoansSummary(BaseModel):
    """Pre-aggregated previous loan history features."""
    prev_count: Optional[float] = None
    prev_approval_rate: Optional[float] = None
    inst_pct_late: Optional[float] = None
    inst_days_late_mean: Optional[float] = None
    pos_dpd_max: Optional[float] = None
    pos_cnt_instalment_future_mean: Optional[float] = None
    cc_utilization_mean: Optional[float] = None
    cc_balance_mean: Optional[float] = None

    class Config:
        extra = "allow"


class PredictRequest(BaseModel):
    """Full prediction request body."""
    application: ApplicationData
    bureau_summary: Optional[BureauSummary] = None
    previous_loans_summary: Optional[PreviousLoansSummary] = None
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override classification threshold (default: 0.66)"
    )

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictResponse(BaseModel):
    """Prediction result."""
    applicant_id: Optional[int]
    default_probability: float = Field(..., description="Probability of loan default (0.0–1.0)")
    decision: str             = Field(..., description="APPROVE or REJECT")
    risk_level: str           = Field(..., description="LOW / MEDIUM / HIGH")
    threshold_used: float
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")


class BatchPredictRequest(BaseModel):
    """Batch prediction — up to 100 applicants."""
    applicants: List[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    approved: int
    rejected: int
    avg_default_probability: float


class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool
    model_file_exists: bool
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

MODEL_META_PATH = os.path.join("artifacts", "models", "model_meta.json")
MODEL_PKL_PATH  = os.path.join("artifacts", "models", "lgbm_model.pkl")


def _load_model_meta() -> dict:
    if os.path.exists(MODEL_META_PATH):
        with open(MODEL_META_PATH) as f:
            return json.load(f)
    return {}


def _request_to_dict(req: PredictRequest) -> dict:
    """Convert Pydantic PredictRequest into the dict format predict() expects."""
    d = {
        "application": {
            k: v for k, v in req.application.model_dump().items()
            if v is not None
        }
    }
    if req.bureau_summary:
        d["bureau_summary"] = {
            k: v for k, v in req.bureau_summary.model_dump().items()
            if v is not None
        }
    if req.previous_loans_summary:
        d["previous_loans_summary"] = {
            k: v for k, v in req.previous_loans_summary.model_dump().items()
            if v is not None
        }
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Liveness check. Returns whether the model artifacts are accessible.
    Use this for container health probes.
    """
    model_exists = os.path.exists(MODEL_PKL_PATH)
    artifacts_ok = False
    msg = "Model artifacts not found — run train.py first."

    if model_exists:
        try:
            load_artifacts()   # will raise if any pkl is missing
            artifacts_ok = True
            msg = "All artifacts loaded successfully."
        except Exception as e:
            msg = f"Artifact load error: {str(e)}"

    return HealthResponse(
        status="ok" if artifacts_ok else "degraded",
        artifacts_loaded=artifacts_ok,
        model_file_exists=model_exists,
        message=msg,
    )


@app.get("/model-info", tags=["System"])
def model_info():
    """
    Returns current model metadata: AUC, threshold, MLflow run_id, training date.
    Data is read from artifacts/models/model_meta.json (written by train.py).
    """
    meta = _load_model_meta()
    if not meta:
        raise HTTPException(
            status_code=404,
            detail="No model_meta.json found. Run train.py to train and register a model."
        )
    return {"model_metadata": meta}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_single(request: PredictRequest):
    """
    Predict loan default risk for a single applicant.

    - **application**: Required. Core applicant data (income, credit, age, etc.)
    - **bureau_summary**: Optional. Pre-aggregated bureau credit history.
    - **previous_loans_summary**: Optional. Pre-aggregated previous loan stats.
    - **threshold**: Optional override for classification cutoff (default 0.66).

    Returns default probability, APPROVE/REJECT decision, and LOW/MEDIUM/HIGH risk level.
    """
    t0 = time.time()

    try:
        applicant_dict = _request_to_dict(request)
        threshold = request.threshold if request.threshold is not None else 0.66

        result = predict(applicant_dict, threshold=threshold)

        latency = round((time.time() - t0) * 1000, 2)
        logger.info(
            f"/predict | ID={result['applicant_id']} | "
            f"prob={result['default_probability']} | "
            f"decision={result['decision']} | {latency}ms"
        )

        return PredictResponse(
            applicant_id=result["applicant_id"],
            default_probability=result["default_probability"],
            decision=result["decision"],
            risk_level=result["risk_level"],
            threshold_used=result["threshold_used"],
            latency_ms=latency,
        )

    except FileNotFoundError as e:
        logger.error(f"Missing artifact: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts not found. "
                "Please run 'python -m src.train' to train the model first."
            )
        )

    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """
    Predict loan default risk for a batch of applicants (up to 100).

    Returns individual predictions plus summary statistics:
    total count, approved/rejected counts, and average default probability.
    """
    results = []
    errors  = []

    for i, applicant_req in enumerate(request.applicants):
        t0 = time.time()
        try:
            applicant_dict = _request_to_dict(applicant_req)
            threshold = applicant_req.threshold if applicant_req.threshold is not None else 0.66
            result = predict(applicant_dict, threshold=threshold)
            latency = round((time.time() - t0) * 1000, 2)

            results.append(PredictResponse(
                applicant_id=result["applicant_id"],
                default_probability=result["default_probability"],
                decision=result["decision"],
                risk_level=result["risk_level"],
                threshold_used=result["threshold_used"],
                latency_ms=latency,
            ))
        except Exception as e:
            logger.error(f"Batch item {i} failed: {e}")
            errors.append({"index": i, "error": str(e)})

    if not results and errors:
        raise HTTPException(status_code=500, detail=f"All predictions failed: {errors}")

    approved = sum(1 for r in results if r.decision == "APPROVE")
    rejected = len(results) - approved
    avg_prob = round(sum(r.default_probability for r in results) / len(results), 4) if results else 0.0

    logger.info(
        f"/predict/batch | total={len(results)} | "
        f"approved={approved} | rejected={rejected} | avg_prob={avg_prob}"
    )

    return BatchPredictResponse(
        results=results,
        total=len(results),
        approved=approved,
        rejected=rejected,
        avg_default_probability=avg_prob,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handler — catch anything Pydantic/FastAPI doesn't
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
