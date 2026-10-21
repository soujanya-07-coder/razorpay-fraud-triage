
import logging
import time

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from prepare import add_features
from train import build_final_model
from casenote import run_case


# ============================================================
# 1. APP AND MODEL INITIALIZATION
# ============================================================

app = FastAPI(title="Merchant Risk Desk - Fraud Triage Service")

# Train once when this service process imports app.py.
# Keep the fitted model in memory; do not save it to disk.
MODEL_BUNDLE = build_final_model()


# ============================================================
# 2. REQUEST MODELS
# ============================================================

class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: float | None = None
    is_new_device: int
    account_age_days: int


class CaseIn(Txn):
    question: str


# ============================================================
# 3. COUNTERS AND LOGGING
# ============================================================

METRICS = {
    "requests_total": 0,
    "success_total": 0,
    "error_total": 0,
    "human_review_total": 0,
}

LATENCY_TOTAL_MS = 0.0

logger = logging.getLogger("triage_service")
logger.setLevel(logging.INFO)
logger.propagate = False

# Ensure a single file handler, even if the module is reloaded.
if not logger.handlers:
    file_handler = logging.FileHandler(
        "service.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"
        )
    )
    logger.addHandler(file_handler)


@app.middleware("http")
async def observe_post_requests(request: Request, call_next):
    """
    Observe every POST response, including framework 422 responses.
    GET requests do not change the counters or create log lines.
    """

    global LATENCY_TOTAL_MS

    started = time.perf_counter()

    request.state.info = {
        "warning": False,
        "safe_default": False,
        "guardrail_override": False,
    }

    try:
        response = await call_next(request)
    except Exception:
        # Do not leak a stack trace or terminate the service.
        response = JSONResponse(
            status_code=500,
            content={"error": "internal service error"},
        )

    latency_ms = (time.perf_counter() - started) * 1000.0

    if (
        request.method == "POST"
        and request.url.path in {"/score", "/case"}
    ):
        status = response.status_code
        info = request.state.info

        METRICS["requests_total"] += 1

        if status < 400:
            METRICS["success_total"] += 1
        else:
            METRICS["error_total"] += 1

        # Rejected requests never increase human-review counters.
        if status < 400 and info.get("human_review") is True:
            METRICS["human_review_total"] += 1

        LATENCY_TOTAL_MS += latency_ms

        if status >= 500:
            level = logging.ERROR
        elif (
            status >= 400
            or info.get("warning", False)
            or info.get("safe_default", False)
            or info.get("guardrail_override", False)
        ):
            level = logging.WARNING
        else:
            level = logging.INFO

        message = (
            f"endpoint={request.url.path} "
            f"status_code={status} "
            f"latency_ms={latency_ms:.2f}"
        )

        if info.get("txn_id") is not None:
            message += f" txn_id={info['txn_id']}"

        if info.get("fraud_probability") is not None:
            message += (
                f" fraud_probability="
                f"{info['fraud_probability']:.4f}"
            )

        logger.log(level, message)

    return response


# ============================================================
# 4. INPUT BUSINESS-RULE VALIDATION
# ============================================================

KNOWN_CATEGORIES = {
    "Grocery",
    "Electronics",
    "Travel",
    "Food Delivery",
    "Fashion",
    "Utilities",
}


def validate_business_rules(txn: Txn):
    """Return a 400 JSON response for impossible business values."""

    if not (1 <= txn.amount_inr <= 1_000_000):
        return {
            "error": "amount_inr must be between 1 and 1000000",
            "field": "amount_inr",
        }

    if not (0 <= txn.hour_of_day <= 23):
        return {
            "error": "hour_of_day must be between 0 and 23",
            "field": "hour_of_day",
        }

    if txn.txn_count_last_24h < 0:
        return {
            "error": "txn_count_last_24h cannot be negative",
            "field": "txn_count_last_24h",
        }

    if txn.avg_amount_30d_inr <= 0:
        return {
            "error": "avg_amount_30d_inr must be greater than zero",
            "field": "avg_amount_30d_inr",
        }

    category = txn.merchant_category.strip().title()

    if category not in KNOWN_CATEGORIES:
        return {
            "error": "Unknown merchant_category",
            "field": "merchant_category",
        }

    return None


# ============================================================
# 5. SHARED SCORING PATH
# ============================================================

RAW_NUMERIC_COLUMNS = [
    "amount_inr",
    "hour_of_day",
    "txn_count_last_24h",
    "avg_amount_30d_inr",
    "distance_from_home_km",
    "is_new_device",
    "account_age_days",
]

ENGINEERED_COLUMNS = [
    "amount_vs_avg_ratio",
    "is_night",
    "high_value",
]

CATEGORICAL_COLUMNS = [
    "merchant_category",
    "city",
    "payment_mode",
]


def score_transaction(txn: Txn, request: Request):
    """Validate and score one transaction using the trained model."""

    info = request.state.info

    # The payload has passed Pydantic's type validation at this point.
    info["txn_id"] = txn.txn_id

    validation_error = validate_business_rules(txn)

    if validation_error is not None:
        return JSONResponse(
            status_code=400,
            content=validation_error,
        )

    row = txn.model_dump()

    # Apply the same category/city normalization used in Part A.
    row["merchant_category"] = (
        row["merchant_category"].strip().title()
    )
    row["city"] = row["city"].strip().title()

    frame = pd.DataFrame([row])

    # Use Part A's required feature-engineering function.
    frame = add_features(frame)

    # Use the training-only median learned in Part B.
    frame["distance_from_home_km"] = (
        frame["distance_from_home_km"].fillna(
            MODEL_BUNDLE["median"]
        )
    )

    numeric_features = frame[
        RAW_NUMERIC_COLUMNS + ENGINEERED_COLUMNS
    ]

    categorical_features = pd.get_dummies(
        frame[CATEGORICAL_COLUMNS],
        dtype=int,
    )

    features = pd.concat(
        [numeric_features, categorical_features],
        axis=1,
    )

    # Ensure identical feature names and order to training.
    features = features.reindex(
        columns=MODEL_BUNDLE["columns"],
        fill_value=0,
    )

    model = MODEL_BUNDLE["model"]

    probability = float(model.predict_proba(features)[0, 1])
    flagged = probability >= MODEL_BUNDLE["threshold"]

    decision = (
        "hold_for_review" if flagged else "approve"
    )

    guardrail_override = False
    large_payment_hold = False

    # Extra business guardrail for an unflagged, very large payment
    # from a device first seen recently.
    if (
        not flagged
        and txn.amount_inr >= 100_000
        and txn.is_new_device == 1
    ):
        decision = "hold_for_review"
        guardrail_override = True
        large_payment_hold = True

    human_review = decision == "hold_for_review"

    info.update({
        "fraud_probability": probability,
        "human_review": human_review,
        "guardrail_override": guardrail_override,
        "large_payment_hold": large_payment_hold,
        "warning": guardrail_override,
    })

    return {
        "txn_id": txn.txn_id,
        "fraud_probability": round(probability, 4),
        "flagged": bool(flagged),
        "decision": decision,
        "human_review": bool(human_review),
        "guardrail_override": guardrail_override,
    }


# ============================================================
# 6. GET /health
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# 7. POST /score
# ============================================================

@app.post("/score")
def score(txn: Txn, request: Request):
    result = score_transaction(txn, request)

    if isinstance(result, JSONResponse):
        return result

    return {
        "txn_id": result["txn_id"],
        "fraud_probability": result["fraud_probability"],
        "flagged": result["flagged"],
        "decision": result["decision"],
        "human_review": result["human_review"],
    }


# ============================================================
# 8. POST /case
# ============================================================

@app.post("/case")
def case_endpoint(case_input: CaseIn, request: Request):
    scored = score_transaction(case_input, request)

    if isinstance(scored, JSONResponse):
        return scored

    probability = scored["fraud_probability"]
    flagged = scored["flagged"]

    # If no fraud alert is raised, no case note is generated.
    if not flagged:
        return {
            "txn_id": scored["txn_id"],
            "fraud_probability": probability,
            "flagged": False,
            "decision": scored["decision"],
            "human_review": scored["human_review"],
            "case_note": None,
            "attempts": 0,
            "safe_default": False,
            "guardrail_override": scored["guardrail_override"],
        }

    # Only flagged transactions reach Part C's case pipeline.
    try:
        result = run_case(
            txn_id=case_input.txn_id,
            fraud_probability=probability,
            amount_inr=case_input.amount_inr,
            question=case_input.question,
            max_attempts=3,
        )
    except ConnectionError:
        return JSONResponse(
            status_code=503,
            content={
                "error": (
                    "case-note service temporarily unavailable, "
                    "try again later"
                )
            },
        )

    # An out-of-scope query cannot produce a policy-grounded note.
    if result["status"] == "no_policy_match":
        decision = "hold_for_review"
        case_note = None
        attempts = 0
        safe_default = False
        guardrail_override = False
    else:
        case_note = result["case_note"]
        attempts = result["attempts"]
        safe_default = result["safe_default"]
        guardrail_override = result["guardrail_override"]

        # A flagged payment must never be auto-approved.
        # This service guardrail also applies below probability 0.5.
        if (
            case_note is not None
            and case_note["recommended_action"] == "approve"
        ):
            case_note["recommended_action"] = "hold_for_review"
            guardrail_override = True

        decision = case_note["recommended_action"]

    # Both holding for review and block-and-notify require a person.
    human_review = (
        decision in {"hold_for_review", "block_and_notify"}
        or safe_default
    )

    info = request.state.info
    info.update({
        "fraud_probability": probability,
        "human_review": bool(human_review),
        "safe_default": bool(safe_default),
        "guardrail_override": bool(
            guardrail_override
        ),
        "warning": bool(
            safe_default or guardrail_override
        ),
    })

    return {
        "txn_id": scored["txn_id"],
        "fraud_probability": probability,
        "flagged": True,
        "decision": decision,
        "human_review": bool(human_review),
        "case_note": case_note,
        "attempts": attempts,
        "safe_default": bool(safe_default),
        "guardrail_override": bool(guardrail_override),
    }


# ============================================================
# 9. GET /metrics
# ============================================================

@app.get("/metrics")
def metrics():
    total = METRICS["requests_total"]
    success = METRICS["success_total"]
    errors = METRICS["error_total"]
    reviews = METRICS["human_review_total"]

    if total:
        success_rate = round(success / total, 4)
        error_rate = round(errors / total, 4)
        human_review_rate = round(reviews / total, 4)
        average_latency = round(
            LATENCY_TOTAL_MS / total,
            2,
        )
    else:
        success_rate = 0.0
        error_rate = 0.0
        human_review_rate = 0.0
        average_latency = 0.0

    return {
        "requests_total": total,
        "success_total": success,
        "error_total": errors,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "avg_latency_ms": average_latency,
        "human_review_total": reviews,
        "human_review_rate": human_review_rate,
    }
