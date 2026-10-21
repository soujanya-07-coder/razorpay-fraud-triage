
import json
import sys
import time

import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 10

BENIGN = {
    "merchant_category": "Grocery",
    "city": "Pune",
    "payment_mode": "UPI",
    "amount_inr": 1500,
    "hour_of_day": 13,
    "txn_count_last_24h": 0,
    "avg_amount_30d_inr": 1600,
    "distance_from_home_km": 3.0,
    "is_new_device": 0,
    "account_age_days": 900,
}

RISKY = {
    "merchant_category": "Electronics",
    "city": "Mumbai",
    "payment_mode": "Card",
    "amount_inr": 45000,
    "hour_of_day": 2,
    "txn_count_last_24h": 9,
    "avg_amount_30d_inr": 3000,
    "distance_from_home_km": 60.0,
    "is_new_device": 1,
    "account_age_days": 40,
}

BIG = {
    **BENIGN,
    "merchant_category": "Utilities",
    "payment_mode": "NetBanking",
    "amount_inr": 150000,
    "avg_amount_30d_inr": 250000,
    "distance_from_home_km": 0.0,
    "is_new_device": 1,
    "account_age_days": 2200,
}


def with_id(payload, txn_id):
    return {"txn_id": txn_id, **payload}


def check_health(data):
    return data.get("status") == "ok"


def check_benign(data):
    return (
        data.get("flagged") is False
        and data.get("decision") == "approve"
        and data.get("human_review") is False
    )


def check_risky(data):
    return (
        data.get("flagged") is True
        and data.get("decision") == "hold_for_review"
        and data.get("human_review") is True
    )


def check_big(data):
    probability = data.get("fraud_probability", -1)
    return (
        data.get("flagged") is False
        and 0.17 <= probability <= 0.26
        and data.get("decision") == "hold_for_review"
        and data.get("human_review") is True
    )


def check_bad_amount(data):
    return (
        "error" in data
        and data.get("field") == "amount_inr"
    )


def check_case_p2(data):
    note = data.get("case_note") or {}
    return (
        data.get("flagged") is True
        and data.get("attempts") == 2
        and data.get("safe_default") is False
        and note.get("policy_id") == "P2"
        and data.get("human_review") is True
    )


def check_safe_default(data):
    return (
        data.get("attempts") == 3
        and data.get("safe_default") is True
        and data.get("decision") == "hold_for_review"
        and data.get("human_review") is True
    )


def check_approved_without_note(data):
    return (
        data.get("case_note") is None
        and data.get("attempts") == 0
        and data.get("decision") == "approve"
        and data.get("human_review") is False
    )


def check_no_policy(data):
    return (
        data.get("case_note") is None
        and data.get("attempts") == 0
        and data.get("decision") == "hold_for_review"
        and data.get("human_review") is True
    )


def wait_for_service():
    print("Waiting for FastAPI service...")

    deadline = time.time() + 120

    while time.time() < deadline:
        try:
            response = requests.get(
                f"{BASE_URL}/health",
                timeout=3,
            )

            if (
                response.status_code == 200
                and response.json().get("status") == "ok"
            ):
                print("Service is ready.\n")
                return True

        except requests.RequestException:
            pass

        time.sleep(2)

    print("ERROR: Service did not start within 120 seconds.")
    return False


def main():
    if not wait_for_service():
        sys.exit(1)

    invalid_payload = with_id(BENIGN, "T800005")
    invalid_payload.pop("hour_of_day")

    cases = [
        {
            "id": "Q1",
            "category": "happy",
            "method": "GET",
            "path": "/health",
            "payload": None,
            "status": 200,
            "check": check_health,
            "expected": "Health status is ok",
        },
        {
            "id": "Q2",
            "category": "happy",
            "method": "POST",
            "path": "/score",
            "payload": with_id(BENIGN, "T800001"),
            "status": 200,
            "check": check_benign,
            "expected": "Benign payment approved",
        },
        {
            "id": "Q3",
            "category": "escalation",
            "method": "POST",
            "path": "/score",
            "payload": with_id(RISKY, "T800002"),
            "status": 200,
            "check": check_risky,
            "expected": "Risky payment held for review",
        },
        {
            "id": "Q4",
            "category": "escalation",
            "method": "POST",
            "path": "/score",
            "payload": with_id(BIG, "T800003"),
            "status": 200,
            "check": check_big,
            "expected": "Large-payment guardrail",
        },
        {
            "id": "Q5",
            "category": "edge",
            "method": "POST",
            "path": "/score",
            "payload": {
                **with_id(BENIGN, "T800004"),
                "amount_inr": -5,
            },
            "status": 400,
            "check": check_bad_amount,
            "expected": "Impossible amount rejected",
        },
        {
            "id": "Q6",
            "category": "edge",
            "method": "POST",
            "path": "/score",
            "payload": invalid_payload,
            "status": 422,
            "check": lambda data: True,
            "expected": "Missing required field rejected",
        },
        {
            "id": "Q7",
            "category": "escalation",
            "method": "POST",
            "path": "/case",
            "payload": {
                **with_id(RISKY, "T900002"),
                "question": (
                    "more than 6 UPI payments in 24 hours "
                    "velocity alert"
                ),
            },
            "status": 200,
            "check": check_case_p2,
            "expected": "P2 policy and successful retry",
        },
        {
            "id": "Q8",
            "category": "escalation",
            "method": "POST",
            "path": "/case",
            "payload": {
                **with_id(RISKY, "T900005"),
                "question": (
                    "electronics merchant payment on an account "
                    "younger than 30 days"
                ),
            },
            "status": 200,
            "check": check_safe_default,
            "expected": "Safe default after three failed attempts",
        },
        {
            "id": "Q9",
            "category": "edge",
            "method": "POST",
            "path": "/case",
            "payload": {
                **with_id(RISKY, "T900099"),
                "question": (
                    "card payment above INR 50000 from a new device"
                ),
            },
            "status": 503,
            "check": lambda data: "error" in data,
            "expected": "Model connection error handled",
        },
        {
            "id": "Q10",
            "category": "happy",
            "method": "POST",
            "path": "/case",
            "payload": {
                **with_id(BENIGN, "T800001"),
                "question": (
                    "card payment above INR 50000 from a new device"
                ),
            },
            "status": 200,
            "check": check_approved_without_note,
            "expected": "Unflagged payment has no case note",
        },
        {
            "id": "Q11",
            "category": "edge",
            "method": "POST",
            "path": "/case",
            "payload": {
                **with_id(RISKY, "T800006"),
                "question": "what is the cafeteria menu today",
            },
            "status": 200,
            "check": check_no_policy,
            "expected": "Out-of-scope question requires review",
        },
    ]

    rows = []

    # Send the 11 requests in assignment order.
    for test in cases:
        try:
            url = BASE_URL + test["path"]

            if test["method"] == "GET":
                response = requests.get(url, timeout=TIMEOUT)
            else:
                response = requests.post(
                    url,
                    json=test["payload"],
                    timeout=TIMEOUT,
                )

            try:
                data = response.json()
            except ValueError:
                data = {}

            status_ok = response.status_code == test["status"]
            body_ok = test["check"](data) if status_ok else False
            passed = status_ok and body_ok

            rows.append({
                "ID": test["id"],
                "Category": test["category"],
                "Expected": test["status"],
                "Observed": response.status_code,
                "Result": "PASS" if passed else "FAIL",
                "Details": test["expected"],
            })

        except requests.RequestException as exc:
            rows.append({
                "ID": test["id"],
                "Category": test["category"],
                "Expected": test["status"],
                "Observed": "NO RESPONSE",
                "Result": "FAIL",
                "Details": str(exc),
            })

    results = pd.DataFrame(rows)

    print("========== SMOKE TEST RESULTS ==========")
    print(results.to_string(index=False))

    print("\n========== CATEGORY PASS COUNTS ==========")

    for category in ("happy", "escalation", "edge"):
        subset = results[results["Category"] == category]
        passed = int((subset["Result"] == "PASS").sum())
        print(f"{category}: {passed}/{len(subset)}")

    # GET /metrics does not affect the POST counters.
    try:
        metrics_response = requests.get(
            f"{BASE_URL}/metrics",
            timeout=TIMEOUT,
        )
        metrics_response.raise_for_status()
        metrics = metrics_response.json()

        print("\n========== METRICS JSON ==========")
        print(json.dumps(metrics, indent=2))

        print("\n========== METRICS CHECK ==========")

        expected_metrics = {
            "requests_total": 10,
            "success_total": 7,
            "error_total": 3,
            "success_rate": 0.7,
            "error_rate": 0.3,
            "human_review_total": 5,
            "human_review_rate": 0.5,
        }

        metrics_ok = True

        for key, expected in expected_metrics.items():
            actual = metrics.get(key)
            passed = actual == expected
            metrics_ok = metrics_ok and passed

            print(
                f"{key}: actual={actual}, expected={expected}, "
                f"{'PASS' if passed else 'FAIL'}"
            )

        latency = metrics.get("avg_latency_ms", 0)
        latency_ok = isinstance(latency, (int, float)) and latency > 0
        print(
            f"avg_latency_ms: {latency} — "
            f"{'PASS' if latency_ok else 'FAIL'}"
        )

    except (requests.RequestException, ValueError) as exc:
        print("\nMetrics request failed:", exc)
        metrics_ok = False

    total_pass = int((results["Result"] == "PASS").sum())

    print(f"\nOverall: {total_pass}/11 PASS")

    if total_pass != 11 or not metrics_ok:
        print("One or more checks failed. Inspect the output before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
