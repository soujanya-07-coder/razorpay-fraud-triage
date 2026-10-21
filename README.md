# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview

This project implements a fraud-triage service for synthetic payment transactions.
It audits and cleans payment data and engineers features for fraud detection.
It compares imbalance-handling techniques and tunes a Random Forest classifier.
It selects an alert threshold using validation recall and evaluates the final model on a held-out test set.
It retrieves relevant fraud policies and generates validated case notes using the supplied mock language model.
A FastAPI service exposes health, scoring, case-note and metrics endpoints.
The service includes input validation, business guardrails, retries, safe defaults, error handling, logging and metrics.
All transaction data and policies in this project are synthetic and fictional; the project is an educational prototype, not a production payment-risk system.

## Reproduce

Run commands from the repository root in the order shown below. Python 3.11 or newer is required.

```powershell
git clone https://github.com/soujanya-07-coder/razorpay-fraud-triage.git
cd razorpay-fraud-triage
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py
python prepare.py
python train.py
python retrieval.py
python casenote.py
python -c "from pathlib import Path; Path('service.log').unlink(missing_ok=True)"
```

Replace `<YOUR_USERNAME>` with your GitHub username after creating the repository. On macOS or Linux, activate the environment with `source .venv/bin/activate` instead of the PowerShell activation command.

Start the service in Terminal 1:

```powershell
uvicorn app:app --port 8000
```

Wait for the application startup to complete. Keep the service running, open Terminal 2 in the repository root, activate the same virtual environment, and run:

```powershell
python smoke_test.py
```

The smoke test sends 11 requests, prints the response checks and metrics, and should report 11/11 PASS. After the test completes, stop the server with Ctrl+C. The resulting `service.log` contains one log line for each of the 10 POST requests.

## Part A results

### Audit and cleaning

The raw dataset contains 6,090 rows and 14 columns, including 90 exact duplicate rows. After deduplication, 6,000 rows remain.

| Audit metric | Result |
|---|---:|
| Raw rows | 6,090 |
| Raw columns | 14 |
| Exact duplicate rows | 90 |
| Rows after deduplication | 6,000 |
| Missing `amount_inr` after deduplication | 76 |
| Missing `distance_from_home_km` after deduplication | 193 |
| Amounts below INR 1 or above INR 1,000,000 | 5 |
| Distinct raw `merchant_category` spellings | 24 |
| Distinct raw `city` spellings | 24 |

`clean_transactions(raw_df)` removes exact duplicates, strips surrounding spaces, standardizes the merchant-category and city names to title case, removes rows with missing or invalid amounts, and resets the index. The original DataFrame remains unchanged.

The cleaned dataset contains 5,919 rows, no exact duplicates, six merchant categories and six distinct cities. The missing `distance_from_home_km` values are intentionally retained.

| Merchant category | Rows |
|---|---:|
| Grocery | 1,406 |
| Food Delivery | 1,326 |
| Fashion | 976 |
| Electronics | 845 |
| Utilities | 685 |
| Travel | 681 |
| **Total** | **5,919** |

| Payment mode | Rows |
|---|---:|
| UPI | 3,221 |
| Card | 1,450 |
| NetBanking | 731 |
| Wallet | 517 |
| **Total** | **5,919** |

The minimum cleaned transaction amount is INR 141.37 and the maximum is INR 105,883.95. There are 191 missing distance values after cleaning.

Missing distance values are left unfilled in Part A because the median must be calculated using the training split only in Part B, preventing information from validation and test rows from leaking into training.

### Feature engineering

The `add_features(df)` function returns a copy with three additional columns:

- `amount_vs_avg_ratio`: `amount_inr / avg_amount_30d_inr`.
- `is_night`: 1 when `hour_of_day` is 5 or less, otherwise 0.
- `high_value`: 1 when `amount_inr` is greater than INR 50,000, otherwise 0.

| Feature checkpoint | Result |
|---|---:|
| Sum of `is_night` | 365 |
| Sum of `high_value` | 17 |
| Maximum `amount_vs_avg_ratio` | 13.11 |

Both functions work on a one-row DataFrame without error, and `clean_transactions` leaves its input unchanged.

### Fraud rate and baseline

The cleaned dataset contains 366 fraud rows among 5,919 transactions, giving a fraud rate of 6.18%. A rule that predicts every transaction as non-fraud achieves 93.82% accuracy but has zero fraud recall. Accuracy is therefore misleading as the headline metric because the majority non-fraud class dominates the dataset, while the rule fails to detect every fraudulent transaction.

### Leakage audit and model-input columns

| `chargeback_filed` | Rows | Fraud rows |
|---:|---:|---:|
| 0 | 5,546 | 62 |
| 1 | 373 | 304 |

`chargeback_filed` records a dispute outcome that may occur after settlement, so using it as an input would leak information that may not be available at prediction time. `txn_id` and `customer_id` are identifiers that may encourage memorization rather than generalizable fraud detection. `is_fraud` is the target and must not be included among model inputs.

The model-input columns before one-hot encoding are:

`amount_inr`, `hour_of_day`, `txn_count_last_24h`, `avg_amount_30d_inr`, `distance_from_home_km`, `is_new_device`, `account_age_days`, `amount_vs_avg_ratio`, `is_night`, `high_value`, `merchant_category`, `city`, and `payment_mode`.

The excluded columns are `txn_id`, `customer_id`, `chargeback_filed`, and `is_fraud`.

## Part B results

### Data split and training-only imputation

The dataset was divided with two successive stratified splits using `random_state=42`, producing a 60/20/20 train/validation/test split. The missing-distance median was learned from the training split and then applied unchanged to the validation and test splits.

| Split | Rows | Fraud rows |
|---|---:|---:|
| Training | 3,551 | 220 |
| Validation | 1,184 | 73 |
| Test | 1,184 | 73 |

The training median for `distance_from_home_km` is 12.6. All three splits have zero missing distance values after imputation.

The model has 26 one-hot-encoded input columns in the following order:

```text
amount_inr
hour_of_day
txn_count_last_24h
avg_amount_30d_inr
distance_from_home_km
is_new_device
account_age_days
amount_vs_avg_ratio
is_night
high_value
merchant_category_Electronics
merchant_category_Fashion
merchant_category_Food Delivery
merchant_category_Grocery
merchant_category_Travel
merchant_category_Utilities
city_Bengaluru
city_Chennai
city_Delhi
city_Kolkata
city_Mumbai
city_Pune
payment_mode_Card
payment_mode_NetBanking
payment_mode_UPI
payment_mode_Wallet
```

### Imbalance experiments

All three logistic-regression models were evaluated on the validation split at probability threshold 0.5.

| Model | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain logistic regression | 0.625 | 0.342 | 0.442 | 1,096 | 15 | 48 | 25 |
| Balanced logistic regression | 0.252 | 0.781 | 0.381 | 942 | 169 | 16 | 57 |
| Randomly oversampled logistic regression | 0.257 | 0.781 | 0.386 | 946 | 165 | 16 | 57 |

The oversampled training table contained 6,662 rows: 3,331 non-fraud rows and 3,331 fraud rows. Oversampling was applied only to training data, never to validation or test data.

Both imbalance treatments increased recall from 0.342 to 0.781, but precision fell and more legitimate payments were falsely flagged. A risk desk that prioritizes detecting fraud would prefer balanced weighting or oversampling for their higher recall, accepting the increased cost of manual reviews.

### Cross-validation and hyperparameter tuning

Five-fold stratified cross-validation on the training split gave balanced logistic regression a mean F1 of 0.402 with standard deviation 0.015. The untuned Random Forest achieved mean F1 of 0.453 with standard deviation 0.052.

The Random Forest grid search tested 12 parameter combinations over five folds and selected `max_depth=None` and `min_samples_leaf=5`. The best mean cross-validation F1 was 0.465.

The tuned Random Forest was selected for its stronger cross-validation F1 and ability to model nonlinear relationships. Cross-validation provides a more stable comparison than a single split, though the observed spread across folds shows that performance still varies with the data. Compared with logistic regression, the forest sacrifices direct coefficient-level interpretability and requires more computational work.

### Validation threshold table

The alert threshold was selected using validation recall, with the rule that `t_star` is the highest tested threshold with recall of at least 0.65, or 0.1 if none qualifies.

| Threshold | Precision | Recall | F1 | Payments flagged |
|---:|---:|---:|---:|---:|
| 0.1 | 0.110 | 0.973 | 0.200 | 638 |
| 0.2 | 0.158 | 0.945 | 0.271 | 437 |
| 0.3 | 0.209 | 0.863 | 0.336 | 302 |
| 0.4 | 0.272 | 0.712 | 0.394 | 191 |
| 0.5 | 0.352 | 0.589 | 0.441 | 122 |
| 0.6 | 0.507 | 0.507 | 0.507 | 73 |
| 0.7 | 0.528 | 0.260 | 0.349 | 36 |
| 0.8 | 0.778 | 0.192 | 0.308 | 18 |
| 0.9 | 0.800 | 0.055 | 0.103 | 5 |

The selected threshold is `t_star = 0.4`, the highest threshold in the tested grid with validation recall of at least 0.65.

### Final test and risk-committee note

The final model was evaluated on the test split once, at the threshold selected using validation data.

| Test metric | Result |
|---|---:|
| Threshold | 0.4 |
| Precision | 0.281 |
| Recall | 0.781 |
| F1 | 0.413 |
| True negatives (TN) | 965 |
| False positives (FP) | 146 |
| False negatives (FN) | 16 |
| True positives (TP) | 57 |
| Payments flagged | 203 of 1,184 |
| Legitimate payments held per fraud caught | 2.561 |
| Fraud cases missed | 16 |

**Risk-committee note:** The selected alert threshold is 0.4, the highest tested threshold whose validation recall was at least 0.65. On validation, threshold 0.4 achieved precision 0.272 and recall 0.712, whereas threshold 0.5 gave precision 0.352 and recall 0.589. On the held-out test set, the model flagged 203 payments, caught 57 fraud cases and missed 16. It falsely held 146 legitimate payments, or about 2.56 legitimate holds per fraud caught. These results favor higher fraud recall but create a substantial manual-review workload, while a stricter threshold would reduce flags at the cost of missing more fraud.

### Leakage experiment

At threshold 0.5, the clean tuned Random Forest achieved validation F1 of 0.441, while the deliberately leaky model using `chargeback_filed` achieved validation F1 of 0.797. The apparently improved result is not trustworthy because `chargeback_filed` contains post-event dispute information and is not used in the final model.

### Feature importance

| Feature | Importance |
|---|---:|
| `amount_vs_avg_ratio` | 0.2700 |
| `txn_count_last_24h` | 0.1779 |
| `amount_inr` | 0.1339 |
| `account_age_days` | 0.0835 |
| `distance_from_home_km` | 0.0709 |

The two strongest features, transaction amount relative to the customer's average and transaction frequency in the previous 24 hours, are consistent with suspicious behavior involving unusually large purchases or repeated payments.

### Reusable model builder

`build_final_model()` repeats the data preparation, split, training-only median imputation, model tuning and validation-threshold selection. It returns a dictionary with the keys `model`, `columns`, `median`, and `threshold`, and keeps the fitted model in memory rather than saving it to disk.

## Part C results

### Policy retrieval

The retriever uses a `TfidfVectorizer` with default settings on the six policy texts and retrieves the policy with the highest cosine similarity. The refusal threshold is `MIN_SCORE = 0.46`.

| Query | Expected policy | Retrieved policy | Similarity score | Accepted result |
|---|---|---|---:|---|
| Card payment above INR 50000 from a new device | P1 | P1 | 0.5501 | P1 |
| How many UPI payments in 24 hours trigger a velocity alert | P2 | P2 | 0.5622 | P2 |
| Payment at 3 AM far away from the home city | P3 | P3 | 0.4948 | P3 |
| Customer reports unauthorised payment and chargeback case | P4 | P4 | 0.5689 | P4 |
| Electronics merchant payment on an account younger than 30 days | P5 | P5 | 0.5641 | P5 |
| Which payments are approved automatically without manual review | P6 | P6 | 0.6250 | P6 |
| Temporary hold on repeated UPI payments within a day | P2 | P2 | 0.4862 | P2 |
| What happens after a customer reports an unauthorised payment | P4 | P4 | 0.4867 | P4 |
| What is the cafeteria menu today | None | P3 | 0.4251 | `no_policy_match` |
| How do I reset my email password | None | P1 | 0.0000 | `no_policy_match` |

All eight in-scope queries retrieved the expected policy (8/8). The highest out-of-scope score was 0.4251 and the lowest in-scope score was 0.4862. The selected threshold of 0.46 is above the highest out-of-scope score and at or below the lowest in-scope score, so both supplied out-of-scope queries are refused.

### Case-note generation and validation

The generated case note must contain five string keys: `summary`, `priority`, `recommended_team`, `recommended_action`, and `policy_id`. Validation has three gates: valid JSON object, required keys and string values, and allowed enum values with the policy ID matching the retrieved policy.

| Scenario | Retrieved policy | Attempts | Failed gates | Status | Safe default | Final action | Override |
|---|---|---:|---|---|---|---|---|
| S1 (T900001) | P1 | 1 | [] | ok | no | hold_for_review | no |
| S2 (T900002) | P2 | 2 | [1] | ok | no | hold_for_review | no |
| S3 (T900003) | P3 | 2 | [2] | ok | no | hold_for_review | no |
| S4 (T900004) | P4 | 2 | [3] | ok | no | block_and_notify | no |
| S5 (T900005) | P5 | 3 | [1, 1, 1] | ok | yes | hold_for_review | no |
| S6 (T900006) | P2 | 1 | [] | ok | no | hold_for_review | yes |
| S7 (T900007) | P1 | 2 | [3] | ok | no | hold_for_review | no |
| S8 (T900008) | P4 | 1 | [] | ok | no | block_and_notify | no |
| S9 (T900009) | P3 | 3 | [1, 2] | ok | no | hold_for_review | no |
| S10 (T900010) | P2 | 2 | [3] | ok | no | hold_for_review | no |

Totals: 19 model calls, 3 scenarios valid on the first call (S1, S6, S8), 6 scenarios recovered by retry (S2, S3, S4, S7, S9, S10), 1 safe default (S5), and 1 guardrail override (S6).

The supplied mock language model injects scripted defects such as malformed JSON, missing keys and invalid values, so a response can fail validation and a retry can fail again. Validation gates, bounded retries and the conservative safe default prevent invalid recommendations from being trusted, while the business guardrail changes an `approve` action to `hold_for_review` when the fraud probability is at least 0.5.

A `ConnectionError` from the model propagates from `run_case` so the service can return a 503 response. An out-of-scope query returns `no_policy_match`, with zero model attempts and no case note.

## Part D results

### Smoke-test results

The smoke test was run against the locally running FastAPI service. It passed all 11 requests.

| ID | Category | Request | Expected status | Observed status | Result |
|---|---|---|---:|---:|---|
| Q1 | happy | `GET /health` | 200 | 200 | PASS |
| Q2 | happy | `POST /score` — benign payment T800001 | 200 | 200 | PASS |
| Q3 | escalation | `POST /score` — risky payment T800002 | 200 | 200 | PASS |
| Q4 | escalation | `POST /score` — large payment T800003 | 200 | 200 | PASS |
| Q5 | edge | `POST /score` — invalid amount T800004 | 400 | 400 | PASS |
| Q6 | edge | `POST /score` — missing `hour_of_day` | 422 | 422 | PASS |
| Q7 | escalation | `POST /case` — UPI velocity question, T900002 | 200 | 200 | PASS |
| Q8 | escalation | `POST /case` — electronics policy, T900005 | 200 | 200 | PASS |
| Q9 | edge | `POST /case` — model connection failure, T900099 | 503 | 503 | PASS |
| Q10 | happy | `POST /case` — benign payment T800001 | 200 | 200 | PASS |
| Q11 | edge | `POST /case` — out-of-scope cafeteria question | 200 | 200 | PASS |

The category totals were happy 3/3, escalation 4/4 and edge 4/4. Invalid requests were rejected before model execution, the model connection failure returned 503 without exposing a stack trace, and the service continued to answer subsequent requests.

### Metrics

The `/metrics` endpoint reported these counters after the smoke-test run.

```json
{
  "requests_total": 10,
  "success_total": 7,
  "error_total": 3,
  "success_rate": 0.7,
  "error_rate": 0.3,
  "avg_latency_ms": 30.25,
  "human_review_total": 5,
  "human_review_rate": 0.5
}
```

The success and error rates are decimal fractions. The average latency is measured in milliseconds. The human-review rate is calculated as `human_review_total / requests_total`.

### Service log evidence

The committed `service.log` contains exactly 10 lines, one for each POST request and none for GET requests. Its levels were INFO for five lines, WARNING for four lines and ERROR for one line, with no stack traces.

Representative lines from the recorded run:

```text
2026-10-21 19:01:52,825 INFO endpoint=/score status_code=200 latency_ms=47.52 txn_id=T800001 fraud_probability=0.0430
2026-10-21 19:01:52,963 WARNING endpoint=/score status_code=400 latency_ms=1.57 txn_id=T800004
2026-10-21 19:01:53,131 ERROR endpoint=/case status_code=503 latency_ms=29.23 txn_id=T900099 fraud_probability=0.9597
```

## Limitations

- The transaction data is synthetic and generated from a hand-built fraud rule, so the measured performance may not generalize to real payment fraud.
- The TF-IDF retriever relies on lexical similarity and may fail to match semantically related policy questions with substantially different wording.
- The supplied mock language model uses scripted response defects and does not establish the reliability of a real hosted language model.
- The alert threshold was selected using a validation set containing only 73 fraud cases, so its estimated recall may be unstable.
- The service is an educational local prototype and has not been validated for production payment processing, security, throughput or operational reliability.