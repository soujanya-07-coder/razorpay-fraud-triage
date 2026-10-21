
import json
import pandas as pd

import mock_llm
from retrieval import retrieve, MIN_SCORE


# ============================================================
# 1. LOAD THE TEN SUPPLIED SCENARIOS
# ============================================================

with open("case_scenarios.json", "r", encoding="utf-8") as file:
    CASE_SCENARIOS = json.load(file)


# ============================================================
# 2. DEFINE THE CASE-NOTE SCHEMA
# ============================================================

CASE_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "recommended_team": {
            "type": "string",
            "enum": [
                "card_risk",
                "upi_risk",
                "merchant_ops",
                "customer_care",
            ],
        },
        "recommended_action": {
            "type": "string",
            "enum": [
                "approve",
                "hold_for_review",
                "block_and_notify",
            ],
        },
        "policy_id": {
            "type": "string",
            "enum": ["P1", "P2", "P3", "P4", "P5", "P6"],
        },
    },
    "required": [
        "summary",
        "priority",
        "recommended_team",
        "recommended_action",
        "policy_id",
    ],
}


# ============================================================
# 3. BUILD THE PROMPT
# ============================================================

def build_prompt(
    txn_id,
    fraud_probability,
    amount_inr,
    policy_id,
    policy_text,
):
    """Build a policy-grounded prompt in the required section order."""

    priority_values = CASE_NOTE_SCHEMA["properties"]["priority"]["enum"]
    team_values = CASE_NOTE_SCHEMA["properties"]["recommended_team"]["enum"]
    action_values = CASE_NOTE_SCHEMA["properties"]["recommended_action"]["enum"]
    policy_values = CASE_NOTE_SCHEMA["properties"]["policy_id"]["enum"]

    prompt = f"""ROLE:
You are a payment-risk analyst writing a short case note for a human risk desk.

TASK:
Produce a structured case note for the flagged payment using the supplied fraud policy.

CONSTRAINTS:
Use only the policy text supplied below to determine the policy-grounded recommendation. Do not invent policy rules or introduce unsupported facts. The fraud probability and transaction amount provide context, but the policy text is the only source for the policy recommendation.

POLICY TEXT:
{policy_text}

OUTPUT FORMAT:
Return exactly one valid JSON object and nothing else. Do not use Markdown fences or explanatory text outside the JSON object.
The object must contain these five keys, and every value must be a string:
summary, priority, recommended_team, recommended_action, policy_id.
Allowed priority values: {", ".join(priority_values)}.
Allowed recommended_team values: {", ".join(team_values)}.
Allowed recommended_action values: {", ".join(action_values)}.
Allowed policy_id values: {", ".join(policy_values)}.
The policy_id must equal the supplied retrieved policy ID.

TXN_ID: {txn_id}
FRAUD_PROBABILITY: {float(fraud_probability)}
AMOUNT_INR: {float(amount_inr)}
POLICY_ID: {policy_id}"""

    return prompt


# ============================================================
# 4. VALIDATE A GENERATED CASE NOTE IN THREE GATES
# ============================================================

def validate(text, retrieved_policy_id):
    """
    Return (parsed_note, gate).

    Gate 0: all checks pass
    Gate 1: invalid JSON or JSON is not an object
    Gate 2: missing required keys or non-string values
    Gate 3: invalid allowed values or incorrect policy ID
    """

    # Gate 1: valid JSON object.
    try:
        note = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, 1

    if not isinstance(note, dict):
        return None, 1

    # Gate 2: all five required values must be strings.
    required_keys = CASE_NOTE_SCHEMA["required"]

    for key in required_keys:
        if key not in note:
            return None, 2

        if not isinstance(note[key], str):
            return None, 2

    # Gate 3: enforce the schema's allowed enum values.
    properties = CASE_NOTE_SCHEMA["properties"]

    for key in (
        "priority",
        "recommended_team",
        "recommended_action",
        "policy_id",
    ):
        allowed_values = properties[key]["enum"]

        if note[key] not in allowed_values:
            return None, 3

    # The generated note must use the policy actually retrieved.
    if note["policy_id"] != retrieved_policy_id:
        return None, 3

    return note, 0


# ============================================================
# 5. SAFE DEFAULT
# ============================================================

def make_safe_default(retrieved_policy_id):
    """Return a conservative case note if every attempt fails."""

    return {
        "summary": (
            "Automated case note unavailable; manual review required."
        ),
        "priority": "high",
        "recommended_team": "customer_care",
        "recommended_action": "hold_for_review",
        "policy_id": retrieved_policy_id,
    }


# ============================================================
# 6. RUN ONE CASE: RETRIEVE, GENERATE, RETRY AND GUARDRAIL
# ============================================================

def run_case(
    txn_id,
    fraud_probability,
    amount_inr,
    question,
    max_attempts=3,
):
    """Process one payment case. ConnectionError intentionally propagates."""

    # Retrieve the most similar policy.
    retrieved_policy_id, policy_text, score = retrieve(question)

    # Do not call the model for a query below the refusal threshold.
    if score < MIN_SCORE:
        return {
            "status": "no_policy_match",
            "attempts": 0,
            "failures": [],
            "case_note": None,
            "safe_default": False,
            "guardrail_override": False,
        }

    prompt = build_prompt(
        txn_id,
        fraud_probability,
        amount_inr,
        retrieved_policy_id,
        policy_text,
    )

    failures = []
    note = None
    attempts = 0

    # Retry with the exact same prompt after validation failures.
    for _ in range(max_attempts):
        attempts += 1

        # Do NOT swallow ConnectionError here; it must propagate.
        generated_text = mock_llm.generate(prompt)

        note, gate = validate(
            generated_text,
            retrieved_policy_id,
        )

        if gate == 0:
            break

        failures.append(gate)

    safe_default = False

    # Use a safe fallback if all attempts failed validation.
    if note is None or (len(failures) == max_attempts):
        note = make_safe_default(retrieved_policy_id)
        safe_default = True

    # Business guardrail: never approve a case with probability >= 0.5.
    guardrail_override = False

    if (
        note["recommended_action"] == "approve"
        and float(fraud_probability) >= 0.5
    ):
        note["recommended_action"] = "hold_for_review"
        guardrail_override = True

    return {
        "status": "ok",
        "attempts": attempts,
        "failures": failures,
        "case_note": note,
        "safe_default": safe_default,
        "guardrail_override": guardrail_override,
    }


# ============================================================
# 7. RUN THE TEN SCENARIOS IN FILE ORDER
# ============================================================

def main():
    # Reset the supplied mock's scripted behaviour.
    mock_llm.reset()

    rows = []

    for index, scenario in enumerate(CASE_SCENARIOS, start=1):
        txn_id = scenario["txn_id"]
        probability = scenario["fraud_probability"]
        amount = scenario["amount_inr"]
        question = scenario["question"]

        result = run_case(
            txn_id,
            probability,
            amount,
            question,
        )

        # Identify the top policy for the scenario table.
        retrieved_id, _, score = retrieve(question)
        if score < MIN_SCORE:
            retrieved_label = "no_policy_match"
        else:
            retrieved_label = retrieved_id

        note = result["case_note"]

        if note is None:
            final_action = "no_policy_match"
        else:
            final_action = note["recommended_action"]

        rows.append({
            "Scenario": f"S{index} ({txn_id})",
            "Retrieved policy": retrieved_label,
            "Attempts": result["attempts"],
            "Failures": result["failures"],
            "Status": result["status"],
            "Safe default": result["safe_default"],
            "Final action": final_action,
            "Override": result["guardrail_override"],
        })

    table = pd.DataFrame(rows)

    print("\n========== PART C SCENARIO RESULTS ==========")
    print(table.to_string(index=False))

    total_calls = int(table["Attempts"].sum())
    first_call_valid = sum(
        row["Attempts"] == 1
        and row["Failures"] == []
        and not row["Safe default"]
        for row in rows
    )
    recovered_by_retry = sum(
        len(row["Failures"]) > 0
        and not row["Safe default"]
        and row["Status"] == "ok"
        for row in rows
    )
    safe_defaults = int(table["Safe default"].sum())
    overrides = int(table["Override"].sum())

    print("\n========== PART C TOTALS ==========")
    print("Total model calls:", total_calls)
    print("Valid on first call:", first_call_valid)
    print("Recovered by retry:", recovered_by_retry)
    print("Safe defaults:", safe_defaults)
    print("Guardrail overrides:", overrides)

    # Verify that ConnectionError is not swallowed by run_case.
    # This test is deliberately kept outside run_case.
    print("\n========== CONNECTION ERROR CHECK ==========")
    mock_llm.reset()

    try:
        run_case(
            "T900099",
            0.80,
            50000,
            "card payment above INR 50000 from a new device",
        )
    except ConnectionError:
        print("ConnectionError propagated as required.")
    else:
        print(
            "Warning: the supplied mock did not raise ConnectionError "
            "for the T900099 check."
        )

    # Verify that out-of-scope queries never call the model.
    print("\n========== OUT-OF-SCOPE CHECK ==========")
    mock_llm.reset()

    unmatched = run_case(
        "T900100",
        0.70,
        2500,
        "what is the cafeteria menu today",
    )

    print(unmatched)

    print("\nPart C case-note processing completed.")


if __name__ == "__main__":
    main()
