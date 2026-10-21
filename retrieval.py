
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# 1. LOAD POLICIES AND BUILD THE SEARCH INDEX
# ------------------------------------------------------------

with open("fraud_policies.json", "r", encoding="utf-8") as file:
    POLICIES = json.load(file)

with open("retrieval_queries.json", "r", encoding="utf-8") as file:
    QUERIES = json.load(file)

# Keep the default TfidfVectorizer settings.
# In particular, do not remove stop words.
VECTORIZER = TfidfVectorizer()

POLICY_TEXTS = [policy["text"] for policy in POLICIES]

# Convert each policy's text into a numerical TF-IDF vector.
POLICY_MATRIX = VECTORIZER.fit_transform(POLICY_TEXTS)


# ------------------------------------------------------------
# 2. RETRIEVE THE MOST SIMILAR POLICY
# ------------------------------------------------------------

def retrieve(query):
    """
    Return (policy_id, policy_text, score) for the policy
    with the highest cosine similarity to the query.
    """

    query_vector = VECTORIZER.transform([query])

    scores = cosine_similarity(
        query_vector,
        POLICY_MATRIX,
    )[0]

    best_index = int(scores.argmax())
    best_policy = POLICIES[best_index]
    best_score = float(scores[best_index])

    return (
        best_policy["policy_id"],
        best_policy["text"],
        best_score,
    )


# ------------------------------------------------------------
# 3. CALIBRATE THE REFUSAL THRESHOLD
# ------------------------------------------------------------

# Provisional value based on the assignment's reference scores.
# Verify the measured score interval printed below.
MIN_SCORE = 0.46


# ------------------------------------------------------------
# 4. TEST THE TEN SUPPLIED QUERIES
# ------------------------------------------------------------

def main():
    rows = []
    in_scope_scores = []
    out_of_scope_scores = []
    correct_in_scope = 0
    in_scope_total = 0

    for item in QUERIES:
        query = item["query"]
        expected = item["expected_policy"]

        policy_id, policy_text, score = retrieve(query)

        # A below-threshold result must be refused by downstream code.
        accepted_policy = policy_id if score >= MIN_SCORE else None

        rows.append({
            "query": query,
            "expected_policy": expected,
            "retrieved_policy": policy_id,
            "score": f"{score:.4f}",
            "accepted_policy": accepted_policy or "no_policy_match",
        })

        if expected is not None:
            in_scope_total += 1
            in_scope_scores.append(score)

            if policy_id == expected:
                correct_in_scope += 1
        else:
            out_of_scope_scores.append(score)

    print("\n========== POLICY RETRIEVAL RESULTS ==========")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n========== TOP-1 RETRIEVAL ACCURACY ==========")
    print(
        f"In-scope top-1 hits: "
        f"{correct_in_scope}/{in_scope_total}"
    )

    highest_out_of_scope = max(out_of_scope_scores)
    lowest_in_scope = min(in_scope_scores)

    print("\n========== REFUSAL THRESHOLD ==========")
    print(
        "Highest out-of-scope score:",
        f"{highest_out_of_scope:.4f}",
    )
    print(
        "Lowest in-scope score:",
        f"{lowest_in_scope:.4f}",
    )
    print("MIN_SCORE:", MIN_SCORE)

    if highest_out_of_scope < MIN_SCORE <= lowest_in_scope:
        print("Threshold check: PASS")
    else:
        print(
            "Threshold check: FAIL — adjust MIN_SCORE so that "
            "it is above the highest out-of-scope score and "
            "at or below the lowest in-scope score."
        )

    print("\nPart C retrieval completed.")


if __name__ == "__main__":
    main()
