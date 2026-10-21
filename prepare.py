import pandas as pd


# --------------------------------------------------
# 1. CLEAN THE TRANSACTION DATA
# --------------------------------------------------

def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned copy of the transaction DataFrame.
    The original DataFrame must remain unchanged.
    """

    # Step A: Remove exact duplicate rows.
    df = raw_df.drop_duplicates().copy(deep=True)

    # Step B: Remove surrounding spaces and standardize text.
    for column in ["merchant_category", "city"]:
        df[column] = df[column].str.strip().str.title()

    # Step C: Remove rows with a missing transaction amount.
    df = df.dropna(subset=["amount_inr"])

    # Step D: Remove impossible or out-of-range amounts.
    df = df[
        (df["amount_inr"] >= 1)
        & (df["amount_inr"] <= 1_000_000)
    ]

    # Step E: Reset the row index.
    df = df.reset_index(drop=True)

    return df


# --------------------------------------------------
# 2. CREATE THE THREE REQUIRED FEATURES
# --------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with three engineered features.
    Works for a full dataset or a single transaction.
    """

    result = df.copy(deep=True)

    # Transaction amount compared with the customer's average.
    result["amount_vs_avg_ratio"] = (
        result["amount_inr"] / result["avg_amount_30d_inr"]
    )

    # A night-time payment is one made at hour 0 to hour 5.
    result["is_night"] = (
        result["hour_of_day"] <= 5
    ).astype(int)

    # High-value means strictly greater than Rs. 50,000.
    result["high_value"] = (
        result["amount_inr"] > 50_000
    ).astype(int)

    return result


# --------------------------------------------------
# 3. AUDIT THE RAW DATA AND PRINT THE CHECKPOINTS
# --------------------------------------------------

def main():
    # Read the original CSV.
    raw_df = pd.read_csv("transactions_raw.csv")

    # Keep a copy so we can verify the input was not changed.
    original_copy = raw_df.copy(deep=True)

    # Audit duplicate-free data BEFORE cleaning amounts.
    deduplicated = raw_df.drop_duplicates()

    audit = [
        ("Rows as read", len(raw_df)),
        ("Columns as read", len(raw_df.columns)),
        ("Exact duplicate rows", int(raw_df.duplicated().sum())),
        ("Rows after removing duplicates", len(deduplicated)),
        (
            "Missing amount_inr after deduplication",
            int(deduplicated["amount_inr"].isna().sum()),
        ),
        (
            "Missing distance_from_home_km after deduplication",
            int(deduplicated["distance_from_home_km"].isna().sum()),
        ),
        (
            "Amounts below 1 or above 1,000,000",
            int(
                (
                    (deduplicated["amount_inr"] < 1)
                    | (deduplicated["amount_inr"] > 1_000_000)
                ).sum()
            ),
        ),
        (
            "Distinct raw merchant_category spellings",
            int(raw_df["merchant_category"].nunique()),
        ),
        (
            "Distinct raw city spellings",
            int(raw_df["city"].nunique()),
        ),
    ]

    print("\n========== RAW DATA AUDIT ==========")
    print(pd.DataFrame(audit, columns=["Metric", "Value"]).to_string(index=False))

    # Clean the dataset.
    clean_df = clean_transactions(raw_df)

    # Verify that clean_transactions did not modify its input.
    pd.testing.assert_frame_equal(raw_df, original_copy)

    print("\n========== CLEANING CHECKPOINTS ==========")
    print("Cleaned rows:", len(clean_df))
    print("Remaining exact duplicates:", int(clean_df.duplicated().sum()))
    print("Distinct merchant categories:", clean_df["merchant_category"].nunique())
    print("Distinct cities:", clean_df["city"].nunique())
    print("Missing distance values:", int(clean_df["distance_from_home_km"].isna().sum()))
    print("Minimum amount:", round(clean_df["amount_inr"].min(), 2))
    print("Maximum amount:", round(clean_df["amount_inr"].max(), 2))

    print("\nMerchant category counts:")
    print(clean_df["merchant_category"].value_counts().to_string())

    print("\nPayment mode counts:")
    print(clean_df["payment_mode"].value_counts().to_string())

    # Engineer features.
    featured_df = add_features(clean_df)

    # Verify feature engineering also works on one row.
    one_row = add_features(clean_df.iloc[[0]])
    assert len(one_row) == 1
    assert {
        "amount_vs_avg_ratio",
        "is_night",
        "high_value",
    }.issubset(one_row.columns)

    print("\n========== FEATURE CHECKPOINTS ==========")
    print("is_night sum:", int(featured_df["is_night"].sum()))
    print("high_value sum:", int(featured_df["high_value"].sum()))
    print(
        "Maximum amount_vs_avg_ratio:",
        round(featured_df["amount_vs_avg_ratio"].max(), 2),
    )

    # Fraud rate and the always-predict-not-fraud baseline.
    fraud_count = int(featured_df["is_fraud"].sum())
    fraud_rate = featured_df["is_fraud"].mean()
    baseline_accuracy = (featured_df["is_fraud"] == 0).mean()

    print("\n========== FRAUD CHECKPOINTS ==========")
    print(f"Fraud rows: {fraud_count} of {len(featured_df)}")
    print(f"Fraud rate: {fraud_rate * 100:.2f}%")
    print(f"'Never fraud' baseline accuracy: {baseline_accuracy * 100:.2f}%")
    print("Never-fraud baseline recall: 0.00")

    # Leakage audit: chargeback is recorded after a dispute.
    leakage = (
        featured_df.groupby("chargeback_filed")
        .agg(
            rows=("is_fraud", "size"),
            fraud_rows=("is_fraud", "sum"),
        )
        .reset_index()
    )

    print("\n========== LEAKAGE AUDIT ==========")
    print(leakage.to_string(index=False))

    # Exclude identifiers, post-event information and the target.
    excluded_columns = {
        "txn_id",
        "customer_id",
        "chargeback_filed",
        "is_fraud",
    }

    # Put numeric and engineered inputs first, then categorical inputs.
    model_input_columns = [
        "amount_inr",
        "hour_of_day",
        "txn_count_last_24h",
        "avg_amount_30d_inr",
        "distance_from_home_km",
        "is_new_device",
        "account_age_days",
        "amount_vs_avg_ratio",
        "is_night",
        "high_value",
        "merchant_category",
        "city",
        "payment_mode",
    ]

    assert all(column not in excluded_columns for column in model_input_columns)

    print("\n========== MODEL INPUT COLUMNS ==========")
    print(", ".join(model_input_columns))

    print("\nPart A completed successfully.")


if __name__ == "__main__":
    main()