
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from prepare import clean_transactions, add_features


# ============================================================
# 1. CONFIGURATION AND REQUIRED COLUMN ORDER
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

DISTANCE_COLUMN = "distance_from_home_km"


# ============================================================
# 2. LOAD, CLEAN, ENCODE, SPLIT AND IMPUTE
# ============================================================

def load_model_data():
    """Prepare all data for training without leaking held-out data."""

    raw_df = pd.read_csv("transactions_raw.csv")

    clean_df = clean_transactions(raw_df)
    featured_df = add_features(clean_df)

    y = featured_df["is_fraud"].astype(int)

    # The categorical vocabularies are fixed and known in this dataset.
    # The assignment permits encoding before splitting.
    numeric_features = featured_df[
        RAW_NUMERIC_COLUMNS + ENGINEERED_COLUMNS
    ]

    categorical_features = pd.get_dummies(
        featured_df[CATEGORICAL_COLUMNS],
        dtype=int,
    )

    X = pd.concat(
        [numeric_features, categorical_features],
        axis=1,
    )

    # The supplied assignment requires this exact model-input count.
    if X.shape[1] != 26:
        raise ValueError(
            f"Expected 26 model columns, got {X.shape[1]}."
        )

    # Separate leakage-experiment data. Keep chargeback_filed last.
    X_leaky = X.copy()
    X_leaky["chargeback_filed"] = (
        featured_df["chargeback_filed"].astype(int)
    )

    # First split: 60% train and 40% remaining.
    X_train_raw, X_remaining_raw, y_train_raw, y_remaining_raw = (
        train_test_split(
            X,
            y,
            test_size=0.4,
            stratify=y,
            random_state=42,
        )
    )

    # Second split: divide the remaining 40% equally.
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = (
        train_test_split(
            X_remaining_raw,
            y_remaining_raw,
            test_size=0.5,
            stratify=y_remaining_raw,
            random_state=42,
        )
    )

    # Save the original row indexes for the leakage experiment.
    X_leak_train_raw = X_leaky.loc[X_train_raw.index].copy()
    X_leak_val_raw = X_leaky.loc[X_val_raw.index].copy()
    X_leak_test_raw = X_leaky.loc[X_test_raw.index].copy()

    # Learn the imputation value from TRAINING DATA ONLY.
    training_median = X_train_raw[DISTANCE_COLUMN].median()

    def prepare_split(frame):
        frame = frame.copy()
        frame[DISTANCE_COLUMN] = frame[DISTANCE_COLUMN].fillna(
            training_median
        )
        return frame.reset_index(drop=True)

    X_train = prepare_split(X_train_raw)
    X_val = prepare_split(X_val_raw)
    X_test = prepare_split(X_test_raw)

    # Apply the same training median to the leakage-experiment splits.
    X_leak_train = prepare_split(X_leak_train_raw)
    X_leak_val = prepare_split(X_leak_val_raw)
    X_leak_test = prepare_split(X_leak_test_raw)

    y_train = y_train_raw.reset_index(drop=True)
    y_val = y_val_raw.reset_index(drop=True)
    y_test = y_test_raw.reset_index(drop=True)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_leak_train": X_leak_train,
        "X_leak_val": X_leak_val,
        "X_leak_test": X_leak_test,
        "median": float(training_median),
        "columns": list(X.columns),
    }


# ============================================================
# 3. COMMON EVALUATION FUNCTION
# ============================================================

def evaluate_probabilities(y_true, probabilities, threshold=0.5):
    """Calculate classification metrics for a probability threshold."""

    y_true_array = np.asarray(y_true)
    predictions = (
        np.asarray(probabilities) >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": precision_score(
            y_true_array,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true_array,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true_array,
            predictions,
            zero_division=0,
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "flagged": int(predictions.sum()),
    }


def evaluate_model(model, X, y, threshold=0.5):
    probabilities = model.predict_proba(X)[:, 1]
    return evaluate_probabilities(y, probabilities, threshold)


def print_metrics_table(results):
    table = pd.DataFrame(results).T
    columns = [
        "precision", "recall", "f1",
        "TN", "FP", "FN", "TP",
    ]
    print(table[columns].round(3).to_string())


# ============================================================
# 4. PART B TASK 2: IMBALANCE EXPERIMENTS
# ============================================================

def run_imbalance_experiments(data):
    X_train = data["X_train"]
    X_val = data["X_val"]
    y_train = data["y_train"]
    y_val = data["y_val"]

    results = {}

    # Model 1: Plain logistic regression.
    plain_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    plain_model.fit(X_train, y_train)
    results["Plain logistic regression"] = evaluate_model(
        plain_model, X_val, y_val
    )

    # Model 2: Class-weighted logistic regression.
    balanced_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    balanced_model.fit(X_train, y_train)
    results["Balanced logistic regression"] = evaluate_model(
        balanced_model, X_val, y_val
    )

    # Model 3: Random oversampling on TRAINING DATA ONLY.
    fraud_mask = y_train == 1
    nonfraud_mask = y_train == 0

    X_fraud = X_train.loc[fraud_mask]
    X_nonfraud = X_train.loc[nonfraud_mask]
    y_nonfraud = y_train.loc[nonfraud_mask]

    X_fraud_sampled = X_fraud.sample(
        n=len(X_nonfraud),
        replace=True,
        random_state=42,
    )

    X_oversampled = pd.concat(
        [
            X_nonfraud.reset_index(drop=True),
            X_fraud_sampled.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    y_oversampled = pd.concat(
        [
            y_nonfraud.reset_index(drop=True),
            pd.Series(
                np.ones(len(X_fraud_sampled), dtype=int)
            ),
        ],
        ignore_index=True,
    )

    oversampled_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    oversampled_model.fit(X_oversampled, y_oversampled)
    results["Oversampled logistic regression"] = evaluate_model(
        oversampled_model, X_val, y_val
    )

    print("\n========== IMBALANCE EXPERIMENTS ==========")
    print("Oversampled rows:", len(X_oversampled))
    print("Oversampled fraud rows:", int(y_oversampled.sum()))
    print(
        "Oversampled non-fraud rows:",
        int((y_oversampled == 0).sum()),
    )
    print("\nValidation metrics at threshold 0.5:")
    print_metrics_table(results)

    print(
        "\nConfusion matrix counts are shown as TN, FP, FN and TP."
    )

    return results


# ============================================================
# 5. PART B TASK 3: CROSS-VALIDATION AND RANDOM FOREST SEARCH
# ============================================================

def make_cv():
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )


def fit_final_forest(data):
    """Fit the required grid search using training data only."""

    X_train = data["X_train"]
    y_train = data["y_train"]

    cv = make_cv()

    balanced_logistic = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    logistic_scores = cross_val_score(
        balanced_logistic,
        X_train,
        y_train,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    base_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    forest_scores = cross_val_score(
        base_forest,
        X_train,
        y_train,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    print("\n========== CROSS-VALIDATION ==========")
    print(
        "Balanced logistic regression: "
        f"mean F1={logistic_scores.mean():.3f}, "
        f"standard deviation={logistic_scores.std():.3f}"
    )
    print(
        "Untuned random forest: "
        f"mean F1={forest_scores.mean():.3f}, "
        f"standard deviation={forest_scores.std():.3f}"
    )

    parameter_grid = {
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [5, 10, 20],
    }

    search = GridSearchCV(
        estimator=base_forest,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    print("\n========== RANDOM FOREST GRID SEARCH ==========")
    print("Parameter combinations:", len(search.cv_results_["params"]))
    print("Best parameters:", search.best_params_)
    print(f"Best mean CV F1: {search.best_score_:.3f}")

    return search.best_estimator_, search


# ============================================================
# 6. PART B TASK 4: VALIDATION THRESHOLD
# ============================================================

def build_threshold_table(model, X_val, y_val):
    probabilities = model.predict_proba(X_val)[:, 1]
    rows = []

    for threshold in [round(i / 10, 1) for i in range(1, 10)]:
        metrics = evaluate_probabilities(
            y_val,
            probabilities,
            threshold,
        )

        rows.append({
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "flagged": metrics["flagged"],
        })

    table = pd.DataFrame(rows)

    # Choose the highest threshold with validation recall >= 0.65.
    qualifying = table.loc[table["recall"] >= 0.65, "threshold"]

    if qualifying.empty:
        chosen_threshold = 0.1
    else:
        chosen_threshold = float(qualifying.max())

    return table, chosen_threshold


# ============================================================
# 7. PART B TASK 6: LEAKAGE EXPERIMENT
# ============================================================

def run_leakage_experiment(data, best_params, clean_model):
    clean_metrics = evaluate_model(
        clean_model,
        data["X_val"],
        data["y_val"],
        threshold=0.5,
    )

    leaky_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )

    # This deliberately leaky model is for comparison only.
    leaky_model.fit(
        data["X_leak_train"],
        data["y_train"],
    )

    leaky_metrics = evaluate_model(
        leaky_model,
        data["X_leak_val"],
        data["y_val"],
        threshold=0.5,
    )

    print("\n========== LEAKAGE EXPERIMENT ==========")
    print(
        "Clean tuned forest validation F1 at 0.5:",
        round(clean_metrics["f1"], 3),
    )
    print(
        "Leaky forest validation F1 at 0.5:",
        round(leaky_metrics["f1"], 3),
    )
    print(
        "The leaky model is for demonstration only: "
        "chargeback_filed must not be used as a model input."
    )

    return clean_metrics, leaky_metrics


# ============================================================
# 8. PART B TASK 8: REUSABLE BUILDER FOR PART D
# ============================================================

def build_final_model():
    """
    Rebuild the final model and return the values required by Part D.

    The model is intentionally kept in memory rather than saved to disk.
    """

    data = load_model_data()
    model, search = fit_final_forest(data)

    threshold_table, threshold = build_threshold_table(
        model,
        data["X_val"],
        data["y_val"],
    )

    return {
        "model": model,
        "columns": data["columns"],
        "median": data["median"],
        "threshold": threshold,
    }


# ============================================================
# 9. MAIN: PRINT THE REQUIRED PART B RESULTS
# ============================================================

def main():
    data = load_model_data()

    print("\n========== PART B: DATA SPLIT ==========")

    for name in ("train", "val", "test"):
        X_split = data[f"X_{name}"]
        y_split = data[f"y_{name}"]

        print(
            f"{name}: rows={len(X_split)}, "
            f"fraud={int(y_split.sum())}"
        )

        assert X_split[DISTANCE_COLUMN].isna().sum() == 0

    print("\nTraining distance median:", data["median"])
    print("Number of model-input columns:", len(data["columns"]))
    print("\nPinned model-input column order:")
    for column in data["columns"]:
        print(column)

    # Three ways to handle the imbalance.
    run_imbalance_experiments(data)

    # Cross-validation and hyperparameter tuning.
    final_model, search = fit_final_forest(data)

    # Choose the alert threshold from VALIDATION data only.
    threshold_table, threshold = build_threshold_table(
        final_model,
        data["X_val"],
        data["y_val"],
    )

    print("\n========== VALIDATION THRESHOLD TABLE ==========")
    print(threshold_table.round(3).to_string(index=False))
    print(f"\nSelected threshold t_star: {threshold:.1f}")

    # Evaluate the final model on the test set exactly once,
    # at the threshold chosen using the validation set.
    test_metrics = evaluate_model(
        final_model,
        data["X_test"],
        data["y_test"],
        threshold=threshold,
    )

    print("\n========== FINAL TEST RESULT ==========")
    print(
        "Threshold:", round(threshold, 1),
        "| Precision:", round(test_metrics["precision"], 3),
        "| Recall:", round(test_metrics["recall"], 3),
        "| F1:", round(test_metrics["f1"], 3),
    )
    print(
        "Confusion matrix counts (TN, FP, FN, TP):",
        (
            test_metrics["TN"],
            test_metrics["FP"],
            test_metrics["FN"],
            test_metrics["TP"],
        ),
    )
    print("Payments flagged:", test_metrics["flagged"])

    tp = test_metrics["TP"]
    fp = test_metrics["FP"]
    fn = test_metrics["FN"]

    if tp > 0:
        legitimate_per_fraud = fp / tp
        print(
            "Legitimate payments held per fraud caught:",
            round(legitimate_per_fraud, 3),
        )
    else:
        print("Legitimate payments held per fraud caught: undefined (TP=0)")

    print("Fraud cases missed:", fn)

    # Print an evidence-based risk-committee note.
    row_04 = threshold_table.loc[
        threshold_table["threshold"] == 0.4
    ].iloc[0]
    row_05 = threshold_table.loc[
        threshold_table["threshold"] == 0.5
    ].iloc[0]

    print("\n========== RISK-COMMITTEE NOTE DRAFT ==========")
    print(
        f"The selected alert threshold is {threshold:.1f}, "
        "the highest tested threshold whose validation recall "
        "is at least 0.65 (or 0.1 if none qualifies)."
    )
    print(
        f"On the test split, {test_metrics['flagged']} of "
        f"{len(data['y_test'])} payments were flagged."
    )
    print(
        f"The model caught {tp} fraud cases and missed {fn}."
    )
    if tp > 0:
        print(
            f"It falsely held {fp} legitimate payments, or "
            f"{legitimate_per_fraud:.2f} legitimate holds per fraud caught."
        )
    print(
        f"On validation, threshold 0.4 gave precision "
        f"{row_04['precision']:.3f} and recall {row_04['recall']:.3f}; "
        f"threshold 0.5 gave precision {row_05['precision']:.3f} "
        f"and recall {row_05['recall']:.3f}."
    )

    # Leakage experiment; does not change the final model.
    run_leakage_experiment(
        data,
        search.best_params_,
        final_model,
    )

    # Report the five most important features.
    importance_table = pd.DataFrame({
        "feature": data["columns"],
        "importance": final_model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    print("\n========== TOP FIVE FEATURES ==========")
    print(importance_table.head(5).round(4).to_string(index=False))

    # Confirm the final-builder return contract used by Part D.
    print("\n========== REUSABLE BUILDER ==========")
    print('build_final_model() returns keys: "model", "columns", "median", "threshold"')
    print("Number of model columns:", len(data["columns"]))
    print("Part B completed.")


if __name__ == "__main__":
    main()
