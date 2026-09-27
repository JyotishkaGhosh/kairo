"""
ml_utils.py - small helpers shared by Kairo's models (lead scoring, win
probability, revenue forecast), so each model script can focus on its own logic.
"""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PICK_COMPLEX_IF_BETTER_BY = 0.01  # gradient boosting must cut log loss by 1% to be chosen


def logistic_model(categorical, numeric, C=0.5):
    """Simple and explainable: every feature adds a fixed amount to the log-odds."""
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numeric),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=C, max_iter=5000))])


def boosting_model(categorical, numeric):
    """Many small decision trees: can learn curves and combinations, harder to explain."""
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ("num", "passthrough", numeric),
    ])
    model = HistGradientBoostingClassifier(
        categorical_features=list(range(len(categorical))), max_iter=300,
        learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    return Pipeline([("prep", prep), ("model", model)])


MODEL_KINDS = {"logistic regression": logistic_model, "gradient boosting": boosting_model}


def fit(model, df, features, weight=None):
    kwargs = {"model__sample_weight": df[weight]} if weight else {}
    model.fit(df[features], df["label"], **kwargs)
    return model


def predict(model, df, features):
    return model.predict_proba(df[features])[:, 1]


def evaluate(y, p, w=None):
    return {"auc": roc_auc_score(y, p, sample_weight=w),
            "log_loss": log_loss(y, p, sample_weight=w, labels=[0, 1]),
            "brier": brier_score_loss(y, p, sample_weight=w)}


def compare_models(train, test, categorical, numeric, weight=None):
    """Backtest both model kinds against a 'everyone gets the average' baseline.

    Keeps logistic regression unless gradient boosting is clearly better.
    Returns (results, test predictions per model, chosen model name).
    """
    features = categorical + numeric
    w_train = train[weight] if weight else None
    w_test = test[weight] if weight else None
    base_rate = np.average(train["label"], weights=w_train)
    results = {"baseline (everyone gets the average)":
               evaluate(test["label"], np.full(len(test), base_rate), w_test)}
    preds = {}
    for name, make in MODEL_KINDS.items():
        preds[name] = predict(fit(make(categorical, numeric), train, features, weight), test, features)
        results[name] = evaluate(test["label"], preds[name], w_test)

    print(f"{'model':<40}{'AUC':>7}{'log loss':>10}{'Brier':>8}")
    for name, m in results.items():
        print(f"{name:<40}{m['auc']:>7.3f}{m['log_loss']:>10.4f}{m['brier']:>8.4f}")
    print("(AUC: 0.5 = random, 1.0 = perfect ranking. Log loss / Brier: lower is better.)")

    lr_loss = results["logistic regression"]["log_loss"]
    gb_loss = results["gradient boosting"]["log_loss"]
    chosen = ("gradient boosting" if gb_loss < lr_loss * (1 - PICK_COMPLEX_IF_BETTER_BY)
              else "logistic regression")
    print(f"\nChosen model: {chosen}")
    return results, preds, chosen


def print_calibration(y, p, w=None, noun="rows", bins=(0, 0.1, 0.2, 0.3, 0.45, 1.0)):
    """When the model says 30%, do about 30% actually happen?"""
    w = np.ones(len(p)) if w is None else np.asarray(w)
    df = pd.DataFrame({"bin": pd.cut(p, list(bins)), "p": p, "y": np.asarray(y), "w": w})
    print("\n  Calibration (predicted vs actual)")
    for b, g in df.groupby("bin", observed=True):
        print(f"    score {str(b):<13} {g.w.sum():>7.0f} {noun}   "
              f"predicted {np.average(g.p, weights=g.w):>4.0%}   actual {np.average(g.y, weights=g.w):>4.0%}")


def plural(n, word, many=None):
    """'1 meeting', '3 meetings' (many = irregular plural, e.g. 'activities')."""
    n = int(n)
    return f"{n} {word if n == 1 else many or word + 's'}"


def save_table(con, name, df):
    """Write a DataFrame to DuckDB (text columns passed as plain Python objects)."""
    df = df.astype({c: object for c in df.select_dtypes(["str", "string"]).columns})
    con.register("_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
    con.unregister("_df")


def save_metrics(con, model_name, as_of, metrics):
    """Store backtest numbers in analytics.model_metrics (one row per metric)."""
    con.execute("""CREATE TABLE IF NOT EXISTS analytics.model_metrics
                   (as_of_date DATE, model VARCHAR, metric VARCHAR, value VARCHAR)""")
    con.execute("DELETE FROM analytics.model_metrics WHERE model = ?", [model_name])
    con.executemany("INSERT INTO analytics.model_metrics VALUES (?, ?, ?, ?)",
                    [(as_of, model_name, k, json.dumps(v if isinstance(v, str) else round(float(v), 4)))
                     for k, v in metrics.items()])
