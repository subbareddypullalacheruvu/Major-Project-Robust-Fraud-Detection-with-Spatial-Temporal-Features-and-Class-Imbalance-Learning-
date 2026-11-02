"""
Helper functions for our fraud detection prototype (FraudLens).

The prototype takes any card transaction table, builds the temporal and
spatial features from our experiments, trains a Random Forest with the same
leak-free protocol as our notebooks, and ranks transactions by fraud risk.

What carries over from our experiments:
  - Temporal features: time since the card's previous transaction (gap_log)
    and night-time transactions (is_night). The time gap was the main signal
    on Sparkov.
  - Spatial features: distance from the cardholder's home, merchant outside
    the home state, and merchant abroad. Location was the main signal on
    TabFormer.
  - Amount, category risk, age and gender, as in our final model.
  - The protocol: split by time, fit category risk on the training part only,
    choose the F2-optimal threshold on validation, and report PR-AUC, F2,
    MCC and alert budgets on the test part.

Each dataset has different column names, so the user tells the prototype
which column plays which role (a "column mapping"). Roles that are missing
are skipped, and the features that need them are left out.
"""

import io
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from sklearn.metrics import fbeta_score


# ---------------------------------------------------------------------------
# 0. Shared helpers (the same definitions as src/utils.py and src/evaluation.py,
#    copied here so the app folder runs on its own)
# ---------------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two GPS points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def fit_category_risk(train_df, category_column, label_column):
    """Fraud rate of each category, from the training data only."""
    return {"risk_by_category": train_df.groupby(category_column)[label_column].mean().to_dict(),
            "fallback_rate": train_df[label_column].mean()}


def apply_category_risk(df, category_column, fitted):
    return df[category_column].map(fitted["risk_by_category"]).fillna(fitted["fallback_rate"])


def tune_threshold(y_val, val_proba):
    """Try thresholds from 0.01 to 0.99 and keep the one with the best F2 on validation."""
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    f2 = [fbeta_score(y_val, (val_proba >= t).astype(int), beta=2, zero_division=0) for t in thresholds]
    return float(thresholds[np.argmax(f2)])


def point_metrics(y, proba, threshold):
    """PR-AUC, F1, F2, MCC, precision and recall at the threshold."""
    y = np.asarray(y)
    pred = (proba >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y == 1)); fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1)); tn = np.sum((pred == 0) & (y == 0))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall > 0 else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {"PR-AUC": average_precision_score(y, proba), "F1": f1, "F2": f2, "MCC": mcc,
            "Precision": precision, "Recall": recall}


# ---------------------------------------------------------------------------
# 1. Column roles
# ---------------------------------------------------------------------------

# role -> (description, common column names in the datasets we used)
ROLES = {
    "label":          ("Fraud label (0/1, Yes/No)", ["is_fraud", "isFraud", "Is Fraud?", "fraud", "Class", "label"]),
    "time":           ("Transaction date and time", ["trans_date_trans_time", "transaction_datetime", "timestamp",
                                                     "datetime", "TransactionDT", "date", "time"]),
    "card":           ("Card or customer ID", ["cc_num", "card_id", "card", "customer_id", "sender_account",
                                               "account_id", "user_id", "nameOrig", "card1"]),
    "amount":         ("Amount", ["amt", "amount", "transaction_amount", "TransactionAmt", "Amount"]),
    "category":       ("Merchant category", ["category", "merchant_category", "MCC", "mcc", "ProductCD", "type"]),
    "age":            ("Customer age", ["age", "customer_age"]),
    "dob":            ("Customer date of birth", ["dob", "date_of_birth", "birth_date"]),
    "gender":         ("Customer gender", ["gender", "customer_gender", "Gender", "sex"]),
    "home_lat":       ("Home latitude", ["lat", "home_lat", "Latitude", "customer_lat"]),
    "home_long":      ("Home longitude", ["long", "home_long", "Longitude", "customer_long"]),
    "merch_lat":      ("Merchant latitude", ["merch_lat", "merchant_lat"]),
    "merch_long":     ("Merchant longitude", ["merch_long", "merchant_long"]),
    "home_state":     ("Home state", ["state", "State", "customer_state", "home_state"]),
    "merchant_state": ("Merchant state or country", ["merchant_state", "Merchant State", "merchant_country"]),
}

# feature -> (group, roles it needs, plain description)
FEATURES = {
    "amt_log":            ("Amount",   ["amount"], "Log of the transaction amount"),
    "category_risk":      ("Amount",   ["category", "label"], "Fraud rate of the merchant category (training part only)"),
    "gap_log":            ("Temporal", ["card", "time"], "Log of hours since the card's previous transaction"),
    "is_night":           ("Temporal", ["time"], "Transaction between 22:00 and 03:59"),
    "age":                ("Customer", ["age|dob", "time"], "Customer age at the time of the transaction"),
    "gender_flag":        ("Customer", ["gender"], "1 for male, 0 otherwise"),
    "dist_home_km":       ("Spatial",  ["home_lat", "home_long", "merch_lat", "merch_long"],
                           "Distance between the merchant and the cardholder's home"),
    "outside_home_state": ("Spatial",  ["merchant_state", "home_state"], "Merchant is outside the home state"),
    "abroad":             ("Spatial",  ["merchant_state"], "Merchant is in another country (not the USA)"),
}

NIGHT_HOURS = [22, 23, 0, 1, 2, 3]
HOME_COUNTRY = {"US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"}


def guess_columns(df):
    """Guess which column plays each role from common names. Missing roles are None."""
    lower = {c.lower(): c for c in df.columns}
    mapping = {}
    for role, (_, names) in ROLES.items():
        mapping[role] = next((lower[n.lower()] for n in names if n.lower() in lower), None)
    # the home latitude/longitude guesses only make sense next to merchant coordinates
    if mapping["merch_lat"] is None:
        mapping["home_lat"] = mapping["home_long"] = None
    # a merchant country cannot be compared with a home state
    if mapping["merchant_state"] and "country" in mapping["merchant_state"].lower():
        mapping["home_state"] = None
    return mapping


def available_features(mapping):
    """The features we can build with this mapping."""
    out = []
    for feat, (_, needs, _) in FEATURES.items():
        ok = True
        for need in needs:
            if "|" in need:
                ok &= any(mapping.get(r) for r in need.split("|"))
            else:
                ok &= mapping.get(need) is not None
        if ok:
            out.append(feat)
    return out


# ---------------------------------------------------------------------------
# 2. Preparing the data and building the features
# ---------------------------------------------------------------------------

def _to_number(s):
    # amounts and incomes are sometimes stored as text such as "$1,234.50"
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")


def _to_label(s):
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).clip(0, 1)
    return s.astype(str).str.strip().str.lower().isin(["1", "yes", "y", "true", "fraud"]).astype(int)


def _to_time(s):
    # a numeric time column (such as IEEE-CIS TransactionDT) is read as seconds from a reference date
    if pd.api.types.is_numeric_dtype(s):
        return pd.Timestamp("2000-01-01") + pd.to_timedelta(s, unit="s")
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s, errors="coerce")


def prepare(df, mapping):
    """Parse the mapped columns and build every feature that does not depend on the labels.

    Returns a copy sorted by time, with _time, _y (if there is a label), _card,
    _amount and the features. The time gap only looks back at the same card's
    earlier transactions, so it is safe to build before splitting.
    """
    m = mapping
    d = df.copy()
    d["_time"] = _to_time(d[m["time"]]) if m.get("time") else pd.NaT
    if m.get("label"):
        d["_y"] = _to_label(d[m["label"]])
    d["_card"] = d[m["card"]].astype(str) if m.get("card") else "all"
    if m.get("time"):
        d = d.dropna(subset=["_time"])

    if m.get("amount"):
        d["_amount"] = _to_number(d[m["amount"]])
        d["amt_log"] = np.log1p(d["_amount"].clip(lower=0)).fillna(0)
    if m.get("time"):
        d["is_night"] = d["_time"].dt.hour.isin(NIGHT_HOURS).astype(int)
    if m.get("card") and m.get("time"):
        d = d.sort_values(["_card", "_time"])
        d["_gap_h"] = d.groupby("_card")["_time"].diff().dt.total_seconds() / 3600
        d["gap_log"] = np.log1p(d["_gap_h"]).fillna(-1)      # -1 = first transaction of the card
    if m.get("age"):
        d["age"] = _to_number(d[m["age"]]).fillna(-1)
    elif m.get("dob") and m.get("time"):
        d["age"] = ((d["_time"] - pd.to_datetime(d[m["dob"]], errors="coerce")).dt.days / 365.25).fillna(-1)
    if m.get("gender"):
        d["gender_flag"] = d[m["gender"]].astype(str).str.strip().str.upper().str.startswith("M").astype(int)
    if all(m.get(r) for r in ["home_lat", "home_long", "merch_lat", "merch_long"]):
        d["dist_home_km"] = haversine_distance(_to_number(d[m["home_lat"]]), _to_number(d[m["home_long"]]),
                                               _to_number(d[m["merch_lat"]]), _to_number(d[m["merch_long"]])).fillna(-1)
    if m.get("merchant_state"):
        ms = d[m["merchant_state"]]
        has = ms.notna() & (ms.astype(str).str.strip() != "")
        # abroad: a country name instead of a two-letter state code, other than the home country
        name = ms.astype(str).str.strip()
        d["abroad"] = (has & (name.str.len() > 2) & ~name.str.upper().isin(HOME_COUNTRY)).astype(int)
        if m.get("home_state"):
            d["outside_home_state"] = (has & (ms.astype(str).str.strip() != d[m["home_state"]].astype(str).str.strip())).astype(int)

    return d.sort_values("_time" if m.get("time") else "_card").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Splitting by time
# ---------------------------------------------------------------------------

def time_split(d, test_frac=0.2, val_frac=0.2):
    """Last 20% of the period is test; last 20% of the rest is validation (as in our notebooks)."""
    t = d["_time"]
    cut_test = t.min() + (1 - test_frac) * (t.max() - t.min())
    train = d[t < cut_test]
    cut_val = train["_time"].min() + (1 - val_frac) * (train["_time"].max() - train["_time"].min())
    return (train[train["_time"] < cut_val].copy(), train[train["_time"] >= cut_val].copy(),
            d[t >= cut_test].copy(), cut_val, cut_test)


def sample_cards(d, max_rows, seed=42):
    """Keep a random sample of whole cards, so the time gaps stay correct."""
    if len(d) <= max_rows:
        return d
    cards = d["_card"].unique()
    keep = np.random.default_rng(seed).choice(cards, size=max(1, int(len(cards) * max_rows / len(d))), replace=False)
    return d[d["_card"].isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Leakage check: each raw column on its own
# ---------------------------------------------------------------------------

def leak_check(train, test, columns):
    """ROC-AUC of each raw column on its own (fitted on train, scored on test).

    0.95 or more: the column almost gives the answer, so check it for leakage.
    Below 0.55: little or no signal.
    """
    y_tr, y_te = train["_y"], test["_y"]
    rows = []
    if y_te.nunique() < 2 or y_tr.nunique() < 2:
        return pd.DataFrame(columns=["column", "ROC-AUC", "note"])
    for col in columns:
        num_tr = _to_number(train[col])
        if num_tr.notna().mean() > 0.99:
            fill = num_tr.median()
            x_tr, x_te = num_tr.fillna(fill), _to_number(test[col]).fillna(fill)
            if x_tr.nunique() > 1 and roc_auc_score(y_tr, x_tr) < 0.5:
                x_te = -x_te
        elif train[col].nunique() <= 1000:
            rates = y_tr.groupby(train[col].astype(str)).mean()
            x_te = test[col].astype(str).map(rates).fillna(y_tr.mean())
        else:
            continue
        if x_te.nunique() < 2:
            continue
        roc = roc_auc_score(y_te, x_te)
        note = "check for leakage" if roc >= 0.95 else ("little or no signal" if roc < 0.55 else "some signal")
        rows.append({"column": col, "ROC-AUC": round(roc, 4), "note": note})
    if not rows:
        return pd.DataFrame(columns=["column", "ROC-AUC", "note"])
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Training and evaluation
# ---------------------------------------------------------------------------

def alert_budgets(y, proba, amount=None, budgets=(0.005, 0.01, 0.02)):
    """Recall, dollar recall and precision when analysts review only the top share of transactions."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(proba), kind="stable")
    rows = []
    for b in budgets:
        k = max(1, int(b * len(y)))
        flagged = order[:k]
        row = {"Budget": f"{b:.1%}", "Alerts": k,
               "Recall": y[flagged].sum() / max(y.sum(), 1),
               "Precision": y[flagged].mean()}
        if amount is not None:
            a = np.asarray(amount, dtype=float)
            row["Dollar recall"] = np.nansum(a[flagged] * y[flagged]) / max(np.nansum(a * y), 1e-9)
        rows.append(row)
    return pd.DataFrame(rows)


def train_and_evaluate(d, mapping, features, n_estimators=200, seed=42):
    """Split by time, fit category risk on the training part, train a Random Forest
    (our Random Forest Tuned settings), pick the F2 threshold on validation and score the test part."""
    train_sub, val, test, cut_val, cut_test = time_split(d)
    risk_map = None
    if "category_risk" in features:
        risk_map = fit_category_risk(train_sub, mapping["category"], "_y")
        for part in (train_sub, val, test):
            part["category_risk"] = apply_category_risk(part, mapping["category"], risk_map)

    for name, part in [("training", train_sub), ("validation", val), ("test", test)]:
        if part["_y"].nunique() < 2:
            raise ValueError(f"the {name} part has no fraud cases (or only fraud cases), so we cannot train "
                             "and evaluate on it. Use more data or a longer time period.")

    model = RandomForestClassifier(n_estimators=n_estimators, max_features=min(3, len(features)),
                                   min_samples_leaf=5, random_state=seed, n_jobs=-1)
    model.fit(train_sub[features], train_sub["_y"])
    val_proba = model.predict_proba(val[features])[:, 1]
    threshold = tune_threshold(val["_y"], val_proba)
    test_proba = model.predict_proba(test[features])[:, 1]

    metrics = point_metrics(test["_y"], test_proba, threshold)
    metrics["Val PR-AUC"] = average_precision_score(val["_y"], val_proba) if val["_y"].nunique() > 1 else np.nan
    metrics["ROC-AUC"] = roc_auc_score(test["_y"], test_proba) if test["_y"].nunique() > 1 else np.nan
    metrics["Threshold"] = threshold
    metrics["Test fraud rate"] = test["_y"].mean()

    precision, recall, _ = precision_recall_curve(test["_y"], test_proba)
    pr_curve = pd.DataFrame({"Recall": recall, "Precision": precision})

    importance = (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    # reference values from normal training transactions, used to explain alerts
    normal = train_sub[train_sub["_y"] == 0]
    reference = {f: (normal[f].quantile(0.05), normal[f].quantile(0.95)) for f in features}

    bundle = {"model": model, "features": features, "mapping": mapping, "threshold": threshold,
              "risk_map": risk_map, "reference": reference, "metrics": metrics}
    test = test.assign(fraud_score=test_proba, alert=(test_proba >= threshold).astype(int))
    split_info = {"validation starts": cut_val, "test starts": cut_test,
                  "train rows": len(train_sub), "validation rows": len(val), "test rows": len(test),
                  "train fraud": int(train_sub["_y"].sum()), "validation fraud": int(val["_y"].sum()),
                  "test fraud": int(test["_y"].sum())}
    return bundle, test, pr_curve, importance, split_info


# ---------------------------------------------------------------------------
# 6. Scoring new data and explaining alerts
# ---------------------------------------------------------------------------

def score(bundle, df):
    """Score a new table with a trained model. The table needs the same column names
    as the training data. Its own history is used for the time gaps."""
    m = dict(bundle["mapping"])
    m["label"] = m["label"] if m.get("label") in df.columns else None
    d = prepare(df, m)
    if "category_risk" in bundle["features"]:
        d["category_risk"] = apply_category_risk(d, m["category"], bundle["risk_map"])
    missing = [f for f in bundle["features"] if f not in d.columns]
    if missing:
        raise ValueError(f"the new data cannot build these features: {missing}")
    d["fraud_score"] = bundle["model"].predict_proba(d[bundle["features"]])[:, 1]
    d["alert"] = (d["fraud_score"] >= bundle["threshold"]).astype(int)
    return d


def explain(row, reference):
    """Short reasons for an alert: the features that look unusual compared with normal transactions."""
    reasons = []
    low, high = reference.get("gap_log", (None, None))
    if "gap_log" in row and low is not None and 0 <= row["gap_log"] <= low:
        minutes = np.expm1(row["gap_log"]) * 60
        reasons.append(f"very soon after the card's previous transaction ({minutes:.0f} min)")
    if row.get("is_night", 0) == 1:
        reasons.append("at night")
    if "amt_log" in row and row["amt_log"] >= reference.get("amt_log", (0, np.inf))[1]:
        reasons.append(f"unusually large amount ({np.expm1(row['amt_log']):,.2f})")
    if "category_risk" in row and row["category_risk"] >= reference.get("category_risk", (0, np.inf))[1]:
        reasons.append("high-risk merchant category")
    if "dist_home_km" in row and row["dist_home_km"] >= reference.get("dist_home_km", (0, np.inf))[1]:
        reasons.append(f"far from home ({row['dist_home_km']:,.0f} km)")
    if row.get("abroad", 0) == 1:
        reasons.append("merchant abroad")
    elif row.get("outside_home_state", 0) == 1:
        reasons.append("merchant outside the home state")
    return "; ".join(reasons) if reasons else "combination of features"


# ---------------------------------------------------------------------------
# 7. Saving and loading a trained model
# ---------------------------------------------------------------------------

def to_bytes(bundle):
    buf = io.BytesIO()
    pickle.dump(bundle, buf)
    return buf.getvalue()


def from_bytes(data):
    # only load model files that we saved ourselves: pickle files can run code when loaded
    return pickle.loads(data)
