"""
Shared helper functions for the fraud detection pipeline.

split.py, the EDA notebook, and the feature-engineering script (coming
next) all need the same handful of calculations - distance between two
GPS points, how long since a card's last transaction, how fast someone
would've had to travel between transactions, and so on. Instead of
writing these separately in every file and risking them drifting out of
sync, we define them once here and import them wherever they're needed.
"""

import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Straight-line distance in km between two GPS points, accounting for
    the Earth being a sphere rather than a flat plane. Used to measure
    how far a transaction happened from a cardholder's home, and how far
    apart two consecutive transactions were.
    """
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_prev_txn_features(df, id_column, time_column, lat_column, lon_column):
    """
    Adds three columns to a transactions dataframe, each comparing a row
    to that same cardholder's previous transaction:

      - time_gap_hours        : hours since the cardholder's last transaction
      - dist_from_prev_txn_km : how far this transaction is from the last one
      - implied_velocity_kmh  : distance divided by time - "how fast would
                                 this person have had to travel?"

    All three only ever look backward in time for a given cardholder, so
    they're safe to compute on the full dataset before splitting - each
    row's value only depends on that same cardholder's past, never on
    anyone else's future.
    """
    df = df.sort_values([id_column, time_column]).copy()

    prev_lat = df.groupby(id_column)[lat_column].shift(1)
    prev_lon = df.groupby(id_column)[lon_column].shift(1)
    prev_time = df.groupby(id_column)[time_column].shift(1)

    df["dist_from_prev_txn_km"] = haversine_distance(df[lat_column], df[lon_column], prev_lat, prev_lon)
    df["time_gap_hours"] = (df[time_column] - prev_time).dt.total_seconds() / 3600

    # add a tiny amount of time (1 second) so we never divide by zero
    # for two transactions that happened at the exact same instant
    epsilon = 1 / 3600
    df["implied_velocity_kmh"] = df["dist_from_prev_txn_km"] / (df["time_gap_hours"] + epsilon)

    return df


def add_rolling_txn_count(df, id_column, time_column, window="24h", amount_column="amt"):
    """
    Adds a column counting how many transactions this cardholder made in
    the trailing time window (default: past 24 hours), up to and
    including the current transaction. Backward-looking, same as above -
    safe to compute before splitting.
    """
    df = df.sort_values([id_column, time_column]).copy()
    df_indexed = df.set_index(time_column)

    column_name = f"txn_count_{window}"
    df[column_name] = (
        df_indexed.groupby(id_column)[amount_column]
        .rolling(window)
        .count()
        .reset_index(level=0, drop=True)
        .values
    )
    return df


def fit_category_risk(train_df, category_column, label_column):
    """
    Calculates the fraud rate for each merchant category, using TRAIN
    DATA ONLY. This is different from the functions above: it's built
    from the fraud label itself, so if we calculated it on the full
    dataset (train + test), we'd be leaking test-set fraud information
    into a feature the model sees during training. Fit here, then apply
    the same frozen numbers to test with apply_category_risk() below.
    """
    risk_by_category = train_df.groupby(category_column)[label_column].mean().to_dict()
    fallback_rate = train_df[label_column].mean()  # used for any category test has that train never saw
    return {"risk_by_category": risk_by_category, "fallback_rate": fallback_rate}


def apply_category_risk(df, category_column, fitted):
    """Applies a risk lookup produced by fit_category_risk() to any dataframe."""
    return df[category_column].map(fitted["risk_by_category"]).fillna(fitted["fallback_rate"])


def id_overlap_report(train_df, test_df, id_column):
    """
    Reports how many cardholders appear in both train and test. This is
    just informational, not a red flag - for our chronological split,
    the same cardholder legitimately shows up in both (their earlier
    transactions in train, later ones in test). Recording this here
    means it's a documented, expected fact instead of a surprise later.
    """
    train_ids = set(train_df[id_column].unique())
    test_ids = set(test_df[id_column].unique())
    overlap = train_ids & test_ids

    return {
        "cardholders_in_train": len(train_ids),
        "cardholders_in_test": len(test_ids),
        "cardholders_in_both": len(overlap),
    }
