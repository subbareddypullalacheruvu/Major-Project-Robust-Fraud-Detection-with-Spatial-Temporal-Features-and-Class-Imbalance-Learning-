"""
Train/test split for the fraud detection pipeline (Stage 4 of our methodology).

Run this once. Every later stage - EDA, feature engineering, baseline
training, model comparison, ablation - reads from data/processed/ instead
of splitting the data again. Keeping the split in one place is what
actually makes the leak-free guarantee hold across the rest of the
project; re-deriving it in different notebooks is how that guarantee
quietly breaks.

ULB doesn't come with a train/test split, so we cut it ourselves along
time (see chronological_split() below). Sparkov is different - Kaggle
already ships fraudTrain.csv and fraudTest.csv, and we didn't take that
on faith before using it. We checked: no overlap between the two files'
transaction times (train's last transaction is 2020-06-21 12:13:37,
test's first is 2020-06-21 12:14:25), zero transaction IDs shared between
them, zero duplicate rows in either file. Given that, we use Kaggle's
split as-is instead of re-splitting fraudTrain.csv ourselves, which
would've thrown away all 555,719 rows of fraudTest.csv for nothing.
"""

import pandas as pd
from pathlib import Path

# Where the raw Kaggle data lives, and where the split files get saved
RAW_DIR = Path("data")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def chronological_split(df, time_column, train_fraction=0.8):
    """
    Splits a dataframe into train/test by TIME, not randomly - everything
    before the cutoff goes to train, everything after goes to test.
    Used for ULB, which has no ready-made split.
    """
    # Step 1: put every row in true time order
    df = df.sort_values(time_column).reset_index(drop=True)

    # Step 2: find the row that marks 80% of the way through
    cutoff_row = int(len(df) * train_fraction)

    # Step 3: everything before that row is train, everything after is test
    train = df.iloc[:cutoff_row].copy()
    test = df.iloc[cutoff_row:].copy()

    print(f"Train: {len(train):,} rows, from {train[time_column].min()} to {train[time_column].max()}")
    print(f"Test:  {len(test):,} rows, from {test[time_column].min()} to {test[time_column].max()}")

    # Safety check: train must fully end before test begins
    assert train[time_column].max() <= test[time_column].min(), "Something's wrong - the split is not chronological!"

    return train, test


def split_sparkov():
    """
    Loads Sparkov's train/test files exactly as Kaggle provides them.
    We're not splitting anything here - we already checked by hand that
    Kaggle's files are clean and chronological, so we just load and save them.
    """
    train = pd.read_csv(RAW_DIR / "sparkov" / "fraudTrain.csv")
    test = pd.read_csv(RAW_DIR / "sparkov" / "fraudTest.csv")

    train["trans_date_trans_time"] = pd.to_datetime(train["trans_date_trans_time"])
    test["trans_date_trans_time"] = pd.to_datetime(test["trans_date_trans_time"])

    # These checks should always pass - we already verified this by hand.
    # If they ever fail, something changed and needs a look.
    assert train["trans_date_trans_time"].max() < test["trans_date_trans_time"].min(), \
        "Train and test transaction times overlap - stop and check!"

    shared_ids = set(train["trans_num"]) & set(test["trans_num"])
    assert len(shared_ids) == 0, "Found the same transaction in both train and test - stop and check!"

    print(f"Train: {len(train):,} rows, from {train['trans_date_trans_time'].min()} to {train['trans_date_trans_time'].max()}")
    print(f"Test:  {len(test):,} rows, from {test['trans_date_trans_time'].min()} to {test['trans_date_trans_time'].max()}")

    train.to_csv(PROCESSED_DIR / "sparkov_train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "sparkov_test.csv", index=False)

    return train, test


def split_ulb():
    """Loads ULB, removes duplicate rows, splits it by time, and saves the result."""
    ulb = pd.read_csv(RAW_DIR / "ulb" / "creditcard.csv")
    ulb = ulb.drop_duplicates()  # duplicates found during EDA

    # ULB's "Time" column is just seconds since the first transaction,
    # so it's already something we can sort by directly
    train, test = chronological_split(ulb, time_column="Time", train_fraction=0.8)

    train.to_csv(PROCESSED_DIR / "ulb_train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "ulb_test.csv", index=False)

    return train, test


# Lets this file be run directly from a terminal with: python src/split.py
if __name__ == "__main__":
    print("Splitting Sparkov...")
    split_sparkov()

    print("\nSplitting ULB...")
    split_ulb()
