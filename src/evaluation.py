"""
Evaluation helpers for the paper experiments.

We use these functions in every experiment (the final model comparison, the
velocity breakdown, the location-shuffle test, alert budgets and fairness),
so that all of our reported numbers are produced in exactly the same way.

Our protocol:
  1. Validation is the last 20% of the training period by time, so it comes
     after the data we fit on, just like the test set does.
  2. We pick the F2-optimal threshold on validation and apply it once to test.
  3. We train each model with five different seeds and average the results.
  4. We put a 95% confidence interval on every number by resampling whole
     test days (a day-block bootstrap). Sparkov creates fraud in bursts of
     7-16 transactions in one day, so we resample days rather than single
     transactions to keep each burst together.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score


# ---------------------------------------------------------------------------
# 1. Splitting the training data by time
# ---------------------------------------------------------------------------

def chronological_val_split(train_df, time_col, val_frac=0.2):
    """Use the first 80% of the training period for fitting and the last 20%
    for validation. Returns (train_sub, val, cutoff)."""
    times = pd.to_datetime(train_df[time_col])

    # the cutoff date sits 80% of the way through the training period
    cutoff = times.min() + (1 - val_frac) * (times.max() - times.min())

    train_sub = train_df[times < cutoff].copy()
    val = train_df[times >= cutoff].copy()

    # check that no validation transaction is earlier than the fitting data
    assert pd.to_datetime(train_sub[time_col]).max() < pd.to_datetime(val[time_col]).min()
    return train_sub, val, cutoff


# ---------------------------------------------------------------------------
# 2. Choosing the threshold and scoring the predictions
# ---------------------------------------------------------------------------

def tune_threshold(y_val, val_proba):
    """Try thresholds from 0.01 to 0.99 and keep the one with the best F2 on validation."""
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    f2_scores = [fbeta_score(y_val, (val_proba >= t).astype(int), beta=2, zero_division=0)
                 for t in thresholds]
    return float(thresholds[np.argmax(f2_scores)])


def point_metrics(y, proba, threshold, weights=None):
    """PR-AUC, F1, F2, MCC, precision and recall.

    weights is only used by the bootstrap: it says how many times each row
    was drawn. Without weights, every row counts once.
    """
    y = np.asarray(y)
    if weights is None:
        weights = np.ones(len(y))
    pred = (proba >= threshold).astype(int)

    # weighted counts of true/false positives and negatives
    tp = np.sum(weights * ((pred == 1) & (y == 1)))
    fp = np.sum(weights * ((pred == 1) & (y == 0)))
    fn = np.sum(weights * ((pred == 0) & (y == 1)))
    tn = np.sum(weights * ((pred == 0) & (y == 0)))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall > 0 else 0.0
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0 else 0.0

    return {
        "PR-AUC": average_precision_score(y, proba, sample_weight=weights),
        "F1": f1, "F2": f2, "MCC": mcc,
        "Precision": precision, "Recall": recall,
    }


# ---------------------------------------------------------------------------
# 3. Training one model over several seeds
# ---------------------------------------------------------------------------

def run_seeds(name, make_model, X_tr, y_tr, X_val, y_val, X_test, seeds,
              checkpoint_dir=None, fit_kwargs=None):
    """Train the model once per seed, tune its threshold on validation and
    keep its test probabilities.

    make_model(seed) returns a new, unfitted model.
    fit_kwargs(seed) can return extra fit() arguments (CatBoost's eval_set).

    We save every finished seed to checkpoint_dir. If Colab disconnects, we
    only lose the seed that was running, and re-running skips saved seeds.
    """
    results = []
    for seed in seeds:
        file_path = None
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            file_path = os.path.join(checkpoint_dir, f"{name.replace(' ', '_')}__seed{seed}.npz")

            # this seed was already finished in an earlier session
            if os.path.exists(file_path):
                saved = np.load(file_path)
                results.append({"seed": seed,
                                "threshold": float(saved["threshold"]),
                                "val_pr_auc": float(saved["val_pr_auc"]),
                                "test_proba": saved["test_proba"]})
                print(f"{name} seed {seed}: loaded from checkpoint")
                continue

        model = make_model(seed)
        extra = fit_kwargs(seed) if fit_kwargs else {}
        model.fit(X_tr, y_tr, **extra)

        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1].astype(np.float32)
        threshold = tune_threshold(y_val, val_proba)
        val_pr_auc = average_precision_score(y_val, val_proba)

        if file_path:
            np.savez_compressed(file_path, threshold=threshold,
                                val_pr_auc=val_pr_auc, test_proba=test_proba)

        results.append({"seed": seed, "threshold": threshold,
                        "val_pr_auc": val_pr_auc, "test_proba": test_proba})
        print(f"{name} seed {seed}: threshold={threshold:.2f}  val PR-AUC={val_pr_auc:.4f}")
    return results


# ---------------------------------------------------------------------------
# 4. Day-block bootstrap
# ---------------------------------------------------------------------------

def bootstrap_model(y_test, runs, days, n_boot=500, boot_seed=0):
    """Resample whole test days n_boot times and score the model on each resample.

    runs is a list of (test_proba, threshold), one pair per seed. On each
    resample we score every seed and average them, because the number we
    report is the average over seeds.

    Instead of copying rows, we give each row a weight equal to how many
    times its day was drawn. A day drawn twice counts twice; a day not drawn
    counts zero times.

    We use the same boot_seed for every model, so all models are scored on
    the same resampled days and can be compared fairly.
    """
    y_test = np.asarray(y_test)
    days = pd.Series(days).reset_index(drop=True)
    unique_days = days.unique()
    rng = np.random.default_rng(boot_seed)

    rows = []
    for _ in range(n_boot):
        # draw as many days as the test period has, with replacement
        drawn = rng.choice(unique_days, size=len(unique_days), replace=True)
        times_drawn = pd.Series(drawn).value_counts()
        weights = days.map(times_drawn).fillna(0).values

        seed_scores = [point_metrics(y_test, proba, t, weights) for proba, t in runs]
        rows.append(pd.DataFrame(seed_scores).mean().to_dict())
    return pd.DataFrame(rows)


def summarise(y_test, runs, boot_df, metrics=("PR-AUC", "F1", "F2", "MCC")):
    """For each metric: the average over seeds on the full test set, the
    standard deviation across seeds, and the 95% bootstrap interval."""
    per_seed = pd.DataFrame([point_metrics(y_test, proba, t) for proba, t in runs])
    summary = {}
    for m in metrics:
        summary[m] = per_seed[m].mean()
        summary[f"{m} seed SD"] = per_seed[m].std()
        summary[f"{m} CI low"] = np.percentile(boot_df[m], 2.5)
        summary[f"{m} CI high"] = np.percentile(boot_df[m], 97.5)
    summary["Thresholds"] = ", ".join(f"{t:.2f}" for _, t in runs)
    return summary


def paired_difference(boot_a, boot_b, metric="PR-AUC"):
    """Interval for (model A - model B) on the same resampled days.
    If the interval does not include 0, we treat the gap as real."""
    diff = boot_a[metric].values - boot_b[metric].values
    return {"mean diff": diff.mean(),
            "CI low": np.percentile(diff, 2.5),
            "CI high": np.percentile(diff, 97.5),
            "P(A > B)": np.mean(diff > 0)}


def format_ci(summary, metric, digits=4):
    """Formats a result as '0.8326 [0.8100, 0.8500]' for our tables."""
    return (f"{summary[metric]:.{digits}f} "
            f"[{summary[metric + ' CI low']:.{digits}f}, {summary[metric + ' CI high']:.{digits}f}]")
