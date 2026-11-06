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

def _sorted_for_pr_auc(proba):
    """Sort one seed's test scores once, from the highest to the lowest, in the same
    way as scikit-learn, and mark the last row of each group of tied scores."""
    proba = np.asarray(proba)
    order = np.argsort(proba, kind="mergesort")[::-1]
    sorted_scores = proba[order]
    group_ends = np.r_[np.where(np.diff(sorted_scores))[0], len(sorted_scores) - 1]
    return order, group_ends


def _weighted_pr_auc(times_drawn, day_sorted, pos_day, pos_before_end, group_ends):
    """Weighted average precision from rows that are already sorted by score.
    This follows scikit-learn's average_precision_score step by step: rows with
    zero weight are ignored, tied scores form one point, and the area is the
    step-function integral of precision over recall.

    Every weight is a whole number (how many times a day was drawn), so the
    running counts of true and false positives are whole numbers too, and adding
    them up in a different order gives exactly the same totals. We therefore add
    up the weights of the fraud rows only (a few thousand) for the true positives,
    and take the false positives as all weight minus the true positives."""
    w_end = np.cumsum(times_drawn[day_sorted])[group_ends]
    tps = np.r_[0.0, np.cumsum(times_drawn[pos_day])][pos_before_end]
    fps = w_end - tps
    # a group whose rows were all drawn zero times is not a point on the curve
    keep = np.diff(np.r_[0.0, w_end]) > 0
    tps, fps = tps[keep], fps[keep]
    if len(tps) == 0 or tps[-1] == 0:
        return 0.0
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    return float(max(0.0, -np.sum(np.diff(recall) * precision[:-1])))


def bootstrap_model(y_test, runs, days, n_boot=500, boot_seed=0, n_check=3):
    """Resample whole test days n_boot times and score the model on each resample.

    runs is a list of (test_proba, threshold), one pair per seed. On each
    resample we score every seed and average them, because the number we
    report is the average over seeds.

    Instead of copying rows, we give each row a weight equal to how many
    times its day was drawn. A day drawn twice counts twice; a day not drawn
    counts zero times.

    We use the same boot_seed for every model, so all models are scored on
    the same resampled days and can be compared fairly.

    This gives the same resampled days and the same numbers as
    bootstrap_model_reference, but faster: each seed's scores are sorted once
    instead of once per resample, and the threshold counts (for F1, F2, MCC,
    precision and recall) are added up per day once. On the first n_check
    resamples we also score with point_metrics and stop if any number differs.
    """
    y_test = np.asarray(y_test)
    days = pd.Series(days).reset_index(drop=True)
    unique_days = days.unique()
    day_index = pd.Index(unique_days)
    day_of_row = day_index.get_indexer(days)
    n_days = len(unique_days)
    rng = np.random.default_rng(boot_seed)

    prepared = []
    positive = y_test == 1
    for proba, t in runs:
        order, group_ends = _sorted_for_pr_auc(proba)
        pred = np.asarray(proba) >= t
        # weighted counts per day; a resample's count is the sum over days of count x times drawn
        per_day = {name: np.bincount(day_of_row, weights=mask.astype(float), minlength=n_days)
                   for name, mask in [("tp", pred & positive), ("fp", pred & ~positive),
                                      ("fn", ~pred & positive), ("tn", ~pred & ~positive)]}
        y_sorted, day_sorted = y_test[order], day_of_row[order]
        pos_rows = np.where(y_sorted == 1)[0]
        # how many fraud rows sit at or above the end of each group of tied scores
        pos_before_end = np.searchsorted(pos_rows, group_ends, side="right")
        prepared.append((group_ends, day_sorted, day_sorted[pos_rows], pos_before_end, per_day))

    rows = []
    for b in range(n_boot):
        # draw as many days as the test period has, with replacement
        drawn = rng.choice(unique_days, size=len(unique_days), replace=True)
        times_drawn = np.bincount(day_index.get_indexer(drawn), minlength=n_days).astype(float)

        seed_scores = []
        for group_ends, day_sorted, pos_day, pos_before_end, per_day in prepared:
            tp, fp, fn, tn = (float(per_day[k] @ times_drawn) for k in ("tp", "fp", "fn", "tn"))
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall > 0 else 0.0
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0 else 0.0
            pr_auc = _weighted_pr_auc(times_drawn, day_sorted, pos_day, pos_before_end, group_ends)
            seed_scores.append({"PR-AUC": pr_auc, "F1": f1, "F2": f2, "MCC": mcc,
                                "Precision": precision, "Recall": recall})

        if b < n_check:
            # the same resample scored the original way, to confirm the numbers match
            weights = days.map(pd.Series(drawn).value_counts()).fillna(0).values
            for (proba, t), fast in zip(runs, seed_scores):
                slow = point_metrics(y_test, proba, t, weights)
                for m in fast:
                    if not np.isclose(fast[m], slow[m], rtol=0, atol=1e-10):
                        raise RuntimeError(f"fast bootstrap differs from point_metrics on {m}: "
                                           f"{fast[m]!r} vs {slow[m]!r}")
        rows.append(pd.DataFrame(seed_scores).mean().to_dict())
    return pd.DataFrame(rows)


def bootstrap_model_reference(y_test, runs, days, n_boot=500, boot_seed=0):
    """The original (slower) day-block bootstrap. bootstrap_model gives the same
    numbers faster; we keep this version to check that against.

    Resample whole test days n_boot times and score the model on each resample.

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
