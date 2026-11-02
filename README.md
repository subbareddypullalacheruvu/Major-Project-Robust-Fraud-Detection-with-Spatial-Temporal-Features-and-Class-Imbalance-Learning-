# Robust Fraud Detection with Spatial-Temporal Features and Class-Imbalance Learning

COMP8851 Major Project · Group 2 · Macquarie University, School of Computing
Supervisor: Dr Mahmoud Bekhit

## What this project is about

Credit card fraud is rare. In our data, fewer than 1 in 170 transactions are fraud, so a model can be 99% accurate and still miss every fraud. We build fraud-detection models that cope with this imbalance, and we test which features really help.

Our starting point is Bekhit et al. (2025), which found that a merchant-category risk score helped detect fraud but that distance between transactions barely did. We add a stronger movement feature, **implied velocity** (distance between two transactions divided by the time between them), and then ask a simple question:

> **Does velocity work because of *where* transactions happen, or because of *when* they happen?**

Our answer, on the Sparkov data, is **when**. Merchant locations add nothing, and the useful part of velocity is the short time gap between fraudulent transactions.

## Main results

All numbers are on the test period (Kaggle's `fraudTest.csv`, 21 June to 31 December 2020). Each model is trained with five seeds, and the range in brackets is a 95% confidence interval from a bootstrap that resamples whole days.

### Best model

| Model | PR-AUC [95% CI] | F2 | MCC |
|---|---|---|---|
| **Random Forest Tuned** | **0.834 [0.814, 0.853]** | 0.789 | 0.718 |
| CatBoost Tuned | 0.819 [0.797, 0.839] | 0.773 | 0.687 |
| Random Forest (default settings) | 0.814 [0.792, 0.835] | 0.780 | 0.712 |
| XGBoost Weighted | 0.794 [0.770, 0.815] | 0.753 | 0.668 |
| Balanced Random Forest | 0.727 [0.697, 0.753] | 0.760 | 0.673 |

Random Forest Tuned beats every other model, and every gap is larger than chance. Models with built-in imbalance handling never beat the same model without it.

### Location or time?

| Features | PR-AUC [95% CI] |
|---|---|
| Six features (with velocity) | 0.834 [0.814, 0.853] |
| Six features, **time gap instead of velocity** | **0.844 [0.825, 0.863]** |
| Six features, merchant locations shuffled | 0.833 [0.814, 0.852] |
| Six features, no velocity at all | 0.817 [0.796, 0.836] |
| Distance only / time gap only / velocity only | about 0.004 each |

- Shuffling the merchant locations within each card leaves the score unchanged, so **location adds nothing**.
- Fraud transactions are about 1.3 hours apart, against 4.6 hours for genuine ones, while the distances are almost the same (98 km vs 101 km).
- Using the time gap instead of velocity gives a slightly **higher** score.
- No movement feature works on its own. The time gap is useful only together with the amount, the merchant category and the time of day.

### Performance over time

Month by month, test PR-AUC stays between 0.82 and 0.87 from June to November. It drops to 0.78 in December, when transaction volume doubles and the fraud rate halves. ROC-AUC (0.99) and recall (about 87%) stay steady, so the drop comes from fraud being rarer, not from the model getting worse.

## Datasets

| Dataset | What it is | How we use it |
|---|---|---|
| [Sparkov](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | Synthetic card transactions, 1.85 million rows, 2019–2020, with time, amount, category and merchant location | Main dataset for all feature and model work |
| [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Real European card transactions, anonymised features | First baseline |
| [IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection) | Real e-commerce transactions, no card number and no location | Checks whether the timing signal holds on real data |

The data files are not stored in this repository. The notebooks download them from Kaggle when they run.

**Sparkov is synthetic.** We read the code of its generator: merchant locations are random points near the cardholder's home, and fraud is created as a burst of 7–16 transactions on one day. We designed our velocity tests around this, and we use IEEE-CIS to check the findings on real data.

## How we evaluate

We use the same rules in every notebook so that results are comparable:

1. **Time order is never broken.** Training data comes before validation data, which comes before test data. Validation is the last 20% of the training period.
2. **No information leaks from test.** Features that use fraud labels (such as `category_risk`) are learned from the training data only.
3. **The threshold is chosen on validation.** We pick the threshold with the best F2 score on validation and use the test set once.
4. **Metrics suited to rare fraud:** PR-AUC, F1, F2 and MCC, not accuracy.
5. **Five seeds and confidence intervals.** Every number is an average over five seeds, with a 95% interval from a day-block bootstrap. We resample whole days because fraud comes in bursts within a day.

The shared code for these rules is in `src/evaluation.py`.

## Repository structure

```
├── data/               Datasets are placed here when a notebook runs (not committed)
├── notebooks/          One notebook per project stage (see below)
├── results/            Results tables (CSV), saved models and figures
├── src/
│   ├── split.py        Loads Sparkov and ULB and makes the time-based train/test split
│   ├── utils.py        Feature helpers: distance, time gap, implied velocity, category risk
│   └── evaluation.py   Time-based validation split, threshold choice, bootstrap intervals
├── requirements.txt
└── README.md
```

## Notebooks, in the order we built them

| Notebook | What it does |
|---|---|
| `week2_setupipynb.ipynb` | First environment and data setup |
| `session_bootstrap.ipynb` | Loads the data and helpers at the start of a working session |
| `data_setup_split.ipynb` | Makes the time-based train/test splits and checks that they do not overlap |
| `eda_train_only.ipynb` | Explores the training data only: imbalance, amount, time of day, category, velocity |
| `baseline_ulb.ipynb` | Logistic Regression baseline on ULB |
| `feature_engineering.ipynb` | Builds and tests the six Sparkov features one step at a time |
| `imbalance_comparison.ipynb` | Compares class weights, SMOTE and focal loss, with Logistic Regression and Random Forest |
| `boosting_comparison.ipynb` | XGBoost and LightGBM, with the threshold tuned on validation |
| `final_model_comparison.ipynb` | Adds CatBoost and Balanced Random Forest, tunes the best models, selects Random Forest Tuned |
| `time_honest_evaluation.ipynb` | Reruns the final comparison with time-based validation, five seeds and confidence intervals |
| `velocity_breakdown.ipynb` | Separates distance, time gap and velocity, and runs the location-shuffle test |
| `bank_metrics_fairness.ipynb` | Alert budgets (recall and dollars caught when reviewing the top 0.5%, 1% or 2%), and a fairness check on gender and age |
| `ieee_transfer_and_drift.ipynb` | Month-by-month drift on Sparkov, and the timing features on IEEE-CIS |

The six final features are: `amt_log` (log of the amount), `category_risk` (fraud rate of the merchant category), `is_night` (10 pm to 3 am), `velocity_log` (log of implied velocity), `age` and `gender_flag`.

## How to run

We run every notebook in **Google Colab** on a standard CPU.

1. Open a notebook in Colab using the "Open in Colab" badge at its top.
2. Have these ready. Each is typed in when the notebook asks, and none is saved in the notebook:
   - a **Kaggle username and API key** (from your Kaggle account settings);
   - a **GitHub token**, to clone this repository.
3. For IEEE-CIS, first join the [competition](https://www.kaggle.com/c/ieee-fraud-detection) on Kaggle and accept its rules, otherwise the download fails with a 401 error. You can also download the files by hand and upload them when the notebook asks.
4. Run the cells from top to bottom.

**Long runs.** The later notebooks train many models and take 2 to 4 hours. They save each finished model to Google Drive, so if Colab disconnects, running the notebook again skips everything already finished. Allow access to Google Drive when the notebook asks.

To run locally instead:

```bash
pip install -r requirements.txt
pip install catboost kaggle
```

## Limitations

- Sparkov is synthetic, so findings about velocity describe how its generator works. IEEE-CIS is our check on real data.
- IEEE-CIS has no card number, so we build an approximate card identifier from `card1`, `addr1` and the account start date.
- The Random Forest settings were chosen on an earlier random validation split.
- Removing gender from a model does not remove information that other features carry about it.

## Team

- Subbareddy Pullalacheruvu
- Uday Reddy Poli
- Niharika Kunaminna
- Trilok Kumar Reddy Ande

## Key references

- Bekhit, M., et al. (2025). The study this project extends: merchant-category risk and distance features for card-fraud detection.
- Kabane, S. (2024). On data leakage from resampling before splitting in fraud detection.
- Imani, M., et al. (2026). On choosing evaluation metrics for imbalanced classification.
- Elor, Y., & Averbuch-Elor, H. (2022). *To SMOTE, or not to SMOTE?*

The full reference list is in our project report.
