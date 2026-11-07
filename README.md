# Robust Fraud Detection with Spatial-Temporal Features and Class-Imbalance Learning

COMP8851 Major Project, Group 2, Macquarie University, School of Computing
Supervisor: Dr Mahmoud Bekhit

## What this project is about

Credit card fraud is rare. In our main dataset, fewer than 1 in 170 transactions are fraud, so a model can be 99% accurate and still miss every fraud. We build fraud-detection models that cope with this imbalance, and we test which features really help.

Our starting point is Bekhit et al. (2025), which found that a merchant-category risk score helped detect fraud but that the distance between transactions barely did. We add a stronger movement feature, **implied velocity** (distance between two transactions divided by the time between them), and then ask a simple question:

> **Does velocity work because of *where* transactions happen, or because of *when* they happen?**

On Sparkov, our main dataset, the answer is **when**. Merchant locations add nothing, and the useful part of velocity is the short time gap between fraudulent transactions. So our final model uses the **time gap** instead of velocity.

We then ran the same protocol on three more datasets. The answer depends on how fraud is created in each one. In TabFormer and Credit Card Fraud 2026 the signal is **where** (the merchant's location), not when. On IEEE-CIS, which is real data, the time gap still carries signal on top of the dataset's own count columns.

Finally, we built **FraudLens**, a web app that takes any card transaction table, builds both kinds of features, trains a model with our leak-free protocol and ranks transactions for review.

## Main results

All Sparkov numbers are on the test period (Kaggle's `fraudTest.csv`, 21 June to 31 December 2020). Each model is trained with five seeds, and the range in brackets is a 95% confidence interval from a bootstrap that resamples whole days.

### Final model

Random Forest Tuned with six features: `amt_log` (log of the amount), `category_risk` (fraud rate of the merchant category, learned from training data only), `is_night` (10 pm to 3 am), `gap_log` (log of the hours since the card's previous transaction), `age` and `gender_flag`.

| Model | PR-AUC [95% CI] | F1 | F2 | MCC |
|---|---|---|---|---|
| **Random Forest Tuned, time gap (final)** | **0.844 [0.825, 0.863]** | 0.715 | 0.794 | 0.724 |
| Random Forest Tuned, velocity | 0.834 [0.814, 0.853] | 0.709 | 0.789 | 0.718 |

Switching from velocity to the time gap raises PR-AUC by +0.010 [0.006, 0.015]. Validation PR-AUC also rises (0.884 against 0.877), so the switch is supported by validation and not only by the test set.

### Model comparison

We chose the model with the velocity feature set, before the switch to the time gap.

| Model | PR-AUC [95% CI] | F2 | MCC |
|---|---|---|---|
| **Random Forest Tuned** | **0.834 [0.814, 0.853]** | 0.789 | 0.718 |
| CatBoost Tuned | 0.819 [0.797, 0.839] | 0.773 | 0.687 |
| Random Forest (default settings) | 0.814 [0.792, 0.835] | 0.780 | 0.712 |
| XGBoost Weighted | 0.794 [0.770, 0.815] | 0.753 | 0.668 |
| Balanced Random Forest | 0.727 [0.697, 0.753] | 0.760 | 0.673 |

Random Forest Tuned beats every other model, and every gap is larger than chance. Models with built-in imbalance handling never beat the same model without it.

### Location or time on Sparkov?

| Features | PR-AUC [95% CI] |
|---|---|
| Six features, time gap (final) | **0.844 [0.825, 0.863]** |
| Six features, velocity | 0.834 [0.814, 0.853] |
| Six features, merchant locations shuffled | 0.833 [0.814, 0.852] |
| Six features, no movement feature at all | 0.817 [0.796, 0.836] |
| Distance only / time gap only / velocity only | about 0.004 each |

- Shuffling the merchant locations within each card leaves the score unchanged, so **location adds nothing**.
- Fraud transactions are about 1.3 hours apart, against 4.6 hours for genuine ones, while the distances are almost the same (98 km against 101 km).
- No movement feature works on its own. The time gap is useful only together with the amount, the merchant category and the time of day.

### Alert budgets

A bank's analysts can only review a small share of transactions. With the final model:

| Review budget | Alerts | Fraud caught | Fraud dollars caught | Precision |
|---|---|---|---|---|
| Top 0.5% | 2,778 | 84.3% | 94.4% | 65.1% |
| Top 1% | 5,557 | 92.0% | 98.1% | 35.5% |
| Top 2% | 11,114 | 95.2% | 99.4% | 18.4% |

Dollar recall is higher than transaction recall at every budget, so the frauds the model misses tend to be the smaller ones. Going from 1% to 2% doubles the workload for about 3 more points of recall.

### Does the signal transfer to other datasets?

We ran one protocol on three datasets and compared the same seven feature groups (test PR-AUC; random ranking scores the fraud rate).

| Feature group | Sparkov | TabFormer | Credit Card Fraud 2026 |
|---|---|---|---|
| Fraud rate (random ranking) | 0.004 | 0.001 | 0.021 |
| Distance only | 0.004 | 0.001 | 0.021 |
| Time gap only | 0.005 | 0.001 | 0.022 |
| Velocity only | 0.004 | 0.001 | 0.022 |
| Location only | 0.004 | 0.121 | 0.050 |
| Our six features | **0.845** | 0.018 | 0.024 |
| Six features + location | 0.833 | 0.346 | 0.066 |
| All of the dataset's own columns | 0.151 | **0.887** | **0.143** |

The shuffle tests confirm this directly:

| Dataset | Six + location, intact | Drop when locations are shuffled | Drop when time gaps are shuffled |
|---|---|---|---|
| Sparkov | 0.833 [0.812, 0.851] | −0.002 [−0.004, 0.001] | **0.023 [0.016, 0.030]** |
| TabFormer | 0.346 [0.283, 0.407] | **0.329 [0.269, 0.382]** | −0.027 [−0.058, 0.003] |

- **Sparkov:** the signal is temporal. Fraud comes in bursts on one card within a day, and normal card use is sparse, so a burst stands out.
- **TabFormer:** the signal is spatial. Cards are used so often that normal activity also has many short gaps. Almost all of the signal comes from `Merchant State`: fraud is concentrated at merchants in a small set of places outside the cardholder's home state. We checked this for leakage. The column does not record the outcome, and the result holds for users the model never saw (0.858 against 0.887).
- **Credit Card Fraud 2026:** there are no bursts and the movement features are at chance. Location is the only feature group we build that is clearly above chance on its own.
- **IEEE-CIS (real data):** timing carries fraud signal. 14.2% of transactions are fraud when the card's previous transaction was under a minute earlier, against 1.2% when it was more than a week earlier. With LightGBM, the time features add +0.013 [0.007, 0.018] PR-AUC on top of IEEE's own count columns, and a placebo with shuffled card identifiers adds nothing. A linear model cannot use this signal.
- **Financial Transactions (negative control):** every column scores at chance, and our features score the same with real and shuffled labels (lift 1.00). This shows that our protocol does not find signal where there is none.

### Fairness

The final model misses more fraud on women than on men: 19.9% of fraud on women's cards is missed, against 7.5% on men's. Removing gender narrows the gap from −0.124 to −0.109, but it stays large, and removing age as well does not close it. The gap does not come from the gender and age columns alone. Sparkov's customers are synthetic, so this tests our method, not the treatment of real people.

### Performance over time

Month by month, test PR-AUC stays between 0.82 and 0.87 from June to November. It drops to 0.78 in December, when transaction volume doubles and the fraud rate halves. ROC-AUC (0.99) and recall (about 87%) stay steady, so the drop comes from fraud being rarer, not from the model getting worse.

## FraudLens: our prototype

FraudLens is a Streamlit app in `app/`. In five steps it:

1. loads a transaction table (CSV upload, a demo file or a file path);
2. matches the table's columns to roles such as label, time, card, amount, category, customer and location;
3. checks each raw column for leakage, splits the data by time, trains a Random Forest with our settings and picks the alert threshold with the best F2 on validation;
4. lists the riskiest test transactions, each with its reasons and the card's history;
5. scores new transactions with the trained model.

On the results page you can drag a **review budget** and see how much fraud and how many fraud dollars it catches, and move the **alert threshold** to see fraud caught, missed and false alarms change on the precision-recall curve.

The app builds the features from all our experiments: the time gap and night-time (temporal), distance from home, merchant outside the home state and merchant abroad (spatial), and the amount, category risk, age and gender. It only builds the features that a table's columns allow. See `app/README.md` for details.

## Datasets

| Dataset | What it is | How we use it |
|---|---|---|
| [Sparkov](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | Synthetic card transactions, 1.85 million rows, 2019–2020, with time, amount, category and merchant location | Main dataset for all feature and model work |
| [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Real European card transactions, anonymised features | First baseline |
| [IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection) | Real e-commerce transactions, no card number and no location | Checks whether the timing signal holds on real data |
| [IBM TabFormer](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions) | Synthetic card transactions, 24 million rows from 2,000 users, 1991–2020, with merchant city and state | Second validation dataset; we use 400 users from 2015 onwards |
| [Credit Card Fraud 2026](https://www.kaggle.com/datasets/mohankrishnathalla/credit-card-fraud-detection-dataset-2026) | 100,000 transactions on 8,000 cards in 2026, 2.06% fraud | Third validation dataset |
| [Financial Transactions](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection) | Synthetic transactions whose labels carry no signal | Negative control |

The data files are not stored in this repository. The notebooks download them from Kaggle when they run.

**Sparkov is synthetic.** We read the code of its generator: merchant locations are random points near the cardholder's home, and fraud is created as a burst of 7–16 transactions on one day. We designed our velocity tests around this, and we use the other datasets to check the findings.

## How we evaluate

We use the same rules in every notebook so that results are comparable:

1. **Time order is never broken.** Training data comes before validation data, which comes before test data. Validation is the last 20% of the training period.
2. **No information leaks from test.** Features that use fraud labels (such as `category_risk`) are learned from the training data only.
3. **The threshold is chosen on validation.** We pick the threshold with the best F2 score on validation and use the test set once.
4. **Metrics suited to rare fraud:** PR-AUC, F1, F2 and MCC, not accuracy. Across datasets we also report lift (PR-AUC divided by the fraud rate).
5. **Five seeds and confidence intervals.** Every number is an average over five seeds, with a 95% interval from a day-block bootstrap. We resample whole days because fraud comes in bursts within a day.

The shared code for these rules is in `src/evaluation.py`.

## Repository structure

```
├── .streamlit/         App colours and settings (read when the app starts from this folder)
├── app/                FraudLens, our prototype web app (see app/README.md)
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
| `week2_setup.ipynb` | First environment and data setup |
| `session_bootstrap.ipynb` | Loads the data and helpers at the start of a working session |
| `data_setup_split.ipynb` | Makes the time-based train/test splits and checks that they do not overlap |
| `eda_train_only.ipynb` | Explores the training data only: imbalance, amount, time of day, category, velocity |
| `baseline_ulb.ipynb` | Logistic Regression baseline on ULB |
| `feature_engineering.ipynb` | Builds and tests the Sparkov features one step at a time |
| `imbalance_comparison.ipynb` | Compares class weights, SMOTE and focal loss, with Logistic Regression and Random Forest |
| `boosting_comparison.ipynb` | XGBoost and LightGBM, with the threshold tuned on validation |
| `final_model_comparison.ipynb` | Adds CatBoost and Balanced Random Forest, tunes the best models, selects Random Forest Tuned |
| `time_honest_evaluation.ipynb` | Reruns the final comparison with time-based validation, five seeds and confidence intervals |
| `velocity_breakdown.ipynb` | Separates distance, time gap and velocity, and runs the location-shuffle test |
| `bank_metrics_fairness.ipynb` | Alert budgets, out-of-fold `category_risk`, and a fairness check on gender and age |
| `ieee_transfer_and_drift.ipynb` | Month-by-month drift on Sparkov, and the timing features on IEEE-CIS |
| `final_model_gap.ipynb` | Makes the time-gap model our final model: results, alert budgets and fairness |
| `external_dataset_check.ipynb` | First look at Credit Card Fraud 2026 and Financial Transactions |
| `tabformer_validation.ipynb` | Our features on IBM TabFormer, and where its signal comes from |
| `cross_dataset_protocol.ipynb` | One protocol, seven feature groups, three datasets, side by side |
| `cross_dataset_alert_budgets.ipynb` | Alert budgets for every feature group on every dataset |
| `shuffle_tests.ipynb` | Shuffles locations and time gaps within each card on Sparkov and TabFormer |
| `tabformer_leakage_check.ipynb` | Checks the TabFormer result for leakage: `Merchant State`, `Errors?` and unseen users |
| `negative_control_fintx.ipynb` | Financial Transactions as a negative control, with real and shuffled labels |
| `app/prototype_demo.ipynb` | Makes the demo files and starts FraudLens in Colab |

## How to run

We run every notebook in **Google Colab** on a standard CPU.

1. Open a notebook in Colab using the "Open in Colab" badge at its top.
2. Have these ready. Each is typed in when the notebook asks, and none is saved in the notebook:
   - a **Kaggle username and API key** (from your Kaggle account settings);
   - a **GitHub token**, to clone this repository.
3. For IEEE-CIS, first join the [competition](https://www.kaggle.com/c/ieee-fraud-detection) on Kaggle and accept its rules, otherwise the download fails with a 401 error. You can also download the files by hand and upload them when the notebook asks.
4. Run the cells from top to bottom.

**Long runs.** The later notebooks train many models and take 2 to 4 hours. They save each finished model to Google Drive, so if Colab disconnects, running the notebook again skips everything already finished. Allow access to Google Drive when the notebook asks.

**The app.** Open `app/prototype_demo.ipynb` in Colab and run it from the top, or run it on your own computer from this folder:

```bash
pip install -r app/requirements.txt
streamlit run app/fraud_app.py
```

To run the notebooks locally instead:

```bash
pip install -r requirements.txt
pip install catboost kaggle
```

## Limitations

- Sparkov and TabFormer are synthetic, so what drives fraud in each one reflects how its generator works: bursts in Sparkov, merchant locations in TabFormer. Our six features fit data where fraud comes in bursts and do not transfer to data where fraud is created another way.
- The origin of Credit Card Fraud 2026 is not documented in detail, so we treat it as a secondary dataset.
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
