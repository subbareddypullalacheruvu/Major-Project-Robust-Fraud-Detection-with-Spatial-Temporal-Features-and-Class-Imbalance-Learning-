# FraudLens: our fraud detection prototype

FraudLens is a small web app that takes a card transaction table (CSV), builds the features from our experiments, trains a fraud model with our leak-free protocol, and lists the riskiest transactions with short reasons.

This folder runs on its own. It does not depend on the experiment code in `src/` or `notebooks/`.

## Files

| File | What it is |
|---|---|
| `fraud_app.py` | The Streamlit app (five tabs) |
| `prototype.py` | Helper functions: column mapping, features, time split, training, scoring, reasons |
| `prototype_demo.ipynb` | Colab notebook: makes the demo files and starts the app |
| `requirements.txt` | Packages the app needs |
| `demo_data/` | Demo files made by the notebook (not on GitHub, too large) |

## Features the app builds

| Group | Feature | What it means |
|---|---|---|
| Temporal | `gap_log` | Time since the card's previous transaction |
| Temporal | `is_night` | Transaction between 22:00 and 03:59 |
| Spatial | `dist_home_km` | Distance between the merchant and the cardholder's home |
| Spatial | `outside_home_state` | Merchant is outside the home state |
| Spatial | `abroad` | Merchant is in another country |
| Amount | `amt_log` | Log of the amount |
| Amount | `category_risk` | Fraud rate of the merchant category (training part only) |
| Customer | `age`, `gender_flag` | Customer age and gender |

The app builds only the features that the table's columns allow.

## Steps in the app

1. **Data:** upload a CSV or type a file path.
2. **Columns:** choose which column is the label, time, card, amount, category, age or date of birth, gender, and home and merchant location. The app fills in its guesses.
3. **Train and evaluate:** split by time (train, validation, test), fit `category_risk` on the training part, train a Random Forest with our settings, choose the F2 threshold on validation, and show PR-AUC, F2, MCC, the precision-recall curve, feature importance, alert budgets and a leakage check on the test part.
4. **Alerts:** the riskiest test transactions with short reasons, downloadable as CSV.
5. **Score new data:** score a new CSV with the trained model; save or load a model.

## Running it

**In Colab:** open `prototype_demo.ipynb` and run it from the top.

**On a computer**, from the repository folder:

```bash
pip install -r app/requirements.txt
streamlit run app/fraud_app.py
```
