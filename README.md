# Hotel Booking Cancellation Risk

Predicting hotel booking cancellations at the time of booking, and grouping bookings into segments, to support deposit decisions.

**Module:** IT3091 Machine Learning · Year 3, Semester 1 · 2026
**Group:** 2026-DS-06K · Kandy UNI · Guided Data Track (Scenario 06: Tourism & Hospitality)

## Team

| Member | Name | Student ID | Role |
|---|---|---|---|
| M1 | Mahdhi S.M.M | IT24100667 | Team lead. Multivariate EDA, clustering (secondary lens), deposit threshold/policy, recommendation, repo setup |
| M2 | Amri B.S.M | IT24102955 | Univariate EDA, shared preprocessing, Logistic Regression baseline, Decision Tree, bias & risk check |
| M3 | Wickremasinghe N.N.D.R | IT24100442 | Bivariate EDA, feature engineering, Random Forest, model interpretation, reproducibility check |
| M4 | Wijesundara W.M.R.T | IT24101979 | Data quality, basic cleaning, train/test split, evaluation setup, Gradient Boosting, model comparison |

## Business problem

- **Stakeholder:** the hotel chain's revenue manager.
- **Decision:** whether a booking should be asked for a deposit at confirmation.
- **Primary lens: cancellation risk.** Binary classification on `is_canceled`. Output: a cancellation probability per booking, using only information available at the time of booking.
- **Secondary lens: booking segmentation.** Clustering bookings into a small number of types, then combining each segment with predicted risk to produce a deposit policy per segment, not just a list of individual scores.

## Dataset

- **Source:** [Hotel Booking Demand (Kaggle)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **File:** `data/raw/hotel_bookings.csv`
- **Shape:** 119390 rows × 32 columns


## Folder structure

```
hotel-cancellation/
├── data/
│   ├── raw/hotel_bookings.csv     ← original data, never edited
│   └── processed/                 ← train/test files and model outputs
├── notebooks/                     ← run in number order
├── src/
│   ├── data.py                    ← load_basic_clean()
│   ├── features.py                ← add_features()
│   ├── preprocessing.py           ← shared prep
│   └── evaluate.py                ← evaluate() + shared CV folds
├── models/                        ← saved pipelines (joblib)
├── logs/                          ← decision log, AI-use log, data dictionary (exported)
├── requirements.txt
└── README.md
```

## Setup

Python version: **3.14.6**

```
git clone https://github.com/mahdhis/hotel-cancellation.git
cd hotel-cancellation
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

In VS Code, select the `.venv` interpreter as the notebook kernel.

## How to run

Run the notebooks in number order. Each one depends on outputs from the earlier ones.

| # | Notebook | Owner |
|---|---|---|
| 01 | data_quality | M4 |
| 02 | eda_univariate | M2 |
| 03 | eda_bivariate | M3 |
| 04 | eda_multivariate | M1 |
| 05 | train_test_split | M4 |
| 06 | feature_engineering | M3 |
| 07 | shared_prep | M2 |
| 08 | model_logistic_regression | M2 |
| 09 | model_random_forest | M3 |
| 10 | model_gradient_boosting | M4 |
| 11 | model_decision_tree | M2 |
| 12 | clustering | M1 |
| 13 | model_comparison (+ 13a interpretation, 13b bias check) | M4 / M3 / M2 |
| 14 | segments_x_risk | M1 |


## Logs

- Live working logs (team only): [Google Sheet link]
- Final exported copies: `logs/decision_log.md`, `logs/ai_use_log.md`, `logs/data_dictionary.md`