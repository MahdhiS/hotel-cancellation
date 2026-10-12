# Hotel Booking Cancellation Risk

Predicting hotel booking cancellations at the time of booking, and grouping bookings into segments, to support deposit decisions.

**Module:** IT3091 Machine Learning, Year 3, Semester 1, 2026
**Group:** 2026-DS-06K, Kandy UNI, Guided Data Track (Scenario 06: Tourism & Hospitality)

**Demo video (3 minutes):** https://youtu.be/-P9RIEfNi5k

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
- **Secondary lens: booking segmentation.** Clustering bookings into a small number of types, then combining each segment with predicted risk to produce a deposit policy, not just a list of individual scores.

## Key results

- **Final model:** XGBoost (Gradient Boosting), chosen on validation ROC-AUC (0.7758). Test ROC-AUC: 0.7378.
- **Segments:** K-Means with 3 segments: Short solo stays, Short stays for two, Long planned holidays. Long planned holidays are 37.6% of training bookings but 68.6% of the money lost to cancellations.
- **Deposit rule:** ask for a deposit when the predicted risk is 0.35 or higher, for all segments. On the test set this catches 66% of cancellations, with an estimated net gain of 326,278 (based on a 20% deposit and 10% of asked guests leaving).

## Dataset

- **Source:** [Hotel Booking Demand (Kaggle)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand), originally from Antonio, de Almeida and Nunes (2019), Data in Brief, 22, 41-49.
- **File:** `data/raw/hotel_bookings.csv` (never edited)

**Fingerprint.** If these numbers match, you have the same file:

| Check | Expected value |
|---|---|
| Rows and columns | 119,390 rows, 32 columns |
| Bookings with `is_canceled` = 1 | 44,224 (37.0%) |
| Exact duplicate rows | 31,994 |
| Rows after basic cleaning | 86,940 |

## Folder structure

```
hotel-cancellation/
|-- data/
|   |-- raw/hotel_bookings.csv     # original data, never edited
|   |-- processed/                 # cleaned, split and prepared files, model outputs
|-- notebooks/                     # run in number order
|-- src/
|   |-- data.py                    # load_basic_clean()
|   |-- features.py                # add_features()
|   |-- preprocessing.py           # shared prep (SharedPreprocessor)
|   |-- evaluate.py                # evaluate() + shared CV folds
|-- models/                        # saved pipelines (joblib)
|-- logs/                          # decision log, AI-use log, data dictionary (exported)
|-- requirements.txt
|-- README.md
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

| # | Notebook | Owner | Main output |
|---|---|---|---|
| 01 | data_quality | M4 | List of data issues |
| 02 | eda_univariate | M2 | Univariate EDA findings |
| 03 | eda_bivariate | M3 | Bivariate EDA findings |
| 04 | eda_multivariate | M1 | Candidate clustering features |
| 05a | preprocessing | M4 | Cleaned data (86,940 rows) |
| 05 | train_test_split | M4 | Train, validation and test files |
| 06 | feature_engineering | M3 | Files with 5 new features |
| 07 | shared_prep | M2 | Shared prepared files (29 columns) |
| 08 | model_logistic_regression | M2 | Logistic Regression pipeline |
| 09 | model_random_forest | M3 | Random Forest pipeline |
| 10 / 10a | model_gradient_boosting | M4 | Final XGBoost pipeline |
| 11 | model_decision_tree | M2 | Decision Tree pipeline and rules |
| 12 | clustering | M1 | Segment labels and clustering model |
| 13 | model_comparison (+ 13a interpretation, 13b bias check) | M4 / M3 / M2 | Final model choice, feature importance, risk screen |
| 14 | segments_x_risk | M1 | Deposit policy, test results and demo |

## Team rules

- `random_state=42` everywhere, including the shared 5 cross-validation folds.
- Never edit `data/raw/`. Use relative paths only.
- Everything learned from data (fill values, rare categories, scaling, encoding, clustering) is fitted on the training data only.
- `booking_id` is kept in every saved file, so segments and risk scores can be joined.

## Logs

- Decision log, AI-use log and data dictionary: `logs/decision_log.md`, `logs/ai_use_log.md`, `logs/data_dictionary.md`
- In the final report: the data dictionary is Section 3.2, the decision log is Appendix A and the AI-use log is Appendix B.