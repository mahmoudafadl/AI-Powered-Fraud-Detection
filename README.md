# AI-Powered Fraud Detection — Machine Learning Classification

A machine learning project for detecting potentially fraudulent financial transactions and distinguishing **fraudulent** transactions from **legitimate** ones.

The project covers the complete workflow from data inspection and cleaning to feature engineering, preprocessing, model training, evaluation, hyperparameter tuning, ensemble learning, model persistence, and Streamlit deployment.

## Project Objective

The main business objective is to identify potentially fraudulent transactions accurately while balancing fraud-detection performance against false-positive errors.

Because fraud detection is an imbalanced classification problem, the project focuses on more than accuracy alone. **Precision, Recall, F1-score, ROC-AUC, and confusion-matrix behavior** are used to understand how well each model detects fraud and how many legitimate transactions it incorrectly flags.

## Dataset

The project uses the local dataset:

`data/Credit-Card-Transactions.csv`

### Dataset overview

- **Initial rows:** 5,300
- **Initial columns:** 15
- **Target:** `risk_label`
- **Target classes:**
  - `0` — Legitimate
  - `1` — Fraud
- **Initial class distribution:**
  - Legitimate: 4,773 (90.06%)
  - Fraud: 527 (9.94%)
- **Initial missing values:** 167
- **Initial duplicate rows:** 300

The final modeling dataset contains **5,000 rows** after duplicate removal and retains a **90% legitimate / 10% fraud** class distribution.

## Features

The original transaction data contains identifiers, numerical features, binary indicators, and categorical features.

### Raw modeling features

The identifiers `transaction_id` and `customer_id` are removed from the modeling feature set because they are identifiers rather than behavioral predictors.

The raw input features used for modeling are:

- `transaction_hour`
- `account_age_days`
- `previous_chargebacks`
- `merchant_category`
- `transaction_country`
- `device_type`
- `is_international`
- `is_high_risk_merchant`
- `transaction_amount`
- `transaction_velocity_1h`
- `transaction_velocity_24h`
- `avg_transaction_amount_30d`

## Machine Learning Workflow

The notebook follows these main stages:

```text
Raw Dataset
    ↓
Data Overview & EDA
    ↓
Duplicate Removal
    ↓
Invalid-Value Handling
    ↓
Feature Analysis & Fraud Patterns
    ↓
Feature Engineering
    ↓
Train / Test Split
    ↓
Missing-Value Handling
    ↓
Outlier Analysis
    ↓
Preprocessing
    ↓
Class-Imbalance Handling
    ↓
Cross-Validation
    ↓
Baseline Model Training
    ↓
Test-Set Evaluation
    ↓
Hyperparameter Tuning
    ↓
Tuned Model Evaluation
    ↓
Stacking Ensemble
    ↓
Final Model Selection
    ↓
Model & Preprocessor Saving
    ↓
Streamlit Deployment
```

## 1. Data Cleaning

### Duplicate removal

The dataset initially contained **300 duplicate rows**. These rows were removed, reducing the dataset from 5,300 to **5,000 rows**.

### Invalid values

Domain-based validation rules were applied before modeling. Invalid values were converted to `NaN` so they could be handled later by the preprocessing stage.

Detected invalid values included:

- Negative `transaction_amount` values
- Negative `avg_transaction_amount_30d` values
- Negative `account_age_days` values
- Invalid `transaction_hour` values outside the valid hour range

A total of **32 invalid values** were converted to missing values.

After cleaning:

- Duplicate rows: **0**
- Invalid domain values: **0**
- Missing values were intentionally retained until after the train/test split

## 2. Feature Analysis & Fraud Patterns

The exploratory analysis examined numerical correlations, categorical fraud rates, and binary-feature fraud rates.

The strongest absolute numerical relationships with the fraud target were observed for:

- `transaction_velocity_24h`
- `transaction_velocity_1h`
- `previous_chargebacks`
- `transaction_amount`

Fraud-rate analysis also showed higher fraud rates for some transaction contexts. For example, international transactions and high-risk merchants had higher observed fraud rates than their corresponding non-international and non-high-risk groups.

These relationships were used as analytical insights rather than as manual rules for prediction.

## 3. Feature Engineering

Two additional features were created:

### `amount_to_avg_ratio`

```text
transaction_amount / avg_transaction_amount_30d
```

This measures the current transaction amount relative to the customer's recent 30-day average transaction amount.

A zero 30-day average is converted to `NaN` before division to avoid invalid ratios.

### `is_night_transaction`

A binary indicator derived from `transaction_hour`:

- `1` for transactions from **00:00 through 05:59**
- `0` otherwise

The same feature-engineering logic is reproduced in `app.py` during inference so that deployment uses the same feature definitions as training.

## 4. Train / Test Split

The cleaned and engineered dataset was split using a stratified 80/20 split:

- **Training set:** 4,000 rows
- **Test set:** 1,000 rows
- `random_state=42`
- `stratify=y`

Both sets preserve the 90% legitimate / 10% fraud target distribution.

## 5. Missing-Value Handling

Missing values were handled using training-set information only.

The final preprocessing pipeline applies:

### Numerical features

- Median imputation
- Standard scaling

### Binary features

- Most-frequent imputation

### Categorical features

- Most-frequent imputation
- One-hot encoding
- `handle_unknown="ignore"`

The final fitted `ColumnTransformer` produces **26 processed features**.

## 6. Outlier Analysis

Outliers were analyzed using IQR-based bounds calculated from the training data for selected continuous numerical features.

The analysis also compared fraud rates between outlier and non-outlier observations.

No outliers were removed or capped because unusually large or unusual transaction behavior can itself be informative in fraud detection.

## 7. Preprocessing Pipeline

The preprocessing pipeline uses a `ColumnTransformer` with three branches:

```text
Numerical
    → Median Imputation
    → StandardScaler

Binary
    → Most-Frequent Imputation

Categorical
    → Most-Frequent Imputation
    → OneHotEncoder(handle_unknown="ignore")
```

The fitted preprocessor is saved as:

`models/preprocessor.joblib`

## 8. Class-Imbalance Handling

Fraud represents only 10% of the final modeling data, so class imbalance was explicitly addressed.

Balanced class weights were calculated from the training target:

- **Class 0 — Legitimate:** `0.5556`
- **Class 1 — Fraud:** `5.0000`

Logistic Regression, Decision Tree, and Random Forest use these class weights.

XGBoost uses `scale_pos_weight`, calculated from the training-class ratio.

## 9. Cross-Validation

A **5-fold StratifiedKFold** setup was used with:

- `n_splits=5`
- `shuffle=True`
- `random_state=42`

The baseline cross-validation evaluated:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

For this stage, preprocessing was placed inside a pipeline so that each fold learned its preprocessing transformations from its corresponding training portion.

## 10. Baseline Models

Four baseline classifiers were trained:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

### Baseline Test-Set Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 95.40% | 74.11% | 83.00% | **78.30%** | 97.89% |
| Random Forest | 95.00% | **83.78%** | 62.00% | 71.26% | 97.74% |
| Decision Tree | 93.10% | 66.32% | 63.00% | 64.62% | 79.72% |
| Logistic Regression | 88.70% | 46.86% | **97.00%** | 63.19% | 96.49% |

The baseline results highlight different model behaviors:

- **XGBoost** provided the strongest overall individual-model F1-score and a strong balance between precision and recall.
- **Random Forest** achieved the highest precision among the baseline models but lower fraud recall.
- **Logistic Regression** achieved the highest fraud recall but produced many more false positives, resulting in lower precision and F1-score.
- **Decision Tree** was retained as a benchmark and showed weaker overall discrimination than the ensemble models.

## 11. Hyperparameter Tuning

Hyperparameter tuning was applied to the two strongest ensemble candidates:

- XGBoost
- Random Forest

`RandomizedSearchCV` was used with:

- **30 candidate configurations**
- **5-fold stratified cross-validation**
- `scoring="f1"`
- `random_state=42`
- `n_jobs=-1`

### Best XGBoost configuration

- `n_estimators=500`
- `max_depth=4`
- `learning_rate=0.05`
- `subsample=0.9`
- `colsample_bytree=0.8`
- `min_child_weight=5`

Best cross-validation F1-score: **0.7780**.

### Best Random Forest configuration

- `n_estimators=300`
- `max_depth=20`
- `min_samples_split=5`
- `min_samples_leaf=4`
- `max_features="sqrt"`
- `bootstrap=True`

Best cross-validation F1-score: **0.7541**.

The tuned models were evaluated again on the unseen test set before the final ensemble stage.

## 12. Stacking Ensemble

A Stacking Ensemble was created to combine the strengths of different classifiers.

### Base models

- XGBoost
- Tuned Random Forest
- Logistic Regression

### Meta-learner

- Logistic Regression

The stack uses model probability outputs through `stack_method="predict_proba"` and 5-fold cross-validation.

### Stacking Test-Set Performance

| Metric | Stacking Ensemble |
|---|---:|
| Accuracy | 95.90% |
| Precision | 78.64% |
| Recall | 81.00% |
| F1-score | **79.80%** |
| ROC-AUC | 98.22% |

The stacking model was selected as the **primary deployment model** because it provided the strongest overall balance between precision and recall and the highest F1-score among the final deployment candidates.

## 13. Final Deployment Models

Three models were selected for deployment:

### 1. Stacking Ensemble — Primary

- Accuracy: **95.90%**
- Precision: **78.64%**
- Recall: **81.00%**
- F1-score: **79.80%**
- ROC-AUC: **98.22%**

### 2. XGBoost — Alternative

- Accuracy: **95.40%**
- Precision: **74.11%**
- Recall: **83.00%**
- F1-score: **78.30%**
- ROC-AUC: **97.89%**

### 3. Tuned Random Forest — High-Recall Alternative

- Accuracy: **94.80%**
- Precision: **68.75%**
- Recall: **88.00%**
- F1-score: **77.19%**
- ROC-AUC: **98.15%**

Logistic Regression and Decision Tree remain useful as baseline benchmarks but are not part of the final deployment set.

## 14. Feature Importance

Feature importance was analyzed using the Tuned Random Forest.

The most important features included:

1. `transaction_velocity_24h`
2. `previous_chargebacks`
3. `transaction_velocity_1h`
4. `transaction_amount`
5. `amount_to_avg_ratio`
6. `account_age_days`
7. `avg_transaction_amount_30d`
8. `transaction_hour`
9. `is_high_risk_merchant`
10. `is_international`
11. `transaction_country_US`
12. `is_night_transaction`

The importance analysis is descriptive and does not replace model evaluation or causal analysis.

## 15. Model Persistence

The following trained artifacts are saved for deployment:

```text
models/
├── preprocessor.joblib
├── xgboost_model.joblib
├── random_forest_model.joblib
└── stacking_model.joblib
```

The deployment app loads these fitted artifacts instead of retraining them.

## 16. Streamlit Deployment

The Streamlit application is designed as a thin inference layer over the training pipeline.

### Inference flow

```text
Raw User Input
      ↓
Feature Engineering
      ↓
Saved Fitted Preprocessor
      ↓
Selected Saved Model
      ↓
Prediction + Fraud Probability
      ↓
Risk Presentation
```

The app does **not** retrain, refit, retune, or recreate the saved models.

For inference, the saved preprocessor is used with:

```python
preprocessor.transform(...)
```

and never with `.fit()` or `.fit_transform()`.

### Available deployment models

The user can select:

- Stacking Ensemble
- XGBoost
- Tuned Random Forest

### Input fields

The application collects the raw transaction features required by the trained pipeline, including:

- Transaction hour
- Account age
- Previous chargebacks
- Merchant category
- Transaction country
- Device type
- International transaction indicator
- High-risk merchant indicator
- Transaction amount
- Transaction velocity over 1 hour
- Transaction velocity over 24 hours
- Average transaction amount over 30 days

The engineered features are calculated automatically inside the application and are not entered manually.

### Risk presentation

The app displays the model's fraud probability and a presentation-level risk band:

- **High Risk:** probability ≥ 70%
- **Moderate Risk:** probability ≥ 30% and < 70%
- **Low Risk:** probability < 30%

These risk bands are **presentation-only**. They do not alter the model's own Fraud/Legitimate prediction threshold.

## 17. Environment

### Runtime environment

The deployment environment was tested with:

- Python 3.10.0
- NumPy 2.2.6
- Pandas 2.3.3
- scikit-learn 1.7.2
- XGBoost 3.2.0
- Joblib 1.5.3
- Streamlit 1.63.0

### Development environment

The notebook-based development environment also used:

- Jupyter 1.1.1
- Notebook 7.5.0
- JupyterLab 4.5.1
- Matplotlib 3.10.8
- Seaborn 0.13.2

The runtime packages required by the deployed Streamlit application are kept in `requirements.txt`, while notebook and visualization tooling is considered part of the development environment.

## 18. Installation

### Runtime / deployment

Create or activate a Python 3.10 environment, then install:

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

From the project root:

```bash
streamlit run app.py
```

The project should have the following deployment structure:

```text
Fraud_Detection_Project/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── preprocessor.joblib
│   ├── xgboost_model.joblib
│   ├── random_forest_model.joblib
│   └── stacking_model.joblib
└── data/
    └── Credit-Card-Transactions.csv
```

## 19. Project Notes

- The test set is used for final model evaluation after training and tuning.
- Fraud detection performance is interpreted using precision, recall, F1-score, ROC-AUC, and confusion-matrix behavior rather than accuracy alone.
- Outliers were analyzed but intentionally retained because unusual transaction values may contain fraud-related information.
- The saved preprocessor and saved models are treated as deployment artifacts and are not refit by the Streamlit application.
- The baseline cross-validation stage uses preprocessing inside the fold pipeline. The later RandomizedSearchCV stage operates on the already transformed training matrix used by the notebook.

## License

This project is an educational and portfolio-oriented machine learning project.
