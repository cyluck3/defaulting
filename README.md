# Credit Default Risk Prediction App

An interactive credit risk scoring application built with Streamlit, LightGBM, and scikit-learn to predict client default probability using financial features and engineered ratios.

## Overview

This project provides a machine learning workflow for predicting credit default risk from historical customer financial data (`dataset_morosidad.csv`). The core classifier uses a LightGBM gradient boosting model wrapped inside an `imbalanced-learn` pipeline. To handle imbalanced credit risk targets, **SMOTE** is applied strictly to training data within the pipeline context.

The application computes key financial metrics in real time (e.g., debt-to-income and interest burden ratios) and exposes user-configurable parameters to inspect live prediction probabilities and feature importances.

## Key Features

* **Feature Engineering**:
  * **Debt-to-Income Ratio (`dti_ratio`)**: $\text{Installment Amount} / \text{Monthly Income}$
  * **Utilization per Tenure (`utilization_per_tenure`)**: $\text{Line Utilization} / (\text{Months Tenure} + 1)$
  * **Monthly Interest Amount (`monthly_interest_amount`)**: Estimated monthly interest obligations based on interest rates.
  * **Interest Burden Ratio (`interest_burden_ratio`)**: Proportion of monthly income spent purely on interest.

* **Imbalanced Dataset Handling**: Pipeline-level SMOTE oversampling to stabilize model performance across default and non-default classes without data leakage.

* **LightGBM Classification**: Gradient boosted decision trees configured for fast, reproducible binary classification (`n_estimators=120`, `learning_rate=0.05`).

* **Dynamic Risk Scoring**: Real-time slider inputs for credit utilization, delinquency history, income, interest rates, and loan terms.

* **Exploratory Data Analysis**: Visual breakdown of income distribution tiers (quantiles) and 6-month delinquency frequencies using Plotly.

## Technical Stack

* **Frontend / UI**: Streamlit
* **Data Manipulation**: Pandas, NumPy
* **Machine Learning**: LightGBM (`LGBMClassifier`), scikit-learn (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`)
* **Resampling**: Imbalanced-learn (`SMOTE`, `ImbPipeline`)
* **Data Visualization**: Plotly Express, Plotly Graph Objects

## Pipeline Architecture

```
Raw CSV Dataset (dataset_morosidad.csv)
   │
   ▼
Feature Engineering Module
(dti_ratio, interest_burden_ratio, utilization_per_tenure, etc.)
   │
   ▼
Train/Test Split (80/20)
   │
   ▼
Imbalanced-Learn Pipeline
   ├── Numeric Preprocessing ──> Median Imputer + StandardScaler
   ├── Categorical Preprocessing ──> Frequent Imputer + OneHotEncoder
   ├── Resampling ──> SMOTE Oversampling
   └── Estimator ──> LightGBM Classifier
   │
   ▼
Streamlit Dashboard Inference & Feature Importance
```

## Installation and Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/credit-default-risk.git
   cd credit-default-risk
   ```

2. **Ensure `dataset_morosidad.csv` is present in the root directory.**

3. **Set up a virtual environment and activate it:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```bash
   pip install numpy pandas streamlit plotly scikit-learn lightgbm imbalanced-learn
   ```

5. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

## Model Evaluation Metrics

The pipeline measures real-time performance on held-out test data across five standard metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **ROC AUC**

## Repository Structure

```
.
├── dataset_morosidad.csv   # Historical credit data
├── app.py                  # Streamlit application and ML code
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```