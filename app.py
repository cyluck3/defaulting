import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


@st.cache_resource
def load_and_train_model():
    data = pd.read_csv("dataset_morosidad.csv")

    columns = [
        "client_id", "age", "months_tenure", "monthly_income", "product_type", 
        "installment_amount", "anual_interest_rate", "line_utilization_rate", 
        "average_days_past_due", "delinquency_frequency_6m", "target_default"
    ]
    data.columns = columns

    data["product_type"] = data.product_type.map({
        "Tarjeta de Credito": "Credit Card",
        "Prestamo Personal": "Personal Loan",
        "Microcredito": "Microloan"
    })

    # --- FEATURE ENGINEERING SIMPLE Y EFECTIVO ---
    data["dti_ratio"] = data["installment_amount"] / data["monthly_income"]
    data["utilization_per_tenure"] = data["line_utilization_rate"] / (data["months_tenure"] + 1)
    
    # Nuevas variables enfocadas en la tasa de interés:
    data["monthly_interest_amount"] = (data["installment_amount"] * (data["anual_interest_rate"] / 100)) / 12
    data["interest_burden_ratio"] = data["monthly_interest_amount"] / data["monthly_income"]

    X = data.drop(columns=["client_id", "target_default", "average_days_past_due", "age", "product_type"])
    y = data["target_default"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', cat_transformer, categorical_features)
        ]
    )

    # Modelo estándar y limpio sin restricciones manuales
    model = ImbPipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LGBMClassifier(
                n_estimators=120, 
                learning_rate=0.05, 
                random_state=42, 
                verbose=-1
            ))
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }

    return model, X_train, y_train, metrics

class App:
    def __init__(self):
        self.config()
        self.model, X_train, y_train, self.metrics = load_and_train_model()
        self.df = X_train.copy()
        self.df["target_default"] = y_train
        self.run()

    def run(self):
        self.sidebar()
        self.header()
        self.body()
        self.footer()

    def config(self):
        st.set_page_config(page_title="📈 Defaulting Risk Analysis", layout="wide", initial_sidebar_state="expanded")

    def sidebar(self):
        with st.sidebar:
            st.title("⚙️ Settings")
            self.line_utilization_rate = st.slider("Line Utilization Rate: ", 0.0, 1.0, 0.70)
            self.delinquency_frequency_6m = st.slider("Delinquency Frequency: ", 0, 6, 4)
            self.installment_amount = st.slider("Installment Amount: ", 1, 1263, 244)
            self.month_tenure = st.slider("Month Tenure: ", 1, 119, 20)
            self.anual_interest_rate = st.slider("Annual Interest Rate: ", 1, 44, 12)
            self.monthly_income = st.slider("Monthly Income: ", 1, 4983, 3204)

    def header(self):
        st.title("📈 Default Risk Prediction")
        st.markdown("*This app predicts the defaulting status of a client based on their financial features.*")
        st.write("---")
        st.header("LGBM Classifier Prediction")
        self.model_()

    def model_(self):
        st.markdown("### Settings & Inference")

        dti_calc = self.installment_amount / self.monthly_income if self.monthly_income > 0 else 0
        util_tenure_calc = self.line_utilization_rate / (self.month_tenure + 1)

        monthly_interest_calc = (self.installment_amount * (self.anual_interest_rate / 100)) / 12
        interest_burden_calc = monthly_interest_calc / self.monthly_income if self.monthly_income > 0 else 0

        user_df = pd.DataFrame({
            "months_tenure": [self.month_tenure],
            "monthly_income": [self.monthly_income],
            "installment_amount": [self.installment_amount],
            "anual_interest_rate": [self.anual_interest_rate],
            "line_utilization_rate": [self.line_utilization_rate],
            "delinquency_frequency_6m": [self.delinquency_frequency_6m],
            "dti_ratio": [dti_calc],
            "utilization_per_tenure": [util_tenure_calc],
            "monthly_interest_amount": [monthly_interest_calc],
            "interest_burden_ratio": [interest_burden_calc]
        })

        prob = float(self.model.predict_proba(user_df)[0][1])
        result = int(self.model.predict(user_df)[0])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Prediction", "Default (1)" if result == 1 else "Non-Default (0)")
        with col2:
            st.metric("Default Probability", f"{prob:.2%}")

        st.progress(prob)

        st.markdown("#### Model Performance Metrics")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Accuracy", f"{self.metrics['accuracy']:.2%}")
        with m_col2:
            st.metric("Recall", f"{self.metrics['recall']:.2%}")
        with m_col3:
            st.metric("Precision", f"{self.metrics['precision']:.2%}")
        with m_col4:
            st.metric("F1 Score", f"{self.metrics['f1']:.2%}")
        with m_col5:
            st.metric("ROC AUC", f"{self.metrics['roc_auc']:.2%}")

        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        importances = self.model.named_steps['classifier'].feature_importances_

        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        st.markdown("#### Feature Importances")
        st.dataframe(df_importance, use_container_width=True)

    def body(self):
        self.section_1()
        st.write("---")
        self.section_2()

    def section_1(self):
        st.title("Dataset Overview")
        st.header("Income Distribution")
        co1, co2, co3, co4 = st.columns(4)

        self.df['income_category'] = pd.qcut(
            self.df['monthly_income'], 
            q=[0, 0.33, 0.66, 1.0], 
            labels=['Lower Income Clients', 'Medium Income Clients', 'Higher Income Clients']
        )
        
        category_totals = self.df['income_category'].value_counts().reset_index()
        category_totals.columns = ['Category', 'Total Clients']
        
        with co1:
            n = category_totals[category_totals["Category"] == "Higher Income Clients"]["Total Clients"].values[0]
            st.metric("Higher Income Clients", n, f"{n/self.df.shape[0]:.2%}", delta_description="Low Risk")
            
        with co2:
            n = category_totals[category_totals["Category"] == "Medium Income Clients"]["Total Clients"].values[0]
            st.metric("Medium Income Clients", n, f"{n/self.df.shape[0]:.2%}")

        with co3:
            n = category_totals[category_totals["Category"] == "Lower Income Clients"]["Total Clients"].values[0]
            st.metric("Lower Income Clients", n, f"{n/self.df.shape[0]:.2%}")
        with co4:
            st.metric("Total Clients", self.df.shape[0])

        co1, co2, co3, co4 = st.columns(4)

        with co1:
            st.metric("Average Income", f"${round(self.df.monthly_income.mean(), 2)}")
        with co2:
            st.metric("Highest Income", f"${round(self.df.monthly_income.max(), 2)}", "Low Risk", delta_color="green")
        with co3:
            st.metric("Lowest Income", f"${round(self.df.monthly_income.min(), 2)}", "High Risk", delta_color="red")

        fig = px.histogram(self.df, x="monthly_income", y="income_category", color="income_category", labels={
            "monthly_income": "Monthly Income",
            "income_category": "Income Category"
        }, title="Distribution By Category", color_discrete_map={
                "Lower Income Clients": "#756d14",
                "Medium Income Clients": "#b0ab1c",
                "Higher Income Clients": "#F9EA7D",
            })
        st.plotly_chart(fig, use_container_width=True)

        

    def section_2(self):
        st.header("Months Tenure Distribution")
        co1, co2, co3, co4 = st.columns(4)
        mt = {
            "Average Months Tenure": int(self.df.months_tenure.mean()),
            "Highest Month Tenure": round(self.df.months_tenure.max(), 0),
            "Lowest Month Tenure": round(self.df.months_tenure.min(), 0),
            "Total Months": self.df.months_tenure.sum()
        }

        with co1:
            st.metric("Average Months Tenure", mt["Average Months Tenure"])
        with co2:
            st.metric("Highest Month Tenure", mt["Highest Month Tenure"])
        with co3:
            st.metric("Lowest Month Tenure", mt["Lowest Month Tenure"])
        with co4:
            st.metric("Total Months", mt["Total Months"])

        mt_ = pd.DataFrame(mt, index=[0])
        mt_melted = mt_.melt(var_name="Metric", value_name="Value")
        g1 = mt_melted.drop(3, axis=0)

        fig = px.line(g1, x="Metric", y="Value", title="Visualization")
        st.plotly_chart(fig, use_container_width=True)
            
        st.header("Delinquency Frequency")

        co1, co2, co3, co4 = st.columns(4)
        defq = self.df["delinquency_frequency_6m"].value_counts().reset_index()
        defq.columns = ["delinquency_frequency_6m", "count"]

        def get_count(freq):
            val = defq[defq["delinquency_frequency_6m"] == freq]["count"].values
            return val[0] if len(val) > 0 else 0

        with co1:
            st.metric("Delinquency Frequency: 0", get_count(0))
        with co2:
            st.metric("Delinquency Frequency: 1", get_count(1))
        with co3:
            st.metric("Delinquency Frequency: 2", get_count(2))
        with co4:
            st.metric("Delinquency Frequency: 3", get_count(3))

        co1, co2, co3, _ = st.columns(4)
        with co1:
            st.metric("Delinquency Frequency: 4", get_count(4))
        with co2:
            st.metric("Delinquency Frequency: 5", get_count(5))
        with co3:
            st.metric("Delinquency Frequency: 6", get_count(6))

        fig = px.bar(defq, x="delinquency_frequency_6m", y="count", labels={
            "delinquency_frequency_6m": "Delinquency Frequency",
            "count": "Clients"
        }, title="Visualization", color="count")

        st.plotly_chart(fig, use_container_width=True)

    def footer(self):
        st.header("Raw Data for Training")
        st.dataframe(self.df.drop(columns=["income_category"], errors="ignore"), use_container_width=True)


app = App()