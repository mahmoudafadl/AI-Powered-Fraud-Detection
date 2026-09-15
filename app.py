"""
Fraud Detection — Streamlit Deployment App
============================================

Thin inference layer on top of the trained fraud-detection pipeline from
`Fraud_Detection_Machine_Learning.ipynb`. This file only presents results —
it does not redesign preprocessing, change feature engineering, retrain
models, retune models, or touch the test set.

Pipeline (unchanged):
    Raw User Input -> Feature Engineering -> Saved Fitted Preprocessor
    -> Selected Saved Model -> Prediction -> Fraud Probability -> Result

Artifacts loaded (Section 23. Saving For Deployment in the notebook):
    models/preprocessor.joblib        -> fitted ColumnTransformer
    models/xgboost_model.joblib       -> XGBoost
    models/random_forest_model.joblib -> Tuned Random Forest
    models/stacking_model.joblib      -> Stacking Ensemble (primary model)

All three models were trained and evaluated on the OUTPUT of the fitted
preprocessor (`preprocessor.transform(...)`), never on raw feature values,
so inference always calls `preprocessor.transform()` — never `.fit()` or
`.fit_transform()` — before handing data to any model.
"""

from pathlib import Path
from html import escape

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# PROJECT PATH HANDLING
# ============================================================================
# Resolve every path relative to this file's location, never the cwd.

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
STACKING_MODEL_PATH = MODELS_DIR / "stacking_model.joblib"


# ============================================================================
# STATIC CONFIG — copy, colors, metrics (presentation only)
# ============================================================================

BUSINESS_OBJECTIVE = (
    "Identify potentially fraudulent transactions accurately while "
    "balancing fraud detection performance and false-positive errors."
)

HERO_BADGES = [
    "3 ML Models",
    "Binary Classification",
    "Fraud Detection",
    "Class Imbalance Handling",
    "Stratified Cross-Validation",
    "Hyperparameter Tuning",
]

COLOR_BRAND_START = "#4338CA"
COLOR_BRAND_END = "#6366F1"
COLOR_TEXT_MUTED = "#64748B"
COLOR_CARD_BORDER = "#E2E8F0"
COLOR_FRAUD = "#DC2626"
COLOR_LEGIT = "#16A34A"
COLOR_MODERATE = "#D97706"

MODEL_REGISTRY = {
    "Stacking Ensemble": {
        "path": STACKING_MODEL_PATH,
        "color": "#7C3AED",
        "metrics": {
            "Accuracy": 0.9590,
            "Precision": 0.7864,
            "Recall": 0.81,
            "F1": 0.7980,
            "ROC-AUC": 0.9822,
        },
        "description": (
            "Primary model combining XGBoost, Random Forest, and "
            "Logistic Regression through a stacked meta-learner. Selected "
            "for the strongest overall balance between precision and recall."
        ),
    },
    "XGBoost": {
        "path": XGBOOST_MODEL_PATH,
        "color": "#2563EB",
        "metrics": {
            "Accuracy": 0.9540,
            "Precision": 0.7411,
            "Recall": 0.83,
            "F1": 0.7830,
            "ROC-AUC": 0.9789,
        },
        "description": (
            "Strong standalone fraud-detection model with the highest "
            "F1-score among the individual models."
        ),
    },
    "Random Forest": {
        "path": RANDOM_FOREST_MODEL_PATH,
        "color": "#0D9488",
        "metrics": {
            "Accuracy": 0.9480,
            "Precision": 0.6875,
            "Recall": 0.88,
            "F1": 0.7719,
            "ROC-AUC": 0.9815,
        },
        "description": (
            "Alternative ensemble model with strong recall, useful when "
            "detecting more fraudulent transactions is prioritized."
        ),
    },
}

MERCHANT_CATEGORIES = [
    "grocery",
    "restaurant",
    "electronics",
    "travel",
    "online_services",
    "fashion",
    "fuel",
]
TRANSACTION_COUNTRIES = ["US", "CA", "UK", "AU", "IN"]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]

RAW_FEATURE_COLUMNS = [
    "transaction_hour",
    "account_age_days",
    "previous_chargebacks",
    "merchant_category",
    "transaction_country",
    "device_type",
    "is_international",
    "is_high_risk_merchant",
    "transaction_amount",
    "transaction_velocity_1h",
    "transaction_velocity_24h",
    "avg_transaction_amount_30d",
]


# ============================================================================
# LOAD ARTIFACTS (cached — loaded once per session, transform-only, never fit)
# ============================================================================

@st.cache_resource(show_spinner="Loading preprocessing pipeline...")
def _load_preprocessor():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at: {PREPROCESSOR_PATH}\n"
            f"Expected project structure:\n"
            f"  {BASE_DIR.name}/\n"
            f"  ├── app.py\n"
            f"  └── models/preprocessor.joblib"
        )
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_resource(show_spinner="Loading model...")
def _load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


def load_models():
    """Load the fitted preprocessor + the three deployment models once.
    Returns (preprocessor, {model_name: fitted_model}). Never fits or
    refits anything — every artifact is loaded exactly as saved."""
    try:
        preprocessor = _load_preprocessor()
        models = {
            name: _load_model(info["path"])
            for name, info in MODEL_REGISTRY.items()
        }
        return preprocessor, models
    except FileNotFoundError as e:
        st.error(
            "**Could not load one or more deployment artifacts.**\n\n"
            f"{e}\n\n"
            "Make sure `app.py` sits at the root of the project and that "
            "`models/preprocessor.joblib`, `models/xgboost_model.joblib`, "
            "`models/random_forest_model.joblib`, and "
            "`models/stacking_model.joblib` all exist."
        )
        st.stop()


# ============================================================================
# FEATURE ENGINEERING (identical to notebook Section 6 — do not modify)
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact same feature engineering used during training."""
    df = df.copy()

    df["amount_to_avg_ratio"] = (
        df["transaction_amount"] /
        df["avg_transaction_amount_30d"].replace(0, np.nan)
    )

    df["is_night_transaction"] = np.where(
        df["transaction_hour"].isna(),
        np.nan,
        (
            (df["transaction_hour"] >= 0) &
            (df["transaction_hour"] < 6)
        ).astype(int)
    )

    return df


def validate_input(raw_record: dict) -> list:
    """Defensive input validation. Streamlit's widgets already constrain
    ranges and categorical choices, but this keeps the pipeline safe
    against any future non-widget caller. Returns a list of human-readable
    error messages (empty list = valid). Never silently modifies values —
    the caller is expected to block prediction and show these as st.error()."""
    errors = []

    hour = raw_record["transaction_hour"]
    if not (0 <= hour <= 23):
        errors.append("Transaction hour must be between 0 and 23.")

    non_negative_fields = {
        "account_age_days": "Account age",
        "previous_chargebacks": "Previous chargebacks",
        "transaction_amount": "Transaction amount",
        "transaction_velocity_1h": "Transactions in last 1 hour",
        "transaction_velocity_24h": "Transactions in last 24 hours",
        "avg_transaction_amount_30d": "Average transaction amount (30d)",
    }
    for field, label in non_negative_fields.items():
        if raw_record[field] < 0:
            errors.append(f"{label} cannot be negative.")

    if raw_record["merchant_category"] not in MERCHANT_CATEGORIES:
        errors.append("Merchant category is not a recognized value.")
    if raw_record["transaction_country"] not in TRANSACTION_COUNTRIES:
        errors.append("Transaction country is not a recognized value.")
    if raw_record["device_type"] not in DEVICE_TYPES:
        errors.append("Device type is not a recognized value.")
    if raw_record["is_international"] not in (0, 1):
        errors.append("International transaction must map to Yes/No.")
    if raw_record["is_high_risk_merchant"] not in (0, 1):
        errors.append("High-risk merchant must map to Yes/No.")

    return errors


def prepare_input(raw_record: dict) -> pd.DataFrame:
    """Turn a validated raw-feature dict into a one-row DataFrame with the
    correct dtypes, then apply feature engineering. Users never enter
    `amount_to_avg_ratio` / `is_night_transaction` directly — both are
    always derived here from the raw fields."""
    df = pd.DataFrame([raw_record], columns=RAW_FEATURE_COLUMNS)

    numeric_raw_cols = [
        "transaction_hour",
        "account_age_days",
        "previous_chargebacks",
        "transaction_amount",
        "transaction_velocity_1h",
        "transaction_velocity_24h",
        "avg_transaction_amount_30d",
        "is_international",
        "is_high_risk_merchant",
    ]
    df[numeric_raw_cols] = df[numeric_raw_cols].astype(float)

    return engineer_features(df)


def get_risk_level(fraud_probability: float) -> str:
    """Presentation-only risk banding derived from the model's own output
    probability. Does not affect the model's Fraud/Legitimate prediction
    or any prediction threshold."""
    if fraud_probability >= 0.70:
        return "High Risk"
    if fraud_probability >= 0.30:
        return "Moderate Risk"
    return "Low Risk"


def predict_transaction(engineered_df: pd.DataFrame, preprocessor, model) -> dict:
    """Run preprocessor.transform() (never .fit / .fit_transform) followed
    by the selected saved model's own predict / predict_proba. This is the
    single inference path used everywhere in the app — no duplicated
    scoring logic, no new thresholds."""
    processed = preprocessor.transform(engineered_df)
    proba = model.predict_proba(processed)[0]
    predicted_label = model.predict(processed)[0]

    fraud_probability = float(proba[1])
    legitimate_probability = float(proba[0])
    prediction = "Fraud" if predicted_label == 1 else "Legitimate"

    return {
        "prediction": prediction,
        "fraud_probability": fraud_probability,
        "legitimate_probability": legitimate_probability,
        "risk_level": get_risk_level(fraud_probability),
    }


# ============================================================================
# STYLING
# ============================================================================

def inject_custom_css():
    st.markdown(
        f"""
        <style>
        .hero-banner {{
            background: linear-gradient(135deg, {COLOR_BRAND_START}, {COLOR_BRAND_END});
            color: #FFFFFF;
            padding: 2.2rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.4rem;
        }}
        .hero-banner h1 {{
            margin: 0 0 .4rem 0;
            font-size: 2rem;
            font-weight: 700;
        }}
        .hero-banner p {{
            margin: 0 0 1rem 0;
            opacity: .92;
            font-size: 1.02rem;
        }}
        .badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: .5rem;
        }}
        .badge-pill {{
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.35);
            padding: .3rem .8rem;
            border-radius: 999px;
            font-size: .8rem;
            font-weight: 500;
        }}
        .model-card {{
            border-radius: 10px;
            border: 1px solid {COLOR_CARD_BORDER};
            border-left: 4px solid #94A3B8;
            padding: .85rem 1.1rem;
            background: #FFFFFF;
            margin-top: .4rem;
        }}
        .model-card p {{
            margin: 0;
            color: {COLOR_TEXT_MUTED};
            font-size: .92rem;
            line-height: 1.5;
        }}
        .sidebar-row {{
            display: flex;
            justify-content: space-between;
            font-size: .88rem;
            padding: .15rem 0;
        }}
        .sidebar-row span:first-child {{ color: {COLOR_TEXT_MUTED}; }}
        .sidebar-row span:last-child {{ font-weight: 600; }}
        .sidebar-model-item {{
            font-size: .9rem;
            padding: .2rem 0;
        }}
        .result-banner {{
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            font-weight: 700;
            font-size: 1.25rem;
            text-align: center;
            margin-bottom: .8rem;
        }}
        .result-fraud {{
            background: #FEF2F2;
            color: {COLOR_FRAUD};
            border: 1px solid #FCA5A5;
        }}
        .result-legit {{
            background: #F0FDF4;
            color: {COLOR_LEGIT};
            border: 1px solid #86EFAC;
        }}
        .performance-accent {{
            height: 4px;
            border-radius: 2px;
            margin-bottom: .7rem;
        }}
        .recommended-banner {{
            background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
            border: 1px solid #C7D2FE;
            border-left: 5px solid #6366F1;
            border-radius: 12px;
            padding: 1rem 1.15rem;
            margin: .9rem 0 1rem 0;
            color: #1E3A8A;
            font-size: .95rem;
            line-height: 1.55;
        }}
        .recommended-banner strong {{
            color: #3730A3;
        }}
        .feature-chart {{
            background: #FFFFFF;
            border: 1px solid {COLOR_CARD_BORDER};
            border-radius: 12px;
            padding: 1rem 1.1rem 1.15rem 1.1rem;
            margin-top: .6rem;
        }}
        .feature-row {{
            display: grid;
            grid-template-columns: 235px 1fr 72px;
            gap: .7rem;
            align-items: center;
            margin: .68rem 0;
        }}
        .feature-label {{
            color: #334155;
            font-size: .86rem;
            font-weight: 500;
            text-align: right;
            overflow-wrap: anywhere;
        }}
        .feature-track {{
            height: 17px;
            background: #F1F5F9;
            border-radius: 999px;
            overflow: hidden;
        }}
        .feature-bar {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #312E81 0%, #7E22CE 48%, #EC4899 78%, #FACC15 100%);
        }}
        .feature-value {{
            color: #475569;
            font-size: .82rem;
            font-weight: 600;
            text-align: left;
        }}
        .chart-legend {{
            color: #64748B;
            font-size: .78rem;
            margin-top: .8rem;
            text-align: right;
        }}
        .business-intro {{
            color: #334155;
            font-size: .98rem;
            line-height: 1.65;
            margin: .2rem 0 .7rem 0;
        }}
        .business-list {{
            margin: .4rem 0 0 1.1rem;
            padding-left: .8rem;
            color: #334155;
        }}
        .business-list li {{
            margin: .55rem 0;
            line-height: 1.55;
        }}
        .section-note {{
            color: #64748B;
            font-size: .78rem;
            margin-top: .65rem;
        }}
        @media (max-width: 850px) {{
            .feature-row {{
                grid-template-columns: 150px 1fr 62px;
            }}
            .feature-label {{
                font-size: .78rem;
            }}
        }}
        .section-gap {{
            display: block;
            height: 2.4rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# UI RENDERING
# ============================================================================

def render_hero():
    badge_html = "".join(f'<span class="badge-pill">{b}</span>' for b in HERO_BADGES)
    st.markdown(
        f"""
        <div class="hero-banner">
            <h1>💳 Credit Card Fraud Detection</h1>
            <p>Detect potentially fraudulent transactions using trained machine learning models.</p>
            <div class="badge-row">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.subheader("Project Information")
        st.markdown("**Credit Card Fraud Detection**")
        rows = {
            "Target": "Fraud",
            "Classes": "Legitimate / Fraud",
            "Deployment Models": str(len(MODEL_REGISTRY)),
        }
        for label, value in rows.items():
            st.markdown(
                f'<div class="sidebar-row"><span>{label}</span><span>{value}</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("Business Objective")
        st.caption(BUSINESS_OBJECTIVE)

        st.divider()
        st.subheader("Deployment Models")
        for name, info in MODEL_REGISTRY.items():
            st.markdown(
                f'<div class="sidebar-model-item">'
                f'<span style="color:{info["color"]};">●</span> {name}'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_model_selection() -> str:
    st.markdown("### 1. Model Selection")
    st.caption("Choose the trained model used for this prediction.")

    selected_model_name = st.selectbox(
        "Model",
        options=list(MODEL_REGISTRY.keys()),
        index=0,
        label_visibility="collapsed",
    )

    info = MODEL_REGISTRY[selected_model_name]
    st.markdown(
        f'<div class="model-card" style="border-left-color:{info["color"]};">'
        f'<p>{info["description"]}</p></div>',
        unsafe_allow_html=True,
    )
    return selected_model_name


def render_transaction_form():
    st.markdown("### 2. Transaction Details")

    with st.form("transaction_form"):
        with st.container(border=True):
            st.markdown("**Transaction**")
            c1, c2, c3 = st.columns(3)
            transaction_hour = c1.number_input(
                "Transaction Hour (0-23)", min_value=0, max_value=23, value=12, step=1
            )
            transaction_amount = c2.number_input(
                "Transaction Amount", min_value=0.0, value=150.0, step=10.0, format="%.2f"
            )
            avg_transaction_amount_30d = c3.number_input(
                "Average Transaction Amount (30d)",
                min_value=0.0, value=300.0, step=10.0, format="%.2f",
            )

        with st.container(border=True):
            st.markdown("**Account & History**")
            c1, c2 = st.columns(2)
            account_age_days = c1.number_input(
                "Account Age (days)", min_value=0.0, value=365.0, step=1.0
            )
            previous_chargebacks = c2.number_input(
                "Previous Chargebacks", min_value=0.0, value=0.0, step=1.0
            )

        with st.container(border=True):
            st.markdown("**Transaction Behavior**")
            c1, c2 = st.columns(2)
            transaction_velocity_1h = c1.number_input(
                "Transactions in Last 1 Hour", min_value=0.0, value=1.0, step=1.0
            )
            transaction_velocity_24h = c2.number_input(
                "Transactions in Last 24 Hours", min_value=0.0, value=3.0, step=1.0
            )

        with st.container(border=True):
            st.markdown("**Merchant & Context**")
            c1, c2, c3 = st.columns(3)
            merchant_category = c1.selectbox("Merchant Category", MERCHANT_CATEGORIES)
            transaction_country = c2.selectbox("Transaction Country", TRANSACTION_COUNTRIES)
            device_type = c3.selectbox("Device Type", DEVICE_TYPES)

            c4, c5 = st.columns(2)
            is_international = c4.selectbox("International Transaction?", ["No", "Yes"])
            is_high_risk_merchant = c5.selectbox("High-Risk Merchant?", ["No", "Yes"])

        st.markdown("")
        submitted = st.form_submit_button(
            "Check Transaction", type="primary", use_container_width=True
        )

    raw_record = {
        "transaction_hour": transaction_hour,
        "account_age_days": account_age_days,
        "previous_chargebacks": previous_chargebacks,
        "merchant_category": merchant_category,
        "transaction_country": transaction_country,
        "device_type": device_type,
        "is_international": 1 if is_international == "Yes" else 0,
        "is_high_risk_merchant": 1 if is_high_risk_merchant == "Yes" else 0,
        "transaction_amount": transaction_amount,
        "transaction_velocity_1h": transaction_velocity_1h,
        "transaction_velocity_24h": transaction_velocity_24h,
        "avg_transaction_amount_30d": avg_transaction_amount_30d,
    }
    return submitted, raw_record


def render_prediction_result(result):
    st.markdown("### 3. Prediction Result")

    if result is None:
        st.info("Fill in the transaction details above and click **Check Transaction** to see the result.")
        return

    is_fraud = result["prediction"] == "Fraud"
    banner_class = "result-fraud" if is_fraud else "result-legit"
    icon = "🚨" if is_fraud else "✅"
    st.markdown(
        f'<div class="result-banner {banner_class}">{icon} {result["prediction"].upper()}</div>',
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Fraud Probability", f"{result['fraud_probability']:.1%}")
    r2.metric("Legitimate Probability", f"{result['legitimate_probability']:.1%}")
    r3.metric("Risk Level", result["risk_level"])
    r4.metric("Model Used", result["model_name"])


def render_model_performance(selected_model_name: str):
    st.markdown("### 4. Model Performance")

    info = MODEL_REGISTRY[selected_model_name]
    metrics = info["metrics"]

    st.markdown(
        f'<div class="performance-accent" style="background:{info["color"]};"></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    c2.metric("Precision", f"{metrics['Precision']:.1%}")
    c3.metric("Recall", f"{metrics['Recall']:.1%}")
    c4.metric("F1-score", f"{metrics['F1']:.1%}")
    c5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.1%}")


def render_recommended_model():
    st.markdown("### 5. Recommended Model for Deployment")

    recommended = MODEL_REGISTRY["Stacking Ensemble"]
    metrics = recommended["metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", "Stacking Ensemble")
    c2.metric("Fraud F1-score", f"{metrics['F1']:.3f}")
    c3.metric("Fraud Recall", f"{metrics['Recall']:.3f}")
    c4.metric("Test Accuracy", f"{metrics['Accuracy']:.3f}")

    st.markdown(
        '<div class="recommended-banner">'
        "<strong>The Stacking Ensemble is the recommended model</strong> "
        "because it provides the strongest overall balance between fraud detection "
        "and false-positive control among the deployed models. It combines "
        "XGBoost, Random Forest, and Logistic Regression through a stacked "
        "meta-learner."
        "</div>",
        unsafe_allow_html=True,
    )


FEATURE_DISPLAY_NAMES = {
    "transaction_hour": "Transaction Hour",
    "account_age_days": "Account Age (days)",
    "previous_chargebacks": "Previous Chargebacks",
    "transaction_amount": "Transaction Amount",
    "transaction_velocity_1h": "Transactions in Last 1 Hour",
    "transaction_velocity_24h": "Transactions in Last 24 Hours",
    "avg_transaction_amount_30d": "Average Transaction Amount (30d)",
    "amount_to_avg_ratio": "Amount-to-Average Ratio",
    "is_international": "International Transaction",
    "is_high_risk_merchant": "High-Risk Merchant",
    "is_night_transaction": "Night Transaction",
    "merchant_category": "Merchant Category",
    "transaction_country": "Transaction Country",
    "device_type": "Device Type",
}

CATEGORICAL_FEATURES = {
    "merchant_category",
    "transaction_country",
    "device_type",
}


def _clean_feature_name(name: str) -> str:
    for prefix in ("num__", "binary__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    for base in CATEGORICAL_FEATURES:
        prefix = f"{base}_"
        if name.startswith(prefix) and name != base:
            value = name[len(prefix):]
            if base == "transaction_country":
                value = value.upper()
            else:
                value = value.replace("_", " ").title()
            return f"{FEATURE_DISPLAY_NAMES[base]}: {value}"

    return FEATURE_DISPLAY_NAMES.get(
        name,
        name.replace("_", " ").title(),
    )


def get_top_feature_importances(random_forest_model, preprocessor, top_n: int = 10):
    feature_names = preprocessor.get_feature_names_out()
    importances = random_forest_model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            "Feature-name count does not match Random Forest importance count."
        )

    rows = sorted(
        zip(feature_names, importances),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_n]

    return [
        {
            "name": _clean_feature_name(str(name)),
            "importance": float(importance),
        }
        for name, importance in rows
    ]


def render_feature_importance(random_forest_model, preprocessor):
    st.markdown("### 6. What Drives Fraud Detection?")
    st.markdown("**Top 10 Features**")

    features = get_top_feature_importances(
        random_forest_model,
        preprocessor,
        top_n=10,
    )

    max_importance = max((item["importance"] for item in features), default=1.0)

    rows_html = []
    for item in features:
        width = 100 * item["importance"] / max_importance if max_importance else 0
        rows_html.append(
            '<div class="feature-row">'
            f'<div class="feature-label">{escape(item["name"])}</div>'
            '<div class="feature-track">'
            f'<div class="feature-bar" style="width:{width:.2f}%;"></div>'
            '</div>'
            f'<div class="feature-value">{item["importance"]:.3f}</div>'
            '</div>'
        )

    st.markdown(
        '<div class="feature-chart">'
        + "".join(rows_html)
        + '<div class="chart-legend">'
        "Importance from the Random Forest model"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-note">'
        "Feature importance shows how strongly the Random Forest model relies on "
        "each feature across the dataset. It does not represent fraud probability "
        "or prove a causal relationship."
        "</div>",
        unsafe_allow_html=True,
    )


def render_business_interpretation():
    st.markdown("### 7. Business Interpretation")

    st.markdown(
        '<p class="business-intro">'
        "This fraud prediction system helps identify transactions that may require "
        "additional review, allowing fraud teams to focus their attention on the "
        "highest-risk activity while balancing unnecessary alerts."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<ul class="business-list">'
        "<li>Prioritize transactions with high fraud probability for further review.</li>"
        "<li>Use the Stacking Ensemble as the primary model when overall balance "
        "between fraud detection and alert quality matters.</li>"
        "<li>Use XGBoost as a strong standalone alternative, especially when a "
        "slightly higher fraud detection rate is preferred.</li>"
        "<li>Use Random Forest when missing a fraudulent transaction is more costly, "
        "while recognizing that stronger recall can generate more false alerts.</li>"
        "</ul>",
        unsafe_allow_html=True,
    )


def render_technical_details(engineered_df, selected_model_name: str, submitted: bool):
    with st.expander("Technical Details"):
        st.markdown(f"**Selected model:** {selected_model_name}")
        st.markdown(
            "**Preprocessing:** fitted preprocessor loaded from "
            "`preprocessor.joblib` — only `.transform()` is called, "
            "never `.fit()` or `.fit_transform()`."
        )
        if submitted and engineered_df is not None:
            ratio_value = engineered_df.loc[0, "amount_to_avg_ratio"]
            night_value = engineered_df.loc[0, "is_night_transaction"]
            t1, t2 = st.columns(2)
            t1.metric(
                "amount_to_avg_ratio",
                "N/A" if pd.isna(ratio_value) else f"{ratio_value:.3f}",
            )
            t2.metric(
                "is_night_transaction",
                "N/A" if pd.isna(night_value) else int(night_value),
            )
        else:
            st.caption("Engineered feature values will appear here after you check a transaction.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")
    inject_custom_css()

    render_hero()
    render_sidebar()

    preprocessor, models = load_models()

    selected_model_name = render_model_selection()
    submitted, raw_record = render_transaction_form()

    result = None
    engineered_df = None

    if submitted:
        errors = validate_input(raw_record)
        if errors:
            for err in errors:
                st.error(err)
        else:
            engineered_df = prepare_input(raw_record)
            selected_model = models[selected_model_name]
            result = predict_transaction(engineered_df, preprocessor, selected_model)
            result["model_name"] = selected_model_name

    render_prediction_result(result)
    render_model_performance(selected_model_name)
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    render_recommended_model()
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    render_feature_importance(models["Random Forest"], preprocessor)
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    render_business_interpretation()
    render_technical_details(engineered_df, selected_model_name, submitted)

    st.divider()
    st.caption(
        "Inference only — this app uses models and preprocessing trained and "
        "saved outside of it. No retraining, refitting, or retuning occurs here."
    )


if __name__ == "__main__":
    main()
