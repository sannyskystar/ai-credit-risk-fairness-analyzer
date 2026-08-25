
import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Fairness Analyzer",
    page_icon="⚖️",
    layout="wide"
)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

MODEL_DIR = "fairness_models"

baseline_model = joblib.load(
    os.path.join(MODEL_DIR, "baseline_model.pkl")
)

fairness_model = joblib.load(
    os.path.join(MODEL_DIR, "fairness_model.pkl")
)

preprocessor = joblib.load(
    os.path.join(MODEL_DIR, "preprocessor.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.title("⚖️ AI Loan Credit Fairness Analyzer")

st.markdown(
    """
This application demonstrates how a machine learning credit-risk model
can be evaluated for **group-level fairness** and improved using
**Reweighing-based bias mitigation**.

The underlying model predicts whether an applicant represents:

- **Good credit risk**
- **Bad credit risk**

The fairness analysis compares model behaviour across **male and female**
applicants.
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "Credit Risk Predictor",
        "Fairness Dashboard",
        "About the Project"
    ]
)


# ============================================================
# PAGE 1 — CREDIT RISK PREDICTOR
# ============================================================

if page == "Credit Risk Predictor":

    st.header("Credit Risk Prediction")

    st.write(
        "Enter the applicant's information below. "
        "The application will compare predictions from the "
        "original model and the fairness-mitigated model."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

        job = st.selectbox(
            "Job",
            [0, 1, 2, 3],
            index=2,
            help="German Credit dataset job category."
        )

        housing = st.selectbox(
            "Housing",
            ["own", "rent", "free"]
        )

        saving_accounts = st.selectbox(
            "Saving accounts",
            ["unknown", "little", "moderate", "quite rich", "rich"]
        )

    with col2:

        checking_account = st.selectbox(
            "Checking account",
            ["unknown", "little", "moderate", "rich"]
        )

        credit_amount = st.number_input(
            "Credit amount",
            min_value=0,
            max_value=50000,
            value=3000,
            step=100
        )

        duration = st.number_input(
            "Loan duration (months)",
            min_value=1,
            max_value=72,
            value=12,
            step=1
        )

        purpose = st.selectbox(
            "Purpose",
            [
                "radio/TV",
                "education",
                "furniture/equipment",
                "car",
                "business",
                "domestic appliances",
                "repairs",
                "vacation/others"
            ]
        )

    st.divider()

    predict_button = st.button(
        "🔍 Analyze Credit Risk",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        # --------------------------------------------------------
        # Create input dataframe
        # --------------------------------------------------------

        input_data = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "Job": [job],
            "Housing": [housing],
            "Saving accounts": [saving_accounts],
            "Checking account": [checking_account],
            "Credit amount": [credit_amount],
            "Duration": [duration],
            "Purpose": [purpose]
        })


        # --------------------------------------------------------
        # Apply the exact preprocessing used during training
        # --------------------------------------------------------

        processed_input = preprocessor.transform(input_data)

        scaled_input = scaler.transform(processed_input)


        # --------------------------------------------------------
        # Predictions
        # --------------------------------------------------------

        baseline_prediction = baseline_model.predict(
            scaled_input
        )[0]

        fairness_prediction = fairness_model.predict(
            scaled_input
        )[0]


        baseline_probability = baseline_model.predict_proba(
            scaled_input
        )[0][1]

        fairness_probability = fairness_model.predict_proba(
            scaled_input
        )[0][1]


        # --------------------------------------------------------
        # Display results
        # --------------------------------------------------------

        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.markdown("### Original Model")

            if baseline_prediction == 1:
                st.success("✅ Good Credit Risk")
            else:
                st.error("⚠️ Bad Credit Risk")

            st.metric(
                "Good-risk probability",
                f"{baseline_probability:.1%}"
            )


        with result_col2:

            st.markdown("### Fairness-Mitigated Model")

            if fairness_prediction == 1:
                st.success("✅ Good Credit Risk")
            else:
                st.error("⚠️ Bad Credit Risk")

            st.metric(
                "Good-risk probability",
                f"{fairness_probability:.1%}"
            )


        # --------------------------------------------------------
        # Explain difference
        # --------------------------------------------------------

        if baseline_prediction != fairness_prediction:

            st.warning(
                """
                The two models produced different predictions for this
                applicant. This demonstrates how bias mitigation can
                change individual model decisions.
                """
            )

        else:

            st.info(
                """
                Both models produced the same prediction for this
                applicant. This does not mean the models behave identically
                across the entire population.
                """
            )


# ============================================================
# PAGE 2 — FAIRNESS DASHBOARD
# ============================================================

elif page == "Fairness Dashboard":

    st.header("Fairness Evaluation Dashboard")

    st.write(
        """
        These results come from the project's **200-record held-out
        test set**. They compare the original model with the model
        trained using Reweighing.
        """
    )


    # --------------------------------------------------------
    # Experimental results
    # --------------------------------------------------------

    baseline_metrics = {
        "Accuracy": {
            "Female": 70.00,
            "Male": 75.00
        },
        "Predicted Good Rate": {
            "Female": 60.00,
            "Male": 83.57
        },
        "True Positive Rate": {
            "Female": 72.50,
            "Male": 91.00
        },
        "False Positive Rate": {
            "Female": 35.00,
            "Male": 65.00
        }
    }


    mitigated_metrics = {
        "Accuracy": {
            "Female": 68.33,
            "Male": 75.00
        },
        "Predicted Good Rate": {
            "Female": 71.67,
            "Male": 79.29
        },
        "True Positive Rate": {
            "Female": 80.00,
            "Male": 88.00
        },
        "False Positive Rate": {
            "Female": 55.00,
            "Male": 57.50
        }
    }


    # --------------------------------------------------------
    # Headline metrics
    # --------------------------------------------------------

    st.subheader("Key Fairness Metrics")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:

        st.metric(
            "Statistical Parity Difference",
            "-0.0762",
            delta="+0.1595"
        )

        st.caption(
            "Before: -0.2357"
        )


    with metric_col2:

        st.metric(
            "Disparate Impact",
            "0.9039",
            delta="+0.1860"
        )

        st.caption(
            "Before: 0.7179"
        )


    with metric_col3:

        st.metric(
            "Overall Accuracy",
            "73.0%",
            delta="-0.5%"
        )

        st.caption(
            "Before: 73.5%"
        )


    st.divider()


    # --------------------------------------------------------
    # Before vs After table
    # --------------------------------------------------------

    st.subheader("Before vs After Reweighing")

    comparison_data = []

    for metric in baseline_metrics:

        comparison_data.append({
            "Metric": metric,
            "Female Before": baseline_metrics[metric]["Female"],
            "Male Before": baseline_metrics[metric]["Male"],
            "Female After": mitigated_metrics[metric]["Female"],
            "Male After": mitigated_metrics[metric]["Male"]
        })


    comparison_df = pd.DataFrame(comparison_data)

    st.dataframe(
        comparison_df.style.format(
            {
                "Female Before": "{:.2f}%",
                "Male Before": "{:.2f}%",
                "Female After": "{:.2f}%",
                "Male After": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Visual comparison
    # --------------------------------------------------------

    st.subheader("Model Behaviour by Gender")

    selected_metric = st.selectbox(
        "Select metric:",
        list(baseline_metrics.keys())
    )


    chart_df = pd.DataFrame({
        "Female": [
            baseline_metrics[selected_metric]["Female"],
            mitigated_metrics[selected_metric]["Female"]
        ],
        "Male": [
            baseline_metrics[selected_metric]["Male"],
            mitigated_metrics[selected_metric]["Male"]
        ]
    }, index=["Before Reweighing", "After Reweighing"])


    st.bar_chart(chart_df)


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.subheader("What changed?")

    st.markdown(
        """
### Before mitigation

The original model showed substantial differences between male and
female applicants:

- Statistical Parity Difference: **-0.2357**
- Disparate Impact: **0.7179**
- TPR gap: **18.5 percentage points**
- FPR gap: **30 percentage points**

### After Reweighing

The disparities were substantially reduced:

- Statistical Parity Difference: **-0.0762**
- Disparate Impact: **0.9039**
- TPR gap: **8 percentage points**
- FPR gap: **2.5 percentage points**

Overall accuracy changed only from **73.5% to 73.0%**.

This demonstrates the central trade-off explored by the project:
**improving fairness while attempting to preserve predictive performance.**
"""
    )


# ============================================================
# PAGE 3 — ABOUT
# ============================================================

else:

    st.header("About the Project")

    st.markdown(
        """
## AI Loan Credit Fairness Analyzer

This project explores fairness in machine learning using a German Credit
risk dataset.

### Research workflow

**Dataset → Preprocessing → Logistic Regression → Fairness Evaluation
→ Reweighing → New Model → Comparison**

### Machine Learning

The project uses:

- Logistic Regression
- One-Hot Encoding
- Feature Scaling
- AIF360 Reweighing
- Statistical Parity Difference
- Disparate Impact
- True Positive Rate
- False Positive Rate

### Research objective

The objective is not simply to maximize prediction accuracy.

Instead, the project investigates whether a machine learning model can
reduce group-level disparities while maintaining reasonable predictive
performance.

### Important limitation

Fairness metrics are statistical measurements. A disparity does not
automatically prove intentional discrimination or causal bias.

The results depend on the dataset, selected protected attribute,
model, preprocessing choices and evaluation sample.
"""
    )

    st.divider()

    st.caption(
        "Research-based AI/ML portfolio project • German Credit dataset"
    )

