"""Streamlit application for the loan-eligibility project.

Three tabs: Overview and Data, Model Results, Eligibility Predictor.

Run from the project root:

    streamlit run app.py
"""

import logging

import matplotlib.pyplot as plt
import streamlit as st

from src.data import get_data_summary, load_data, validate_data
from src.logging_config import configure_logging
from src.modeling import predict_applicant, train_and_evaluate_models
from src.preprocessing import prepare_applicant_input

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Loan Eligibility Prediction", layout="wide")


# Streamlit re-runs the script on every interaction. Data and trained models are
# cached using the same pattern as Project 1.
@st.cache_data
def get_data():
    """Load and validate the dataset once per session."""
    df = load_data()
    validate_data(df)
    return df


@st.cache_resource
def get_model_results():
    """Train and evaluate all three classifiers once per session."""
    return train_and_evaluate_models(get_data())


# Fail once at startup with a controlled message instead of inside a tab.
try:
    df = get_data()
    results = get_model_results()
    summary = get_data_summary(df)
except FileNotFoundError:
    logger.exception("Credit dataset was not found")
    st.error("The credit dataset could not be found.")
    st.stop()
except ValueError as exc:
    logger.exception("Data validation failed")
    st.error(str(exc))
    st.stop()
except Exception:
    logger.exception("Unexpected application failure")
    st.error("The application encountered an unexpected error.")
    st.stop()


st.title("Loan Eligibility Prediction")
st.caption(
    "A modular rebuild of the Loan Eligibility notebook. Logistic regression, decision tree, and random forest are compared, with the strongest test-accuracy model is used by the predictor."
)

tab_data, tab_models, tab_predict = st.tabs(
    ["Overview and Data", "Model Results", "Eligibility Predictor"]
)


# --- Tab 1 -----------------------------------------------------------------
# Dataset shape, class balance, missing values, and preview.
with tab_data:
    st.header("Overview and Data")
    st.write(
        "The raw course dataset contains applicant, income, loan, credit, and property-area fields. Missing values are prepared inside the model pipelines."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{summary['rows']:,}")
    col2.metric("Columns", summary["columns"])
    col3.metric("Missing values", summary["missing_values"])
    col4.metric("Target", "Loan_Approved")

    st.subheader("Approval summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Approved", summary["approved"])
    col2.metric("Denied", summary["denied"])
    col3.metric("Approval rate", f"{summary['approval_rate']:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            ["Approved", "Denied"],
            [summary["approved"], summary["denied"]],
            color=["#4C72B0", "#C44E52"],
        )
        ax.set_ylabel("Applications")
        ax.set_title("Approval distribution")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col2:
        missing = summary["missing_per_column"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(missing.index[::-1], missing.values[::-1], color="#4C72B0")
        ax.set_xlabel("Missing values")
        ax.set_title("Missing values by column")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.subheader("Data preview")
    st.dataframe(df.head(20), width="stretch")


# --- Tab 2 -----------------------------------------------------------------
# Same split and preprocessing rules for all three notebook models.
with tab_models:
    st.header("Model Results")
    st.write(
        "All models use the same 80/20 stratified split. Preprocessing is fitted on training data only, and five-fold cross-validation pipeline."
    )

    metrics = results["metrics"]
    st.subheader("Model comparison")
    st.dataframe(
        metrics.style.format({
            "Train Accuracy": "{:.3f}",
            "Test Accuracy": "{:.3f}",
            "Accuracy Gap": "{:.3f}",
            "Test F1": "{:.3f}",
            "CV Mean Accuracy": "{:.3f}",
            "CV Std Accuracy": "{:.3f}",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Accuracy Gap is train accuracy minus test accuracy; larger positive values indicate more overfitting."
    )

    st.subheader(f"Selected model: {results['selected_model_name']}")
    st.caption("Selection rule: highest held-out test accuracy.")

    confusion = results["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted denied", "Predicted approved"])
    ax.set_yticks([0, 1], ["Actually denied", "Actually approved"])
    for row in range(2):
        for column in range(2):
            ax.text(column, row, confusion[row, column], ha="center", va="center")
    ax.set_title("Selected model — test set")
    st.pyplot(fig, width="content")
    plt.close(fig)


# --- Tab 3 -----------------------------------------------------------------
# Eleven inputs become one row in the same feature order used for training.
with tab_predict:
    st.header("Eligibility Predictor")
    st.write(
        f"Enter an application below. The prediction uses "
        f"{results['selected_model_name']}."
    )

    with st.form("applicant_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self employed", ["No", "Yes"])
            property_area = st.selectbox(
                "Property area", ["Urban", "Semiurban", "Rural"]
            )

        with col2:
            applicant_income = st.number_input(
                "Applicant income", min_value=150, max_value=81000, value=4000, step=100
            )
            coapplicant_income = st.number_input(
                "Coapplicant income", min_value=0, max_value=41667, value=1200, step=100
            )
            loan_amount = st.number_input(
                "Loan amount (thousands)", min_value=1, max_value=700, value=130, step=5
            )
            loan_term = st.selectbox(
                "Loan term (months)",
                [360, 480, 300, 240, 180, 120, 84, 60, 36, 12],
            )
            credit_history = st.selectbox(
                "Credit history",
                ["Has credit history", "No credit history"],
            )

        submitted = st.form_submit_button("Check eligibility")

    if submitted:
        try:
            applicant_df = prepare_applicant_input({
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": 1 if credit_history == "Has credit history" else 0,
                "Property_Area": property_area,
            })
            probability, approved = predict_applicant(
                results["selected_model"], applicant_df
            )

            if approved:
                st.success(f"Predicted result: Approved ({probability:.1%})")
            else:
                st.warning(f"Predicted result: Not approved ({probability:.1%})")
        except ValueError as exc:
            logger.exception("Applicant input validation failed")
            st.error(str(exc))
        except Exception:
            logger.exception("Unexpected prediction failure")
            st.error("The prediction could not be completed.")

    st.info(
        "Educational demo only, Not for use to make real lending decisions."
        
    )
