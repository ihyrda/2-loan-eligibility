"""Test suite for the loan-eligibility project.

Run from the project root inside the activated environment:

    python -m pytest -q

The dataset and complete model result bundle are session fixtures so the three
models and cross-validation are not repeated for every test.
"""

import numpy as np
import pytest

from src.data import feats, required_cols, target, load_data, validate_data
from src.modeling import (
    build_models,
    predict_applicant,
    split_data,
    train_and_evaluate_models,
)
from src.preprocessing import (
    build_preprocessor,
    prepare_applicant_input,
    split_features_target,
)


@pytest.fixture(scope="session")
def df():
    """The validated course dataset."""
    frame = load_data()
    validate_data(frame)
    return frame


@pytest.fixture(scope="session")
def results(df):
    """The full trained result bundle."""
    return train_and_evaluate_models(df)


@pytest.fixture
def applicant_values():
    """One complete applicant row using source-column names."""
    return {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 130,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    }


# --- data ------------------------------------------------------------------

# The bundled CSV is the expected course dataset.
def test_dataset_loads_with_expected_shape_and_columns(df):
    assert df.shape == (614, 13)
    assert list(df.columns) == required_cols


# A missing file raises instead of returning an empty frame.
def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


# A swapped or incomplete source file is rejected.
def test_missing_column_raises_error(df):
    with pytest.raises(ValueError, match="Loan_Approved"):
        validate_data(df.drop(columns=[target]))


# Training labels must remain the notebook's N/Y values.
def test_invalid_target_raises_error(df):
    invalid = df.copy()
    invalid.loc[0, target] = "Maybe"
    with pytest.raises(ValueError, match="target"):
        validate_data(invalid)


# --- preprocessing ---------------------------------------------------------

# Loan_ID is not a predictor, the target is separated, and order is frozen.
def test_features_and_target_are_prepared(df):
    X, y = split_features_target(df)
    assert list(X.columns) == feats
    assert "Loan_ID" not in X.columns
    assert target not in X.columns
    assert set(y.unique()) == {0, 1}


# The transformer must fill every source missing value before modelling.
def test_preprocessor_handles_missing_values(df):
    X, _ = split_features_target(df)
    transformed = build_preprocessor().fit_transform(X)
    assert transformed.shape[0] == len(df)
    assert not np.isnan(transformed).any()


# The app's dictionary becomes exactly the eleven training columns.
def test_applicant_input_has_expected_columns(applicant_values):
    row = prepare_applicant_input(applicant_values)
    assert row.shape == (1, 11)
    assert list(row.columns) == feats


# Numeric fields cannot describe negative income or loan values.
def test_negative_input_is_rejected(applicant_values):
    applicant_values["ApplicantIncome"] = -1
    with pytest.raises(ValueError, match="ApplicantIncome"):
        prepare_applicant_input(applicant_values)


# --- modelling -------------------------------------------------------------

# Stratification retains the dataset's approval share on both sides.
def test_split_is_stratified(df):
    X, y = split_features_target(df)
    _, _, y_train, y_test = split_data(X, y)
    assert y_train.mean() == pytest.approx(y.mean(), abs=0.01)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.01)


# The candidate set is faithful to the source notebook.
def test_model_candidates_match_notebook():
    assert set(build_models()) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }


# Contract between modeling.py and the model-results tab.
def test_metrics_table_matches_app(results):
    metrics = results["metrics"]
    assert set(metrics["Model"]) == set(results["models"])
    for column in (
        "Train Accuracy",
        "Test Accuracy",
        "Accuracy Gap",
        "Test F1",
        "CV Mean Accuracy",
        "CV Std Accuracy",
    ):
        assert column in metrics.columns
        assert metrics[column].notna().all()


# The selected model follows the documented highest-test-accuracy rule.
def test_selection_rule_is_highest_test_accuracy(results):
    metrics = results["metrics"]
    expected = metrics.loc[metrics["Test Accuracy"].idxmax(), "Model"]
    assert results["selected_model_name"] == expected
    assert results["selected_model"] is results["models"][expected]


# Confusion counts cover every held-out observation.
def test_confusion_matrix_shape_and_total(results):
    confusion = results["confusion_matrix"]
    assert confusion.shape == (2, 2)
    assert confusion.sum() == len(results["test_actual"])


# Full app path: eleven raw inputs become a valid probability and decision.
def test_prediction_path_runs_end_to_end(results, applicant_values):
    row = prepare_applicant_input(applicant_values)
    probability, approved = predict_applicant(results["selected_model"], row)
    assert 0.0 <= probability <= 1.0
    assert approved == (probability >= 0.50)


# The split and randomized classifiers have fixed seeds.
def test_fixed_seed_is_reproducible(df):
    X, y = split_features_target(df)
    first_split = split_data(X, y)[0].index.tolist()
    second_split = split_data(X, y)[0].index.tolist()
    assert first_split == second_split

    X_train, X_test, y_train, _ = split_data(X, y)
    first = build_models()["Random Forest"].fit(X_train, y_train)
    second = build_models()["Random Forest"].fit(X_train, y_train)
    assert np.array_equal(first.predict(X_test), second.predict(X_test))
