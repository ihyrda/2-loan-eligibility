"""Splitting, training, evaluation, selection, and prediction.

The source notebook compares logistic regression, a decision tree, and a
random forest. All three are reproduced with fixed seeds. Each is a pipeline,
so imputation, encoding, and scaling are fitted on training data only.
"""

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor, split_features_target

logger = logging.getLogger(__name__)

random_state = 42
t_size = 0.20
CV_folds = 5


def split_data(X: pd.DataFrame, y: pd.Series):
    """Create reproducible stratified 80/20 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = t_size,
        random_state = random_state,
        stratify = y,
    )
    logger.info("Split: %d train / %d test rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    """Create the three notebook classifier candidates as pipelines."""
    return {
        "Logistic Regression": Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "Decision Tree": Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("model", DecisionTreeClassifier(random_state=random_state)),
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
            )),
        ]),
    }


def train_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit every candidate model."""
    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
    return models


def evaluate_models(models, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Return train/test accuracy and held-out F1 for each classifier."""
    rows = []
    for name, model in models.items():
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        rows.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, train_predictions),
            "Test Accuracy": accuracy_score(y_test, test_predictions),
            "Accuracy Gap": (
                accuracy_score(y_train, train_predictions)
                - accuracy_score(y_test, test_predictions)
            ),
            "Test F1": f1_score(y_test, test_predictions),
        })
        logger.info(
            "%s: test accuracy %.4f, test F1 %.4f",
            name,
            rows[-1]["Test Accuracy"],
            rows[-1]["Test F1"],
        )
    return pd.DataFrame(rows)


def cross_validate_models(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Return five-fold stratified accuracy mean and standard deviation."""
    folds = StratifiedKFold(
        n_splits=CV_folds,
        shuffle=True,
        random_state=random_state,
    )
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=folds, scoring="accuracy")
        rows.append({
            "Model": name,
            "CV Mean Accuracy": scores.mean(),
            "CV Std Accuracy": scores.std(),
        })
    return pd.DataFrame(rows)


def select_model(metrics_df: pd.DataFrame, trained_models: dict):
    """Select the classifier with the highest held-out test accuracy."""
    ranked = metrics_df.sort_values("Test Accuracy", ascending=False)
    name = ranked.iloc[0]["Model"]
    logger.info("Selected model: %s", name)
    return name, trained_models[name]


def predict_applicant(model, applicant_df: pd.DataFrame) -> tuple[float, bool]:
    """Return approval probability and the standard 0.50 decision."""
    probability = float(model.predict_proba(applicant_df)[0][1])
    approved = bool(probability >= 0.50)
    logger.info("Applicant probability %.4f, approved %s", probability, approved)
    return probability, approved


def train_and_evaluate_models(df: pd.DataFrame) -> dict:
    """Run the complete modular workflow and return one result bundle."""
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = train_models(build_models(), X_train, y_train)
    metrics = evaluate_models(models, X_train, X_test, y_train, y_test)
    cv_metrics = cross_validate_models(build_models(), X, y)
    metrics = metrics.merge(cv_metrics, on="Model")
    selected_name, selected_model = select_model(metrics, models)
    selected_predictions = selected_model.predict(X_test)

    return {
        "models": models,
        "metrics": metrics,
        "selected_model_name": selected_name,
        "selected_model": selected_model,
        "confusion_matrix": confusion_matrix(y_test, selected_predictions),
        "test_actual": y_test,
        "test_predictions": selected_predictions,
    }
