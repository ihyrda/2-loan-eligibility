"""Preparation for model training and single-applicant prediction.

The source file is raw, so the returned preprocessor imputes missing values,
one-hot encodes the notebook's six text fields, and applies min-max scaling.
It remains unfitted until a model pipeline is trained.
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.data import (
    cat_feats,
    med_feats,

    feats,
    mode_feats,
    numeric_feats,
    target,
    target_map,
)

logger = logging.getLogger(__name__)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove Loan_ID, preserve feature order, and encode N/Y as 0/1."""
    X = df[feats].copy()
    y = df[target].map(target_map).astype(int)
    y.name = target
    logger.info("Prepared training data: X=%s, y=%s", X.shape, y.shape)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create the notebook-equivalent preprocessing steps, unfitted.

    Separating the two numeric imputation rules preserves the notebook: loan
    amount uses a median, while term and credit history use their modes. The
    transformer lives inside every model pipeline so it learns from training
    data only.
    """
    continuous_steps = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", MinMaxScaler()),
    ])
    mode_numeric_steps = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", MinMaxScaler()),
    ])
    categorical_steps = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(transformers=[
        ("continuous", continuous_steps, med_feats),
        ("mode_numeric", mode_numeric_steps, mode_feats),
        ("categorical", categorical_steps, cat_feats),
    ])


def prepare_applicant_input(values: dict) -> pd.DataFrame:
    """Create one raw applicant row in the exact training feature order."""
    unknown = sorted(set(values) - set(feats))
    if unknown:
        raise ValueError(f"Unknown input fields: {unknown}")

    for column in numeric_feats:
        value = values.get(column)
        if value is not None and float(value) < 0:
            raise ValueError(f"'{column}' cannot be negative.")

    row = pd.DataFrame(
        [{column: values.get(column) for column in feats}],
        columns=feats,
    )
    logger.info("Prepared one applicant row with %d features", row.shape[1])
    return row
