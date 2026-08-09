"""Dataset loading, validation, and app summary metrics.

credit.csv is the raw course dataset. It contains missing predictor values and
categorical columns that are prepared later inside each model pipeline.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

root = Path(__file__).resolve().parents[1]
d_path = root / "data" / "credit.csv"

target = "Loan_Approved"
id = "Loan_ID"
target_map = {"N": 0, "Y": 1}

# The notebook one-hot encodes these six text fields.
cat_feats = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

# Loan amount is median-imputed in the notebook.
med_feats = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
]

# The notebook mode-imputes these fields, then passes them to MinMaxScaler.
mode_feats = ["Loan_Amount_Term", "Credit_History"]

feats = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

numeric_feats = med_feats + mode_feats
required_cols = [id] + feats + [target]


def load_data(path=None) -> pd.DataFrame:
    """Read the course dataset from disk; handle file-level failures only."""
    path = Path(path) if path is not None else d_path
    logger.info("Loading dataset from %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Credit dataset not found at {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset at {path} is empty or unreadable.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset at {path} is malformed: {exc}") from exc

    logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Confirm required columns, numeric fields, and Y/N target values."""
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    non_numeric = [
        column
        for column in numeric_feats
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"Expected numeric columns are non-numeric: {non_numeric}")

    if df[target].isna().any():
        raise ValueError(f"target '{target}' contains missing values.")

    unexpected = sorted(set(df[target].unique()) - set(target_map))
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")

    logger.info(
        "Validation passed: %d rows, %d columns, %d missing values",
        df.shape[0],
        df.shape[1],
        int(df.isna().sum().sum()),
    )


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return the dataset metrics displayed on the overview tab."""
    missing = df.isna().sum()
    approved = int((df[target] == "Y").sum())
    denied = int((df[target] == "N").sum())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(missing.sum()),
        "missing_per_column": missing[missing > 0].sort_values(ascending=False),
        "approved": approved,
        "denied": denied,
        "approval_rate": float(approved / len(df)),
    }
