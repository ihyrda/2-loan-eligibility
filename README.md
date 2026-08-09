# Loan Eligibility Prediction

A modular rebuild of `Loan_Eligibility_Model_Solution.ipynb`. The project prepares the raw course dataset, compares logistic regression, decision tree, and random forest classifiers, selects the best held-out accuracy, and serves the selected model through a Streamlit application.

## Setup

Requires **Python 3.11**. Run commands from the project root, where `app.py` is located. The virtual environment is not included in the repository.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run test verifications (pytest)

```powershell
python -m pytest -v
```

### Run the application

```powershell
streamlit run app.py
```


## Project structure

```text
2_loan_eligibility/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── credit.csv
├── original_notebook/
│   └── Loan_Eligibility_Model_Solution.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── modeling.py
│   └── logging_config.py
└── tests/
    └── test_project.py
```

## How it works

Three classifiers are compared -- logistic regression, a decision tree, and a random forest -- on an 80/20 split stratified by approval. Selection is on highest held-out test accuracy. Read that figure against the 68.73% approval rate in this dataset: a model that approves every application would score roughly that, so F1 is reported alongside accuracy as supporting evidence.

Every model is a `Pipeline` whose first step is an unfitted `ColumnTransformer`. Imputation, encoding, and scaling are therefore fitted on each split's training portion alone, and refitted per fold during cross-validation. Fitting a transformer on the full dataset and splitting afterwards would leak test-set medians and category frequencies into training; building it into the pipeline makes that impossible to express rather than merely discouraged. Three tests enforce it.

The application has three tabs: dataset overview, model comparison with the selected model's confusion matrix, and a prediction form for one application.

## Limitations

**No model in this project beats a single column.** Approving whenever `Credit_History` is 1, using the same imputer the models use, scores a higher test F1 than any of the three.

This is an educational demonstration using a small teaching dataset. It is not a lending-decision system and must not be used to assess a real application.

## Links

- GitHub repository: _to be added after upload_
- Deployed Streamlit app: _to be added after deployment_
