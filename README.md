# Osteoporosis Risk Prediction Demo

An educational machine-learning web application that predicts osteoporosis risk
from demographic and bone-health measurements.

> **Important:** This repository is a learning project, not a clinical diagnostic
> system. Its output must not be used for medical decisions.

## What the project demonstrates

- Data loading and validation with pandas
- A leakage-aware feature set that excludes `T-score`
- Reproducible stratified train/test splitting
- Preprocessing inside a scikit-learn `Pipeline`
- Five-fold cross-validation for model and hyperparameter selection
- Comparison of Logistic Regression and Perceptron classifiers
- Final evaluation on a held-out test set using Accuracy, Precision, Recall and F1
- An interactive Streamlit interface for inference

## Dataset

The included CSV contains 169 records: 78 positive and 91 negative examples.
The model uses six features:

1. Gender
2. Age
3. Height
4. Weight
5. Lumbar spine measurement (L1-L4)
6. Bone mineral density (BMD)

`T-score` is excluded from training because it is strongly associated with the
clinical definition of osteoporosis and may leak information about the target.
The provenance and gender encoding of the dataset should be independently verified
before drawing substantive conclusions.

## Evaluation design

The dataset is divided once into stratified training and test sets using a fixed
random seed. Candidate models and hyperparameters are selected using five-fold
cross-validation only on the training set. The held-out test set is used once for
the final reported metrics.

This replaces the previous approach of repeatedly trying random validation splits
until a target accuracy was reached, which could produce an optimistically biased
estimate.

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run Project.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run Project.py
```

## Repository structure

```text
Project.py       Streamlit application
database.py      Data validation, model selection and evaluation
du_lieu.csv      Demonstration dataset
requirements.txt Python dependencies
```

## Limitations

- The dataset is very small, so performance estimates have high uncertainty.
- Dataset provenance and feature definitions are not documented.
- The app has not been clinically validated.
- A larger external dataset is required before considering real-world use.
