from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# T-score is intentionally excluded. In this dataset it is strongly tied to the
# target definition, so including it would risk target leakage and inflate scores.
FEATURE_COLS = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "lumbar spine(L1-L4)",
    "BMD(Bone Mineral Density)",
]
TARGET_COL = "Osteoporosis"


def load_data(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tại {path.resolve()}")

    df = pd.read_csv(path).dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    if y.nunique() != 2:
        raise ValueError("Target phải có đúng hai lớp 0 và 1.")

    return X, y


def build_search():
    pipeline = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    )

    parameter_grid = [
        {
            "poly__degree": [1, 2],
            "classifier": [LogisticRegression(max_iter=5000, random_state=42)],
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
        {
            "poly__degree": [1, 2],
            "classifier": [
                Perceptron(
                    max_iter=5000,
                    penalty="l2",
                    alpha=0.0001,
                    random_state=42,
                )
            ],
            "classifier__eta0": [0.001, 0.01, 0.1],
            "classifier__class_weight": [None, "balanced"],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )


def train_and_evaluate(csv_path: str, test_size: float = 0.2):
    X, y = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    search = build_search()
    search.fit(X_train, y_train)
    predictions = search.best_estimator_.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    classifier_name = search.best_estimator_.named_steps["classifier"].__class__.__name__
    return {
        "model": search.best_estimator_,
        "model_name": classifier_name,
        "best_params": search.best_params_,
        "cv_f1": search.best_score_,
        "metrics": metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def patient_input_form():
    with st.form("patient-input"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Giới tính", ["Nữ", "Nam"])
            age = st.number_input("Tuổi", min_value=0, max_value=120, value=50)
            height = st.number_input("Chiều cao (cm)", min_value=50.0, value=165.0)
        with col2:
            weight = st.number_input("Cân nặng (kg)", min_value=10.0, value=60.0)
            lumbar_spine = st.number_input(
                "Chỉ số cột sống thắt lưng (L1-L4)", min_value=0.0, value=1.0
            )
            bmd = st.number_input("Mật độ khoáng xương (BMD)", min_value=0.0, value=0.8)

        submitted = st.form_submit_button("Dự đoán", type="primary")

    if not submitted:
        return None

    # This mapping follows the convention used by the original application.
    gender_value = 1 if gender == "Nam" else 0
    return pd.DataFrame(
        [[gender_value, age, height, weight, lumbar_spine, bmd]],
        columns=FEATURE_COLS,
    )
