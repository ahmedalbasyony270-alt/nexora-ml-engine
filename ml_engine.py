import os
import tempfile
import traceback
from typing import Any, List

import numpy as np
import pandas as pd

from flask import Flask, jsonify, request

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)


# ============================================================
# APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

ALLOWED_MODELS = {
    "logistic_regression",
    "linear_regression",
    "decision_tree",
    "random_forest",
    "knn",
    "svm"
}


# ============================================================
# JSON SAFE
# ============================================================

def json_safe(value: Any):
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if pd.isna(value):
        return None

    return value


# ============================================================
# LOAD CSV
# ============================================================

def load_dataset(file_path: str) -> pd.DataFrame:

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            "Dataset file was not found."
        )

    if not file_path.lower().endswith(".csv"):
        raise ValueError(
            "Only CSV files are supported."
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    # Remove completely empty rows
    df = df.dropna(
        axis=0,
        how="all"
    )

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove duplicated rows
    df = df.drop_duplicates()

    # Clean column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# AUTO TASK DETECTION
# ============================================================

def detect_task(target: pd.Series) -> str:

    if not pd.api.types.is_numeric_dtype(target):
        return "classification"

    unique_count = target.nunique(
        dropna=True
    )

    if unique_count <= 10:
        return "classification"

    return "regression"


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    X: pd.DataFrame
) -> ColumnTransformer:

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    transformers = []

    if numeric_columns:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    if not transformers:
        raise ValueError(
            "No usable feature columns were found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    model_name: str,
    task_type: str
):

    if task_type == "classification":

        if model_name == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )

        if model_name == "decision_tree":
            return DecisionTreeClassifier(
                random_state=RANDOM_STATE
            )

        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=150,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

        if model_name == "knn":
            return KNeighborsClassifier(
                n_neighbors=5
            )

        if model_name == "svm":
            return SVC(
                probability=True,
                random_state=RANDOM_STATE
            )

    if task_type == "regression":

        if model_name == "linear_regression":
            return LinearRegression()

        if model_name == "decision_tree":
            return DecisionTreeRegressor(
                random_state=RANDOM_STATE
            )

        if model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=150,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

        if model_name == "svm":
            return SVR()

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def get_classification_metrics(
    y_true,
    y_pred
):

    labels = sorted(
        list(
            set(
                y_true.astype(str)
            )
            |
            set(
                y_pred.astype(str)
            )
        )
    )

    cm = confusion_matrix(
        y_true.astype(str),
        y_pred.astype(str),
        labels=labels
    )

    report = classification_report(
        y_true.astype(str),
        y_pred.astype(str),
        labels=labels,
        zero_division=0,
        output_dict=True
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "labels": labels
    }


# ============================================================
# REGRESSION METRICS
# ============================================================

def get_regression_metrics(
    y_true,
    y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = float(
        np.sqrt(mse)
    )

    return {
        "r2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),
        "mse": float(mse),
        "rmse": rmse
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    pipeline: Pipeline
):

    try:

        preprocessor = pipeline.named_steps[
            "preprocessor"
        ]

        model = pipeline.named_steps[
            "model"
        ]

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            importances = (
                model.feature_importances_
            )

        elif hasattr(
            model,
            "coef_"
        ):

            coefficients = model.coef_

            if len(
                getattr(
                    coefficients,
                    "shape",
                    ()
                )
            ) > 1:

                importances = np.mean(
                    np.abs(coefficients),
                    axis=0
                )

            else:

                importances = np.abs(
                    coefficients
                )

        else:

            return []

        result = []

        for name, importance in zip(
            feature_names,
            importances
        ):

            result.append({
                "feature": str(name),
                "importance": float(
                    importance
                )
            })

        result.sort(
            key=lambda item:
                item["importance"],
            reverse=True
        )

        return result[:20]

    except Exception:
        return []


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_one_model(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    model_name: str
):

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    if model_name not in ALLOWED_MODELS:

        raise ValueError(
            f"Unsupported model '{model_name}'."
        )

    work_df = df.copy()

    # Remove rows with missing target
    work_df = work_df[
        work_df[target_column].notna()
    ].copy()

    if len(work_df) < 10:

        raise ValueError(
            "Dataset needs at least 10 valid rows."
        )

    X = work_df.drop(
        columns=[target_column]
    )

    y = work_df[target_column]

    # Remove obvious ID-like columns
    drop_columns = []

    for column in X.columns:

        if X[column].dtype == "object":

            unique_ratio = (
                X[column].nunique(
                    dropna=False
                )
                /
                len(X)
            )

            if unique_ratio > 0.98:
                drop_columns.append(column)

    if drop_columns:

        X = X.drop(
            columns=drop_columns
        )

    if X.shape[1] == 0:

        raise ValueError(
            "No usable feature columns remain."
        )

    # Target preparation
    if task_type == "classification":

        y = y.astype(str)

        if y.nunique() < 2:

            raise ValueError(
                "Classification requires at least 2 classes."
            )

    elif task_type == "regression":

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

        valid = y.notna()

        X = X.loc[valid]
        y = y.loc[valid]

    else:

        raise ValueError(
            "Task must be classification or regression."
        )

    if len(X) < 10:

        raise ValueError(
            "Not enough usable rows."
        )

    # Check if stratification is safe
    stratify = None

    if task_type == "classification":

        class_counts = y.value_counts()

        if (
            len(class_counts) > 1
            and class_counts.min() >= 2
        ):

            stratify = y

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=stratify
        )
    )

    preprocessor = build_preprocessor(
        X_train
    )

    model = create_model(
        model_name,
        task_type
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    predictions = pipeline.predict(
        X_test
    )

    result = {
        "model": model_name,
        "task": task_type,
        "train_rows": int(
            len(X_train)
        ),
        "test_rows": int(
            len(X_test)
        ),
        "features": X.columns.tolist(),
        "dropped_columns": drop_columns
    }

    # Metrics
    if task_type == "classification":

        metrics = get_classification_metrics(
            y_test,
            predictions
        )

        result["metrics"] = metrics

        result["classes"] = (
            metrics["labels"]
        )

    else:

        result["metrics"] = (
            get_regression_metrics(
                y_test,
                predictions
            )
        )

    # Feature importance
    result["feature_importance"] = (
        get_feature_importance(
            pipeline
        )
    )

    # Prediction sample
    sample = []

    sample_count = min(
        len(X_test),
        10
    )

    for index in range(sample_count):

        sample.append({
            "actual": json_safe(
                y_test.iloc[index]
            ),
            "predicted": json_safe(
                predictions[index]
            )
        })

    result["prediction_sample"] = sample

    return result


# ============================================================
# TRAIN MULTIPLE MODELS
# ============================================================

def train_models(
    file_path: str,
    target_column: str,
    task_type: str,
    models: List[str]
):

    df = load_dataset(
        file_path
    )

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    if task_type == "auto":

        task_type = detect_task(
            df[target_column]
        )

    valid_models = [
        model
        for model in models
        if model in ALLOWED_MODELS
    ]

    if not valid_models:

        raise ValueError(
            "No valid models were selected."
        )

    results = []
    failures = []

    for model_name in valid_models:

        try:

            result = train_one_model(
                df=df,
                target_column=target_column,
                task_type=task_type,
                model_name=model_name
            )

            results.append(result)

        except Exception as error:

            failures.append({
                "model": model_name,
                "error": str(error)
            })

    if not results:

        raise RuntimeError(
            "All selected models failed."
        )

    # Technical best-model selection
    if task_type == "classification":

        results.sort(
            key=lambda item:
                item["metrics"]["f1"],
            reverse=True
        )

    else:

        results.sort(
            key=lambda item:
                item["metrics"]["r2"],
            reverse=True
        )

    return {
        "success": True,
        "dataset_rows": int(
            len(df)
        ),
        "dataset_columns": int(
            len(df.columns)
        ),
        "columns": df.columns.tolist(),
        "target": target_column,
        "task": task_type,
        "models_trained": len(results),
        "results": results,
        "best_model": results[0]["model"],
        "failed_models": failures
    }


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return jsonify({
        "name": "NEXORA AI ML Engine",
        "status": "online",
        "version": "1.0.0",
        "service": "machine-learning"
    })


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return jsonify({
        "status": "ok"
    })


# ============================================================
# TRAIN API
# ============================================================

@app.post("/train")
def train_api():

    temp_path = None

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "CSV file is required."
            }), 400

        uploaded_file = request.files[
            "file"
        ]

        if not uploaded_file.filename:

            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400

        filename = (
            uploaded_file.filename
            .lower()
        )

        if not filename.endswith(
            ".csv"
        ):

            return jsonify({
                "success": False,
                "error":
                    "Only CSV files are supported."
            }), 400

        target_column = request.form.get(
            "target_column",
            ""
        ).strip()

        task_type = request.form.get(
            "task_type",
            "auto"
        ).strip().lower()

        models_raw = request.form.get(
            "models",
            ""
        ).strip()

        if not target_column:

            return jsonify({
                "success": False,
                "error":
                    "target_column is required."
            }), 400

        if task_type not in {
            "auto",
            "classification",
            "regression"
        }:

            return jsonify({
                "success": False,
                "error":
                    "Invalid task_type."
            }), 400

        models = [
            item.strip()
            for item in models_raw.split(",")
            if item.strip()
        ]

        if not models:

            models = [
                "decision_tree",
                "random_forest"
            ]

        # Save temporary CSV
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        ) as temp_file:

            uploaded_file.save(
                temp_file.name
            )

            temp_path = temp_file.name

        result = train_models(
            file_path=temp_path,
            target_column=target_column,
            task_type=task_type,
            models=models
        )

        return jsonify(
            json_safe(result)
        )

    except Exception as error:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

    finally:

        if (
            temp_path
            and
            os.path.isfile(temp_path)
        ):

            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )