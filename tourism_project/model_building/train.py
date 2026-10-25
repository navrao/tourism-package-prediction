import os
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the splits created by the data-prep job
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# Column groups
categorical = ["TypeofContact", "Occupation", "Gender", "ProductPitched",
               "MaritalStatus", "Designation"]
numeric = [c for c in Xtrain.columns if c not in categorical]

# Preprocessing: fill gaps, scale numbers, one-hot encode text columns
preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    (make_pipeline(SimpleImputer(strategy="most_frequent"),
                   OneHotEncoder(handle_unknown="ignore")), categorical),
)

# Model: XGBoost. scale_pos_weight helps with fewer buyers than non-buyers.
ratio = (ytrain == 0).sum() / (ytrain == 1).sum()
model = xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42,
                          eval_metric="logloss")
pipe = make_pipeline(preprocessor, model)

# Parameters to tune
param_grid = {
    "xgbclassifier__n_estimators": [200, 300],
    "xgbclassifier__max_depth": [5, 7, 9],
    "xgbclassifier__learning_rate": [0.1, 0.2],
}

mlflow.set_experiment("tourism-package-prediction")

with mlflow.start_run(run_name="xgboost-tuning"):
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # Log every parameter combination that was tried
    results = grid.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"trial-{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_f1", results["mean_test_score"][i])

    # Log the best parameters
    best = grid.best_estimator_
    mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})

    # Evaluate the best model
    pred_train = best.predict(Xtrain)
    pred_test = best.predict(Xtest)
    proba_test = best.predict_proba(Xtest)[:, 1]

    metrics = {
        "train_f1": f1_score(ytrain, pred_train),
        "test_accuracy": accuracy_score(ytest, pred_test),
        "test_precision": precision_score(ytest, pred_test),
        "test_recall": recall_score(ytest, pred_test),
        "test_f1": f1_score(ytest, pred_test),
        "test_roc_auc": roc_auc_score(ytest, proba_test),
    }
    mlflow.log_metrics(metrics)

    print("Best parameters:", grid.best_params_)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
    print()
    print(classification_report(ytest, pred_test))

    # Save the best model where the app and the workflow expect it
    os.makedirs("/content/drive/My Drive/Python Course/tourism_project/deployment", exist_ok=True)
    model_path = "/content/drive/My Drive/Python Course/tourism_project/deployment/model.joblib"
    joblib.dump(best, model_path)
    mlflow.log_artifact(model_path)
    print("Model saved to", model_path)
