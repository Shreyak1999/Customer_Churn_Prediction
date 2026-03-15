import os
import json

from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# -----------------------------
# Evaluate classification model
# -----------------------------
def evaluate_classification_model(model, X_test, y_test):
    logger.info("Evaluating classification model")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_pred_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


# -----------------------------
# Save evaluation metrics
# -----------------------------
def save_metrics(metrics, metrics_path):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    logger.info(f"Metrics saved to {metrics_path}")