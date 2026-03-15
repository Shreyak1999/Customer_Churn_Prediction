import os
import pandas as pd
import yaml
from loguru import logger

from transformations.encoding import prepare_model_data
from models.train_model import train_model_pipeline
from models.evaluate_model import evaluate_classification_model, save_metrics


# -----------------------------
# Load configuration
# -----------------------------
def load_config():
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


# -----------------------------
# Setup logging
# -----------------------------
def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.add(log_path, rotation="1 MB")


# -----------------------------
# Load feature store data
# -----------------------------
def load_feature_store(path):
    logger.info(f"Loading feature store data from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Feature store shape: {df.shape}")
    return df


# -----------------------------
# Run training pipeline
# -----------------------------
def run_training_pipeline():
    config = load_config()

    feature_store_path = config["data"]["feature_store_path"]
    log_path = config["logging"]["log_file"]

    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]
    model_path = config["model"]["model_output_path"]
    feature_path = config["model"]["feature_columns_path"]
    metrics_path = config["model"]["metrics_output_path"]

    setup_logging(log_path)

    logger.info("Starting training pipeline")

    # Load feature store data
    df = load_feature_store(feature_store_path)

    # Prepare model-ready data
    X, y = prepare_model_data(df, config)

    # Train model
    model, X_train, X_test, y_train, y_test = train_model_pipeline(
        X=X,
        y=y,
        model_path=model_path,
        feature_path=feature_path,
        test_size=test_size,
        random_state=random_state
    )

    # Evaluate model
    metrics = evaluate_classification_model(model, X_test, y_test)

    # Save metrics
    save_metrics(metrics, metrics_path)

    logger.info("Training pipeline completed successfully")
    logger.info(f"Final model metrics: {metrics}")


if __name__ == "__main__":
    run_training_pipeline()