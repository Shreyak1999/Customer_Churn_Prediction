
import os
import pandas as pd
import yaml
from loguru import logger

from transformations.feature_engineering import engineer_features


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
# Load processed data
# -----------------------------
def load_processed_data(path):
    logger.info(f"Loading processed data from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Processed data shape: {df.shape}")
    return df


# -----------------------------
# Save feature store data
# -----------------------------
def save_feature_store(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Feature store data saved to {path}")


# -----------------------------
# Run feature pipeline
# -----------------------------
def run_feature_pipeline():
    config = load_config()

    processed_path = config["data"]["processed_data_path"]
    feature_store_path = config["data"]["feature_store_path"]
    log_path = config["logging"]["log_file"]

    setup_logging(log_path)

    logger.info("Starting feature engineering pipeline")

    # Load cleaned data
    df = load_processed_data(processed_path)

    # Engineer features
    df_features = engineer_features(df)

    # Save feature store
    save_feature_store(df_features, feature_store_path)

    logger.info("Feature engineering pipeline completed successfully")


if __name__ == "__main__":
    run_feature_pipeline()