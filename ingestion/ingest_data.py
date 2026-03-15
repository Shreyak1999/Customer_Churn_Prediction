import os
import pandas as pd
import yaml
from loguru import logger


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
# Load dataset
# -----------------------------
def load_dataset(path):
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df


# -----------------------------
# Basic validation
# -----------------------------
def validate_dataset(df):
    logger.info("Running basic data validation")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.duplicated().sum() > 0:
        logger.warning("Dataset contains duplicate rows")

    logger.info("Validation completed")


# -----------------------------
# Save raw dataset
# -----------------------------
def save_raw_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Raw dataset saved to {path}")


# -----------------------------
# Main ingestion pipeline
# -----------------------------
def run_ingestion():
    config = load_config()

    raw_path = config["data"]["raw_data_path"]
    log_path = config["logging"]["log_file"]

    setup_logging(log_path)

    df = load_dataset(raw_path)
    validate_dataset(df)

    save_raw_data(df, raw_path)

    logger.info("Data ingestion pipeline completed")


if __name__ == "__main__":
    run_ingestion()