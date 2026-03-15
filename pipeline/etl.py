import os
import pandas as pd
import yaml
from loguru import logger

from transformations.cleaning import clean_telco_data


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
# Load raw data
# -----------------------------
def load_raw_data(path):
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw data shape: {df.shape}")
    return df


# -----------------------------
# Save processed data
# -----------------------------
def save_processed_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Processed data saved to {path}")


# -----------------------------
# Run ETL pipeline
# -----------------------------
def run_etl():
    config = load_config()

    raw_path = config["data"]["raw_data_path"]
    processed_path = config["data"]["processed_data_path"]
    log_path = config["logging"]["log_file"]

    setup_logging(log_path)

    logger.info("Starting ETL pipeline")

    # Extract
    df = load_raw_data(raw_path)

    # Transform
    df_cleaned = clean_telco_data(df)

    # Load
    save_processed_data(df_cleaned, processed_path)

    logger.info("ETL pipeline completed successfully")


if __name__ == "__main__":
    run_etl()