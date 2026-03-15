import os
import pandas as pd
import yaml
from sqlalchemy import create_engine
from loguru import logger


# -----------------------------
# Load main pipeline configuration
# -----------------------------
def load_main_config():
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


# -----------------------------
# Load database configuration
# -----------------------------
def load_db_config():
    with open("config/db_config.yaml", "r") as file:
        db_config = yaml.safe_load(file)
    return db_config


# -----------------------------
# Setup logging
# -----------------------------
def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.add(log_path, rotation="1 MB")


# -----------------------------
# Create database engine
# -----------------------------
def get_postgres_engine(db_config):
    user = db_config["database"]["user"]
    password = db_config["database"]["password"]
    host = db_config["database"]["host"]
    port = db_config["database"]["port"]
    database = db_config["database"]["database"]

    connection_string = (
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    )

    engine = create_engine(connection_string)
    return engine


# -----------------------------
# Load DataFrame to PostgreSQL
# -----------------------------
def load_dataframe_to_postgres(df, table_name, engine, if_exists="replace"):
    logger.info(f"Loading data into PostgreSQL table: {table_name}")
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    logger.info(f"Loaded {len(df)} rows into table '{table_name}'")


# -----------------------------
# Main load function
# -----------------------------
def run_postgres_load():
    config = load_main_config()
    db_config = load_db_config()

    log_path = config["logging"]["log_file"]
    setup_logging(log_path)

    logger.info("Starting PostgreSQL load process")

    # Load processed and feature store datasets
    processed_path = config["data"]["processed_data_path"]
    feature_store_path = config["data"]["feature_store_path"]

    processed_df = pd.read_csv(processed_path)
    feature_df = pd.read_parquet(feature_store_path)

    logger.info(f"Processed data shape: {processed_df.shape}")
    logger.info(f"Feature store data shape: {feature_df.shape}")

    # Get database engine
    engine = get_postgres_engine(db_config)

    # Load to PostgreSQL
    load_dataframe_to_postgres(processed_df, "telco_processed_customers", engine)
    load_dataframe_to_postgres(feature_df, "telco_churn_features", engine)

    logger.info("PostgreSQL load completed successfully")


if __name__ == "__main__":
    run_postgres_load()