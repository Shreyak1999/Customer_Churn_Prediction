import pandas as pd
from loguru import logger


# -----------------------------
# Separate features and target
# -----------------------------
def split_features_target(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info(f"Separated features and target: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


# -----------------------------
# One-hot encode categorical columns
# -----------------------------
def encode_categorical_features(X: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    available_categorical = [col for col in categorical_columns if col in X.columns]

    X_encoded = pd.get_dummies(
        X,
        columns=available_categorical,
        drop_first=True
    )

    logger.info(f"Applied one-hot encoding to columns: {available_categorical}")
    logger.info(f"Encoded feature shape: {X_encoded.shape}")
    return X_encoded


# -----------------------------
# Ensure boolean columns are int
# -----------------------------
def convert_boolean_columns(X: pd.DataFrame) -> pd.DataFrame:
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
        logger.info(f"Converted boolean columns to int: {list(bool_cols)}")
    return X


# -----------------------------
# Full encoding pipeline
# -----------------------------
def prepare_model_data(df: pd.DataFrame, config: dict):
    logger.info("Starting model data preparation pipeline")

    target_column = config["features"]["target_column"]
    categorical_columns = config["features"]["categorical_columns"] + ["tenure_bucket"]

    X, y = split_features_target(df, target_column)
    X = encode_categorical_features(X, categorical_columns)
    X = convert_boolean_columns(X)

    logger.info("Model data preparation completed")
    return X, y