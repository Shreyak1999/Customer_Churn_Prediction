import pandas as pd
from loguru import logger


# -----------------------------
# Remove duplicates
# -----------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    initial_shape = df.shape
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
    return df


# -----------------------------
# Drop unnecessary columns
# -----------------------------
def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped column: customerID")
    return df


# -----------------------------
# Strip whitespace from object columns
# -----------------------------
def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
    logger.info("Stripped whitespace from object columns")
    return df


# -----------------------------
# Fix TotalCharges datatype
# -----------------------------
def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Business-aware handling
        df.loc[(df["tenure"] == 0) & (df["TotalCharges"].isna()), "TotalCharges"] = 0

        # Fallback median imputation
        median_value = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_value)

        logger.info("Cleaned TotalCharges column")
    return df


# -----------------------------
# Encode target variable
# -----------------------------
def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
        logger.info("Encoded target column: Churn")
    return df


# -----------------------------
# Full cleaning pipeline
# -----------------------------
def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning pipeline")

    df = remove_duplicates(df)
    df = drop_unnecessary_columns(df)
    df = strip_whitespace(df)
    df = clean_total_charges(df)
    df = encode_target(df)

    logger.info(f"Data cleaning completed. Final shape: {df.shape}")
    return df