import pandas as pd
from loguru import logger


# -----------------------------
# Create tenure buckets
# -----------------------------
def create_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "tenure" in df.columns:
        bins = [-1, 12, 24, 48, 72]
        labels = ["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
        df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels)
        logger.info("Created feature: tenure_bucket")
    return df


# -----------------------------
# Count number of subscribed services
# -----------------------------
def create_service_count(df: pd.DataFrame) -> pd.DataFrame:
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    available_cols = [col for col in service_columns if col in df.columns]

    if available_cols:
        df["service_count"] = df[available_cols].apply(
            lambda row: sum(
                value in ["Yes", "Fiber optic", "DSL"]
                for value in row
            ),
            axis=1
        )
        logger.info("Created feature: service_count")
    return df


# -----------------------------
# Monthly to total charge ratio
# -----------------------------
def create_charge_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if {"MonthlyCharges", "TotalCharges", "tenure"}.issubset(df.columns):
        # Avoid division by zero for new customers
        df["avg_monthly_charge_estimate"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
        logger.info("Created feature: avg_monthly_charge_estimate")
    return df


# -----------------------------
# Contract risk flag
# -----------------------------
def create_contract_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    if "Contract" in df.columns:
        df["is_month_to_month"] = df["Contract"].apply(lambda x: 1 if x == "Month-to-month" else 0)
        logger.info("Created feature: is_month_to_month")
    return df


# -----------------------------
# Full feature engineering pipeline
# -----------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering pipeline")

    df = create_tenure_bucket(df)
    df = create_service_count(df)
    df = create_charge_ratio(df)
    df = create_contract_risk_flag(df)

    logger.info(f"Feature engineering completed. Final shape: {df.shape}")
    return df