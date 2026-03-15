import pickle
import pandas as pd
from loguru import logger

from transformations.feature_engineering import engineer_features


# -----------------------------
# Load model artifact
# -----------------------------
def load_pickle_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


# -----------------------------
# Prepare incoming data for prediction
# -----------------------------
def prepare_input_for_prediction(input_data: dict, feature_columns_path: str):
    logger.info("Preparing input data for prediction")

    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])

    # Apply same feature engineering
    df = engineer_features(df)

    # Separate out any target column if accidentally present
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # One-hot encode all categorical columns automatically
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Load training feature columns
    trained_feature_columns = load_pickle_file(feature_columns_path)

    # Align incoming features to training schema
    df_aligned = df_encoded.reindex(columns=trained_feature_columns, fill_value=0)

    logger.info(f"Prediction input prepared with shape: {df_aligned.shape}")
    return df_aligned


# -----------------------------
# Predict churn probability
# -----------------------------
def predict_churn(input_data: dict, model_path: str, feature_columns_path: str):
    logger.info("Starting churn prediction")

    model = load_pickle_file(model_path)
    X_input = prepare_input_for_prediction(input_data, feature_columns_path)

    prediction = model.predict(X_input)[0]
    prediction_probability = model.predict_proba(X_input)[0][1]

    result = {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(prediction_probability), 4)
    }

    logger.info(f"Prediction result: {result}")
    return result