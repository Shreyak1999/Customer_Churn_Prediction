import os
import pickle

from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# -----------------------------
# Split data into train and test
# -----------------------------
def split_train_test(X, y, test_size=0.2, random_state=42):
    logger.info("Splitting data into train and test sets")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(
        f"Train/Test split completed: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )

    return X_train, X_test, y_train, y_test


# -----------------------------
# Train Random Forest model
# -----------------------------
def train_random_forest(X_train, y_train):
    logger.info("Training RandomForestClassifier")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    logger.info("Model training completed")
    return model


# -----------------------------
# Save trained model
# -----------------------------
def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    logger.info(f"Model saved to {model_path}")


# -----------------------------
# Save feature column names
# -----------------------------
def save_feature_columns(X, feature_path):
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)

    with open(feature_path, "wb") as file:
        pickle.dump(list(X.columns), file)

    logger.info(f"Feature columns saved to {feature_path}")


# -----------------------------
# Full training workflow
# -----------------------------
def train_model_pipeline(X, y, model_path, feature_path, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state
    )

    model = train_random_forest(X_train, y_train)

    save_model(model, model_path)
    save_feature_columns(X, feature_path)

    return model, X_train, X_test, y_train, y_test