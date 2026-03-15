from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.predict import predict_churn


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn probability",
    version="1.0.0"
)


# -----------------------------
# Request schema
# -----------------------------
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# -----------------------------
# Model artifact paths
# -----------------------------
MODEL_PATH = "models/churn_model.pkl"
FEATURE_COLUMNS_PATH = "models/model_features.pkl"


# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/")
def health_check():
    return {"message": "Telco Churn Prediction API is running"}


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        input_data = customer.model_dump()

        result = predict_churn(
            input_data=input_data,
            model_path=MODEL_PATH,
            feature_columns_path=FEATURE_COLUMNS_PATH
        )

        return {
            "status": "success",
            "input": input_data,
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))