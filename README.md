# Telco Customer Churn — AI-Enabled Data Engineering Pipeline

An end-to-end **AI-enabled data engineering project** built on the UCI-style Telco Customer Churn dataset.  
This project demonstrates how to design a modular data pipeline that ingests raw data, performs cleaning and feature engineering, stores analytics-ready and ML-ready layers, trains a churn prediction model, exposes real-time inference via API, and provides an interactive dashboard for business users.

---

## Project Objective

Telecom companies lose revenue when customers leave (churn).  
This project simulates a real-world analytics + AI workflow by building a pipeline that:

- Ingests and cleans raw customer data
- Creates engineered features for churn analysis
- Stores curated datasets for analytics and machine learning
- Trains a churn prediction model
- Exposes predictions through a FastAPI endpoint
- Provides a Streamlit dashboard for interactive risk scoring
- Loads processed data into PostgreSQL for downstream consumption
- Includes an Apache Airflow DAG for orchestration design

---

## Tech Stack

- **Language:** Python
- **Data Processing:** pandas, NumPy
- **Feature Store Format:** Parquet (PyArrow)
- **Machine Learning:** scikit-learn (Random Forest)
- **Database:** PostgreSQL
- **ORM / DB Connector:** SQLAlchemy, psycopg2
- **API:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **Orchestration Design:** Apache Airflow
- **Config Management:** YAML
- **Logging:** Loguru

---

## Project Architecture

```text
Raw CSV
   │
   ▼
ETL Pipeline
   │
   ▼
Processed Cleaned Data (CSV)
   │
   ├──► PostgreSQL (analytics-ready layer)
   │
   ▼
Feature Engineering Pipeline
   │
   ▼
Feature Store (Parquet)
   │
   ├──► PostgreSQL (ML-ready layer)
   │
   ▼
Training Pipeline
   │
   ├──► Trained Model (.pkl)
   ├──► Feature Schema (.pkl)
   └──► Evaluation Metrics (.json)
   │
   ▼
Prediction Utility
   │
   ├──► FastAPI Inference Service
   └──► Streamlit Dashboard
```

---

## Project Structure

```text
telco-churn-ai-data-pipeline/
│
├── api/
│   └── prediction_api.py
│
├── config/
│   ├── config.yaml
│   ├── db_config.yaml
│   └── db_config_example.yaml
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── processed/
│   │   └── cleaned_telco_churn.csv
│   └── feature_store/
│       └── churn_features.parquet
│
├── dashboard/
│   └── streamlit_app.py
│
├── database/
│   └── load_to_postgres.py
│
├── logs/
│   └── pipeline.log
│
├── models/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── churn_model.pkl
│   ├── model_features.pkl
│   └── model_metrics.json
│
├── orchestration/
│   └── airflow_dag.py
│
├── pipelines/
│   ├── etl_pipeline.py
│   ├── feature_pipeline.py
│   └── training_pipeline.py
│
├── transformations/
│   ├── cleaning.py
│   ├── feature_engineering.py
│   └── encoding.py
│
├── requirements.txt
├── requirements-airflow.txt
└── README.md
```

---

## Dataset

This project uses the **Telco Customer Churn dataset** (commonly used in churn modeling projects and widely available in public repositories / UCI-style distributions).

Typical fields include:

- Customer demographics
- Contract type
- Internet services
- Payment behavior
- Monthly / total charges
- Churn label

> Place the raw dataset file in:
>
> `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## Configuration Files

### `config/config.yaml`
Contains:
- data paths
- target column
- model settings
- logging settings

### `config/db_config.yaml`
Contains:
- PostgreSQL connection details

### `config/db_config_example.yaml`
Safe example file for GitHub (recommended instead of sharing real credentials).

---

## Setup Instructions

### 1. Create virtual environment (recommended)

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Optional (Airflow):**
```bash
pip install -r requirements-airflow.txt
```

**Note:** Airflow is best run in **Linux / WSL / Docker**.  
For Windows users, the DAG is included as orchestration design and can be tested in WSL or Docker.

---

## Core Pipeline Execution

Run the project in this order:

### 1. ETL Pipeline
```bash
python pipelines/etl_pipeline.py
```

**Output:**
- `data/processed/cleaned_telco_churn.csv`

---

### 2. Feature Engineering Pipeline
```bash
python pipelines/feature_pipeline.py
```

**Output:**
- `data/feature_store/churn_features.parquet`

---

### 3. Training Pipeline
```bash
python pipelines/training_pipeline.py
```

**Outputs:**
- `models/churn_model.pkl`
- `models/model_features.pkl`
- `models/model_metrics.json`

---

### 4. Load Curated Data into PostgreSQL
```bash
python database/load_to_postgres.py
```

**Loads tables:**
- `telco_processed_customers`
- `telco_churn_features`

---

## API Inference Service (FastAPI)

Run:

```bash
uvicorn api.prediction_api:app --reload
```

Open interactive docs:

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

- `GET /` → health check
- `POST /predict` → churn prediction

### Example Request Body

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 79.85,
  "TotalCharges": 956.4
}
```

### Example Response

```json
{
  "status": "success",
  "input": {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 956.4
  },
  "prediction": {
    "churn_prediction": 1,
    "churn_probability": 0.8421
  }
}
```

---

## Interactive Dashboard (Streamlit)

Run:

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard allows users to:

- Enter customer attributes
- Score churn risk in real time
- View churn probability
- See simple business interpretation (low / medium / high risk)

---

## Airflow Orchestration

This project includes:

- `orchestration/airflow_dag.py`

The DAG orchestrates:

```text
ETL → Feature Engineering → Model Training → PostgreSQL Load
```

**Note:**  
Airflow is included as **orchestration design and local testable DAG support**.  
For best compatibility, run Airflow in:
- Linux
- WSL
- Docker

---

## Output Artifacts

### Processed Data
- `data/processed/cleaned_telco_churn.csv`

### Feature Store
- `data/feature_store/churn_features.parquet`

### Model Artifacts
- `models/churn_model.pkl`
- `models/model_features.pkl`

### Metrics
- `models/model_metrics.json`

### Database Tables
- `telco_processed_customers`
- `telco_churn_features`

---

## Key Engineering Highlights

- Built a **modular ETL pipeline** for raw-to-curated data transformation
- Created both **analytics-ready** and **ML-ready** data layers
- Stored ML features in **Parquet feature store** format
- Implemented **schema-aligned prediction logic** to ensure training/inference consistency
- Exposed model inference through **FastAPI**
- Built an interactive **Streamlit dashboard**
- Loaded curated layers into **PostgreSQL**
- Designed an **Apache Airflow DAG** for orchestration

---

## Resume-Ready Project Summary

**Short version:**

> Built an AI-enabled data engineering pipeline for telecom customer churn analysis using Python, PostgreSQL, FastAPI, and Streamlit, including ETL workflows, feature-store creation, model training, real-time inference, and orchestration design with Airflow.

**Detailed version:**

> Developed an end-to-end AI-enabled data engineering pipeline on a public telecom churn dataset, including raw data ingestion, data cleaning, feature engineering, Parquet-based feature-store generation, PostgreSQL loading of curated analytics and ML layers, Random Forest churn modeling, FastAPI inference service, Streamlit dashboard, and Apache Airflow DAG orchestration.

---

## Future Improvements

Potential enhancements for productionization:

- Add Docker / Docker Compose for full local deployment
- Use environment variables (`.env`) for secrets instead of YAML credentials
- Add model versioning (MLflow / DVC)
- Add automated tests (pytest)
- Add data validation (Great Expectations / Pandera)
- Add CI/CD pipeline (GitHub Actions)
- Add containerized Airflow stack
- Deploy API + dashboard to cloud

---

## Notes

- Do **not** commit real database credentials to GitHub.
- Use `config/db_config_example.yaml` as a safe template.
- If using Windows, Airflow is best tested via **WSL** or **Docker**.
- The project is intentionally designed to balance:
  - **data engineering fundamentals**
  - **ML/AI capability**
  - **portfolio practicality for entry-level roles**

---

## License

This project is for learning and portfolio demonstration purposes.