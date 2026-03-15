from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipelines.etl_pipeline import run_etl_pipeline
from pipelines.feature_pipeline import run_feature_pipeline
from pipelines.training_pipeline import run_training_pipeline
from database.load_to_postgres import run_postgres_load


# -----------------------------
# Default DAG arguments
# -----------------------------
default_args = {
    "owner": "shreyak",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# -----------------------------
# Define DAG
# -----------------------------
with DAG(
    dag_id="telco_churn_end_to_end_pipeline",
    default_args=default_args,
    description="End-to-end ETL, feature engineering, model training, and PostgreSQL load for telco churn prediction",
    schedule_interval=None,  # manual trigger
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["data-engineering", "ml-pipeline", "telco-churn"],
) as dag:

    # -----------------------------
    # ETL Task
    # -----------------------------
    etl_task = PythonOperator(
        task_id="run_etl_pipeline",
        python_callable=run_etl_pipeline,
    )

    # -----------------------------
    # Feature Engineering Task
    # -----------------------------
    feature_task = PythonOperator(
        task_id="run_feature_pipeline",
        python_callable=run_feature_pipeline,
    )

    # -----------------------------
    # Model Training Task
    # -----------------------------
    training_task = PythonOperator(
        task_id="run_training_pipeline",
        python_callable=run_training_pipeline,
    )

    # -----------------------------
    # PostgreSQL Load Task
    # -----------------------------
    postgres_load_task = PythonOperator(
        task_id="run_postgres_load",
        python_callable=run_postgres_load,
    )

    # -----------------------------
    # Task dependencies
    # -----------------------------
    etl_task >> feature_task >> training_task >> postgres_load_task