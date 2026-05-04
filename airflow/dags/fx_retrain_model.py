"""Airflow DAG — monthly model retraining pipeline."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def check_new_data() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("check_new_data — stub")


def retrain_model() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("retrain_model — stub (wire to training/train.py)")


def evaluate_model() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("evaluate_model — stub")


try:
    from src.monitoring.airflow_callbacks import on_dag_failure, on_dag_success
except Exception:  # noqa: BLE001
    on_dag_success = on_dag_failure = None  # type: ignore[assignment]

default_args = {
    "owner": "curlit", "retries": 1,
    "on_success_callback": on_dag_success,
    "on_failure_callback": on_dag_failure,
}

with DAG(
    "fx_retrain_model",
    default_args=default_args,
    description="Monthly FinBERT retraining and metrics evaluation",
    schedule_interval="0 0 1 * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["curlit", "training"],
) as dag:

    t_check = PythonOperator(task_id="check_new_data", python_callable=check_new_data)
    t_train = PythonOperator(task_id="retrain_model", python_callable=retrain_model)
    t_eval = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    t_check >> t_train >> t_eval
