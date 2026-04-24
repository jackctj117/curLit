"""
Airflow DAG — weekly CFTC COT ingestion and feature computation.
Runs Friday at 20:30 UTC, after the 3:30 PM ET release.
"""

import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql://{os.environ.get('POSTGRES_USER', 'fx')}:{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@{os.environ.get('POSTGRES_HOST', 'localhost')}:5432/{os.environ.get('POSTGRES_DB', 'fx')}",
)


def ingest_cot() -> None:
    from src.data.cftc import CFTCIngester
    ingester = CFTCIngester(DB_URL)
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    rows = ingester.run(start, end)
    logger.info("COT ingested: %d rows", rows)


def compute_cot_features() -> None:
    import pandas as pd
    from sqlalchemy import create_engine, text

    engine = create_engine(DB_URL)
    query = text("""
        SELECT report_date, symbol, category, longs, shorts
        FROM cot_positioning
        WHERE report_date >= (SELECT MAX(report_date) FROM cot_positioning) - INTERVAL '182 days'
        ORDER BY report_date
    """)
    df = pd.read_sql(query, engine)

    if df.empty:
        logger.info("No COT data for feature computation")
        return

    df["net_position"] = df["longs"] - df["shorts"]
    for sym in df["symbol"].unique():
        for cat in df["category"].unique():
            sub = df[(df["symbol"] == sym) & (df["category"] == cat)].copy()
            if len(sub) < 20:
                continue
            sub = sub.sort_values("report_date")
            roll_mean = sub["net_position"].rolling(156, min_periods=20).mean()
            roll_std = sub["net_position"].rolling(156, min_periods=20).std()
            sub["cot_zscore"] = (sub["net_position"] - roll_mean) / roll_std

            features = []
            for _, row in sub.iterrows():
                if pd.notna(row["cot_zscore"]):
                    features.append({
                        "ts": datetime.combine(row["report_date"].date() if hasattr(row["report_date"], "date") else row["report_date"],
                                                datetime.min.time()),
                        "symbol": f"cot_{sym}_{cat}",
                        "feature_name": "cot_zscore",
                        "value": float(row["cot_zscore"]),
                    })
            if features:
                pd.DataFrame(features).to_sql("features", engine, if_exists="append", index=False)
    logger.info("COT features computed for %d symbols", len(df["symbol"].unique()))


default_args = {
    "owner": "curlit",
    "retries": 3,
    "retry_delay": timedelta(minutes=30),
}

with DAG(
    "fx_weekly_cot",
    default_args=default_args,
    description="Weekly CFTC COT ingestion and feature computation",
    schedule_interval="30 20 * * 5",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["curlit", "cot"],
) as dag:

    t_ingest = PythonOperator(task_id="ingest_cot", python_callable=ingest_cot)
    t_features = PythonOperator(task_id="compute_cot_features", python_callable=compute_cot_features)

    t_ingest >> t_features
