import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", PROJECT_ROOT / "datasets"))
SPARK_TMP_DIR = Path(os.environ.get("SPARK_TMP_DIR", PROJECT_ROOT / ".spark-tmp"))
PROCESSED_DIR = DATASETS_DIR / "processed"
EVENTS_DIR = PROCESSED_DIR / "events"
EDA_DIR = PROCESSED_DIR / "eda"
