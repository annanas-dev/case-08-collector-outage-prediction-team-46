from pyspark.sql import SparkSession

from annanas.paths import SPARK_TMP_DIR


def get_spark(app_name: str = "annanas", cores: int = 8, memory: str = "8g") -> SparkSession:
    """Локальная Spark-сессия с московским временем. Память драйвера не выше 8g (лимит Docker)."""
    SPARK_TMP_DIR.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", memory)
        .config("spark.local.dir", SPARK_TMP_DIR.as_posix())
        .config("spark.sql.session.timeZone", "Europe/Moscow")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
