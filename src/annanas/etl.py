"""Чтение и очистка журнала событий СМВУ. Особенности данных: docs/data.md."""
import re

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

RAW_COLUMNS = ["ид_события", "ид_канала_данных", "дата", "время", "тревожное", "значение_датчика"]
RAW_SCHEMA = StructType([StructField(c, StringType()) for c in RAW_COLUMNS])

# Значение-дата у «Состояние охраны» (01.01.1970 03:00:00) идет в последовательности ID измерений
DATETIME_VALUE_RE = r"^\d{2}\.\d{2}\.\d{4}"

# Столбцы очищенного хранилища событий (Parquet, партиции год/месяц)
EVENT_COLUMNS = ["ид_события", "ид_канала_данных", "дата_время", "тревожное", "значение_датчика",
                 "класс_записи", "числовое_значение", "состояние"]

CHANNEL_COLUMNS = ["ид_канала_данных", "тип_инж_системы", "тип_датчика", "название_датчика", "ид_объект",
                   "иерархия_уровень", "вид_объекта", "диспетчерское_название_объекта", "родитель",
                   "название_родителя", "пикет"]

PK_RE = re.compile(r"ПК\s?(\d+(?:[.,]\d+)?)", re.IGNORECASE)


def read_raw(spark: SparkSession, path) -> DataFrame:
    """Все колонки строками: типы приводим сами, чтобы не потерять «t»/«true» и заголовки внутри файла."""
    return spark.read.option("header", True).schema(RAW_SCHEMA).csv(str(path))


def drop_header_rows(df: DataFrame) -> DataFrame:
    """В файле 2025 года строка заголовка встречается посередине."""
    return df.filter(F.col("дата") != "дата")


def normalize(df: DataFrame) -> DataFrame:
    """Типизация исходных столбцов и разбор значения.

    `ид_события`, `ид_канала_данных`, `тревожное` сохраняют исходные имена, но получают типы
    (long, int, bool); `дата` + `время` -> `дата_время`. `значение_датчика` остается строкой как есть,
    из него выводятся `класс_записи` ('измерение' / 'состояние'), `числовое_значение` и `состояние`.
    """
    value = F.col("значение_датчика")
    value_num = F.expr("try_cast(`значение_датчика` as double)")
    is_measurement = value_num.isNotNull() | value.rlike(DATETIME_VALUE_RE)
    return (
        df
        # try_*: в Spark 4 включен ANSI, обычный cast падает на битой строке вместо null.
        # withColumn с тем же именем заменяет строковый столбец типизированным.
        .withColumn("ид_события", F.expr("try_cast(`ид_события` as long)"))
        .withColumn("ид_канала_данных", F.expr("try_cast(`ид_канала_данных` as int)"))
        .withColumn("дата_время",
                    F.try_to_timestamp(F.concat_ws(" ", "дата", "время"), F.lit("yyyy-MM-dd HH:mm:ss")))
        .withColumn("тревожное", F.lower(F.trim("тревожное")).isin("t", "true", "1"))
        .withColumn("класс_записи", F.when(is_measurement, "измерение").otherwise("состояние"))
        .withColumn("числовое_значение", value_num)
        .withColumn("состояние", F.when(~is_measurement, value))
        .select(*EVENT_COLUMNS)
    )


def extract_pk(name: str) -> float | None:
    """Номер пикета из названия канала – единственная доступная локация (docs/data.md)."""
    m = PK_RE.search(name or "")
    return float(m.group(1).replace(",", ".")) if m else None


def build_channels(channels: pd.DataFrame, objects: pd.DataFrame) -> pd.DataFrame:
    """Справочник каналов + объект + название родителя + пикет. Тег удаляем (разрешено организаторами)."""
    parents = objects[["ид_объект", "диспетчерское_название_объекта"]].rename(columns={
        "ид_объект": "родитель", "диспетчерское_название_объекта": "название_родителя"})
    out = (
        channels.drop(columns=["тег_инженерной_системы"])
        .merge(objects, on="ид_объект", how="left")
        .merge(parents, on="родитель", how="left")
    )
    out["пикет"] = out["название_датчика"].map(extract_pk)
    return out[CHANNEL_COLUMNS]


def clean_journal(spark: SparkSession, path, known_channel_ids: set[int]) -> tuple[DataFrame, dict]:
    """Очистка годового журнала и отчет о каждом шаге.

    1. Один проход по CSV: число строк и строк-заголовков внутри файла.
    2. Удаление заголовков и полных дублей строк (одна дедупликация, результат кешируется).
       По `ид_события` не дедуплицируем: он не уникален, у измерений и состояний свои
       последовательности ID.
    3. Нормализация, отброс строк без `дата_время`, партиции `год` / `месяц`.

    Отчет: строк_в_файле = строк_заголовка + полных_дублей + битых_дат + строк_после_очистки.
    Возвращенный DataFrame опирается на кеш дедупликации: после записи вызвать
    `spark.catalog.clearCache()`.
    """
    raw = read_raw(spark, path)
    raw_stats = raw.agg(
        F.count("*").alias("строк_в_файле"),
        F.count(F.when(F.col("дата") == "дата", 1)).alias("строк_заголовка"),
    ).first().asDict()

    dedup = drop_header_rows(raw).dropDuplicates(RAW_COLUMNS).cache()
    full_duplicates = raw_stats["строк_в_файле"] - raw_stats["строк_заголовка"] - dedup.count()

    norm = normalize(dedup)
    ok = F.col("дата_время").isNotNull()
    norm_stats = norm.agg(
        F.count(F.when(~ok, 1)).alias("битых_дат"),
        F.count(F.when(ok, 1)).alias("строк_после_очистки"),
        F.count(F.when(ok & F.col("тревожное"), 1)).alias("тревожных"),
        F.count(F.when(ok & ~F.col("ид_канала_данных").isin(list(known_channel_ids)), 1))
         .alias("строк_каналов_вне_справочника"),
    ).first().asDict()

    clean = (
        norm.filter(ok)
        .withColumn("год", F.year("дата_время"))
        .withColumn("месяц", F.month("дата_время"))
    )
    report = {**raw_stats, "полных_дублей": full_duplicates, **norm_stats}
    return clean, report


def write_events(df: DataFrame, out_dir) -> None:
    """Дописывает события в партиции год/месяц.

    Append, а не overwrite: строки файла одного года могут попасть в партицию соседнего,
    и overwrite затер бы ранее записанное. Перед полной пересборкой каталог удаляется целиком.
    """
    (df.repartition("год", "месяц")
       .write.mode("append")
       .partitionBy("год", "месяц")
       .parquet(str(out_dir)))
