"""Витрины для EDA и признаков. Вход: events (Parquet из 01_preprocessing) и channels.
Имена столбцов витрин – на русском (исходные сохраняются, новые – snake_case)."""
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def _alarm():
    """Столбец-счетчик тревог. Функция, а не константа: F.col требует активного SparkContext."""
    return F.col("тревожное").cast("int")


def _with_channel(events: DataFrame, channels: DataFrame, col: str) -> DataFrame:
    """Добавляет к событиям атрибут канала из справочника (каналы вне справочника получают null)."""
    return events.join(F.broadcast(channels.select("ид_канала_данных", col)), "ид_канала_данных", "left")


def hourly_type(events: DataFrame, channels: DataFrame) -> DataFrame:
    """Поток по часам × тип датчика × класс записи."""
    return (
        _with_channel(events, channels, "тип_датчика")
        .groupBy(F.date_trunc("hour", "дата_время").alias("час"), "тип_датчика", "класс_записи")
        .agg(F.count("*").alias("количество_событий"), F.sum(_alarm()).alias("количество_тревожных"),
             F.countDistinct("ид_канала_данных").alias("число_активных_каналов"))
    )


def daily_channel(events: DataFrame) -> DataFrame:
    """Активность канала по дням: число записей, тревог, статистика значений."""
    return (
        events.groupBy(F.to_date("дата_время").alias("день"), "ид_канала_данных", "класс_записи")
        .agg(F.count("*").alias("количество_событий"), F.sum(_alarm()).alias("количество_тревожных"),
             F.mean("числовое_значение").alias("среднее_значение"),
             F.min("числовое_значение").alias("минимум_значения"),
             F.max("числовое_значение").alias("максимум_значения"))
    )


def state_vocab(events: DataFrame, channels: DataFrame) -> DataFrame:
    """Словарь состояний по типам датчиков, годам и месяцам."""
    return (
        _with_channel(events.filter(F.col("класс_записи") == "состояние"), channels, "тип_датчика")
        .groupBy("тип_датчика", "состояние", "тревожное", "год", "месяц")
        .agg(F.count("*").alias("количество_событий"),
             F.countDistinct("ид_канала_данных").alias("число_каналов"))
    )


def state_runs(events: DataFrame) -> DataFrame:
    """Каждая смена состояния канала и время до следующей. Сортировка обязательна: CSV не упорядочены.
    `ид_события` – тай-брейкер для записей одного канала в одну секунду (внутри класса «состояние»
    последовательность ID своя, поэтому сравнение корректно)."""
    w = Window.partitionBy("ид_канала_данных").orderBy("дата_время", "ид_события")
    return (
        events.filter(F.col("класс_записи") == "состояние")
        .select("ид_канала_данных", "дата_время", "ид_события", "состояние", "тревожное", "год")
        .withColumn("время_следующего", F.lead("дата_время").over(w))
        .withColumn("следующее_состояние", F.lead("состояние").over(w))
        .withColumn("следующее_тревожное", F.lead("тревожное").over(w))
        .withColumn("секунд_до_следующего",
                    F.col("время_следующего").cast("long") - F.col("дата_время").cast("long"))
    )


def meas_gaps(events: DataFrame, gap_threshold_s: int = 600) -> DataFrame:
    """Интервалы между замерами канала. Первый интервал дня считается от последнего замера предыдущего дня:
    молчание через полночь – тоже молчание. Окно режется по году: газовый канал дает миллионы замеров в год,
    окно за все годы не помещается в одну задачу; цена – теряется один интервал на канал на стыке лет."""
    w = Window.partitionBy("ид_канала_данных", "год").orderBy("дата_время")
    gaps = (
        events.filter(F.col("класс_записи") == "измерение")
        .select("ид_канала_данных", "год", "дата_время")
        .withColumn("интервал_с",
                    F.col("дата_время").cast("long") - F.lag("дата_время").over(w).cast("long"))
    )
    return (
        gaps.groupBy(F.to_date("дата_время").alias("день"), "ид_канала_данных")
        .agg(F.count("*").alias("количество_замеров"),
             F.percentile_approx("интервал_с", 0.5).alias("медианный_интервал_с"),
             F.max("интервал_с").alias("максимальный_интервал_с"),
             F.sum((F.col("интервал_с") > gap_threshold_s).cast("int")).alias("интервалов_дольше_10_мин"))
    )


def undefined_bursts(events: DataFrame, channels: DataFrame) -> DataFrame:
    """Число каналов объекта в «Неопределен» за минуту: пачки – перезапуск контроллера или потеря связи."""
    return (
        _with_channel(events.filter(F.col("состояние") == "Неопределен"), channels, "ид_объект")
        .groupBy(F.date_trunc("minute", "дата_время").alias("минута"), "ид_объект")
        .agg(F.countDistinct("ид_канала_данных").alias("каналов_в_неопределен"))
    )


def undefined_daily(events: DataFrame) -> DataFrame:
    """Число записей «Неопределен» канала за сутки (витрина undefined_daily, notebooks/02_eda)."""
    return (events.filter(F.col("состояние") == "Неопределен")
            .groupBy(F.to_date("дата_время").alias("день"), "ид_канала_данных")
            .agg(F.count("*").alias("количество_неопределен")))


def object_day_states(events: DataFrame, channels: DataFrame) -> DataFrame:
    """Смены состояний каналов справочника: объект × день × тип датчика × состояние (витрина
    features/object_day_states, notebooks/04_features)."""
    return (events.filter(F.col("класс_записи") == "состояние")
            .join(F.broadcast(channels.select("ид_канала_данных", "ид_объект", "тип_датчика")), "ид_канала_данных")
            .groupBy("ид_объект", F.to_date("дата_время").alias("день"), "тип_датчика", "состояние")
            .agg(F.count("*").alias("количество_записей"),
                 F.sum(F.col("тревожное").cast("int")).alias("количество_тревожных"),
                 F.countDistinct("ид_канала_данных").alias("каналов")))


def read_vitrine(path) -> pd.DataFrame:
    """Витрина через pandas с московским временем. Spark хранит timestamp в Parquet как UTC-момент:
    `spark.read` и `toPandas()` в сессии Europe/Moscow отдают Москву, а pandas/pyarrow – наивный UTC (на 3 ч раньше).
    Здесь каждая timestamp-колонка переводится в наивное московское время; даты (date) не трогаются."""
    table = ds.dataset(str(path), format="parquet").to_table()
    ts_cols = [f.name for f in table.schema if pa.types.is_timestamp(f.type)]
    df = table.to_pandas()
    for col in ts_cols:
        s = df[col]
        s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
        df[col] = s.dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    return df


def build_or_load(name: str, builder, out_dir: Path, spark) -> DataFrame:
    """Кэш витрины: строим один раз, дальше читаем Parquet.
    Запись идет в <name>.parquet.tmp и переименовывается по окончании, поэтому
    после падения недописанная витрина не подхватится как готовая."""
    path = Path(out_dir) / f"{name}.parquet"
    if not path.exists():
        tmp = path.with_name(path.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        builder().write.mode("overwrite").parquet(str(tmp))
        tmp.rename(path)
    return spark.read.parquet(str(path))
