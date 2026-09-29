import datetime as dt

import pandas as pd
import pytest
from pyspark.sql import functions as F

from annanas import aggregates as agg

T0 = dt.datetime(2022, 3, 1, 10, 0, 0)


@pytest.fixture
def events(spark):
    rows = [
        # канал, дата_время, класс_записи, тревожное, числовое_значение, состояние
        (1, T0, "состояние", False, None, "Норма"),
        (1, T0 + dt.timedelta(seconds=5), "состояние", True, None, "Неисправен"),
        (1, T0 + dt.timedelta(seconds=65), "состояние", False, None, "Норма"),
        (2, T0, "измерение", False, 0.01, None),
        (2, T0 + dt.timedelta(seconds=10), "измерение", False, 0.03, None),
        (2, T0 + dt.timedelta(minutes=20), "измерение", False, 0.02, None),
        (3, T0, "состояние", False, None, "Неопределен"),
        (4, T0 + dt.timedelta(seconds=30), "состояние", False, None, "Неопределен"),
    ]
    df = spark.createDataFrame(rows, "`ид_канала_данных` int, `дата_время` timestamp, `класс_записи` string, "
                                     "`тревожное` boolean, `числовое_значение` double, `состояние` string")
    return (df.withColumn("ид_события", F.monotonically_increasing_id())
              .withColumn("год", F.year("дата_время")).withColumn("месяц", F.month("дата_время")))


@pytest.fixture
def channels(spark):
    return spark.createDataFrame(
        [(1, "Состояние насоса", 30), (2, "Газовый датчик", 30), (3, "КД Дверь", 30), (4, "КД Люк", 30)],
        "`ид_канала_данных` int, `тип_датчика` string, `ид_объект` int")


def test_state_runs_duration(events):
    out = agg.state_runs(events)
    assert out.columns == ["ид_канала_данных", "дата_время", "ид_события", "состояние", "тревожное", "год",
                           "время_следующего", "следующее_состояние", "следующее_тревожное",
                           "секунд_до_следующего"]
    rows = out.filter("`ид_канала_данных` = 1").orderBy("дата_время").collect()
    alarm = rows[1]
    assert alarm["состояние"] == "Неисправен" and alarm["следующее_состояние"] == "Норма"
    assert alarm["секунд_до_следующего"] == 60 and alarm["следующее_тревожное"] is False
    assert rows[2]["следующее_состояние"] is None and rows[2]["секунд_до_следующего"] is None  # открытое


def test_meas_gaps(events):
    out = agg.meas_gaps(events)
    assert out.columns == ["день", "ид_канала_данных", "количество_замеров", "медианный_интервал_с",
                           "максимальный_интервал_с", "интервалов_дольше_10_мин"]
    row = out.collect()[0]
    assert row["количество_замеров"] == 3 and row["максимальный_интервал_с"] == 1190
    assert row["интервалов_дольше_10_мин"] == 1


def test_undefined_bursts(events, channels):
    out = agg.undefined_bursts(events, channels)
    assert out.columns == ["минута", "ид_объект", "каналов_в_неопределен"]
    row = out.collect()[0]
    assert row["ид_объект"] == 30 and row["каналов_в_неопределен"] == 2


def test_hourly_type(events, channels):
    out = agg.hourly_type(events, channels)
    assert out.columns == ["час", "тип_датчика", "класс_записи", "количество_событий",
                           "количество_тревожных", "число_активных_каналов"]
    rows = out.filter("`тип_датчика` = 'Состояние насоса'").collect()
    assert rows[0]["количество_событий"] == 3 and rows[0]["количество_тревожных"] == 1
    assert rows[0]["число_активных_каналов"] == 1 and rows[0]["класс_записи"] == "состояние"


def test_daily_channel(events):
    out = agg.daily_channel(events)
    assert out.columns == ["день", "ид_канала_данных", "класс_записи", "количество_событий",
                           "количество_тревожных", "среднее_значение", "минимум_значения", "максимум_значения"]
    row = out.filter("`ид_канала_данных` = 2").collect()[0]
    assert row["количество_событий"] == 3 and row["максимум_значения"] == 0.03
    assert row["минимум_значения"] == 0.01 and row["количество_тревожных"] == 0


def test_state_vocab(events, channels):
    out = agg.state_vocab(events, channels)
    assert out.columns == ["тип_датчика", "состояние", "тревожное", "год", "месяц",
                           "количество_событий", "число_каналов"]
    rows = {(r["состояние"], r["тревожное"]): r for r in out.collect()}
    assert rows[("Норма", False)]["количество_событий"] == 2
    assert rows[("Неисправен", True)]["число_каналов"] == 1
    assert rows[("Норма", False)]["месяц"] == 3


def test_build_or_load_caches(spark, tmp_path):
    calls = []
    def builder():
        calls.append(1)
        return spark.range(3)
    agg.build_or_load("x", builder, tmp_path, spark)
    df = agg.build_or_load("x", builder, tmp_path, spark)
    assert len(calls) == 1 and df.count() == 3


def test_build_or_load_ignores_stale_tmp(spark, tmp_path):
    (tmp_path / "x.parquet.tmp").mkdir()          # след упавшей записи
    (tmp_path / "x.parquet.tmp" / "junk").write_text("broken")
    assert agg.build_or_load("x", lambda: spark.range(2), tmp_path, spark).count() == 2
    assert not (tmp_path / "x.parquet.tmp").exists()


def test_read_vitrine_returns_moscow_wall_clock(spark, tmp_path):
    """Spark пишет timestamp как UTC-момент; pandas без перевода отдал бы время на 3 ч раньше."""
    ts = dt.datetime(2021, 4, 22, 7, 0)                      # московское время (сессия в Europe/Moscow)
    df = spark.createDataFrame([(ts, dt.date(2021, 4, 22), 1)], "hour timestamp, day date, n int")
    path = tmp_path / "v.parquet"
    df.write.parquet(str(path))
    out = agg.read_vitrine(path)
    assert out["hour"].iloc[0] == pd.Timestamp("2021-04-22 07:00")
    assert out["hour"].dt.tz is None
    assert out["n"].iloc[0] == 1
    assert pd.Timestamp(out["day"].iloc[0]) == pd.Timestamp("2021-04-22")   # дата не сдвигается
