"""Дозагрузка журнала: CSV формата ext-journal → очищенные события → пересчет производных данных.

1. `etl.clean_journal` (те же правила, что для годовых журналов в 01_preprocessing).
2. Строки, которые уже есть в `events/` (тот же `ид_события, ид_канала_данных, дата_время, тревожное,
   значение_датчика`), отбрасываются – повторная загрузка того же файла ничего не меняет.
3. Новые строки дописываются в `events/` (партиции год/месяц).
4. Производные данные пересчитываются по событиям заново, без инкрементальной логики:
   - суточные витрины (daily_channel, undefined_daily, undefined_bursts, hourly_type, meas_gaps,
     object_day_states) и дни тестового газа / демонтажа – за сутки новых строк плюс OVERLAP_DAYS суток до них;
   - эпизоды тревог и потери связи – за затронутые годы целиком (как в notebooks/03_target: год с запасом сутки).
   Окна ППР АКМ Store строит сам из дней газа и графика.

Каждая витрина заменяется атомарно: запись в `<имя>.tmp`, затем переименование; если процесс упал между удалением
старой витрины и переименованием, следующая замена берет готовый `.tmp`. Диапазон пересчета записывается в
`service/recompute_pending.json` до записи событий и удаляется после пересчета: если сервис упал посередине,
следующая загрузка (хоть того же файла, где новых строк уже нет) доводит пересчет до конца."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from annanas import aggregates as agg
from annanas import episodes as ep
from annanas import etl
from annanas import maintenance as mt
from annanas.inference import features as fi
from annanas.paths import PROCESSED_DIR

OVERLAP_DAYS = 2           # пересчет суточных витрин с запасом: интервалы замеров и эпизоды через полночь
MEAS_LOOKBACK_DAYS = 31    # интервал до первого замера суток ищется не дальше 31 суток назад
EVENT_KEY = ["ид_события", "ид_канала_данных", "дата_время", "тревожное", "значение_датчика"]

# Список тревожных пар (notebooks/03_target, раздел 1.4): доля флага ночью (0-6 ч) до периода смены разметки
PAIRS_STABLE_UNTIL = pd.Timestamp("2025-10-01")
PAIRS_NIGHT_END, PAIRS_MIN_SHARE, PAIRS_MIN_RECORDS = 7, 0.5, 30
COMM_LOSS = {"k": 10, "share": 0.5, "gap_min": 5, "min_minutes": 1}   # comm_loss_k10_s50


def alarm_pairs(root=PROCESSED_DIR) -> list[tuple[str, str]]:
    """Тревожные пары (тип_датчика, состояние), по которым построены эпизоды тревог: пересчет из
    `episodes/flag_hourly.parquet` тем же правилом, что в 03_target, со сверкой с хэшем каталога эпизодов."""
    fh = agg.read_vitrine(Path(root) / "episodes" / "flag_hourly.parquet")
    fh["день"] = pd.to_datetime(fh["день"])
    n = fh[fh["час_суток"] < PAIRS_NIGHT_END]
    g = (n.assign(год=n["день"].dt.year, месяц=n["день"].dt.month)
         .groupby(["тип_датчика", "состояние", "год", "месяц"], as_index=False)
         [["количество_событий", "количество_тревожных"]].sum())
    g["количество_тревожных"] = g["количество_тревожных"].fillna(0)
    vocab = pd.concat([g.assign(тревожное=True, количество_событий=g["количество_тревожных"]),
                       g.assign(тревожное=False, количество_событий=g["количество_событий"] - g["количество_тревожных"])])
    pairs = ep.alarm_pairs(vocab.drop(columns="количество_тревожных"), PAIRS_STABLE_UNTIL,
                           min_share=PAIRS_MIN_SHARE, min_records=PAIRS_MIN_RECORDS)
    out = list(zip(pairs["тип_датчика"], pairs["состояние"]))
    rules = hashlib.md5(repr((ep.NOT_CLOSED, ep.BY_STATE, ep.PRIORITY)).encode()).hexdigest()[:8]
    tag = hashlib.md5(repr((sorted(out), rules)).encode()).hexdigest()[:8]
    ep_dir = fi.episodes_dir(root)
    if ep_dir.name != f"alarms_pairs_{tag}":
        raise RuntimeError(f"список пар не совпадает с каталогом эпизодов {ep_dir.name} (хэш {tag})")
    return out


def _restore(path: Path) -> None:
    """Витрины нет, а `<path>.tmp` дописан (есть _SUCCESS): сбой между удалением и переименованием в _swap."""
    tmp = path.with_name(path.name + ".tmp")
    if not path.exists() and (tmp / "_SUCCESS").exists():
        tmp.rename(path)


def _swap(df: DataFrame, path: Path) -> None:
    """Записать df в `<path>.tmp` и заменить им path."""
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    df.write.mode("overwrite").parquet(str(tmp))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def _replace(spark, path: Path, fresh: DataFrame, col: str, lo, hi) -> None:
    """Заменить в витрине строки с col в [lo, hi) на строки fresh из того же интервала."""
    _restore(path)
    old = spark.read.parquet(str(path))
    inside = (F.col(col) >= F.lit(lo)) & (F.col(col) < F.lit(hi))
    out = old.where(~inside | F.col(col).isNull()).unionByName(fresh.where(inside).select(*old.columns))
    _swap(out, path)


def _events(spark, root, lo, hi) -> DataFrame:
    """События с дата_время в [lo, hi) (с отсечением партиций по году)."""
    ev = spark.read.parquet(str(Path(root) / "events"))
    return ev.where(F.col("год").between(lo.year, hi.year)
                    & (F.col("дата_время") >= F.lit(lo)) & (F.col("дата_время") < F.lit(hi)))


def recompute(spark: SparkSession, lo, hi, root=PROCESSED_DIR) -> None:
    """Пересчитать производные данные по событиям для суток [lo, hi] (включительно)."""
    root = Path(root)
    day = pd.Timedelta(days=1)
    lo, hi = pd.Timestamp(lo).normalize(), pd.Timestamp(hi).normalize() + day   # [lo, hi)
    channels = spark.read.parquet(str(root / "channels.parquet"))
    ev = _events(spark, root, lo, hi).cache()
    eda = root / "eda"
    # суточные витрины
    _replace(spark, eda / "daily_channel.parquet", agg.daily_channel(ev), "день", lo, hi)
    _replace(spark, eda / "undefined_daily.parquet", agg.undefined_daily(ev), "день", lo, hi)
    _replace(spark, eda / "undefined_bursts.parquet", agg.undefined_bursts(ev, channels), "минута", lo, hi)
    _replace(spark, eda / "hourly_type.parquet", agg.hourly_type(ev, channels), "час", lo, hi)
    meas = _events(spark, root, lo - pd.Timedelta(days=MEAS_LOOKBACK_DAYS), hi)
    _replace(spark, eda / "meas_gaps.parquet", agg.meas_gaps(meas), "день", lo, hi)
    _replace(spark, root / "features" / "object_day_states.parquet", agg.object_day_states(ev, channels), "день", lo, hi)
    # дни тестового газа и демонтажа: знаменатель – каналы, писавшие хоть раз (обновленная daily_channel)
    seen = spark.read.parquet(str(eda / "daily_channel.parquet")).select("ид_канала_данных").distinct()
    maint = root / "maintenance"
    for name, state in [("gas_burst_days", mt.GAS_STATE), ("gas_fault_days", mt.FAULT_STATE)]:
        _replace(spark, maint / f"{name}.parquet", mt.gas_burst_days(ev, channels, state, seen_channels=seen),
                 "день", lo, hi)
    ev.unpersist()
    # эпизоды тревог и потери связи: затронутые годы целиком
    pairs = alarm_pairs(root)
    ep_dir = fi.episodes_dir(root)
    cl_path = root / "episodes" / "comm_loss_k10_s50.parquet"
    for y in range(lo.year, (hi - day).year + 1):
        y0, y1 = pd.Timestamp(y, 1, 1), pd.Timestamp(y + 1, 1, 1)
        year_ev = _events(spark, root, y0 - day, y1 + day)
        eps = ep.alarm_episodes(year_ev, channels, "пары", pairs)
        _swap(eps.where((F.col("начало") >= F.lit(y0)) & (F.col("начало") < F.lit(y1))), ep_dir / f"{y}.parquet")
        undef = year_ev.filter(F.col("состояние") == "Неопределен").select("ид_канала_данных", "дата_время", "состояние")
        _replace(spark, cl_path, ep.comm_loss_episodes(undef, channels, **COMM_LOSS), "начало", y0, y1)


def _pending(file: Path):
    """Незавершенный пересчет (с, по) или None."""
    if not file.exists():
        return None
    r = json.loads(file.read_text(encoding="utf-8"))
    return pd.Timestamp(r["с"]), pd.Timestamp(r["по"])


def ingest_csv(spark: SparkSession, path, root=PROCESSED_DIR) -> dict:
    """Загрузить CSV журнала и пересчитать производные данные. Возвращает отчет (поля – в service.schemas)."""
    t0 = time.time()
    root = Path(root)
    events_dir = root / "events"
    pending_file = root / "service" / "recompute_pending.json"
    channels = spark.read.parquet(str(root / "channels.parquet"))
    known = {r[0] for r in channels.select("ид_канала_данных").collect()}
    clean, report = etl.clean_journal(spark, path, known)
    clean = clean.dropDuplicates(EVENT_KEY)          # «t» и «true» после нормализации – одна строка
    months = [(r["год"], r["месяц"]) for r in clean.select("год", "месяц").distinct().collect()]
    new = clean
    if months:
        old = spark.read.parquet(str(events_dir))
        in_months = None
        for y, m in months:
            c = (F.col("год") == y) & (F.col("месяц") == m)
            in_months = c if in_months is None else in_months | c
        old = old.where(in_months).select(*EVENT_KEY)
        new = clean.alias("n").join(old.alias("o"), [F.col(f"n.{c}").eqNullSafe(F.col(f"o.{c}")) for c in EVENT_KEY],
                                    "left_anti")
    # через временный каталог: new читает events/, а запись идет туда же
    tmp = root / "service" / "ingest.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    new.write.parquet(str(tmp))
    spark.catalog.clearCache()
    new = spark.read.parquet(str(tmp))
    stats = new.agg(F.count("*").alias("n"), F.min("дата_время").alias("lo"), F.max("дата_время").alias("hi")).first()
    added = int(stats["n"])
    out = {**{k: int(v) for k, v in report.items()}, "уже_загружено": int(report["строк_после_очистки"]) - added,
           "добавлено": added, "первый_день": None, "последний_день": None,
           "пересчитано_с": None, "пересчитано_по": None}
    rng = _pending(pending_file)
    if added:
        lo, hi = pd.Timestamp(stats["lo"]).normalize(), pd.Timestamp(stats["hi"]).normalize()
        out.update({"первый_день": lo.date(), "последний_день": hi.date()})
        start = lo - pd.Timedelta(days=OVERLAP_DAYS)
        rng = (start, hi) if rng is None else (min(start, rng[0]), max(hi, rng[1]))
    if rng is not None:
        pending_file.parent.mkdir(parents=True, exist_ok=True)
        pending_file.write_text(json.dumps({"с": f"{rng[0]:%Y-%m-%d}", "по": f"{rng[1]:%Y-%m-%d}"}), encoding="utf-8")
        if added:
            etl.write_events(new, events_dir)
        recompute(spark, rng[0], rng[1], root)
        pending_file.unlink()
        out.update({"пересчитано_с": rng[0].date(), "пересчитано_по": rng[1].date()})
    shutil.rmtree(tmp, ignore_errors=True)
    out["секунд"] = round(time.time() - t0, 1)
    return out
