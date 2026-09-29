"""Окна планово-предупредительных работ (ППР) аппаратуры контроля метана (АКМ).

Во время ППР датчики метана демонтируют, возят на поверку и возвращают; при возврате и сдаче комиссии их
проверяют тестовым газом. В журнале это видно как «Обнаружен газ» сразу на большинстве газовых каналов объекта
в рабочее время, ежегодно в том же месяце. Такие дни – артефакт работ, а не инцидент, поэтому окна ППР
маскируются в целевой таблице и в истории признаков.

Источники окон:

- график ППР 2026 (`data/ppr_akm_2026.csv`, см. шапку файла: источник и способ сопоставления объектов);
- детектор по журналу для остальных лет: дни тестового газа (gas_burst_days) → кластеры (detect_windows) →
  начало окна сдвигается на день демонтажа, если он виден как массовое «Неисправен» на газовых каналах
  незадолго до окна (extend_to_dismantling).

Правило объединения – в ppr_windows."""
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

SCHEDULE_FILE = Path(__file__).parent / "data" / "ppr_akm_2026.csv"

GAS_SENSOR_TYPE = "Газовый датчик"
GAS_STATE = "Обнаружен газ"
FAULT_STATE = "Неисправен"     # демонтаж датчиков: массовое «Неисправен» на газовых каналах объекта
MASK_LABEL = "плановые работы АКМ"

# Дни тестового газа
BURST_MIN_SHARE = 0.5          # «Обнаружен газ» хотя бы на половине газовых каналов объекта за сутки
WORK_HOURS = (7, 17)           # первая запись «Обнаружен газ» за сутки в 07:00-17:59
# Кластеры дней в окна (подобраны на графике 2026 H1)
MAX_GAP_DAYS = 21              # следующая вспышка не дальше 21 сут от предыдущей
MAX_SPAN_DAYS = 25             # и не дальше 25 сут от первой вспышки кластера
MIN_BURST_DAYS = 2             # одиночный день – не окно
LEAD_DAYS = 16                 # окно начинается не позже чем за 16 сут до последней вспышки (демонтаж → сдача)
DISMANTLE_MAX_DAYS = 7         # день демонтажа ищется не раньше чем за 7 сут до начала окна детектора
PAD_DAYS = 3                   # запас маски вокруг окна

WINDOW_COLUMNS = ["ид_объект", "начало", "конец", "источник"]


@lru_cache(maxsize=1)
def _read_schedule() -> pd.DataFrame:
    return pd.read_csv(SCHEDULE_FILE, comment="#", parse_dates=["демонтаж", "возврат", "сдача"],
                       dtype={"ид_объект": "Int64", "уверенность": str, "примечание": str})


def load_schedule() -> pd.DataFrame:
    """График ППР АКМ 2026: `объект_графика, ид_объект, датчиков, демонтаж, возврат, сдача, уверенность,
    примечание`. Строка на пару «объект графика × ид_объект»; ид_объект пустой, если объект не сопоставлен."""
    return _read_schedule().copy()


def gas_burst_days(events: DataFrame, channels: DataFrame, state: str = GAS_STATE,
                   seen_channels: DataFrame | None = None) -> DataFrame:
    """Дни тестового газа (Spark): объект × сутки, где «Обнаружен газ» было хотя бы на BURST_MIN_SHARE газовых
    каналов объекта и первая такая запись за сутки пришлась на WORK_HOURS. С state=FAULT_STATE – так же
    считаются дни массового «Неисправен» (демонтаж датчиков).

    events: `ид_канала_данных, дата_время, состояние`; channels: `ид_канала_данных, ид_объект, тип_датчика`.
    Знаменатель доли – газовые каналы объекта, у которых есть хотя бы одна запись в events (любая, включая
    измерения), а не все каналы справочника: неработающие каналы в справочнике занижали бы долю. Поэтому
    передавать нужно журнал за весь период, а не только за интересующие сутки, или список каналов, писавших
    хоть раз (seen_channels: `ид_канала_данных`, например из витрины daily_channel), – тогда events может быть
    только за нужные сутки (так дописывает дни сервис).
    Результат: `ид_объект, день, каналов_с_газом, газовых_каналов, доля_каналов, первый_газ`."""
    gas_ch = (channels.where(F.col("тип_датчика") == GAS_SENSOR_TYPE)
              .select("ид_канала_данных", "ид_объект").distinct())
    is_gas = F.col("состояние") == state
    per_day = (events.select("ид_канала_данных", "дата_время", "состояние")
               .join(F.broadcast(gas_ch), "ид_канала_данных")
               .groupBy("ид_объект", "ид_канала_данных", F.to_date("дата_время").alias("день"))
               .agg(F.min(F.when(is_gas, F.col("дата_время"))).alias("первый_газ")))
    seen = (per_day if seen_channels is None
            else seen_channels.select("ид_канала_данных").distinct().join(F.broadcast(gas_ch), "ид_канала_данных"))
    n_channels = seen.groupBy("ид_объект").agg(F.countDistinct("ид_канала_данных").alias("газовых_каналов"))
    lo, hi = WORK_HOURS
    return (per_day.where(F.col("первый_газ").isNotNull())
            .groupBy("ид_объект", "день")
            .agg(F.countDistinct("ид_канала_данных").alias("каналов_с_газом"), F.min("первый_газ").alias("первый_газ"))
            .join(n_channels, "ид_объект")
            .withColumn("доля_каналов", F.col("каналов_с_газом") / F.col("газовых_каналов"))
            .where((F.col("доля_каналов") >= BURST_MIN_SHARE) & F.hour("первый_газ").between(lo, hi))
            .select("ид_объект", "день", "каналов_с_газом", "газовых_каналов", "доля_каналов", "первый_газ"))


def detect_windows(burst_days: pd.DataFrame) -> pd.DataFrame:
    """Окна ППР по дням тестового газа (`ид_объект, день`, pandas – например, gas_burst_days(...).toPandas()).

    Дни объекта по порядку собираются в кластер, пока следующий день не дальше MAX_GAP_DAYS от предыдущего и
    MAX_SPAN_DAYS от первого; кластеры короче MIN_BURST_DAYS дней отбрасываются. Вспышки – это возврат датчиков
    и сдача, демонтаж газом не виден, поэтому окно = [min(первая вспышка, последняя − LEAD_DAYS), последняя].
    Результат: `ид_объект, начало, конец, источник="журнал", дней_всплеска`."""
    b = (burst_days[["ид_объект", "день"]]
         .assign(день=lambda x: pd.to_datetime(x["день"]).dt.normalize())
         .drop_duplicates().sort_values(["ид_объект", "день"]))
    rows = []
    for obj, days in b.groupby("ид_объект", sort=True)["день"]:
        clusters = []
        for d in days:
            if clusters and (d - clusters[-1][-1]).days <= MAX_GAP_DAYS and (d - clusters[-1][0]).days <= MAX_SPAN_DAYS:
                clusters[-1].append(d)
            else:
                clusters.append([d])
        for c in clusters:
            if len(c) >= MIN_BURST_DAYS:
                rows.append((obj, min(c[0], c[-1] - pd.Timedelta(days=LEAD_DAYS)), c[-1], "журнал", len(c)))
    out = pd.DataFrame(rows, columns=WINDOW_COLUMNS + ["дней_всплеска"])
    out["начало"] = pd.to_datetime(out["начало"])
    out["конец"] = pd.to_datetime(out["конец"])
    return out


def extend_to_dismantling(windows: pd.DataFrame, fault_days: pd.DataFrame,
                          max_days: int = DISMANTLE_MAX_DAYS) -> pd.DataFrame:
    """Сдвигает начало окна детектора на день демонтажа: самый ранний день из fault_days (`ид_объект, день`,
    например gas_burst_days(..., state=FAULT_STATE)) того же объекта в [начало − max_days, начало). Тестовый газ
    демонтаж не показывает, а снятый датчик сообщает «Неисправен» – на всех каналах объекта сразу."""
    out = windows.copy()
    if out.empty or fault_days.empty:
        return out
    f = fault_days[["ид_объект", "день"]].assign(день=lambda x: pd.to_datetime(x["день"]).dt.normalize())
    by_obj = f.groupby("ид_объект")["день"].apply(list).to_dict()
    lag = pd.Timedelta(days=max_days)
    out["начало"] = [min([a] + [d for d in by_obj.get(o, []) if a - lag <= d < a])
                     for o, a in zip(out["ид_объект"], pd.to_datetime(out["начало"]))]
    out["начало"] = pd.to_datetime(out["начало"])
    return out


def _merge_overlaps(w: pd.DataFrame) -> pd.DataFrame:
    """Сливает пересекающиеся окна одного объекта и источника."""
    rows = []
    for (obj, src), g in w.sort_values("начало").groupby(["ид_объект", "источник"], sort=True):
        cur = None
        for a, b in zip(g["начало"], g["конец"]):
            if cur and a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                if cur:
                    rows.append((obj, cur[0], cur[1], src))
                cur = [a, b]
        rows.append((obj, cur[0], cur[1], src))
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def ppr_windows(schedule: pd.DataFrame, detected: pd.DataFrame, until=None) -> pd.DataFrame:
    """Окна ППР из графика и детектора: `ид_объект, начало, конец, источник` (источник – «график» или «журнал»).

    Правило: у объекта из графика в год графика (2026) окна берутся только из графика (демонтаж … сдача),
    окна детектора, начавшиеся в этом году или позже, отбрасываются – график точнее и содержит будущие окна.
    Окно детектора, начавшееся до года графика и заходящее в него (декабрь … январь), остается, а если
    пересекает окно графика своего объекта, его конец обрезается днем перед началом окна графика. Прочие
    годы и объекты вне графика – окна детектора. Несопоставленные строки графика не используются, строки
    одного объекта с пересекающимися датами (составные группы) сливаются.
    until: оставить только окна, начавшиеся не позже этой даты (например, конец данных)."""
    s = schedule.dropna(subset=["ид_объект"])
    sched = _merge_overlaps(pd.DataFrame({
        "ид_объект": s["ид_объект"].astype("int64"), "начало": pd.to_datetime(s["демонтаж"]),
        "конец": pd.to_datetime(s["сдача"]), "источник": "график"}))
    year_start = pd.Timestamp(year=int(pd.to_datetime(schedule["демонтаж"]).dt.year.min()), month=1, day=1)
    det = detected[WINDOW_COLUMNS].copy()
    det = det.loc[~(det["ид_объект"].isin(sched["ид_объект"]) & (det["начало"] >= year_start))]
    det["конец"] = pd.to_datetime([min([b] + [sa - pd.Timedelta(days=1) for so, sa, sb in
                                              zip(sched["ид_объект"], sched["начало"], sched["конец"])
                                              if so == o and sa <= b and sb >= a])
                                   for o, a, b in zip(det["ид_объект"], det["начало"], det["конец"])])
    det = det.loc[det["конец"] >= det["начало"]]
    out = pd.concat([det.astype({"ид_объект": "int64"}), sched], ignore_index=True)
    if until is not None:
        out = out.loc[out["начало"] <= pd.Timestamp(until)]
    return out.sort_values(["ид_объект", "начало"]).reset_index(drop=True)


def ppr_mask(grid: pd.DataFrame, windows: pd.DataFrame, pad_days: int = PAD_DAYS) -> pd.Series:
    """Маска «объект × сутки» в окнах ППР: MASK_LABEL, если день grid попадает в [начало − pad_days,
    конец + pad_days] окна своего объекта, иначе None. Индекс – как у grid (`ид_объект, день`)."""
    pad = pd.Timedelta(days=pad_days)
    keys = [(o, d) for o, a, b in zip(windows["ид_объект"], pd.to_datetime(windows["начало"]),
                                      pd.to_datetime(windows["конец"]))
            for d in pd.date_range(a.normalize() - pad, b.normalize() + pad)]
    days = pd.to_datetime(grid["день"]).dt.normalize()
    hit = pd.MultiIndex.from_arrays([grid["ид_объект"], days]).isin(
        pd.MultiIndex.from_tuples(keys, names=["ид_объект", "день"]) if keys else [])
    return pd.Series([MASK_LABEL if h else None for h in hit], index=grid.index, dtype=object, name="маска")
