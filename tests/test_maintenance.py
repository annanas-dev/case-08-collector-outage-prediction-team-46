from datetime import datetime

import pandas as pd
import pytest

from annanas import maintenance as m

T = pd.Timestamp


# --- график ---------------------------------------------------------------------------------------------------

def test_load_schedule_columns_and_types():
    s = m.load_schedule()
    assert list(s.columns) == ["объект_графика", "ид_объект", "датчиков", "демонтаж", "возврат", "сдача",
                               "уверенность", "примечание"]
    assert s["объект_графика"].between(1, 26).all()
    assert set(s["объект_графика"]) == set(range(1, 27))
    for c in ("демонтаж", "возврат", "сдача"):
        assert pd.api.types.is_datetime64_any_dtype(s[c])
        assert s[c].notna().all()                                   # строки без дат наследуют даты группы
    assert (s["демонтаж"] < s["возврат"]).all() and (s["возврат"] < s["сдача"]).all()
    assert s["ид_объект"].dtype == "Int64"


def test_load_schedule_mapping():
    s = m.load_schedule()
    assert s.loc[s["объект_графика"] == 4, "ид_объект"].isna().all()   # не сопоставлен
    assert set(s.loc[s["объект_графика"].isin([6, 7]), "ид_объект"]) == {4369}
    assert set(s.loc[s["объект_графика"].between(10, 13), "ид_объект"]) == {111}
    assert set(s.loc[s["объект_графика"] == 23, "ид_объект"]) == {5578, 5591}
    assert set(s["уверенность"].dropna()) == {"уверенно", "по_числу"}
    row17 = s.loc[s["объект_графика"] == 17].iloc[0]
    assert row17["демонтаж"] == T("2026-07-28") and row17["сдача"] == T("2026-08-21")


# --- дни тестового газа (Spark) -------------------------------------------------------------------------------

def _events(spark, rows):
    return spark.createDataFrame(
        [(ch, datetime.fromisoformat(ts), st) for ch, ts, st in rows],
        "`ид_канала_данных` long, `дата_время` timestamp, `состояние` string")


def test_gas_burst_days(spark):
    channels = spark.createDataFrame(
        [(1, 10, "Газовый датчик"), (2, 10, "Газовый датчик"), (3, 10, "Газовый датчик"),
         (4, 10, "Газовый датчик"), (5, 10, "Датчик дыма"), (6, 20, "Газовый датчик")],
        "`ид_канала_данных` long, `ид_объект` long, `тип_датчика` string")
    ev = _events(spark, [
        # объект 10: 4 газовых канала пишут, 2026-03-02 газ на 2 из 4 в 09:00 – вспышка (доля 0,5)
        (1, "2026-03-01 12:00:00", None), (2, "2026-03-01 12:00:00", None),
        (3, "2026-03-01 12:00:00", None), (4, "2026-03-01 12:00:00", None),
        (1, "2026-03-02 09:00:00", "Обнаружен газ"), (2, "2026-03-02 10:00:00", "Обнаружен газ"),
        # 2026-03-03: газ на 1 из 4 – доля 0,25, не вспышка
        (1, "2026-03-03 09:00:00", "Обнаружен газ"),
        # 2026-03-04: газ на 3 из 4, но первый в 18:00 – вне рабочего времени
        (1, "2026-03-04 18:00:00", "Обнаружен газ"), (2, "2026-03-04 19:00:00", "Обнаружен газ"),
        (3, "2026-03-04 20:00:00", "Обнаружен газ"),
        # 2026-03-05: первый в 17:59 – в окне 07-17 ч; канал дыма не газовый и не считается
        (1, "2026-03-05 17:59:00", "Обнаружен газ"), (2, "2026-03-05 18:30:00", "Обнаружен газ"),
        (5, "2026-03-05 09:00:00", "Обнаружен газ"),
        # 2026-03-06: другое состояние – не газ
        (1, "2026-03-06 09:00:00", "Норма"), (2, "2026-03-06 09:00:00", "Норма"),
        # объект 20: единственный канал, газ в 07:00 – вспышка
        (6, "2026-03-02 07:00:00", "Обнаружен газ"),
    ])
    out = m.gas_burst_days(ev, channels).toPandas().sort_values(["ид_объект", "день"]).reset_index(drop=True)
    assert list(out.columns) == ["ид_объект", "день", "каналов_с_газом", "газовых_каналов", "доля_каналов",
                                 "первый_газ"]
    got = list(zip(out["ид_объект"], pd.to_datetime(out["день"])))
    assert got == [(10, T("2026-03-02")), (10, T("2026-03-05")), (20, T("2026-03-02"))]
    assert out.loc[0, "газовых_каналов"] == 4 and out.loc[0, "каналов_с_газом"] == 2
    assert out.loc[0, "доля_каналов"] == pytest.approx(0.5)


def test_gas_burst_days_other_state(spark):
    # тот же расчет для «Неисправен» (демонтаж датчиков): доля каналов и рабочие часы
    channels = spark.createDataFrame([(1, 10, "Газовый датчик"), (2, 10, "Газовый датчик")],
                                     "`ид_канала_данных` long, `ид_объект` long, `тип_датчика` string")
    ev = _events(spark, [(1, "2026-03-02 09:00:00", "Неисправен"), (2, "2026-03-02 09:00:05", "Неисправен"),
                         (1, "2026-03-03 09:00:00", "Обнаружен газ"), (2, "2026-03-03 09:00:00", "Обнаружен газ")])
    out = m.gas_burst_days(ev, channels, state=m.FAULT_STATE).toPandas()
    assert list(pd.to_datetime(out["день"])) == [T("2026-03-02")]


def test_gas_burst_days_seen_channels(spark):
    # события только за сутки вспышки, знаменатель – каналы, писавшие хоть раз (4 газовых у объекта 10)
    channels = spark.createDataFrame([(i, 10, "Газовый датчик") for i in range(1, 5)],
                                     "`ид_канала_данных` long, `ид_объект` long, `тип_датчика` string")
    ev = _events(spark, [(1, "2026-03-02 09:00:00", "Обнаружен газ")])
    seen = spark.createDataFrame([(1,), (2,), (3,), (4,), (99,)], "`ид_канала_данных` long")
    out = m.gas_burst_days(ev, channels, seen_channels=seen).toPandas()
    assert out.empty                                                   # 1 из 4 < 0,5
    ev2 = _events(spark, [(1, "2026-03-02 09:00:00", "Обнаружен газ"), (2, "2026-03-02 09:00:00", "Обнаружен газ")])
    out = m.gas_burst_days(ev2, channels, seen_channels=seen).toPandas()
    assert out["газовых_каналов"].tolist() == [4] and out["доля_каналов"].tolist() == [0.5]


# --- детектор окон ----------------------------------------------------------------------------------------

def _burst(obj, *days):
    return pd.DataFrame({"ид_объект": obj, "день": pd.to_datetime(list(days))})


def test_detect_windows_single_cluster_lead():
    # две вспышки через 5 суток: окно начинается за LEAD_DAYS до последней
    w = m.detect_windows(_burst(1, "2026-03-10", "2026-03-15"))
    assert list(w.columns) == ["ид_объект", "начало", "конец", "источник", "дней_всплеска"]
    assert len(w) == 1
    r = w.iloc[0]
    assert r["начало"] == T("2026-03-15") - pd.Timedelta(days=m.LEAD_DAYS)
    assert r["конец"] == T("2026-03-15") and r["источник"] == "журнал" and r["дней_всплеска"] == 2


def test_detect_windows_long_cluster_starts_at_first_burst():
    w = m.detect_windows(_burst(1, "2026-03-01", "2026-03-10", "2026-03-20"))   # 19 суток > LEAD_DAYS
    assert len(w) == 1
    assert w.iloc[0]["начало"] == T("2026-03-01") and w.iloc[0]["конец"] == T("2026-03-20")


def test_detect_windows_single_day_dropped_and_gap_splits():
    # 22 суток между вспышками > MAX_GAP_DAYS: два одиночных дня – окон нет
    assert m.detect_windows(_burst(1, "2026-03-01", "2026-03-23")).empty
    # ровно MAX_GAP_DAYS – один кластер
    assert len(m.detect_windows(_burst(1, "2026-03-01", "2026-03-22"))) == 1


def test_detect_windows_span_limit():
    # разрывы по 10 суток, но кластер не длиннее MAX_SPAN_DAYS от первой вспышки
    w = m.detect_windows(_burst(1, "2026-03-01", "2026-03-11", "2026-03-21", "2026-03-31", "2026-04-10"))
    assert list(zip(w["начало"], w["конец"], w["дней_всплеска"])) == [
        (T("2026-03-01"), T("2026-03-21"), 3), (T("2026-03-25"), T("2026-04-10"), 2)]


def test_detect_windows_per_object_and_duplicates():
    b = pd.concat([_burst(1, "2026-03-01", "2026-03-02", "2026-03-02"), _burst(2, "2026-03-05")])
    w = m.detect_windows(b)
    assert list(w["ид_объект"]) == [1] and w.iloc[0]["дней_всплеска"] == 2


def test_detect_windows_empty():
    w = m.detect_windows(_burst(1)[:0])
    assert w.empty and list(w.columns) == ["ид_объект", "начало", "конец", "источник", "дней_всплеска"]


def _win(rows):
    return pd.DataFrame([(o, T(a), T(b), "журнал", 2) for o, a, b in rows],
                        columns=["ид_объект", "начало", "конец", "источник", "дней_всплеска"])


def test_extend_to_dismantling():
    w = _win([(1, "2025-11-10", "2025-11-26"), (2, "2025-03-10", "2025-03-20"), (3, "2025-05-10", "2025-05-20")])
    fault = pd.DataFrame({"ид_объект": [1, 1, 2, 3, 3],
                          "день": pd.to_datetime(["2025-11-06", "2025-10-01", "2025-03-02", "2025-05-12",
                                                  "2025-05-03"])})
    out = m.extend_to_dismantling(w, fault)
    assert list(out.columns) == list(w.columns)
    got = dict(zip(out["ид_объект"], out["начало"]))
    assert got[1] == T("2025-11-06")                      # за 4 сут до начала – демонтаж
    assert got[2] == T("2025-03-10")                      # 8 сут > DISMANTLE_MAX_DAYS – не трогаем
    assert got[3] == T("2025-05-03")                      # ровно 7 сут; день внутри окна не сдвигает
    assert list(out["конец"]) == list(w["конец"])
    assert m.extend_to_dismantling(w, fault.iloc[:0])["начало"].equals(w["начало"])


# --- объединение графика и детектора ------------------------------------------------------------------------

def _schedule():
    return pd.DataFrame({
        "объект_графика": [1, 2, 3, 4],
        "ид_объект": pd.array([100, 100, 200, pd.NA], dtype="Int64"),
        "демонтаж": pd.to_datetime(["2026-03-01", "2026-03-01", "2026-09-01", "2026-05-01"]),
        "возврат": pd.to_datetime(["2026-03-10", "2026-03-10", "2026-09-10", "2026-05-10"]),
        "сдача": pd.to_datetime(["2026-03-15", "2026-03-17", "2026-09-15", "2026-05-15"]),
    })


def _detected(rows):
    return pd.DataFrame([(o, T(a), T(b), "журнал", 2) for o, a, b in rows],
                        columns=["ид_объект", "начало", "конец", "источник", "дней_всплеска"])


def test_ppr_windows_schedule_replaces_detector_for_schedule_objects_in_2026():
    det = _detected([
        (100, "2025-02-20", "2025-03-08"),     # до года графика – остается
        (100, "2026-02-20", "2026-03-08"),     # объект графика в 2026 – заменяется графиком
        (100, "2025-12-20", "2026-01-05"),     # начался до года графика и не пересекает график – остается
        (300, "2026-04-01", "2026-04-10"),     # объекта нет в графике – остается
        (300, "2019-04-01", "2019-04-10"),
    ])
    w = m.ppr_windows(_schedule(), det)
    assert list(w.columns) == ["ид_объект", "начало", "конец", "источник"]
    got = set(zip(w["ид_объект"], w["начало"], w["конец"], w["источник"]))
    assert got == {
        (100, T("2025-02-20"), T("2025-03-08"), "журнал"),
        (100, T("2025-12-20"), T("2026-01-05"), "журнал"),
        (100, T("2026-03-01"), T("2026-03-17"), "график"),      # строки 1 и 2 одного объекта слиты
        (200, T("2026-09-01"), T("2026-09-15"), "график"),
        (300, T("2026-04-01"), T("2026-04-10"), "журнал"),
        (300, T("2019-04-01"), T("2019-04-10"), "журнал"),
    }
    assert w["ид_объект"].notna().all()                          # несопоставленный объект 4 выброшен


def test_ppr_windows_keeps_detector_window_crossing_year_clipped_at_schedule():
    # окно детектора с декабря по январь у объекта, чей график начинается в январе (как 4177: 2025-12-19 …
    # 2026-01-12 при демонтаже 2026-01-12): часть до графика остается, конец обрезается днем перед графиком
    sched = _schedule().assign(демонтаж=pd.to_datetime(["2026-01-12", "2026-01-12", "2026-09-01", "2026-05-01"]))
    det = _detected([(100, "2025-12-19", "2026-01-12"), (200, "2025-12-28", "2026-01-03")])
    w = m.ppr_windows(sched, det)
    got = set(zip(w["ид_объект"], w["начало"], w["конец"], w["источник"]))
    assert got == {
        (100, T("2025-12-19"), T("2026-01-11"), "журнал"),
        (100, T("2026-01-12"), T("2026-03-17"), "график"),
        (200, T("2025-12-28"), T("2026-01-03"), "журнал"),         # не пересекает график 200 – без изменений
        (200, T("2026-09-01"), T("2026-09-15"), "график"),
    }
    # маска покрывает весь декабрь-январь без разрыва
    days = pd.date_range("2025-12-16", "2026-03-20")
    grid = pd.DataFrame({"ид_объект": 100, "день": days})
    assert m.ppr_mask(grid, w).notna().all()


def test_ppr_windows_until_drops_future_windows():
    det = _detected([(300, "2026-04-01", "2026-04-10")])
    w = m.ppr_windows(_schedule(), det, until="2026-06-30")
    assert set(w["ид_объект"]) == {100, 300}                      # окно 200 (сентябрь) еще не началось


def test_ppr_windows_accepts_empty_detected():
    w = m.ppr_windows(_schedule(), m.detect_windows(_burst(1)[:0]))
    assert set(w["источник"]) == {"график"} and len(w) == 2


# --- маска ----------------------------------------------------------------------------------------------------

def test_ppr_mask_with_pad():
    win = pd.DataFrame({"ид_объект": [1], "начало": [T("2026-03-10")], "конец": [T("2026-03-12")],
                        "источник": ["журнал"]})
    days = pd.date_range("2026-03-05", "2026-03-17")
    grid = pd.DataFrame({"ид_объект": [1] * len(days) + [2], "день": list(days) + [T("2026-03-11")]},
                        index=range(100, 100 + len(days) + 1))
    s = m.ppr_mask(grid, win)
    assert s.index.equals(grid.index) and s.dtype == object
    masked = set(grid.loc[s.notna(), "день"][grid["ид_объект"] == 1])
    assert masked == set(pd.date_range("2026-03-07", "2026-03-15"))       # ±3 сут
    assert s.loc[grid["ид_объект"] == 2].isna().all()                    # другой объект
    assert set(s.dropna()) == {m.MASK_LABEL}
    assert s[s.isna()].map(lambda x: x is None).all()                    # None, не NaN

    s0 = m.ppr_mask(grid, win, pad_days=0)
    assert set(grid.loc[s0.notna(), "день"]) == set(pd.date_range("2026-03-10", "2026-03-12"))


def test_ppr_mask_day_with_time_and_empty_windows():
    grid = pd.DataFrame({"ид_объект": [1], "день": [pd.Timestamp("2026-03-10 00:00")]})
    win = pd.DataFrame(columns=["ид_объект", "начало", "конец", "источник"])
    assert m.ppr_mask(grid, win).isna().all()
