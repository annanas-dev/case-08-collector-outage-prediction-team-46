import numpy as np
import pandas as pd

from annanas import calendar_ru as cal


def _nw(*days):
    return cal.is_non_working(pd.to_datetime(list(days)))


def test_new_year_holidays_2024_non_working():
    assert _nw(*pd.date_range("2024-01-01", "2024-01-08")).all()
    assert not _nw("2024-01-09")[0]


def test_working_saturday_2024_04_27():
    assert not _nw("2024-04-27")[0]
    assert _nw("2024-04-29", "2024-04-30").all()          # перенесенные выходные


def test_transfers_2025_and_2023():
    assert _nw("2025-05-02")[0]                            # пятница, перенос с 4 января
    assert _nw("2023-02-24")[0]                            # пятница, перенос с 1 января


def test_ordinary_days():
    assert not _nw("2024-07-17")[0]                        # обычная среда
    assert _nw("2024-07-20", "2024-07-21").all()          # обычные суббота и воскресенье


def test_decree_days_2020_non_working():
    assert _nw("2020-04-15", "2020-07-01").all()


def test_calendar_table_types():
    t = cal.load_calendar()
    assert set(t["тип"]) == set(cal.DAY_TYPES)
    assert t["дата"].between("2019-01-01", "2026-12-31").all()
    assert t.loc[t["дата"] == "2024-04-27", "тип"].item() == "рабочий_выходной"


def test_calendar_features_definitions():
    days = pd.to_datetime(["2024-04-26", "2024-04-27", "2024-04-28", "2024-05-02", "2024-05-06", "2024-07-17",
                           "2024-02-24"])
    f = cal.calendar_features(days).set_index(days)
    assert list(f.columns) == ["нерабочий_день", "праздник", "дней_до_нерабочего", "дней_после_нерабочего",
                               "первый_рабочий_после_выходных"]
    # 26.04 пятница рабочий, 27.04 рабочая суббота, 28.04-01.05 нерабочие, 02.05 рабочий
    assert f.loc["2024-04-26", "нерабочий_день"] == 0 and f.loc["2024-04-26", "дней_до_нерабочего"] == 2
    assert f.loc["2024-04-27", "нерабочий_день"] == 0 and f.loc["2024-04-27", "дней_после_нерабочего"] == 6
    assert f.loc["2024-04-28", "нерабочий_день"] == 1 and f.loc["2024-04-28", "дней_до_нерабочего"] == 0
    assert f.loc["2024-04-28", "праздник"] == 1           # воскресенье внутри майского блока
    assert f.loc["2024-05-02", "первый_рабочий_после_выходных"] == 1
    assert f.loc["2024-05-02", "дней_после_нерабочего"] == 1
    assert f.loc["2024-05-06", "первый_рабочий_после_выходных"] == 1
    assert f.loc["2024-07-17", "праздник"] == 0 and f.loc["2024-07-17", "первый_рабочий_после_выходных"] == 0
    assert f.loc["2024-07-17", "дней_до_нерабочего"] == 3 and f.loc["2024-07-17", "дней_после_нерабочего"] == 3
    assert f.loc["2024-02-24", "праздник"] == 1            # суббота в блоке 23-25 февраля
    assert (f.loc[:, "дней_до_нерабочего"] <= cal.CAP_DAYS).all()


def test_calendar_features_keep_order_index_and_duplicates():
    days = pd.Series(pd.to_datetime(["2024-01-09", "2024-01-01", "2024-01-09"]), index=[10, 5, 7])
    f = cal.calendar_features(days)
    assert f.index.equals(days.index)
    assert f["нерабочий_день"].tolist() == [0, 1, 0]
    assert f["первый_рабочий_после_выходных"].tolist() == [1, 0, 1]


def test_ordinary_weekend_is_not_holiday():
    f = cal.calendar_features(pd.to_datetime(["2024-07-20"]))
    assert f["нерабочий_день"].item() == 1 and f["праздник"].item() == 0
    assert np.issubdtype(f["праздник"].dtype, np.integer)
