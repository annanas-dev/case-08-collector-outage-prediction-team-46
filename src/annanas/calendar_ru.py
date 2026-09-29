"""Производственный календарь РФ 2019-2026 и календарные признаки дня.

Данные – `data/production_calendar_ru.csv` (источник и дата загрузки – в шапке файла): только особые дни,
обычные субботы и воскресенья вычисляются. Календарь на год вперед утверждается постановлением Правительства,
поэтому признаки дня D известны заранее и утечкой не являются. За пределами COVERED_YEARS известны только
обычные выходные (праздники не учитываются) – для новых лет нужно дополнить файл."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CALENDAR_FILE = Path(__file__).parent / "data" / "production_calendar_ru.csv"
COVERED_YEARS = range(2019, 2027)
DAY_TYPES = ["праздник", "перенесенный_выходной", "нерабочий_по_указу", "рабочий_выходной", "сокращенный"]
NON_WORKING_TYPES = ["праздник", "перенесенный_выходной", "нерабочий_по_указу"]
CAP_DAYS = 10   # «дней до / после нерабочего» обрезаются: дальше 10 суток разница не важна


@lru_cache(maxsize=1)
def load_calendar() -> pd.DataFrame:
    """Особые дни: `дата, тип, название` (тип из DAY_TYPES)."""
    t = pd.read_csv(CALENDAR_FILE, comment="#", parse_dates=["дата"], dtype={"тип": str, "название": str})
    return t


def _days(dates) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(dates)).to_numpy()).normalize()


def is_non_working(dates) -> np.ndarray:
    """Нерабочий день: суббота или воскресенье, если не объявлены рабочими, либо праздник, перенесенный
    выходной или нерабочий день по указу."""
    d = _days(dates)
    t = load_calendar()
    special = d.isin(t.loc[t["тип"].isin(NON_WORKING_TYPES), "дата"])
    working_weekend = d.isin(t.loc[t["тип"] == "рабочий_выходной", "дата"])
    return np.asarray(((d.dayofweek >= 5) & ~working_weekend) | special)


def calendar_features(days) -> pd.DataFrame:
    """Признаки самого дня D (индекс – как у `days`, если это Series):

    - `нерабочий_день` – D нерабочий (is_non_working);
    - `праздник` – D входит в блок подряд идущих нерабочих дней, где есть хотя бы один праздник, перенос или
      нерабочий день по указу (например, суббота 24.02.2024 в блоке 23-25 февраля); обычные выходные – 0;
    - `дней_до_нерабочего` – дней от D до ближайшего нерабочего дня ≥ D (0 – D сам нерабочий), не больше CAP_DAYS;
    - `дней_после_нерабочего` – дней с последнего нерабочего дня строго до D, не больше CAP_DAYS;
    - `первый_рабочий_после_выходных` – D рабочий, D − 1 нерабочий."""
    index = days.index if isinstance(days, pd.Series) else pd.RangeIndex(len(days))
    d = _days(days)
    if len(d) == 0:
        cols = ["нерабочий_день", "праздник", "дней_до_нерабочего", "дней_после_нерабочего",
                "первый_рабочий_после_выходных"]
        return pd.DataFrame({c: pd.Series(dtype="int64") for c in cols}, index=index)
    span = pd.date_range(d.min() - pd.Timedelta(days=CAP_DAYS + 1), d.max() + pd.Timedelta(days=CAP_DAYS + 1))
    nw = is_non_working(span)
    t = load_calendar()
    special = np.asarray(span.isin(t.loc[t["тип"].isin(NON_WORKING_TYPES), "дата"]) & nw)
    # блоки подряд идущих нерабочих дней: праздничный, если в блоке есть особый нерабочий день
    block = np.cumsum(np.r_[True, nw[1:] != nw[:-1]])
    holiday_block = pd.Series(special).groupby(block).transform("max").to_numpy() & nw
    pos = np.arange(len(span))
    big = len(span) + CAP_DAYS
    nxt = np.minimum.accumulate(np.where(nw, pos, big)[::-1])[::-1]              # ближайший нерабочий ≥ D
    prv = np.maximum.accumulate(np.where(nw, pos, -big))
    prv_strict = np.r_[-big, prv[:-1]]                                            # последний нерабочий < D
    i = (d - span[0]).days.to_numpy()
    return pd.DataFrame({
        "нерабочий_день": nw[i].astype(int),
        "праздник": holiday_block[i].astype(int),
        "дней_до_нерабочего": np.minimum(nxt[i] - i, CAP_DAYS),
        "дней_после_нерабочего": np.minimum(i - prv_strict[i], CAP_DAYS),
        "первый_рабочий_после_выходных": (~nw[i] & nw[i - 1]).astype(int),
    }, index=index)
