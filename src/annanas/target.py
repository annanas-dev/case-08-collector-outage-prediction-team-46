"""Целевая таблица «объект × сутки». Выбор единицы, горизонта, классов и масок обоснован в notebooks/03_target."""
import numpy as np
import pandas as pd

from annanas import smvu

HORIZON = pd.Timedelta(hours=24)

# Общие маски на все объекты: (название, начало, конец), выведены в notebooks/03_target (раздел 6). Переход СМВУ
# маскируется с суток перед границей из annanas.smvu (пусконаладка).
GLOBAL_MASKS = [
    ("переход СМВУ", pd.Timestamp(smvu.ПЕРЕХОД_НАЧАЛО).floor("D") - pd.Timedelta(days=1), pd.Timestamp(smvu.ПЕРЕХОД_КОНЕЦ)),
    ("сбой июль 2021", pd.Timestamp("2021-06-29"), pd.Timestamp("2021-07-03")),
    ("обрыв апрель 2024", pd.Timestamp("2024-04-06 01:00"), pd.Timestamp("2024-04-12")),
    ("нет данных 2026-06-01", pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02")),
]
# День смены разметки флага `тревожное` (notebooks/03_target, раздел 1.1): столбец `эпоха_флага`
FLAG_CHANGE = pd.Timestamp("2025-10-29")
BEFORE_FLAG, AFTER_FLAG = "до смены разметки", "после смены разметки"


def flag_epoch(days) -> np.ndarray:
    """`эпоха_флага` для суток."""
    return np.where(pd.to_datetime(days) < FLAG_CHANGE, BEFORE_FLAG, AFTER_FLAG)


def build_grid(objects, start, end) -> pd.DataFrame:
    """Все пары «объект × сутки» с `start` по `end` включительно; `день` – момент прогноза 00:00."""
    days = pd.date_range(start, end, freq="D")
    return (pd.MultiIndex.from_product([sorted(objects), days], names=["ид_объект", "день"])
            .to_frame(index=False))


def add_labels(grid: pd.DataFrame, episodes: pd.DataFrame, classes, t_flap) -> pd.DataFrame:
    """`цель_<класс>` = 1, если в [день, день + 24 ч) на объекте начинается эпизод класса длительностью ≥ t_flap
    (незакрытый эпизод считается недребезговым). episodes: `ид_объект, начало, длительность_с, класс`."""
    dur = pd.to_numeric(episodes["длительность_с"], errors="coerce")
    eps = episodes.loc[dur.isna() | (dur >= t_flap), ["ид_объект", "начало", "класс"]].copy()
    eps["день"] = eps["начало"].dt.floor("D")  # горизонт ровно сутки от 00:00 → день начала эпизода
    out = grid.copy()
    for c in classes:
        hit = eps.loc[eps["класс"] == c, ["ид_объект", "день"]].drop_duplicates().assign(**{f"цель_{c}": 1})
        out = out.merge(hit, on=["ид_объект", "день"], how="left")
        out[f"цель_{c}"] = out[f"цель_{c}"].fillna(0).astype("int8")
    return out


def add_masks(grid: pd.DataFrame, masks) -> pd.DataFrame:
    """masks: [(название, начало, конец)]; день маскируется, если [день, день + 24 ч) пересекает
    [начало, конец). При пересечении нескольких масок остается первая по списку."""
    out = grid.copy()
    out["маска"] = pd.Series(None, index=out.index, dtype=object)
    for name, start, end in masks:
        hit = (out["день"] < end) & (out["день"] + HORIZON > start) & out["маска"].isna()
        out.loc[hit, "маска"] = name
    out["маска"] = out["маска"].astype(object).where(out["маска"].notna(), None)  # pandas пишет NaN вместо None
    return out


NOT_CONNECTED, OFFLINE = "объект не подключен", "объект не на связи"


def connection_mask(grid: pd.DataFrame, daily_records: pd.DataFrame, min_median=200, window=30,
                    gap_days=None) -> pd.Series:
    """Маска прогнозной популяции по связи объекта, только по записям строго до D 00:00:

    - «объект не подключен» – у объекта еще не было ни одной записи (день первой записи ≥ D);
    - «объект не на связи» – объект постоянно пишущий (медиана записей в сутки за [D − window, D − 1] > min_median;
      дни без записей – нули, дни до первой записи не входят) и молчит больше 24 ч к D 00:00, то есть за
      сутки D − 1 нет ни одной записи.

    Тихий день объекта, который пишет редко, не маскируется: для него отсутствие записей – нормальный день
    без событий (отрицательный пример). daily_records: `ид_объект, день, записей` за всю историю (повторы
    складываются). Возвращает Series с индексом grid: название маски или None.

    gap_days – сутки, когда журнал пуст целиком (пропуск выгрузки, а не тишина объектов): медиана берется по
    последним window суткам с данными, пропуски не считаются нулями. Молчание за D − 1 проверяется как
    обычно. Без gap_days (обучение) – календарные window суток."""
    rec = daily_records.assign(день=pd.to_datetime(daily_records["день"]).dt.floor("D"))
    rec = rec.loc[rec["записей"] > 0].groupby(["ид_объект", "день"])["записей"].sum()
    wide = rec.unstack("ид_объект")
    grid_days = pd.to_datetime(grid["день"])
    days = pd.date_range(min(wide.index.min(), grid_days.min()) if len(wide) else grid_days.min(),
                         grid_days.max(), freq="D")
    wide = wide.reindex(days)
    first = wide.notna().idxmax()                                   # день первой записи объекта
    after_first = pd.DataFrame(days.to_numpy()[:, None] >= first.to_numpy()[None, :], index=days,
                               columns=wide.columns)
    counts = wide.fillna(0).where(after_first)                      # до первой записи – NaN, не нули
    if gap_days is None or not len(gap_days):
        median = counts.rolling(window, min_periods=1).median().shift(1)   # [D − window, D − 1]
    else:                                                           # window суток с данными до D
        data = counts.drop(index=pd.DatetimeIndex(gap_days).normalize(), errors="ignore")
        median = data.rolling(window, min_periods=1).median().reindex(days).ffill().shift(1)
    silent = counts.shift(1).fillna(0).eq(0)                        # за сутки D − 1 записей нет
    offline = (median > min_median) & silent
    key = pd.MultiIndex.from_arrays([grid["ид_объект"], grid_days.dt.floor("D")])
    first_g = grid["ид_объект"].map(first).to_numpy(dtype="datetime64[ns]")
    not_connected = np.isnat(first_g) | (first_g >= grid_days.dt.floor("D").to_numpy())
    off = offline.stack().reindex(pd.MultiIndex.from_arrays([key.get_level_values(1), key.get_level_values(0)]))
    off = off.fillna(False).to_numpy(dtype=bool)
    out = np.where(not_connected, NOT_CONNECTED, np.where(off, OFFLINE, None))
    return pd.Series(out, index=grid.index, dtype=object, name="маска")


def add_split(grid: pd.DataFrame, valid_start, test_start) -> pd.DataFrame:
    """`выборка`: обучение до valid_start, валидация до test_start, дальше тест."""
    out = grid.copy()
    out["выборка"] = "обучение"
    out.loc[out["день"] >= pd.Timestamp(valid_start), "выборка"] = "валидация"
    out.loc[out["день"] >= pd.Timestamp(test_start), "выборка"] = "тест"
    return out
