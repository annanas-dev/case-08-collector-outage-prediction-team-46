"""Период внедрения новой СМВУ в 2021 году.

По словам заказчика, в 2021 году старая и новая СМВУ какое-то время работали параллельно.
Границы переходного периода [ПЕРЕХОД_НАЧАЛО, ПЕРЕХОД_КОНЕЦ) выведены и обоснованы по данным
в ноутбуке 02_eda (раздел о внедрении СМВУ в 2021 году); ноутбук проверяет assert-ом,
что константы совпадают с тем, что показывают данные.
"""
import datetime as dt

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

ПЕРЕХОД_НАЧАЛО = dt.datetime(2021, 4, 22, 7, 0)
ПЕРЕХОД_КОНЕЦ = dt.datetime(2021, 6, 8, 10, 0)
# ASCII-синонимы
TRANSITION_START = ПЕРЕХОД_НАЧАЛО
TRANSITION_END = ПЕРЕХОД_КОНЕЦ

ДО, ПЕРЕХОД, ПОСЛЕ = "до", "переход", "после"


def add_smvu_period(df: DataFrame, time_col: str = "дата_время") -> DataFrame:
    """Добавляет столбец `период_смву`: 'до' / 'переход' / 'после' по времени `time_col` (московское)."""
    ts = F.col(time_col)
    return df.withColumn(
        "период_смву",
        F.when(ts < F.lit(ПЕРЕХОД_НАЧАЛО), ДО)
         .when(ts < F.lit(ПЕРЕХОД_КОНЕЦ), ПЕРЕХОД)
         .when(ts.isNotNull(), ПОСЛЕ),
    )


def period_of(series: pd.Series) -> pd.Series:
    """То же для pandas: метка периода СМВУ для каждого момента, NaN для пропусков."""
    ts = pd.to_datetime(series)
    labels = np.select(
        [ts < ПЕРЕХОД_НАЧАЛО, ts < ПЕРЕХОД_КОНЕЦ, ts.notna()],
        [ДО, ПЕРЕХОД, ПОСЛЕ],
        default=None,
    )
    return pd.Series(labels, index=series.index, dtype=object).where(ts.notna(), np.nan)
