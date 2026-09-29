import datetime as dt

import pandas as pd

from annanas import smvu


def test_boundaries():
    assert smvu.ПЕРЕХОД_НАЧАЛО == dt.datetime(2021, 4, 22, 7, 0)
    assert smvu.ПЕРЕХОД_КОНЕЦ == dt.datetime(2021, 6, 8, 10, 0)


def test_add_smvu_period(spark):
    df = spark.createDataFrame(
        [(dt.datetime(2021, 4, 22, 6, 59, 59),), (dt.datetime(2021, 4, 22, 7, 0),),
         (dt.datetime(2021, 6, 8, 9, 59, 59),), (dt.datetime(2021, 6, 8, 10, 0),)],
        "`дата_время` timestamp",
    )
    got = [r["период_смву"] for r in smvu.add_smvu_period(df).orderBy("дата_время").collect()]
    assert got == ["до", "переход", "переход", "после"]


def test_add_smvu_period_custom_column(spark):
    df = spark.createDataFrame([(dt.datetime(2020, 1, 1),)], "`час` timestamp")
    assert smvu.add_smvu_period(df, time_col="час").first()["период_смву"] == "до"


def test_period_of():
    s = pd.Series(pd.to_datetime(["2021-04-22 06:59:59", "2021-04-22 07:00:00", "2021-06-08 09:59:59",
                                  "2021-06-08 10:00:00", None]))
    got = smvu.period_of(s)
    assert list(got[:4]) == ["до", "переход", "переход", "после"]
    assert pd.isna(got.iloc[4])
    assert got.index.equals(s.index)
