import pandas as pd

from annanas import location as loc


def _eps():
    t = pd.to_datetime
    return pd.DataFrame({
        "ид_объект": [1, 1, 1, 1, 1, 1, 2],
        "ид_канала_данных": [10, 11, 11, 12, 13, 14, 20],
        "пикет": [5.0, 5.0, 5.0, 7.0, 9.0, 9.0, 1.0],
        "начало": t(["2022-03-09 10:00", "2022-03-05 08:00", "2022-03-09 23:00", "2022-03-01 00:00",
                     "2022-03-10 00:00", "2022-01-01 00:00", "2022-03-09 12:00"]),
    })


def test_rank_channels_by_recency_then_30_day_count():
    r = loc.rank_channels(_eps(), 1, pd.Timestamp("2022-03-10"), k=3)
    # 10 и 11 – вчера, у 11 два дня с отказом за 30 суток; 13 начался ровно в D 00:00 – еще не известен;
    # 14 – больше 30 суток назад
    assert r["ид_канала_данных"].tolist() == [11, 10, 12]
    assert r["ранг"].tolist() == [1, 2, 3]
    assert r["дней_с_последнего"].tolist() == [1, 1, 9]
    assert r["дней_с_отказом_30д"].tolist() == [2, 1, 1]
    assert r["пикет"].tolist() == [5.0, 5.0, 7.0]


def test_rank_channels_other_object_and_no_history():
    assert loc.rank_channels(_eps(), 2, pd.Timestamp("2022-03-10"))["ид_канала_данных"].tolist() == [20]
    assert loc.rank_channels(_eps(), 3, pd.Timestamp("2022-03-10")).empty
    assert loc.rank_channels(_eps(), 1, pd.Timestamp("2022-01-01")).empty


def test_rank_channels_full_list_and_ties_by_channel_id():
    e = _eps()
    r = loc.rank_channels(e, 1, pd.Timestamp("2022-04-20"), k=10)
    # за 30 суток ничего: порядок по давности (13 – 41 день, 10 и 11 – 42), при равенстве – по ид
    assert r["ид_канала_данных"].tolist() == [13, 10, 11, 12, 14]


def test_rank_pickets():
    r = loc.rank_channels(_eps(), 1, pd.Timestamp("2022-03-10"), k=3, level="пикет")
    # пикет 5: каналы 10 и 11, дни с отказом 05.03 и 09.03; пикет 9: 13 еще не известен, 14 давно
    assert r["пикет"].tolist() == [5.0, 7.0, 9.0]
    assert r["дней_с_последнего"].tolist() == [1, 9, 68] and r["дней_с_отказом_30д"].tolist() == [2, 1, 0]
    assert "ид_канала_данных" not in r.columns


def test_rank_grid_matches_rank_channels():
    e = _eps()
    keys = pd.DataFrame({"ид_объект": [1, 1, 1, 2, 3, 1, 1],
                         "день": pd.to_datetime(["2022-03-10", "2022-04-20", "2022-01-01", "2022-03-10",
                                                 "2022-03-10", "2022-03-06", "2021-12-01"])}, index=[5, 6, 7, 8, 9, 10, 11])
    for level, cols in [("канал", ["ид_канала_данных", "пикет"]), ("пикет", ["пикет"])]:
        g = loc.rank_grid(e, keys, k=3, level=level)
        assert g.index.equals(keys.index)
        for i, r in keys.iterrows():
            ref = loc.rank_channels(e, r["ид_объект"], r["день"], k=3, level=level)
            got = pd.DataFrame(g[i], columns=cols + ["дней_с_последнего", "дней_с_отказом_30д"])
            ref = ref.drop(columns="ранг")
            assert got.values.tolist() == ref.values.tolist(), (level, i)
