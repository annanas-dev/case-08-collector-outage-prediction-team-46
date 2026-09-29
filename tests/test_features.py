import numpy as np
import pandas as pd
import pytest

from annanas import features as fe


def _grid(objects, start, end):
    days = pd.date_range(start, end, freq="D")
    return pd.MultiIndex.from_product([objects, days], names=["ид_объект", "день"]).to_frame(index=False)


# --- Task 1: суточные ряды, окна, время с последнего -------------------------------------------------

def test_daily_counts_by_class():
    eps = pd.DataFrame({"ид_объект": [1, 1, 1, 2],
                        "начало": pd.to_datetime(["2022-01-01 10:00", "2022-01-01 23:59", "2022-01-02 00:00",
                                                  "2022-01-01 05:00"]),
                        "класс": ["отказ", "отказ", "пожар", "отказ"]})
    d = fe.daily_counts(eps, "класс", ["отказ", "пожар"])
    row = d[(d["ид_объект"] == 1) & (d["день"] == pd.Timestamp("2022-01-01"))].iloc[0]
    assert row["отказ"] == 2 and row["пожар"] == 0
    row = d[(d["ид_объект"] == 1) & (d["день"] == pd.Timestamp("2022-01-02"))].iloc[0]
    assert row["пожар"] == 1                           # 00:00 относится к новому дню


def test_daily_counts_empty_events():
    eps = pd.DataFrame({"ид_объект": [1], "начало": pd.to_datetime(["2022-01-01"]), "класс": ["газ"]})
    d = fe.daily_counts(eps, "класс", ["отказ"])
    assert list(d.columns) == ["ид_объект", "день", "отказ"] and len(d) == 0


def test_window_sums_exclude_current_day():
    grid = _grid([1], "2022-01-01", "2022-01-05")
    daily = pd.DataFrame({"ид_объект": [1, 1], "день": pd.to_datetime(["2022-01-02", "2022-01-04"]),
                          "отказ": [3, 1]})
    out = fe.window_sums(grid, daily, ["отказ"], windows=[1, 3])
    s = out.set_index("день")
    assert s.loc["2022-01-02", "отказ_1д"] == 0      # событие того же дня не видно
    assert s.loc["2022-01-03", "отказ_1д"] == 3
    assert s.loc["2022-01-05", "отказ_3д"] == 4      # дни 02, 03, 04
    assert s.loc["2022-01-01", "отказ_3д"] == 0


def test_window_sums_per_object_independent():
    grid = _grid([1, 2], "2022-01-01", "2022-01-02")
    daily = pd.DataFrame({"ид_объект": [1], "день": pd.to_datetime(["2022-01-01"]), "отказ": [5]})
    out = fe.window_sums(grid, daily, ["отказ"], windows=[1]).set_index(["ид_объект", "день"])
    assert out.loc[(2, pd.Timestamp("2022-01-02")), "отказ_1д"] == 0
    assert out.loc[(1, pd.Timestamp("2022-01-02")), "отказ_1д"] == 5


def test_window_sums_keep_grid_order_and_index():
    grid = _grid([30, 2, 17], "2022-01-01", "2022-01-10").sample(frac=1, random_state=0)
    grid.index = grid.index + 1000
    daily = pd.DataFrame({"ид_объект": [2, 30, 17], "день": pd.to_datetime(["2022-01-03"] * 3),
                          "отказ": [1, 2, 3]})
    out = fe.window_sums(grid, daily, ["отказ"], windows=[2])
    assert out.index.equals(grid.index)
    pd.testing.assert_frame_equal(out[["ид_объект", "день"]], grid[["ид_объект", "день"]])
    expected = {2: 1, 30: 2, 17: 3}
    hit = out["день"].isin(pd.to_datetime(["2022-01-04", "2022-01-05"]))
    assert (out.loc[hit, "отказ_2д"] == out.loc[hit, "ид_объект"].map(expected)).all()
    assert (out.loc[~hit, "отказ_2д"] == 0).all()


def test_window_sums_grid_with_gaps_counts_calendar_days():
    # окно – календарные дни, а не строки сетки: пропуск дней в сетке не растягивает окно
    grid = pd.DataFrame({"ид_объект": [1, 1], "день": pd.to_datetime(["2022-01-01", "2022-01-10"])})
    daily = pd.DataFrame({"ид_объект": [1, 1], "день": pd.to_datetime(["2022-01-01", "2022-01-08"]),
                          "отказ": [7, 1]})
    out = fe.window_sums(grid, daily, ["отказ"], windows=[1, 3]).set_index("день")
    assert out.loc["2022-01-10", "отказ_1д"] == 0 and out.loc["2022-01-10", "отказ_3д"] == 1


def test_days_since_last_strictly_before():
    grid = _grid([1], "2022-01-01", "2022-01-04")
    daily = pd.DataFrame({"ид_объект": [1], "день": pd.to_datetime(["2022-01-02"]), "отказ": [1]})
    out = fe.days_since_last(grid, daily, ["отказ"], cap=3650).set_index("день")
    assert out.loc["2022-01-02", "дней_с_последнего_отказ"] == 3650   # в сам день еще не известно
    assert out.loc["2022-01-03", "дней_с_последнего_отказ"] == 1
    assert out.loc["2022-01-04", "дней_с_последнего_отказ"] == 2
    assert out.loc["2022-01-02", "не_было_отказ"] == 1
    assert out.loc["2022-01-04", "не_было_отказ"] == 0


def test_days_since_last_ignores_zero_rows_and_caps():
    grid = _grid([1], "2022-01-01", "2022-01-03")
    daily = pd.DataFrame({"ид_объект": [1, 1], "день": pd.to_datetime(["2021-01-01", "2022-01-01"]),
                          "отказ": [1, 0]})
    out = fe.days_since_last(grid, daily, ["отказ"], cap=100).set_index("день")
    assert (out["дней_с_последнего_отказ"] == 100).all()      # было 365+ дней назад → потолок
    assert (out["не_было_отказ"] == 0).all()                   # но событие было


def test_valid_days_counts_unmasked_days_before_d():
    grid = _grid([1, 2], "2022-01-01", "2022-01-05")
    grid["маска"] = None
    grid.loc[(grid["ид_объект"] == 1) & (grid["день"] == pd.Timestamp("2022-01-03")), "маска"] = "нет данных"
    out = fe.valid_days(grid, windows=[1, 3]).set_index(["ид_объект", "день"])
    assert out.loc[(1, pd.Timestamp("2022-01-01")), "дней_с_данными_3д"] == 0   # дней до сетки нет
    assert out.loc[(1, pd.Timestamp("2022-01-04")), "дней_с_данными_1д"] == 0   # 03 под маской
    assert out.loc[(1, pd.Timestamp("2022-01-05")), "дней_с_данными_3д"] == 2   # 02, 04
    assert out.loc[(2, pd.Timestamp("2022-01-05")), "дней_с_данными_3д"] == 3


def _random_episodes(rng, objects, start, n_days, n):
    t0 = pd.Timestamp(start)
    return pd.DataFrame({
        "ид_объект": rng.choice(objects, n),
        "начало": t0 + pd.to_timedelta(rng.integers(0, n_days * 86400, n), unit="s"),
        "класс": rng.choice(["отказ", "пожар", "газ"], n),
    })


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_leakage_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    classes = ["отказ", "пожар"]
    eps = _random_episodes(rng, [5, 2, 9], "2022-01-01", 60, 150)
    # сетка шире эпизодов с обеих сторон, объект 7 без эпизодов, порядок строк и объектов перемешан
    grid = _grid([9, 7, 5, 2], "2021-12-20", "2022-03-15").sample(frac=1, random_state=seed)
    daily = fe.daily_counts(eps, "класс", classes)
    windows = [1, 7, 30]
    ws = fe.window_sums(grid, daily, classes, windows)
    ds = fe.days_since_last(grid, daily, classes, cap=3650)
    assert ws.index.equals(grid.index) and ds.index.equals(grid.index)

    for i, (obj, day) in enumerate(zip(grid["ид_объект"], grid["день"])):
        past = eps[(eps["ид_объект"] == obj) & (eps["начало"] < day)]   # явный фильтр «строго до D 00:00»
        for c in classes:
            pc = past[past["класс"] == c]
            for w in windows:
                expected = (pc["начало"] >= day - pd.Timedelta(days=w)).sum()
                assert ws[f"{c}_{w}д"].iat[i] == expected, (obj, day, c, w)
            if len(pc):
                assert ds[f"дней_с_последнего_{c}"].iat[i] == (day - pc["начало"].max().floor("D")).days
                assert ds[f"не_было_{c}"].iat[i] == 0
            else:
                assert ds[f"дней_с_последнего_{c}"].iat[i] == 3650
                assert ds[f"не_было_{c}"].iat[i] == 1


def test_future_events_do_not_change_past_features():
    rng = np.random.default_rng(3)
    eps = _random_episodes(rng, [1, 2], "2022-01-01", 40, 100)
    grid = _grid([1, 2], "2022-01-01", "2022-02-15")
    cut = pd.Timestamp("2022-01-20")
    full = fe.window_sums(grid, fe.daily_counts(eps, "класс", ["отказ"]), ["отказ"], [7, 30])
    trunc = fe.window_sums(grid, fe.daily_counts(eps[eps["начало"] < cut], "класс", ["отказ"]), ["отказ"], [7, 30])
    keep = grid["день"] <= cut
    pd.testing.assert_frame_equal(full[keep], trunc[keep])


# --- Task 2: открытые интервалы, соседи, нормировка -----------------------------------------------

def test_active_at_midnight():
    grid = _grid([1], "2022-01-01", "2022-01-04")
    eps = pd.DataFrame({"ид_объект": [1, 1], "ид_канала_данных": [10, 11],
                        "начало": pd.to_datetime(["2022-01-01 12:00", "2022-01-02 01:00"]),
                        "конец": pd.to_datetime(["2022-01-03 06:00", pd.NaT])})
    out = fe.active_at_midnight(grid, eps, "каналов_выведено").set_index("день")
    assert out["каналов_выведено"].tolist() == [0, 1, 2, 1]   # 01: ничего; 02: канал 10; 03: 10 и 11; 04: 11


def test_active_at_midnight_boundaries_and_order():
    grid = _grid([2, 1], "2022-01-01", "2022-01-03").sample(frac=1, random_state=0)
    eps = pd.DataFrame({"ид_объект": [1, 1, 1, 3],
                        "начало": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 10:00", "2022-01-02 00:00",
                                                  "2022-01-01 10:00"]),
                        "конец": pd.to_datetime(["2022-01-02 00:00", "2022-01-01 10:00", "2022-01-02 00:00",
                                                 pd.NaT])})
    out = fe.active_at_midnight(grid, eps, "открыто")
    assert out.index.equals(grid.index)
    s = out.set_index(["ид_объект", "день"])["открыто"]
    # конец ровно в D 00:00 – уже закрыт; начало ровно в D 00:00 – еще не известно; нулевая длина – не открыт
    assert s.loc[(1, pd.Timestamp("2022-01-02"))] == 0
    assert s.loc[(1, pd.Timestamp("2022-01-03"))] == 0
    assert (s.loc[2] == 0).all()                       # объект без интервалов


def test_sibling_sums_exclude_self():
    feats = pd.DataFrame({"ид_объект": [1, 2, 3], "день": pd.to_datetime(["2022-01-01"] * 3),
                          "отказ_7д": [5, 2, 7]})
    parents = pd.Series({1: 100, 2: 100, 3: 200}, name="родитель")
    out = fe.sibling_sums(feats, ["отказ_7д"], parents).set_index("ид_объект")
    assert out.loc[1, "соседи_отказ_7д"] == 2 and out.loc[3, "соседи_отказ_7д"] == 0


def test_sibling_sums_no_parent_is_nan():
    feats = pd.DataFrame({"ид_объект": [1, 4, 5], "день": pd.to_datetime(["2022-01-01"] * 3),
                          "отказ_7д": [5, 2, 7]})
    out = fe.sibling_sums(feats, ["отказ_7д"], pd.Series({1: 100})).set_index("ид_объект")
    assert out.loc[1, "соседи_отказ_7д"] == 0
    assert np.isnan(out.loc[4, "соседи_отказ_7д"])    # объекты без родителя не соседи друг другу


def test_per_channel_rate():
    feats = pd.DataFrame({"ид_объект": [1, 2], "отказ_30д": [10, 10]})
    n_ch = pd.Series({1: 5, 2: 0})
    out = fe.per_channel(feats, "отказ_30д", n_ch)
    assert out.tolist()[0] == 2.0 and np.isnan(out.tolist()[1])


# --- Task 3: проверки сигнала и устойчивости -------------------------------------------------------

def test_average_precision_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert fe.average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert abs(fe.average_precision(y, np.zeros(4)) - 0.5) < 1e-9     # одинаковые оценки = доля положительных


def test_average_precision_ties_are_one_threshold():
    # пороги 0,9 / 0,8 / 0,5: (P, R) = (1, 1/3), (2/3, 2/3), (3/5, 1) → AP = (1 + 2/3 + 3/5) / 3 = 34/45
    y = np.array([1, 0, 1, 0, 1])
    s = np.array([0.9, 0.8, 0.8, 0.5, 0.5])
    assert abs(fe.average_precision(y, s) - 34 / 45) < 1e-12
    # порядок строк внутри связки не влияет
    assert fe.average_precision(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert fe.average_precision(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_average_precision_nan_scores_and_no_positives():
    y = np.array([1, 0, 1, 0])
    assert fe.average_precision(y, np.array([0.9, np.nan, 0.8, 0.1])) == 1.0   # NaN-оценки выбрасываются
    assert np.isnan(fe.average_precision(np.zeros(3), np.array([0.1, 0.2, 0.3])))


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert fe.psi(a, a) < 1e-9
    assert fe.psi(a, a + 2) > 0.25


def test_psi_constant_expected():
    assert fe.psi(np.zeros(10), np.zeros(5)) == 0.0
    assert 0.25 < fe.psi(np.zeros(10), np.ones(5)) < np.inf   # конечный, чтобы сдвиги можно было сравнивать


def test_signal_table_marks_useless_feature():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    X = pd.DataFrame({"хороший": y + rng.normal(0, 0.3, 2000), "шум": rng.normal(size=2000)})
    t = fe.signal_table(X, y).set_index("признак")
    assert t.loc["хороший", "подъем_pr_auc"] > 1.5 and t.loc["шум", "подъем_pr_auc"] < 1.1


def test_signal_table_lift_on_non_missing_rows():
    # шум, у которого пропуски чаще у отрицательных: подъем к общей доле был бы ложным (~1,33)
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 4000)
    v = rng.normal(size=4000)
    v[(y == 0) & (rng.random(4000) < 0.5)] = np.nan
    t = fe.signal_table(pd.DataFrame({"шум": v}), y).set_index("признак")
    assert t.loc["шум", "подъем_pr_auc"] < 1.1
    assert 0.2 < t.loc["шум", "доля_пропусков"] < 0.3


def test_psi_binary_and_zero_heavy():
    a = np.r_[np.zeros(600), np.ones(400)]
    assert fe.psi(a, a) < 1e-9
    assert fe.psi(a, np.ones(1000)) > 0.25          # флаг: все единицы вместо 40 %
    assert fe.psi(np.zeros(100), np.ones(100)) > 0.25  # константа в expected, другое значение в actual
    rng = np.random.default_rng(2)
    z = np.r_[np.zeros(7000), rng.exponential(size=3000)]
    assert fe.psi(z, np.r_[np.zeros(3000), rng.exponential(size=7000)]) > 0.25   # нули не сливаются с хвостом


# --- Часы с последнего эпизода и возраст открытого ------------------------------------------------

def test_hours_since_last_strictly_before_midnight():
    grid = _grid([1, 2], "2022-01-01", "2022-01-04")
    ev = pd.DataFrame({"ид_объект": [1, 1, 1],
                       "момент": pd.to_datetime(["2022-01-01 18:00", "2022-01-03 00:00", "2022-01-02 23:30"]),
                       "класс": ["отказ", "отказ", "пожар"]})
    out = fe.hours_since_last(grid, ev, ["отказ", "пожар"], cap_h=720).set_index(["ид_объект", "день"])
    s = out["часов_с_последнего_отказ"]
    assert s.loc[(1, pd.Timestamp("2022-01-01"))] == 720          # событие дня D еще не известно
    assert s.loc[(1, pd.Timestamp("2022-01-02"))] == 6
    assert s.loc[(1, pd.Timestamp("2022-01-03"))] == 30           # момент ровно D 00:00 – не до D
    assert s.loc[(1, pd.Timestamp("2022-01-04"))] == 24
    assert out.loc[(1, pd.Timestamp("2022-01-03")), "часов_с_последнего_пожар"] == 0.5
    assert (out.loc[2, "часов_с_последнего_отказ"] == 720).all()   # объект без событий


def test_hours_since_last_cap_group_and_order():
    grid = _grid([3, 1], "2022-03-01", "2022-03-02").sample(frac=1, random_state=1)
    grid.index = grid.index + 50
    ev = pd.DataFrame({"ид_объект": [1, 1], "момент": pd.to_datetime(["2022-01-01 00:00", "2022-02-28 12:00"]),
                       "класс": ["отказ", "пожар"]})
    out = fe.hours_since_last(grid, ev, {"любой": ["отказ", "пожар"], "отказ": ["отказ"]}, cap_h=100)
    assert out.index.equals(grid.index)
    s = out.set_index(["ид_объект", "день"])
    assert s.loc[(1, pd.Timestamp("2022-03-01")), "часов_с_последнего_любой"] == 12
    assert s.loc[(1, pd.Timestamp("2022-03-01")), "часов_с_последнего_отказ"] == 100   # 1400 ч → потолок


def test_open_age_at_midnight_boundaries():
    grid = _grid([1], "2022-01-01", "2022-01-05")
    eps = pd.DataFrame({"ид_объект": [1, 1, 1, 1],
                        "начало": pd.to_datetime(["2022-01-01 18:00", "2022-01-02 00:00", "2022-01-03 12:00",
                                                  "2022-01-04 06:00"]),
                        "конец": pd.to_datetime(["2022-01-03 00:00", "2022-01-02 06:00", pd.NaT,
                                                 "2022-01-04 07:00"]),
                        "класс": ["отказ", "отказ", "отказ", "отказ"]})
    out = fe.open_age_at_midnight(grid, eps, ["отказ"]).set_index("день")["часов_открыт_отказ_макс"]
    # 02: открыт первый (6 ч); начало ровно 02 00:00 – не открыт на 02; 03: первый закрыт ровно в 00:00
    assert out.tolist() == [0, 6, 0, 12, 36]


def _random_intervals(rng, objects, start, n_days, n):
    t0 = pd.Timestamp(start)
    begin = t0 + pd.to_timedelta(rng.integers(0, n_days * 24, n) * 3600 + rng.choice([0, 1800], n), unit="s")
    dur = pd.to_timedelta(rng.choice([0, 3600, 86400, 3 * 86400, 10 * 86400], n) + rng.choice([0, 1800], n), unit="s")
    end = pd.Series(begin + dur)
    end[rng.random(n) < 0.15] = pd.NaT
    return pd.DataFrame({"ид_объект": rng.choice(objects, n), "начало": begin, "конец": end.to_numpy(),
                         "класс": rng.choice(["отказ", "пожар"], n)})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hours_features_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    eps = _random_intervals(rng, [5, 2, 9], "2022-01-01", 40, 80)
    grid = _grid([9, 7, 5, 2], "2021-12-25", "2022-02-20").sample(frac=1, random_state=seed)
    ev = eps.rename(columns={"начало": "момент"})
    hs = fe.hours_since_last(grid, ev, ["отказ", "пожар"], cap_h=720)
    oa = fe.open_age_at_midnight(grid, eps, ["отказ", "пожар"])
    act = fe.active_at_midnight(grid, eps[eps["класс"] == "отказ"], "открыто")
    assert hs.index.equals(grid.index) and oa.index.equals(grid.index)
    for i, (obj, day) in enumerate(zip(grid["ид_объект"], grid["день"])):
        for c in ["отказ", "пожар"]:
            e = eps[(eps["ид_объект"] == obj) & (eps["класс"] == c)]
            past = e[e["начало"] < day]
            exp = min((day - past["начало"].max()) / pd.Timedelta(hours=1), 720) if len(past) else 720
            assert hs[f"часов_с_последнего_{c}"].iat[i] == pytest.approx(exp), (obj, day, c)
            open_ = past[past["конец"].isna() | (past["конец"] > day)]
            exp = ((day - open_["начало"].min()) / pd.Timedelta(hours=1)) if len(open_) else 0
            assert oa[f"часов_открыт_{c}_макс"].iat[i] == pytest.approx(exp), (obj, day, c)
            if c == "отказ":   # согласовано с active_at_midnight: открыт ⇔ есть открытый интервал
                assert (act["открыто"].iat[i] > 0) == (len(open_) > 0)


# --- Доли дней с правильным знаменателем ----------------------------------------------------------

def test_window_shares_quiet_days_count_in_denominator():
    # дискретный объект: события 2 дня из 30, остальные дни тихие, но без маски → доля 2/30, а не 2/2
    grid = _grid([1], "2022-01-01", "2022-02-15")
    grid["маска"] = None
    daily = pd.DataFrame({"ид_объект": [1, 1], "день": pd.to_datetime(["2022-01-10", "2022-01-20"]),
                          "дней_отказ": [1, 1]})
    out = fe.window_shares(grid, daily, ["дней_отказ"], [30]).set_index("день")
    assert out.loc["2022-02-01", "доля_дней_отказ_30д"] == pytest.approx(2 / 30)
    vd = fe.valid_days(grid, [30]).set_index("день")
    assert vd.loc["2022-02-01", "дней_с_данными_30д"] == 30


def test_window_shares_masked_days_leave_numerator_and_denominator():
    grid = _grid([1, 2], "2022-01-01", "2022-01-11")
    grid["маска"] = None
    masked = (grid["ид_объект"] == 1) & grid["день"].between("2022-01-03", "2022-01-04")
    grid.loc[masked, "маска"] = "объект не на связи"
    grid.loc[grid["ид_объект"] == 2, "маска"] = "объект не подключен"
    daily = pd.DataFrame({"ид_объект": [1, 1, 1], "день": pd.to_datetime(["2022-01-02", "2022-01-03", "2021-12-31"]),
                          "дней_отказ": [1, 1, 1]})
    out = fe.window_shares(grid, daily, ["дней_отказ"], [10]).set_index(["ид_объект", "день"])
    # D = 11: окно 01-10, маска 03-04 → 8 дней, событие 03 под маской не считается, 31.12 вне сетки
    assert out.loc[(1, pd.Timestamp("2022-01-11")), "доля_дней_отказ_10д"] == pytest.approx(1 / 8)
    assert np.isnan(out.loc[(1, pd.Timestamp("2022-01-01")), "доля_дней_отказ_10д"])   # окно пустое
    assert out.loc[2, "доля_дней_отказ_10д"].isna().all()                              # все дни под маской
    assert out.index.equals(pd.MultiIndex.from_frame(grid[["ид_объект", "день"]]))


def test_window_shares_masks_history_separate_from_grid():
    # сетка из одних суток прогноза: доли и знаменатель берутся из отдельной истории масок, как на полной сетке
    full = _grid([1, 2], "2022-01-01", "2022-01-11")
    full["маска"] = None
    full.loc[(full["ид_объект"] == 1) & (full["день"] == pd.Timestamp("2022-01-05")), "маска"] = "сбой"
    daily = pd.DataFrame({"ид_объект": [1, 1, 2], "день": pd.to_datetime(["2022-01-02", "2022-01-05", "2022-01-09"]),
                          "дней_отказ": [1, 1, 1]})
    day = full[full["день"] == pd.Timestamp("2022-01-11")].reset_index(drop=True)
    one = fe.window_shares(day, daily, ["дней_отказ"], [7, 10], masks=full)
    ref = fe.window_shares(full, daily, ["дней_отказ"], [7, 10])
    ref = ref[ref["день"] == pd.Timestamp("2022-01-11")].reset_index(drop=True)
    pd.testing.assert_frame_equal(one, ref)
    vd = fe.valid_days(day, [10], masks=full)
    assert vd["дней_с_данными_10д"].tolist() == [9, 10]
