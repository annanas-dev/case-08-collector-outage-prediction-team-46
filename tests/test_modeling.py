import time

import numpy as np
import pandas as pd
import pytest

from annanas import modeling as md


def _frame():
    days = pd.to_datetime(["2022-01-01"] * 4 + ["2022-01-02"] * 4)
    return pd.DataFrame({"ид_объект": [1, 2, 3, 4] * 2, "день": days,
                         "y": [1, 0, 0, 1, 0, 0, 1, 0],
                         "p": [0.9, 0.8, 0.1, 0.2, 0.5, 0.4, 0.3, 0.1]})


def test_precision_recall_at_top_n():
    r = md.precision_recall_at_top(_frame(), "y", "p", n=2)
    # день 1: топ-2 = объекты 1, 2 → 1 попадание из 2 инцидентов; день 2: топ-2 = 1, 2 → 0 из 1
    assert r["precision"] == pytest.approx((0.5 + 0.0) / 2)
    assert r["recall"] == pytest.approx((0.5 + 0.0) / 2)
    assert r["recall_pooled"] == pytest.approx(1 / 3)    # пулом: 1 попадание из 3 инцидентов


def test_recall_pooled_differs_from_daily_mean():
    # день 1: 1 инцидент, пойман; день 2: 3 инцидента, пойман 1 → по дням (1 + 1/3) / 2, пулом 2/4
    f = pd.DataFrame({"ид_объект": [1, 2, 1, 2, 3, 4], "день": pd.to_datetime(["2022-01-01"] * 2 + ["2022-01-02"] * 4),
                      "y": [1, 0, 1, 1, 1, 0], "p": [0.9, 0.1, 0.9, 0.2, 0.1, 0.05]})
    r = md.precision_recall_at_top(f, "y", "p", n=1)
    assert r["recall"] == pytest.approx((1 + 1 / 3) / 2) and r["recall_pooled"] == pytest.approx(0.5)
    f["y"] = 0
    assert np.isnan(md.precision_recall_at_top(f, "y", "p", n=1)["recall_pooled"])


def test_recall_ignores_days_without_incidents():
    f = _frame()
    f.loc[f["день"] == pd.Timestamp("2022-01-02"), "y"] = 0
    assert md.precision_recall_at_top(f, "y", "p", n=2)["recall"] == pytest.approx(0.5)


def test_precision_recall_at_top_ties_break_by_object_and_ignore_row_order():
    # одинаковые вероятности: в топ-1 идет меньший ид_объект (3), а не тот, что раньше в таблице
    f = pd.DataFrame({"ид_объект": [5, 3, 4], "день": pd.to_datetime(["2022-01-01"] * 3),
                      "y": [1, 0, 0], "p": [0.5, 0.5, 0.1]})
    r = md.precision_recall_at_top(f, "y", "p", n=1)
    assert r == {"precision": 0.0, "recall": 0.0, "recall_pooled": 0.0}
    r2 = md.precision_recall_at_top(f.iloc[::-1], "y", "p", n=1)
    assert r2 == r


def test_precision_at_top_day_with_fewer_objects_than_n():
    # во второй день всего 1 объект: точность дня = доля попаданий среди тех, кто есть (1/1), а не 1/n
    f = pd.DataFrame({"ид_объект": [1, 2, 3, 1], "день": pd.to_datetime(["2022-01-01"] * 3 + ["2022-01-02"]),
                      "y": [1, 0, 0, 1], "p": [0.9, 0.5, 0.1, 0.7]})
    r = md.precision_recall_at_top(f, "y", "p", n=2)
    assert r["precision"] == pytest.approx((0.5 + 1.0) / 2)
    assert r["recall"] == pytest.approx(1.0)


def test_brier_and_reliability():
    y = np.array([0, 0, 1, 1]); p = np.array([0.0, 0.0, 1.0, 1.0])
    assert md.brier(y, p) == 0.0
    t = md.reliability_table(np.r_[y, y], np.r_[p, p], bins=2)
    assert t["доля_фактическая"].tolist() == [0.0, 1.0]
    assert t["количество"].tolist() == [4, 4]


def test_block_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-01-01", periods=200).repeat(20)
    y = rng.integers(0, 2, len(days)); p = y * 0.6 + rng.random(len(days)) * 0.4
    frame = pd.DataFrame({"день": days, "y": y, "p": p})
    est, lo, hi = md.block_bootstrap(frame, lambda f: md.average_precision(f["y"], f["p"]), block_days=7, n=200, seed=1)
    assert lo <= est <= hi and hi - lo < 0.2


def test_block_bootstrap_keeps_repeated_blocks_as_separate_days():
    # блок, выпавший дважды, не должен слипаться в одни сутки с удвоенным числом объектов (иначе top-N врет)
    days = pd.date_range("2025-01-01", periods=30).repeat(5)
    frame = pd.DataFrame({"день": days, "y": 0})
    est, lo, hi = md.block_bootstrap(frame, lambda f: f.groupby("день").size().max(), block_days=7, n=50, seed=0)
    assert est == lo == hi == 5


def test_block_bootstrap_is_fast():
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-01-01", periods=365).repeat(50)
    frame = pd.DataFrame({"день": days, "y": rng.integers(0, 2, len(days)), "p": rng.random(len(days))})
    t0 = time.perf_counter()
    md.block_bootstrap(frame, lambda f: md.average_precision(f["y"], f["p"]), block_days=7, n=1000, seed=0)
    assert time.perf_counter() - t0 < 30


def test_working_rows_and_features():
    fm = pd.DataFrame({"маска": [None, "переход СМВУ", None], "выборка": ["обучение"] * 3,
                       "a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 0.0]})
    cat = pd.DataFrame({"признак": ["a", "b"], "решение": ["оставить", "убрать"]})
    assert md.feature_list(cat) == ["a"]
    assert len(md.working_rows(fm, "обучение")) == 2


def test_rolling_folds_are_time_ordered():
    days = pd.Series(pd.date_range("2019-01-01", "2024-12-31", freq="D"))
    folds = md.rolling_folds(days, valid_years=[2022, 2023, 2024])
    for train_idx, val_idx, year in folds:
        assert days[train_idx].max() < pd.Timestamp(f"{year}-01-01")
        assert (days[val_idx].dt.year == year).all()


def test_fit_and_predict_lgbm_learns_signal():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=3000), "шум": rng.normal(size=3000)})
    y = (X["a"] + rng.normal(0, 0.5, 3000) > 1).astype(int)
    m = md.fit_lgbm(X, y, params={"num_leaves": 7, "learning_rate": 0.1, "verbose": -1}, n_estimators=100)
    p = m.predict(X)
    assert md.average_precision(y, p) > 0.8


def test_fit_lgbm_early_stops_and_reports_best_iteration():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=4000), "шум": rng.normal(size=4000)})
    y = (X["a"] + rng.normal(0, 0.5, 4000) > 1).astype(int)
    m, best, score = md.fit_lgbm_early(X[:3000], y[:3000], X[3000:], y[3000:],
                                       {"num_leaves": 7, "learning_rate": 0.3}, max_rounds=1000, patience=20)
    assert 1 <= best < 1000
    assert score == pytest.approx(md.average_precision(y[3000:], md.predict_lgbm(m, X[3000:], num_iteration=best)),
                                  abs=1e-6)


def test_fit_lgbm_uses_all_cores_by_default():
    import os
    X = pd.DataFrame({"a": np.arange(200.0)}); y = (X["a"] > 100).astype(int)
    m = md.fit_lgbm(X, y, {"verbose": -1}, 5)
    assert m.params["num_threads"] == os.cpu_count()


def test_fit_lgbm_cyrillic_and_special_feature_names():
    # кириллицу LightGBM принимает как есть; символы JSON ([]{}",:) – нет, их заменяем, порядок столбцов тот же
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"доля_дней_отказ_30д": rng.normal(size=1000), "a:b [1]": rng.normal(size=1000)})
    y = (X["доля_дней_отказ_30д"] > 0.5).astype(int)
    m = md.fit_lgbm(X, y, {"num_leaves": 7, "verbose": -1}, 30)
    assert m.feature_name()[0] == "доля_дней_отказ_30д"
    assert md.average_precision(y, md.predict_lgbm(m, X)) > 0.9
    assert md.predict_lgbm(m, X, pred_contrib=True).shape == (1000, 3)


def test_isotonic_calibration_is_monotone_and_json_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    p = rng.random(2000); y = (rng.random(2000) < p ** 2).astype(int)
    cal = md.Calibrator.fit(p, y, method="изотоническая")
    q = cal.predict(np.sort(p))
    assert np.all(np.diff(q) >= -1e-12)
    path = tmp_path / "c.json"; cal.save(path)
    assert np.allclose(md.Calibrator.load(path).predict(p), cal.predict(p))
    assert md.brier(y, cal.predict(p)) <= md.brier(y, p)


def test_isotonic_matches_sklearn():
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(1)
    p = rng.random(3000); y = (rng.random(3000) < p ** 3).astype(int)
    cal = md.Calibrator.fit(p, y, method="изотоническая")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
    grid = np.r_[-0.5, np.linspace(0, 1, 1001), 1.5, rng.random(500)]
    assert np.allclose(cal.predict(grid), iso.predict(grid), atol=1e-12)


def test_platt_calibration_and_russian_json(tmp_path):
    rng = np.random.default_rng(0)
    p = rng.random(3000) * 0.5; y = (rng.random(3000) < p * 1.5).astype(int)
    cal = md.Calibrator.fit(p, y, method="Платт")
    assert md.brier(y, cal.predict(p)) <= md.brier(y, p)
    path = tmp_path / "c.json"; cal.save(path)
    text = path.read_bytes().decode("utf-8")
    assert "Платт" in text                                  # UTF-8, не \u-экранирование
    assert np.allclose(md.Calibrator.load(path).predict(p), cal.predict(p))


def test_platt_constant_input_gives_mean_rate():
    cal = md.Calibrator.fit(np.full(100, 0.3), np.r_[np.ones(20), np.zeros(80)], method="Платт")
    assert np.allclose(cal.predict([0.1, 0.9]), 0.2)
    cal0 = md.Calibrator.fit(np.random.default_rng(0).random(100), np.zeros(100), method="Платт")
    assert np.all(cal0.predict([0.1, 0.9]) < 1e-3)


def test_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError):
        md.Calibrator.fit([0.1, 0.2], [0, 1], method="platt")


def test_smoothed_rate_shrinks_small_objects():
    hist = pd.DataFrame({"ид_объект": [1] * 10 + [2] * 1000, "y": [1] * 5 + [0] * 5 + [1] * 100 + [0] * 900})
    r = md.smoothed_rate(hist, "y", prior_strength=100)
    glob = hist["y"].mean()
    assert abs(r[1] - glob) < abs(0.5 - glob)          # 10 дней наблюдений – тянем к общей доле
    assert abs(r[2] - 0.1) < 0.01                       # 1000 дней – почти своя доля


def _grid_y():
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-01", periods=60)
    g = pd.DataFrame({"ид_объект": np.repeat([1, 2, 3], 60), "день": np.tile(days, 3),
                      "y": rng.integers(0, 2, 180).astype(float)})
    g.loc[(g["ид_объект"] == 3) & (g["день"] < "2024-01-20"), "y"] = np.nan    # нет разметки – не наблюдение
    return g.sample(frac=1, random_state=0)                                       # порядок строк не важен


def test_smoothed_rate_before_matches_direct_count():
    g = _grid_y()
    r = md.smoothed_rate_before(g, "y", prior_strength=10, window_days=14)
    assert r.index.equals(g.index)
    for i in [0, 17, 99, 150]:
        obj, d = g.iloc[i]["ид_объект"], g.iloc[i]["день"]
        past = g[(g["день"] >= d - pd.Timedelta(days=14)) & (g["день"] <= d - pd.Timedelta(days=1))].dropna()
        own = past[past["ид_объект"] == obj]
        p0 = past["y"].mean()
        expected = (own["y"].sum() + 10 * p0) / (len(own) + 10) if len(past) else np.nan
        assert r.iloc[i] == pytest.approx(expected, nan_ok=True)


def test_smoothed_rate_before_has_no_leakage():
    g = _grid_y()
    cut = pd.Timestamp("2024-02-10")
    r1 = md.smoothed_rate_before(g, "y", 10, window_days=14)
    g2 = g.copy(); g2.loc[g2["день"] >= cut, "y"] = 1 - g2.loc[g2["день"] >= cut, "y"]
    r2 = md.smoothed_rate_before(g2, "y", 10, window_days=14)
    upto = g["день"] <= cut                         # прогноз дня D не видит метку дня D и позже
    assert np.allclose(r1[upto], r2[upto], equal_nan=True)
    assert not np.allclose(r1[~upto], r2[~upto], equal_nan=True)
    first = g["день"] == g["день"].min()            # у первого дня нет прошлого – NaN
    assert r1[first].isna().all()


def test_top_reasons_returns_positive_contributions():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000), "c": rng.normal(size=2000)})
    y = ((X["a"] > 1) | (X["b"] > 1.5)).astype(int)
    m = md.fit_lgbm(X, y, {"num_leaves": 7, "verbose": -1}, 100)
    meaning = {"a": "признак А", "b": "признак Б", "c": "шум"}
    r = md.top_reasons(m, X.iloc[:5], meaning, k=2)
    assert len(r) == 5 and all(len(x) <= 2 for x in r)
    assert all(item["вклад"] > 0 for x in r for item in x)


def test_top_reasons_cyrillic_names_and_order():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"доля_дней_отказ_30д": rng.normal(size=2000), "насос:вкл": rng.normal(size=2000),
                      "шум": rng.normal(size=2000)})
    y = ((X["доля_дней_отказ_30д"] > 1) | (X["насос:вкл"] > 1.5)).astype(int)
    m = md.fit_lgbm(X, y, {"num_leaves": 7, "verbose": -1}, 100)
    row = pd.DataFrame({"доля_дней_отказ_30д": [3.0], "насос:вкл": [3.0], "шум": [0.0]})
    r = md.top_reasons(m, row, {"доля_дней_отказ_30д": "доля дней с отказом за 30 суток"}, k=3)[0]
    names = [x["признак"] for x in r]
    assert set(names[:2]) == {"доля_дней_отказ_30д", "насос:вкл"}      # исходные имена, не замененные
    assert [x["вклад"] for x in r] == sorted([x["вклад"] for x in r], reverse=True)
    item = next(x for x in r if x["признак"] == "доля_дней_отказ_30д")
    assert item["смысл"] == "доля дней с отказом за 30 суток" and item["значение"] == 3.0
    assert next(x for x in r if x["признак"] == "насос:вкл")["смысл"] == "насос:вкл"   # нет в каталоге – имя


def test_frequency_targets_include_guard_fault():
    assert md.FREQUENCY_TARGETS == ["цель_газ", "цель_неисправность_охраны", "цель_потеря_связи"]
    assert not set(md.FREQUENCY_TARGETS) & set(md.MODEL_TARGETS)


# --- Бутстреп по 28 суткам и по объектам ---------------------------------------------------------------

def test_block_bootstrap_default_block_is_28_days():
    import inspect
    assert inspect.signature(md.block_bootstrap).parameters["block_days"].default == 28


def test_cluster_bootstrap_ci_and_duplicated_objects_stay_distinct():
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-01-01", periods=60)
    frame = pd.MultiIndex.from_product([range(40), days], names=["ид_объект", "день"]).to_frame(index=False)
    frame["y"] = rng.integers(0, 2, len(frame))
    frame["p"] = frame["y"] * 0.5 + rng.random(len(frame)) * 0.5
    est, lo, hi = md.cluster_bootstrap(frame, lambda f: md.average_precision(f["y"], f["p"]), n=200, seed=0)
    assert lo <= est <= hi and hi - lo < 0.2
    # объект, выпавший дважды, – два разных объекта: в сутках всегда 40 строк и 40 разных ид
    sizes = md.cluster_bootstrap(frame, lambda f: f.groupby("день")["ид_объект"].nunique().min(), n=30, seed=1)
    assert sizes == (40.0, 40.0, 40.0)


# --- Бэггинг по сидам ----------------------------------------------------------------------------------

def _signal_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"доля:а": rng.normal(size=n), "шум": rng.normal(size=n)})
    y = (X["доля:а"] + rng.normal(0, 0.5, n) > 1).astype(int)
    return X, y


def test_fit_lgbm_seeds_averages_probabilities_and_contributions():
    X, y = _signal_data()
    params = {"num_leaves": 7, "learning_rate": 0.1, "feature_fraction": 0.5, "bagging_fraction": 0.7,
              "bagging_freq": 1}
    bag = md.fit_lgbm_seeds(X, y, params, 50, seeds=(0, 1, 2))
    assert len(bag.models) == 3
    single = [md.fit_lgbm(X, y, params, 50, seed=s) for s in (0, 1, 2)]
    expected = np.mean([md.predict_lgbm(m, X) for m in single], axis=0)
    np.testing.assert_allclose(bag.predict(X), expected)
    np.testing.assert_allclose(md.predict_lgbm(bag, X), expected)
    contrib = md.predict_lgbm(bag, X.iloc[:10], pred_contrib=True)
    exp_c = np.mean([md.predict_lgbm(m, X.iloc[:10], pred_contrib=True) for m in single], axis=0)
    np.testing.assert_allclose(contrib, exp_c)
    assert md.average_precision(y, bag.predict(X)) > 0.8
    r = md.top_reasons(bag, X.iloc[:3], {}, k=2)
    assert len(r) == 3 and all(item["вклад"] > 0 for x in r for item in x)


def test_seed_bag_save_and_load(tmp_path):
    X, y = _signal_data(1000)
    bag = md.fit_lgbm_seeds(X, y, {"num_leaves": 7}, 20, seeds=(0, 1))
    files = bag.save(tmp_path / "отказ")
    assert [f.name for f in files] == ["отказ.seed0.lgb.txt", "отказ.seed1.lgb.txt"]
    loaded = md.SeedBag.load(files)
    np.testing.assert_allclose(loaded.predict(X), bag.predict(X))


def test_params_config39():
    assert md.PARAMS_CONFIG39 == {"num_leaves": 63, "min_data_in_leaf": 50, "learning_rate": 0.03,
                                  "feature_fraction": 0.7, "bagging_fraction": 1.0, "bagging_freq": 1,
                                  "lambda_l2": 10.0, "lambda_l1": 1.0, "min_gain_to_split": 0.0, "max_depth": -1,
                                  "extra_trees": True, "path_smooth": 10.0}


# --- Пороги с потолком нагрузки ------------------------------------------------------------------------

def _load_frame():
    # 2 месяца по 10 суток, 10 объектов; в феврале прогнозы выше
    rows = []
    for month, shift in [(1, 0.0), (2, 0.3)]:
        for d in range(1, 11):
            for o in range(10):
                p = round(o / 10 + shift, 2)
                rows.append({"ид_объект": o, "день": pd.Timestamp(2025, month, d), "p": p, "y": int(o >= 8)})
    return pd.DataFrame(rows)


def test_threshold_report():
    f = _load_frame()
    r = md.threshold_report(f, "y", "p", 0.9)
    # январь: p ≥ 0,9 только объект 9 (1/сут); февраль: объекты 6-9 (4/сут)
    assert r["нагрузка_в_сутки"] == pytest.approx(2.5) and r["нагрузка_макс_месяц"] == pytest.approx(4)
    assert r["точность"] == pytest.approx((10 + 20) / 50)
    assert r["полнота"] == pytest.approx(30 / 40)


def test_thresholds_with_cap_respects_worst_month():
    f = _load_frame()
    r = md.thresholds_with_cap(f, "y", "p", max_per_day=2)
    # самый низкий порог, при котором и в феврале не больше 2 объектов в сутки: p ≥ 1,1 (объекты 8, 9)
    assert r["порог"] == pytest.approx(1.1)
    assert r["нагрузка_макс_месяц"] <= 2
    levels = md.thresholds_with_cap(f, "y", "p", max_per_day={"высокий": 2, "повышенный": 5})
    assert list(levels.index) == ["высокий", "повышенный"]
    assert levels.loc["повышенный", "порог"] < levels.loc["высокий", "порог"]
    assert (levels["нагрузка_макс_месяц"] <= pd.Series({"высокий": 2, "повышенный": 5})).all()
