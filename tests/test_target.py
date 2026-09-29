import pandas as pd

from annanas import target as tg


def _get(out, o, d, c):
    return out.loc[(out["ид_объект"] == o) & (out["день"] == pd.Timestamp(d)), f"цель_{c}"].item()


def test_grid_is_objects_times_days():
    grid = tg.build_grid([31, 30], "2022-03-01", "2022-03-03")
    assert len(grid) == 6 and list(grid.columns) == ["ид_объект", "день"]
    assert grid["ид_объект"].tolist()[:3] == [30, 30, 30]


def test_labels_count_only_non_flapping_in_horizon():
    grid = tg.build_grid([30, 31], "2022-03-01", "2022-03-03")
    eps = pd.DataFrame({
        "ид_объект": [30, 30, 31],
        "начало": pd.to_datetime(["2022-03-01 23:59:00", "2022-03-02 10:00:00", "2022-03-02 12:00:00"]),
        "длительность_с": [120, 10, 3600],
        "класс": ["отказ", "отказ", "пожар"],
    })
    out = tg.add_labels(grid, eps, classes=["отказ", "пожар"], t_flap=60)
    assert len(out) == len(grid)
    assert _get(out, 30, "2022-03-01", "отказ") == 1      # 23:59 того же дня входит в окно
    assert _get(out, 30, "2022-03-02", "отказ") == 0      # 10 с – дребезг
    assert _get(out, 31, "2022-03-02", "пожар") == 1
    assert _get(out, 31, "2022-03-03", "пожар") == 0
    assert _get(out, 30, "2022-03-01", "пожар") == 0


def test_open_episode_counts_as_non_flapping():
    grid = tg.build_grid([30], "2022-03-01", "2022-03-01")
    eps = pd.DataFrame({"ид_объект": [30], "начало": [pd.Timestamp("2022-03-01 05:00")],
                        "длительность_с": [None], "класс": ["отказ"]})
    assert tg.add_labels(grid, eps, ["отказ"], 60)["цель_отказ"].item() == 1


def test_mask_marks_days_overlapping_intervals():
    grid = tg.build_grid([30], "2021-04-21", "2021-04-23")
    masks = [("переход СМВУ", pd.Timestamp("2021-04-22 07:00"), pd.Timestamp("2021-06-08 10:00"))]
    out = tg.add_masks(grid, masks)
    assert out["маска"].tolist() == [None, "переход СМВУ", "переход СМВУ"]


def test_first_matching_mask_wins():
    grid = tg.build_grid([30], "2024-04-06", "2024-04-06")
    masks = [("обрыв", pd.Timestamp("2024-04-06 02:00"), pd.Timestamp("2024-04-10")),
             ("другая", pd.Timestamp("2024-04-06"), pd.Timestamp("2024-04-07"))]
    assert tg.add_masks(grid, masks)["маска"].tolist() == ["обрыв"]


def test_split_by_date():
    grid = tg.build_grid([30], "2024-12-31", "2026-01-01")
    out = tg.add_split(grid, valid_start="2025-01-01", test_start="2026-01-01")
    assert out["выборка"].iloc[[0, 1, -1]].tolist() == ["обучение", "валидация", "тест"]


# --- Маска «объект не подключен / не на связи» ------------------------------------------------------

def _records(obj, start, counts):
    days = pd.date_range(start, periods=len(counts), freq="D")
    return pd.DataFrame({"ид_объект": obj, "день": days, "записей": counts})


def test_connection_mask_before_first_record_and_quiet_days():
    grid = tg.build_grid([1], "2022-01-01", "2022-01-08")
    rec = _records(1, "2022-01-03", [5, 0, 0, 3])                # редкий объект: тихие дни не маскируются
    m = tg.connection_mask(grid, rec)
    assert m.index.equals(grid.index)
    assert m.tolist() == ["объект не подключен"] * 3 + [None] * 5   # 03: записей строго до D еще нет


def test_connection_mask_constant_writer_silence():
    grid = tg.build_grid([1, 2], "2022-01-01", "2022-01-12")
    rec = pd.concat([_records(1, "2022-01-01", [500] * 8 + [0, 0, 0, 500]),     # постоянный замолчал 09-11
                     _records(2, "2022-01-01", [50] * 8 + [0, 0, 0, 50])])       # средний: не маскируется
    m = tg.connection_mask(grid, rec).set_axis(pd.MultiIndex.from_frame(grid))
    s1 = m.loc[1]
    assert s1.loc["2022-01-01"] == "объект не подключен"
    assert s1.loc["2022-01-09"] is None                      # последний раз писал 08: к 09 00:00 тишина ≤ 24 ч
    assert s1.loc["2022-01-10"] == "объект не на связи"      # 09 без записей → тишина > 24 ч
    assert s1.loc["2022-01-12"] == "объект не на связи"      # запись 12-го к 12 00:00 еще неизвестна
    assert m.loc[2].iloc[1:].isna().all()


def test_connection_mask_uses_only_data_before_d():
    grid = tg.build_grid([1], "2022-01-01", "2022-01-20")
    rec = _records(1, "2022-01-01", [500] * 10 + [0] * 3)
    full = tg.connection_mask(grid, pd.concat([rec, _records(1, "2022-01-14", [900] * 7)]))
    trunc = tg.connection_mask(grid, rec)
    keep = (grid["день"] <= pd.Timestamp("2022-01-14")).to_numpy()   # D = 14: запись 14-го – не до D
    assert full[keep].tolist() == trunc[keep].tolist()
    assert full[~keep].isna().all()


def test_connection_mask_median_window_excludes_days_before_first_record():
    # объект подключился недавно: медиана только по дням после первой записи
    grid = tg.build_grid([1], "2022-01-01", "2022-02-15")
    rec = _records(1, "2022-02-01", [400] * 5)
    m = tg.connection_mask(grid, rec).set_axis(grid["день"])
    assert m.loc["2022-02-06"] is None and m.loc["2022-02-07"] == "объект не на связи"


def test_connection_mask_gap_days_skipped_in_median():
    # журнал пуст целиком 02-01 … 02-20 (пропуск выгрузки), 02-21 объект 1 пишет, объект 2 молчит
    grid = tg.build_grid([1, 2], "2022-01-01", "2022-02-22")
    gap = pd.date_range("2022-02-01", "2022-02-20")
    rec = pd.concat([_records(1, "2022-01-01", [500] * 31), _records(2, "2022-01-01", [500] * 31),
                     _records(1, "2022-02-21", [500])])
    plain = tg.connection_mask(grid, rec).set_axis(pd.MultiIndex.from_frame(grid))
    gapped = tg.connection_mask(grid, rec, gap_days=gap).set_axis(pd.MultiIndex.from_frame(grid))
    # по календарю медиана за 30 суток – ноль, объект 2 «не на связи» не считается
    assert plain.loc[(2, pd.Timestamp("2022-02-22"))] is None
    # по суткам с данными медиана 500, объект 2 молчал 02-21 – «не на связи»; объект 1 писал – без маски
    assert gapped.loc[(2, pd.Timestamp("2022-02-22"))] == tg.OFFLINE
    assert gapped.loc[(1, pd.Timestamp("2022-02-22"))] is None
    # внутри пропуска молчание за D − 1 – как обычно
    assert gapped.loc[(1, pd.Timestamp("2022-02-05"))] == tg.OFFLINE
    # без пропусков в окне результат тот же
    early = grid["день"] <= pd.Timestamp("2022-01-31")
    assert plain[early.to_numpy()].tolist() == gapped[early.to_numpy()].tolist()


def test_global_masks_and_flag_epoch():
    assert tg.GLOBAL_MASKS[0][1] == pd.Timestamp("2021-04-21")
    assert [m[0] for m in tg.GLOBAL_MASKS][-1] == "нет данных 2026-06-01"
    assert tg.flag_epoch(pd.Series(pd.to_datetime(["2025-10-28", "2025-10-29"]))).tolist() == [
        "до смены разметки", "после смены разметки"]
