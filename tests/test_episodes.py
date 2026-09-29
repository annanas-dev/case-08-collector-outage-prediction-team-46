import datetime as dt

import pandas as pd
import pytest
from pyspark.sql import functions as F

from annanas import episodes as ep


@pytest.mark.parametrize("тип, состояние, класс", [
    ("Состояние насоса", "Неисправен", "отказ"),
    ("Состояние насоса", "Затоплен", "подтопление"),
    ("Датчик затопления", "Не замкнут", "подтопление"),
    ("Датчик дыма", "Обнаружен дым", "пожар"),
    ("Тепловой датчик", "Не замкнут", "пожар"),
    ("Датчик движения", "Обнаружено движение", "проникновение"),
    ("КД Люк", "Не замкнут", "проникновение"),
    ("Газовый датчик", "Обнаружен газ", "газ"),
    ("ИБП", "Питание от батарей", "отказ"),
    ("Датчик температуры", "Не определено", "отказ"),
    # отдельный класс: плохо прогнозируем и не отказ конкретного оборудования (решение заказчика)
    ("Состояние охраны", "Много неисправных устройств", "неисправность_охраны"),
    ("Состояние насоса", "Отключено устройство", "вывод_из_работы"),
    ("КД Дверь", "Неопределен", "неопределен"),
    ("КД Дверь", "Норма", None),
    ("Датчик дыма", None, None),
    # написание как в журнале: «в АНС», «Рычаг сдернут»; «Рычаг норма» – не тревога
    ("Состояние насоса", "Работают все насосы в АНС", "откачка"),
    ("Ручной извещатель", "Рычаг сдернут", "пожар"),
    ("Состояние УИР-Р", "Рычаг сдернут", "пожар"),
    ("Состояние УИР-Р", "Рычаг норма", None),
    # «Не замкнут» у типов вне NOT_CLOSED и температурные состояния осознанно без класса
    ("Датчик движения", "Не замкнут", None),
    ("Датчик температуры", "Температура выше 40ºC", None),
])
def test_state_class(тип, состояние, класс):
    assert ep.state_class(тип, состояние) == класс


def test_dictionary_uses_journal_spelling():
    # ключи словаря, которых нет в журнале 2019-2026 (опечатки плана)
    for ghost in ["Работают все насосы АНС", "Включены все насосы АНС", "Движение вверх",
                  "Оба рычага сдернуты", "Рычаг сдернут влево"]:
        assert ghost not in ep.BY_STATE
        assert ep.state_class("Состояние насоса", ghost) is None


def test_priority_order():
    assert ep.PRIORITY[:4] == ["подтопление", "пожар", "газ", "отказ"]
    # неисправность охраны слабее отказа и проникновения: цепочка с настоящим отказом остается отказом
    assert ep.RANK["неисправность_охраны"] > max(ep.RANK["отказ"], ep.RANK["проникновение"])
    assert ep.RANK["неисправность_охраны"] < ep.RANK["откачка"]


def test_class_lists_consistent():
    assert set(ep.EPISODE_CLASSES) == set(ep.PRIORITY)
    assert "неисправность_охраны" in ep.TARGET_CLASSES
    assert set(ep.FREQUENCY_CLASSES) == {"газ", "неисправность_охраны"}
    assert set(ep.FREQUENCY_CLASSES) <= set(ep.TARGET_CLASSES)
    assert "неисправность_охраны" not in ep.INCIDENT_CLASSES
    assert set(ep.INCIDENT_CLASSES) == {"подтопление", "пожар", "газ", "отказ"}


def test_state_class_col_matches_python(spark):
    pairs = {(тип, "Не замкнут") for тип in ep.NOT_CLOSED}
    pairs |= {("Состояние насоса", с) for с in ep.BY_STATE}
    pairs |= {("Датчик движения", "Не замкнут"), ("Состояние УИР-Р", "Рычаг норма"), ("КД Дверь", "Норма"),
              (None, "Не замкнут"), (None, "Неисправен"), ("Датчик дыма", None), ("X", "Неизвестное")}
    df = spark.createDataFrame(sorted(pairs, key=str), "`тип_датчика` string, `состояние` string")
    got = {(r["тип_датчика"], r["состояние"]): r["к"]
           for r in df.withColumn("к", ep.state_class_col()).collect()}
    assert got == {p: ep.state_class(*p) for p in pairs}


T0 = dt.datetime(2022, 3, 1, 10, 0, 0)


def _events(spark, rows):
    # канал, секунда от T0, состояние, тревожное; ид_события – порядок в списке
    data = [(c, T0 + dt.timedelta(seconds=s), st, fl, i) for i, (c, s, st, fl) in enumerate(rows)]
    return (spark.createDataFrame(data, "`ид_канала_данных` int, `дата_время` timestamp, `состояние` string, "
                                        "`тревожное` boolean, `ид_события` long")
            .withColumn("класс_записи", F.lit("состояние")))


def _channels(spark):
    return spark.createDataFrame([(1, "Состояние насоса", 30, 12.0), (2, "КД Люк", 30, 12.0)],
                                 "`ид_канала_данных` int, `тип_датчика` string, `ид_объект` int, `пикет` double")


def test_chain_takes_priority_class_and_duration(spark):
    ev = _events(spark, [(1, 0, "Норма", False), (1, 5, "Неисправен", True), (1, 5, "Затоплен", True),
                         (1, 5, "Обесточен", True), (1, 125, "Норма", False)])
    rows = ep.alarm_episodes(ev, _channels(spark)).collect()
    assert len(rows) == 1
    r = rows[0]
    assert r["класс"] == "подтопление" and r["первый_класс"] == "отказ"
    assert r["первое_состояние"] == "Неисправен"
    assert r["длительность_с"] == 120 and r["записей"] == 3 and r["ид_объект"] == 30
    assert r["доля_флага"] == 1.0


def test_guard_fault_chain_with_real_fault_stays_fault(spark):
    ch = spark.createDataFrame([(3, "Состояние охраны", 30, 12.0)],
                               "`ид_канала_данных` int, `тип_датчика` string, `ид_объект` int, `пикет` double")
    ev = _events(spark, [(3, 0, "Много неисправных устройств", True), (3, 10, "Неисправен", True),
                         (3, 20, "Норма", False), (3, 30, "Много неисправных устройств", True),
                         (3, 40, "Норма", False)])
    rows = sorted(ep.alarm_episodes(ev, ch).collect(), key=lambda r: r["начало"])
    assert [(r["класс"], r["первый_класс"]) for r in rows] == [("отказ", "неисправность_охраны"),
                                                              ("неисправность_охраны", "неисправность_охраны")]


def test_same_second_tie_broken_by_event_id(spark):
    # записи в одну секунду: порядок решает ид_события, «Норма» последней закрывает эпизод в ту же секунду
    ev = _events(spark, [(1, 5, "Неисправен", True), (1, 5, "Норма", False), (1, 50, "Норма", False)])
    r = ep.alarm_episodes(ev, _channels(spark)).collect()
    assert len(r) == 1 and r[0]["длительность_с"] == 0


def test_two_episodes_and_open_episode(spark):
    ev = _events(spark, [(2, 0, "Не замкнут", True), (2, 30, "Норма", False),
                         (2, 100, "Не замкнут", True)])
    rows = sorted(ep.alarm_episodes(ev, _channels(spark)).collect(), key=lambda r: r["начало"])
    assert [r["длительность_с"] for r in rows] == [30, None]
    assert rows[0]["класс"] == "проникновение"


def test_flag_source_ignores_state_text(spark):
    ev = _events(spark, [(1, 0, "Неисправен", False), (1, 10, "Норма", False)])
    assert ep.alarm_episodes(ev, _channels(spark), source="флаг").count() == 0
    assert ep.alarm_episodes(ev, _channels(spark), source="состояние").count() == 1


def test_channels_outside_directory_dropped(spark):
    ev = _events(spark, [(99, 0, "Неисправен", True), (99, 10, "Норма", False)])
    assert ep.alarm_episodes(ev, _channels(spark)).count() == 0


def _four_doors(spark):
    return spark.createDataFrame([(i, "КД Дверь", 30, None) for i in (1, 2, 3, 4)],
                                 "`ид_канала_данных` int, `тип_датчика` string, `ид_объект` int, `пикет` double")


def test_comm_loss_merges_minutes_and_filters_short(spark):
    # объект 30, 4 канала; порог = min(10, ceil(0.5*4)) = 2
    rows = []
    for m in [0, 1, 2, 6, 12]:            # 0-2 и 6 склеятся (разрыв 4 мин), 12 – отдельно (разрыв 6)
        for ch in (1, 2):
            rows.append((ch, m * 60, "Неопределен", False))
    got = ep.comm_loss_episodes(_events(spark, rows), _four_doors(spark),
                                k=10, share=0.5, gap_min=5, min_minutes=5).collect()
    assert len(got) == 1
    assert got[0]["длительность_мин"] == 7 and got[0]["ид_объект"] == 30
    assert got[0]["класс"] == "потеря_связи" and got[0]["макс_каналов"] == 2


def test_comm_loss_needs_enough_channels(spark):
    # один канал из четырех – ниже порога 2: не пачка
    rows = [(1, m * 60, "Неопределен", False) for m in range(20)]
    assert ep.comm_loss_episodes(_events(spark, rows), _four_doors(spark), min_minutes=1).count() == 0
    # k ограничивает порог сверху: при k=1 хватает одного канала
    assert ep.comm_loss_episodes(_events(spark, rows), _four_doors(spark), k=1, min_minutes=1).count() == 1


def test_pairs_source_uses_list_not_flag(spark):
    # Неисправен насоса в списке пар – эпизод при флаге false; Не замкнут люка – класс есть, но пары нет в списке
    ev = _events(spark, [(1, 0, "Неисправен", False), (1, 10, "Норма", False),
                         (2, 0, "Не замкнут", True), (2, 10, "Норма", False)])
    for pairs in ([("Состояние насоса", "Неисправен")],
                  spark.createDataFrame([("Состояние насоса", "Неисправен")],
                                        "`тип_датчика` string, `состояние` string")):
        rows = ep.alarm_episodes(ev, _channels(spark), source="пары", pairs=pairs).collect()
        assert [(r["ид_канала_данных"], r["класс"]) for r in rows] == [(1, "отказ")]


def test_pairs_source_requires_known_class(spark):
    # пара в списке, но состояние без класса («Норма») эпизода не дает
    ev = _events(spark, [(1, 0, "Норма", True), (1, 10, "Неисправен", False)])
    rows = ep.alarm_episodes(ev, _channels(spark), source="пары",
                             pairs=[("Состояние насоса", "Норма")]).collect()
    assert rows == []


def test_alarm_episodes_bad_source(spark):
    ev = _events(spark, [(1, 0, "Неисправен", True)])
    with pytest.raises(ValueError):
        ep.alarm_episodes(ev, _channels(spark), source="пары")
    with pytest.raises(ValueError):
        ep.alarm_episodes(ev, _channels(spark), source="что-то")


def test_alarm_pairs_from_vocab():
    def row(тип, сост, флаг, год, мес, n):
        return {"тип_датчика": тип, "состояние": сост, "тревожное": флаг, "год": год, "месяц": мес,
                "количество_событий": n}
    vocab = pd.DataFrame([
        row("Состояние фазы", "Обесточен", False, 2022, 1, 1000),          # 0 % – нет
        row("Состояние насоса", "Неисправен", True, 2022, 1, 800),         # 80 % – да
        row("Состояние насоса", "Неисправен", False, 2022, 2, 200),
        row("Датчик дыма", "Неисправен", False, 2023, 5, 500),             # до границы 0 %,
        row("Датчик дыма", "Неисправен", True, 2025, 11, 10_000),          # после границы – не считается
        row("Газовый датчик", "Обнаружен газ", True, 2022, 1, 50),         # мало записей
        row("Состояние насоса", "Норма", True, 2022, 1, 500),              # нет класса
        row("КД Люк", "Не замкнут", True, 2025, 10, 150),                  # октябрь 2025 – еще до границы
    ])
    got = ep.alarm_pairs(vocab, stable_until="2025-11-01")
    assert list(got.columns) == ["тип_датчика", "состояние", "класс", "доля_флага", "записей"]
    assert set(zip(got["тип_датчика"], got["состояние"])) == {("Состояние насоса", "Неисправен"),
                                                              ("КД Люк", "Не замкнут")}
    r = got.set_index("тип_датчика").loc["Состояние насоса"]
    assert r["класс"] == "отказ" and r["доля_флага"] == pytest.approx(0.8) and r["записей"] == 1000
    assert len(ep.alarm_pairs(vocab, "2025-11-01", min_records=10)) == 3
