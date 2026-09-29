"""Эпизоды тревог и потери связи. Словарь классов и его обоснование – в notebooks/03_target."""
import pandas as pd
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

# «Не замкнут» означает разное у разных типов: у датчика затопления – вода, у извещателей – пожар, у КД – открыто.
# Остальные типы с «Не замкнут» (датчик движения, дыма, УИР-Р, ИБП, фаза...) осознанно без класса:
# смысл шлейфа у них неясен, тревожных записей единицы-сотни.
NOT_CLOSED = {
    "Датчик затопления": "подтопление",
    "Тепловой датчик": "пожар", "Ручной извещатель": "пожар",
    "КД АВ": "проникновение", "КД Дверь": "проникновение", "КД Люк": "проникновение",
    "Стекло": "проникновение", "9-секционный люк": "проникновение",
}
# Написание – как в журнале 2019-2026. «Температура выше 40ºC» / «ниже 3ºC» осознанно без класса:
# это климат коллектора, а не тип инцидента из ТЗ (решение проверяется в notebooks/03_target).
BY_STATE = {
    "Неисправен": "отказ", "Обесточен": "отказ", "Питание от батарей": "отказ", "Батарея разряжена": "отказ",
    "Не определено": "отказ",
    # Только у «Состояние охраны»: почти половина эпизодов отказа, плохо прогнозируема и не указывает на
    # конкретное оборудование. Отдельный класс – фоновая частота, как газ, не входит в «любой инцидент».
    "Много неисправных устройств": "неисправность_охраны",
    "Затоплен": "подтопление",
    "Обнаружен дым": "пожар", "Рычаг сдернут": "пожар",
    "Обнаружен газ": "газ",
    "Обнаружено движение": "проникновение",
    "Отключено устройство": "вывод_из_работы",
    "Работают все насосы в АНС": "откачка",
    "Неопределен": "неопределен",
}
# Чем раньше в списке, тем важнее класс, если в одной цепочке несколько состояний. Неисправность охраны
# ниже отказа и проникновения: цепочка «Много неисправных устройств» + «Неисправен» остается отказом.
PRIORITY = ["подтопление", "пожар", "газ", "отказ", "проникновение", "неисправность_охраны", "откачка",
            "вывод_из_работы"]
TARGET_CLASSES = ["подтопление", "пожар", "газ", "отказ", "проникновение", "неисправность_охраны"]
# Цели, для которых сервис показывает фоновую частоту, а не модель
FREQUENCY_CLASSES = ["газ", "неисправность_охраны"]
# Классы эпизодов тревог в «любом инциденте» (вместе с потерей связи из comm_loss_episodes)
INCIDENT_CLASSES = ["подтопление", "пожар", "газ", "отказ"]


def state_class(тип: str | None, состояние: str | None) -> str | None:
    """Класс записи по типу датчика и тексту состояния; None – не тревога."""
    if состояние == "Не замкнут":
        return NOT_CLOSED.get(тип)
    return BY_STATE.get(состояние)


def state_class_col(type_col: str = "тип_датчика", state_col: str = "состояние"):
    """То же, что state_class, в виде Spark-выражения (без Python UDF – быстро на сотнях миллионов строк)."""
    t, s = F.col(type_col), F.col(state_col)
    nc = F
    for тип, класс in NOT_CLOSED.items():
        nc = nc.when(t == тип, класс)
    expr = F.when(s == "Не замкнут", nc)
    for состояние, класс in BY_STATE.items():
        expr = expr.when(s == состояние, класс)
    return expr


EPISODE_CLASSES = TARGET_CLASSES + ["откачка", "вывод_из_работы"]
RANK = {к: i for i, к in enumerate(PRIORITY)}
SOURCES = ("состояние", "флаг", "пары")
_SEP = "\x1f"  # разделитель ключа «тип + состояние», в текстах журнала не встречается


def _pair_key(тип, состояние):
    return F.concat_ws(_SEP, тип, состояние)


def _in_pairs(pairs) -> Column:
    """Запись входит в список пар (тип_датчика, состояние). Пар – десятки, поэтому isin по ключу:
    без join и перетасовки на сотнях миллионов строк."""
    if isinstance(pairs, DataFrame):
        pairs = [(r["тип_датчика"], r["состояние"]) for r in pairs.select("тип_датчика", "состояние").collect()]
    keys = [f"{т}{_SEP}{с}" for т, с in pairs]
    return _pair_key(F.col("тип_датчика"), F.col("состояние")).isin(keys)


def alarm_pairs(vocab: pd.DataFrame, stable_until, min_share: float = 0.5, min_records: int = 100) -> pd.DataFrame:
    """Тревожные пары (тип_датчика, состояние) по суждению самой системы: доля записей с `тревожное` за
    месяцы строго до `stable_until` (период стабильной разметки флага). Остаются пары с классом из
    EPISODE_CLASSES, долей флага ≥ min_share и ≥ min_records записей. Порог обоснован в notebooks/03_target.
    vocab – витрина state_vocab."""
    month = pd.to_datetime(dict(year=vocab["год"], month=vocab["месяц"], day=1))
    v = vocab.loc[month < pd.Timestamp(stable_until)]
    flagged = v["количество_событий"].where(v["тревожное"].fillna(False).astype(bool), 0)
    g = (v.assign(флаг=flagged)
         .groupby(["тип_датчика", "состояние"], as_index=False)
         .agg(записей=("количество_событий", "sum"), флаг=("флаг", "sum")))
    g["доля_флага"] = g["флаг"] / g["записей"]
    g["класс"] = [state_class(т, с) for т, с in zip(g["тип_датчика"], g["состояние"])]
    keep = g["класс"].isin(EPISODE_CLASSES) & (g["доля_флага"] >= min_share) & (g["записей"] >= min_records)
    return g.loc[keep, ["тип_датчика", "состояние", "класс", "доля_флага", "записей"]].reset_index(drop=True)


def alarm_episodes(events: DataFrame, channels: DataFrame, source: str = "состояние", pairs=None) -> DataFrame:
    """Эпизоды тревог каналов справочника: подряд идущие записи с классом до первой записи без класса.
    Принадлежность к эпизоду: source="состояние" – по словарю, "флаг" – по `тревожное`, "пары" – пара
    (тип_датчика, состояние) из `pairs` (список кортежей или Spark DataFrame), флаг не смотрится.
    Класс всегда по словарю.
    `конец` – время первой записи без класса, null у незакрытого эпизода. Каналы вне справочника
    отбрасываются: у них нет типа и объекта."""
    if source not in SOURCES:
        raise ValueError(f"source: одно из {SOURCES}, получено {source!r}")
    if source == "пары" and pairs is None:
        raise ValueError('source="пары" требует pairs')
    order = ["дата_время", "ид_события"]  # записи одной секунды упорядочивает ид_события
    w = Window.partitionBy("ид_канала_данных").orderBy(*order)
    rank = F.create_map(*[x for к, i in RANK.items() for x in (F.lit(к), F.lit(i))])
    recs = (
        events.filter(F.col("класс_записи") == "состояние")
        .join(F.broadcast(channels.select("ид_канала_данных", "тип_датчика", "ид_объект", "пикет")),
              "ид_канала_данных")
        .withColumn("класс_состояния", state_class_col())
    )
    in_ep = F.col("класс_состояния").isin(EPISODE_CLASSES)
    if source == "флаг":
        in_ep = F.col("тревожное") & in_ep
    elif source == "пары":
        in_ep = _in_pairs(pairs) & in_ep
    recs = (recs.withColumn("в_эпизоде", F.coalesce(in_ep, F.lit(False)))
                .withColumn("старт", F.col("в_эпизоде") & ~F.lag("в_эпизоде", 1, False).over(w))
                .withColumn("след_время", F.lead("дата_время").over(w))
                .withColumn("номер_эпизода", F.sum(F.col("старт").cast("int")).over(w))
                .filter(F.col("в_эпизоде")))
    return (
        recs.groupBy("ид_канала_данных", "номер_эпизода", "тип_датчика", "ид_объект", "пикет")
        .agg(F.min("дата_время").alias("начало"),
             F.max_by("след_время", F.struct(*order)).alias("конец"),
             F.min_by("класс_состояния", rank[F.col("класс_состояния")]).alias("класс"),
             F.min_by("класс_состояния", F.struct(*order)).alias("первый_класс"),
             F.min_by("состояние", F.struct(*order)).alias("первое_состояние"),
             F.count("*").alias("записей"),
             F.avg(F.col("тревожное").cast("double")).alias("доля_флага"))
        .withColumn("длительность_с", F.col("конец").cast("long") - F.col("начало").cast("long"))
        .drop("номер_эпизода")
    )


def comm_loss_episodes(events: DataFrame, channels: DataFrame, k: int = 10, share: float = 0.5,
                       gap_min: int = 5, min_minutes: int = 10) -> DataFrame:
    """Эпизоды потери связи объекта. Минута-пачка: в «Неопределен» ушло ≥ min(k, ⌈share · каналов объекта⌉)
    каналов объекта; пачки с разрывом ≤ gap_min минут склеиваются; эпизод длиной < min_minutes отбрасывается."""
    n_obj = channels.groupBy("ид_объект").agg(F.countDistinct("ид_канала_данных").alias("каналов_объекта"))
    minutes = (
        events.filter(F.col("состояние") == "Неопределен")
        .join(F.broadcast(channels.select("ид_канала_данных", "ид_объект")), "ид_канала_данных")
        .groupBy("ид_объект", F.date_trunc("minute", "дата_время").alias("минута"))
        .agg(F.countDistinct("ид_канала_данных").alias("каналов_в_неопределен"))
        .join(F.broadcast(n_obj), "ид_объект")
        .filter(F.col("каналов_в_неопределен")
                >= F.least(F.lit(k), F.ceil(F.lit(share) * F.col("каналов_объекта"))))
    )
    w = Window.partitionBy("ид_объект").orderBy("минута")
    gap = (F.col("минута").cast("long") - F.lag("минута").over(w).cast("long")) / 60
    return (
        minutes.withColumn("старт", F.coalesce(gap > gap_min, F.lit(True)))
        .withColumn("номер", F.sum(F.col("старт").cast("int")).over(w))
        .groupBy("ид_объект", "номер")
        .agg(F.min("минута").alias("начало"), F.max("минута").alias("последняя_минута"),
             F.max("каналов_в_неопределен").alias("макс_каналов"))
        .withColumn("длительность_мин",
                    ((F.col("последняя_минута").cast("long") - F.col("начало").cast("long")) / 60 + 1).cast("int"))
        .filter(F.col("длительность_мин") >= min_minutes)
        .withColumn("класс", F.lit("потеря_связи"))
        .drop("номер")
    )
