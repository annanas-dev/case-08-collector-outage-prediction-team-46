import pandas as pd

from annanas import etl


def _raw(spark, rows):
    cols = ["ид_события", "ид_канала_данных", "дата", "время", "тревожное", "значение_датчика"]
    return spark.createDataFrame(rows, cols)


def test_normalize_parses_flag_class_and_value(spark):
    df = _raw(spark, [
        ("1", "10", "2021-05-01", "10:00:00", "t", "0.03"),
        ("2", "10", "2021-05-01", "10:00:10", "false", "Норма"),
        ("3", "11", "2021-05-01", "10:00:20", "true", "01.01.1970 03:00:00"),
        ("4", "11", "2021-05-01", "10:00:30", "f", "28"),
    ])
    out = {r["ид_события"]: r for r in etl.normalize(df).collect()}

    assert out[1]["тревожное"] is True
    assert out[2]["тревожное"] is False
    assert out[3]["тревожное"] is True

    assert out[1]["класс_записи"] == "измерение"
    assert out[1]["числовое_значение"] == 0.03
    assert out[1]["состояние"] is None

    assert out[2]["класс_записи"] == "состояние"
    assert out[2]["состояние"] == "Норма"
    assert out[2]["числовое_значение"] is None

    # значение-дата «Состояние охраны» относится к последовательности измерений
    assert out[3]["класс_записи"] == "измерение"
    assert out[3]["числовое_значение"] is None

    assert out[4]["числовое_значение"] == 28.0
    assert str(out[1]["дата_время"]) == "2021-05-01 10:00:00"
    assert out[1]["ид_канала_данных"] == 10


def test_normalize_types_source_columns(spark):
    df = _raw(spark, [("1", "10", "2021-05-01", "10:00:00", "t", "0.03")])
    types = dict(etl.normalize(df).dtypes)
    assert types["ид_события"] == "bigint"
    assert types["ид_канала_данных"] == "int"
    assert types["тревожное"] == "boolean"
    assert types["дата_время"] == "timestamp"
    assert types["значение_датчика"] == "string"


def test_drop_header_rows(spark):
    df = _raw(spark, [
        ("ид_события", "ид_канала_данных", "дата", "время", "тревожное", "значение_датчика"),
        ("1", "10", "2025-01-01", "00:00:00", "f", "0.01"),
    ])
    assert etl.drop_header_rows(df).count() == 1


def test_extract_pk():
    assert etl.extract_pk("ГАЗ Д26 ПК746") == 746.0
    assert etl.extract_pk("Темп. ВШ ПК88,5") == 88.5
    assert etl.extract_pk("ТД ПК86-85") == 86.0
    assert etl.extract_pk("ИБП объекта") is None
    assert etl.extract_pk("ГАЗ Д10 пк212") == 212.0
    assert etl.extract_pk("ОД пк 765 4х секц.люк") == 765.0


def test_build_channels_joins_objects():
    channels = pd.DataFrame({
        "ид_канала_данных": [1], "тип_инж_системы": ["Газовая охрана"], "тип_датчика": ["Газовый датчик"],
        "тег_инженерной_системы": ["15-1."], "название_датчика": ["ГАЗ ПК7"], "ид_объект": [30],
    })
    objects = pd.DataFrame({
        "ид_объект": [30, 20, 10], "иерархия_уровень": [3, 2, 1], "родитель": [20, 10, 3831],
        "вид_объекта": ["guardObject", "controlHouse", "district"],
        "диспетчерское_название_объекта": ["Объект", "Родитель", "Район"],
    })
    out = etl.build_channels(channels, objects)
    row = out.iloc[0]
    assert row["ид_канала_данных"] == 1
    assert row["ид_объект"] == 30 and row["родитель"] == 20
    assert row["диспетчерское_название_объекта"] == "Объект" and row["название_родителя"] == "Родитель"
    assert row["вид_объекта"] == "guardObject" and row["иерархия_уровень"] == 3
    assert row["пикет"] == 7.0
    assert list(out.columns) == [
        "ид_канала_данных", "тип_инж_системы", "тип_датчика", "название_датчика", "ид_объект",
        "иерархия_уровень", "вид_объекта", "диспетчерское_название_объекта", "родитель",
        "название_родителя", "пикет",
    ]


def test_clean_journal_reports_every_step(spark, tmp_path):
    csv = tmp_path / "j.csv"
    csv.write_text(
        "ид_события,ид_канала_данных,дата,время,тревожное,значение_датчика\n"
        "1,10,2025-01-01,00:00:00,f,0.01\n"
        "1,10,2025-01-01,00:00:00,f,0.01\n"          # полный дубль
        "ид_события,ид_канала_данных,дата,время,тревожное,значение_датчика\n"  # заголовок внутри
        "2,99,2025-01-01,00:00:10,t,Неисправен\n",
        encoding="utf-8",
    )
    df, report = etl.clean_journal(spark, csv, known_channel_ids={10})
    assert report == {
        "строк_в_файле": 4, "строк_заголовка": 1, "полных_дублей": 1, "битых_дат": 0,
        "строк_после_очистки": 2, "тревожных": 1, "строк_каналов_вне_справочника": 1,
    }
    assert set(df.columns) == {
        "ид_события", "ид_канала_данных", "дата_время", "тревожное", "значение_датчика",
        "класс_записи", "числовое_значение", "состояние", "год", "месяц",
    }


def test_write_events_roundtrip(spark, tmp_path):
    csv = tmp_path / "j.csv"
    csv.write_text("ид_события,ид_канала_данных,дата,время,тревожное,значение_датчика\n"
                   "1,10,2025-03-01,00:00:00,f,0.01\n", encoding="utf-8")
    df, _ = etl.clean_journal(spark, csv, known_channel_ids={10})
    etl.write_events(df, tmp_path / "events")
    back = spark.read.parquet(str(tmp_path / "events"))
    assert back.count() == 1
    assert (tmp_path / "events" / "год=2025" / "месяц=3").exists()
    row = back.first()
    assert row["год"] == 2025 and row["месяц"] == 3
    assert row["ид_канала_данных"] == 10 and row["класс_записи"] == "измерение"
    assert back.filter("`год` = 2025 and `месяц` = 3").count() == 1   # отбор по партиции


def test_bad_timestamp_is_counted_not_raised(spark, tmp_path):
    csv = tmp_path / "j.csv"
    csv.write_text("ид_события,ид_канала_данных,дата,время,тревожное,значение_датчика\n"
                   "1,10,2025-03-01,00:00:00,f,0.01\n"
                   "2,10,2025-13-45,xx:00:00,f,0.01\n", encoding="utf-8")
    df, report = etl.clean_journal(spark, csv, known_channel_ids={10})
    assert report["битых_дат"] == 1 and report["строк_после_очистки"] == 1
    assert df.count() == 1


def test_write_events_appends_across_files(spark, tmp_path):
    for i, day in enumerate(["2019-12-31", "2020-01-01"]):
        csv = tmp_path / f"j{i}.csv"
        csv.write_text("ид_события,ид_канала_данных,дата,время,тревожное,значение_датчика\n"
                       f"{i},10,{day},00:00:00,f,0.01\n", encoding="utf-8")
        df, _ = etl.clean_journal(spark, csv, known_channel_ids={10})
        etl.write_events(df, tmp_path / "events")
    assert spark.read.parquet(str(tmp_path / "events")).count() == 2
