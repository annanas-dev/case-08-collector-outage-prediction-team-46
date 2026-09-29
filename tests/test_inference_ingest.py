"""Дозагрузка событий: пересчет производных данных доводится до конца после сбоя (нужна Java)."""
import datetime as dt

import pandas as pd
import pytest

from annanas import etl
from annanas.inference import ingest

HEADER = '"ид_события","ид_канала_данных","дата","время","тревожное","значение_датчика"\n'


def _csv(path, day, ids):
    path.write_text(HEADER + "".join(f'{i},1,"{day}","10:00:0{k}",false,"5"\n' for k, i in enumerate(ids)),
                    encoding="utf-8")
    return path


@pytest.fixture
def root(spark, tmp_path):
    root = tmp_path / "processed"
    root.mkdir()
    pd.DataFrame({"ид_канала_данных": [1], "ид_объект": [10]}).to_parquet(root / "channels.parquet")
    old, _ = etl.clean_journal(spark, _csv(tmp_path / "old.csv", "2026-06-30", [1, 2]), {1})
    etl.write_events(old, root / "events")
    spark.catalog.clearCache()
    return root


def test_recompute_resumes_after_crash(spark, root, tmp_path, monkeypatch):
    """Сервис упал после записи событий, но до конца пересчета: повторная загрузка того же файла ничего не
    добавляет, но доводит пересчет; следующая – уже ничего не пересчитывает."""
    calls = []

    def recompute(spark, lo, hi, root):
        calls.append((pd.Timestamp(lo).date(), pd.Timestamp(hi).date()))
        if len(calls) == 1:
            raise RuntimeError("OOM")

    monkeypatch.setattr(ingest, "recompute", recompute)
    new = _csv(tmp_path / "new.csv", "2026-08-01", [3, 4, 5])
    with pytest.raises(RuntimeError):
        ingest.ingest_csv(spark, new, root)

    rep = ingest.ingest_csv(spark, new, root)
    assert rep["добавлено"] == 0 and rep["уже_загружено"] == 3
    assert calls[-1] == (dt.date(2026, 7, 30), dt.date(2026, 8, 1))
    assert rep["пересчитано_с"] == dt.date(2026, 7, 30) and rep["пересчитано_по"] == dt.date(2026, 8, 1)

    rep = ingest.ingest_csv(spark, new, root)
    assert len(calls) == 2 and rep["пересчитано_с"] is None


def test_pending_range_merges_with_new_rows(spark, root, tmp_path, monkeypatch):
    calls = []

    def recompute(spark, lo, hi, root):
        calls.append((pd.Timestamp(lo).date(), pd.Timestamp(hi).date()))
        if len(calls) == 1:
            raise RuntimeError("OOM")

    monkeypatch.setattr(ingest, "recompute", recompute)
    with pytest.raises(RuntimeError):
        ingest.ingest_csv(spark, _csv(tmp_path / "a.csv", "2026-08-01", [3]), root)
    rep = ingest.ingest_csv(spark, _csv(tmp_path / "b.csv", "2026-08-05", [6]), root)
    assert rep["добавлено"] == 1 and rep["первый_день"] == dt.date(2026, 8, 5)
    assert calls[-1] == (dt.date(2026, 7, 30), dt.date(2026, 8, 5))


def test_swap_restores_vitrine_lost_between_remove_and_rename(spark, tmp_path):
    """Сбой между удалением старой витрины и переименованием новой: витрина берется из готового `.tmp`."""
    path = tmp_path / "v.parquet"
    spark.createDataFrame(pd.DataFrame({"день": pd.to_datetime(["2026-08-01"]), "x": [1]})) \
        .write.parquet(str(path.with_name("v.parquet.tmp")))
    fresh = spark.createDataFrame(pd.DataFrame({"день": pd.to_datetime(["2026-08-02"]), "x": [2]}))
    ingest._replace(spark, path, fresh, "день", pd.Timestamp("2026-08-02"), pd.Timestamp("2026-08-03"))
    got = pd.read_parquet(path).sort_values("день")
    assert got["x"].tolist() == [1, 2]
