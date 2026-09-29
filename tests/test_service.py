"""Сервис без данных: уровни риска, преобразование строки кэша в ответ, формат ошибок, синхронность OpenAPI."""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import yaml
from fastapi.testclient import TestClient

from annanas.inference.scorer import level
from annanas.paths import PROJECT_ROOT
from annanas.service import schemas as sc
from annanas.service.app import FREQ_TARGETS, MODEL_TARGETS, create_app, to_forecast

TH = {"высокий": 0.2, "средний": 0.1}


def test_level_thresholds_and_missing():
    got = level([0.25, 0.2, 0.15, 0.1, 0.05, np.nan], TH)
    assert got.tolist() == ["высокий", "высокий", "средний", "средний", "низкий", "нет данных"]


def _row(masked=False):
    reasons = json.dumps([{"признак": "x", "смысл": "икс", "значение": float("nan"), "вклад": 0.5}])
    row = {"ид_объект": 5011, "день": dt.date(2026, 8, 2), "маска": "плановые работы АКМ" if masked else None,
           "каналы_отказ": json.dumps([{"ид_канала_данных": 1, "пикет": 3.5, "дней_с_последнего": 2,
                                        "дней_с_отказом_30д": 1}]),
           "пикеты_отказ": json.dumps([{"пикет": 3.5, "дней_с_последнего": 2, "дней_с_отказом_30д": 1}]),
           "плановые_работы": json.dumps({"начало": "2026-07-28", "конец": "2026-08-21", "источник": "график"})}
    for t in MODEL_TARGETS:
        p = None if masked else 0.15
        row |= {f"вероятность_{t}": p, f"балл_{t}": p, f"уровень_{t}": "нет данных" if masked else "средний",
                f"причины_{t}": "[]" if masked else reasons}
    for t in FREQ_TARGETS:
        row[f"вероятность_{t}"] = 0.01
    return row


def test_to_forecast_matches_schema():
    f = sc.ObjectForecast.model_validate(to_forecast(_row()))
    assert f.прогноз.отказ.уровень == "средний" and f.прогноз.отказ.причины[0].значение is None   # NaN → null
    assert f.плановые_работы.начало == dt.date(2026, 7, 28) and f.локация_отказа.каналы[0].пикет == 3.5
    m = sc.ObjectForecast.model_validate(to_forecast(_row(masked=True)))
    assert m.маска == "плановые работы АКМ" and m.прогноз.пожар.вероятность is None
    assert m.прогноз.пожар.уровень == "нет данных" and m.прогноз.пожар.причины == []


def test_errors_have_russian_format(tmp_path):
    client = TestClient(create_app(tmp_path))            # пустое хранилище: до данных запросы не доходят
    r = client.post("/predict", params={"date": "2026-13-01"})
    assert r.status_code == 422 and r.json()["ошибка"] == "неверные параметры запроса"
    assert r.json()["подробности"][0]["где"] == "query.date"
    r = client.post("/events", content=b"a,b\n1,2\n", headers={"content-type": "text/csv"})
    assert r.status_code == 400 and r.json()["ошибка"] == "неверный формат CSV"
    r = client.post("/events", files={"other": ("x.csv", b"a,b\n")})
    assert r.status_code == 400
    r = client.get("/predictions", params={"date": "2026-08-02"})
    assert r.status_code == 404 and "POST /predict" in r.json()["ошибка"]


def test_openapi_file_is_in_sync():
    """docs/openapi.yaml – выгрузка app.openapi() (scripts/export_openapi.py); после правки API перевыгрузить."""
    saved = yaml.safe_load(Path(PROJECT_ROOT / "docs" / "openapi.yaml").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(create_app().openapi()))


def test_openapi_examples_match_schemas():
    """Примеры ответов в OpenAPI проходят схемы: null-поля (маска, вероятность, плановые_работы) не теряются."""
    models = {"health": sc.Health, "objects": list[sc.ObjectInfo], "events": sc.IngestReport,
              "predict": sc.PredictSummary, "predictions": sc.Predictions, "schedule": list[sc.ScheduleWindow]}
    from pydantic import TypeAdapter
    spec = json.loads(json.dumps(create_app().openapi()))
    seen = set()
    for ops in spec["paths"].values():
        for op in ops.values():
            ex = op["responses"]["200"]["content"]["application/json"].get("example")
            if ex is not None:
                TypeAdapter(models[op["operationId"]]).validate_python(ex)
                seen.add(op["operationId"])
    assert "predictions" in seen


def test_cache_write_is_atomic(tmp_path, monkeypatch):
    """Кэш прогноза заменяется целиком: параллельный GET /predictions не видит недописанный файл."""
    import pandas as pd
    import pytest

    from annanas.service.app import write_parquet_atomic
    f = tmp_path / "2026-08-02.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(f)

    def broken(self, path, **kw):   # запись оборвалась на середине
        Path(path).write_bytes(b"PAR1 half")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken)
    with pytest.raises(OSError):
        write_parquet_atomic(pd.DataFrame({"a": [2]}), f)
    monkeypatch.undo()
    assert pd.read_parquet(f)["a"].tolist() == [1]
    assert list(tmp_path.iterdir()) == [f]
    write_parquet_atomic(pd.DataFrame({"a": [3]}), f)
    assert pd.read_parquet(f)["a"].tolist() == [3] and list(tmp_path.iterdir()) == [f]


def test_predictions_cache_removed_while_reading(tmp_path, monkeypatch):
    """Кэш удален загрузкой событий между проверкой и чтением – 404, а не 500."""
    from annanas.service import app as service_app
    cache = tmp_path / "service" / "predictions" / "2026-08-02.parquet"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"")

    def gone(path, *a, **kw):
        Path(path).unlink()
        raise FileNotFoundError(path)

    monkeypatch.setattr(service_app.pd, "read_parquet", gone)
    r = TestClient(create_app(tmp_path)).get("/predictions", params={"date": "2026-08-02"})
    assert r.status_code == 404 and "POST /predict" in r.json()["ошибка"]


def test_compose_limits_malloc_arenas():
    """Сервис многопоточный: без MALLOC_ARENA_MAX glibc держит арену на поток, и под нагрузкой (загрузка событий
    параллельно с прогнозом и чтением) процесс Python раздувался до 6 ГБ, контейнер падал по OOM (e2e)."""
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert str(compose["services"]["inference"]["environment"].get("MALLOC_ARENA_MAX")) == "2"


def test_ingest_finishing_recompute_drops_stale_forecasts(tmp_path, monkeypatch):
    """Повторная загрузка довела пересчет после сбоя (новых строк 0): прогнозы после пересчитано_с устарели."""
    from annanas.inference import ingest
    from annanas.service.app import Service
    svc = Service(tmp_path)
    svc.pred_dir.mkdir(parents=True)
    for d in ["2026-07-30", "2026-07-31", "2026-08-02"]:
        (svc.pred_dir / f"{d}.parquet").write_bytes(b"")
    monkeypatch.setattr(ingest, "ingest_csv", lambda spark, path, root: {
        "добавлено": 0, "пересчитано_с": dt.date(2026, 7, 30), "пересчитано_по": dt.date(2026, 8, 1)})
    svc.__dict__["spark"] = None
    svc.ingest(tmp_path / "x.csv")
    assert sorted(f.stem for f in svc.pred_dir.glob("*.parquet")) == ["2026-07-30"]
