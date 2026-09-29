"""Сквозной сценарий сервиса на копии хранилища (make test-slow, нужны Java и datasets/processed).

POST /events с демо-днем 2026-08-01 (журнал_событий_пример.csv) → повтор без изменений → POST /predict на
2026-08-02 → GET /predictions по схеме. В данных заказчика нет июля 2026: сутки июля – «нет данных в журнале»,
объекты, которые постоянно пишут, но молчали 2026-08-01, – «объект не на связи»; у 5011 окно ППР по графику."""
import shutil
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from annanas import target as tg
from annanas.inference import features as fi
from annanas.paths import DATASETS_DIR, PROCESSED_DIR
from annanas.service import schemas as sc
from annanas.service.app import create_app

pytestmark = pytest.mark.slow
SAMPLE = DATASETS_DIR / "журнал_событий_пример.csv"
COPY = ["channels.parquet", "features_catalog.parquet", "features_context.json", "models", "maintenance",
        "eda/daily_channel.parquet", "eda/undefined_daily.parquet", "eda/undefined_bursts.parquet",
        "eda/hourly_type.parquet", "eda/meas_gaps.parquet", "features/object_day_states.parquet",
        "episodes/flag_hourly.parquet", "episodes/comm_loss_k10_s50.parquet",
        # события: 2026 целиком (эпизоды года пересчитываются) и конец 2025 (запас в сутки)
        "events/год=2026", "events/год=2025/месяц=12"]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    if not SAMPLE.exists() or not (PROCESSED_DIR / "models" / "config.json").exists():
        pytest.skip("нет демо-журнала или моделей")
    root = tmp_path_factory.mktemp("store")
    for rel in COPY + [f"episodes/{fi.episodes_dir(PROCESSED_DIR).name}"]:
        src, dst = PROCESSED_DIR / rel, root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
    return root


def test_events_predict_predictions(root):
    client = TestClient(create_app(root))
    t0 = time.time()
    with open(SAMPLE, "rb") as f:
        r = client.post("/events", files={"file": ("пример.csv", f, "text/csv")})
    assert r.status_code == 200, r.json()
    rep = r.json()
    print(f"\nPOST /events: {rep}, {time.time() - t0:.0f} с")
    assert rep["строк_в_файле"] == 169_993 and rep["тревожных"] == 207
    assert rep["добавлено"] == rep["строк_после_очистки"] > 0 and rep["первый_день"] == "2026-08-01"
    again = client.post("/events", content=SAMPLE.read_bytes(), headers={"content-type": "text/csv"}).json()
    assert again["добавлено"] == 0 and again["уже_загружено"] == rep["строк_после_очистки"]

    health = client.get("/health").json()
    assert health["последний_день_данных"] == "2026-08-01" and health["прогноз_возможен_по"] == "2026-08-02"
    assert client.post("/predict", params={"date": "2026-08-03"}).status_code == 409

    t0 = time.time()
    r = client.post("/predict", params={"date": "2026-08-02"})
    seconds = time.time() - t0
    assert r.status_code == 200, r.json()
    summary = r.json()
    print(f"POST /predict 2026-08-02: {summary}, {seconds:.1f} с ({seconds / summary['объектов']:.2f} с на объект)")
    assert summary["объектов"] == 78 and summary["предупреждения"]           # пропуск июля в окне 30 суток
    assert seconds / summary["объектов"] < 300                                # ТЗ: не больше 300 с на объект

    p = sc.Predictions.model_validate(client.get("/predictions", params={"date": "2026-08-02"}).json())
    by = {o.ид_объект: o for o in p.объекты}
    sample = pd.read_csv(SAMPLE, usecols=["ид_канала_данных"]).drop_duplicates()
    channels = pd.read_parquet(root / "channels.parquet", columns=["ид_канала_данных", "ид_объект"])
    wrote = set(sample.merge(channels, on="ид_канала_данных")["ид_объект"])
    offline = {o for o, f in by.items() if f.маска == tg.OFFLINE}
    print(f"объектов в демо-дне: {len(wrote)}, «не на связи» на 2026-08-02: {sorted(offline)}")
    assert offline and not offline & wrote
    for o, f in by.items():
        assert (f.маска is None) == (f.прогноз.отказ.вероятность is not None)
        assert f.маска is not None or f.прогноз.любой_инцидент.уровень != "нет данных"
    vita = by[5011]
    assert vita.маска == "плановые работы АКМ"
    assert vita.плановые_работы.model_dump(mode="json") == {"начало": "2026-07-28", "конец": "2026-08-21",
                                                            "источник": "график"}
    one = client.get("/predictions", params={"date": "2026-08-02", "ид_объект": 5011}).json()
    assert [o["ид_объект"] for o in one["объекты"]] == [5011]
    assert client.get("/predictions", params={"date": "2026-08-02", "ид_объект": 1}).status_code == 404
    assert any(w["ид_объект"] == 5011 for w in client.get("/schedule", params={"date": "2026-08-02"}).json())
