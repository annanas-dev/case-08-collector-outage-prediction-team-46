"""Прогноз сервиса на днях теста = журнал прогнозов predictions.parquet из notebooks/05_model (make test-slow).

Вероятности валидации в журнале – с перекрестной калибровкой, поэтому сверка только на тесте."""
import numpy as np
import pandas as pd
import pytest

from annanas.inference.scorer import Scorer, predict_day
from annanas.inference.store import Store
from annanas.paths import PROCESSED_DIR

pytestmark = pytest.mark.slow
DAYS = ["2026-01-05", "2026-02-17", "2026-03-31", "2026-04-20", "2026-05-12", "2026-06-30"]


@pytest.fixture(scope="module")
def setup():
    if not (PROCESSED_DIR / "predictions.parquet").exists():
        pytest.skip("нет datasets/processed/predictions.parquet")
    store = Store(PROCESSED_DIR)
    scorer = Scorer(PROCESSED_DIR / "models", PROCESSED_DIR / "features_catalog.parquet")
    pred = pd.read_parquet(PROCESSED_DIR / "predictions.parquet")
    return store, scorer, pred[pred["выборка"] == "тест"]


@pytest.mark.parametrize("day", DAYS)
def test_predictions_equal_journal(setup, day):
    store, scorer, pred = setup
    out, summary = predict_day(store, scorer, day)
    exp = pred[pred["день"] == pd.Timestamp(day).date()].set_index("ид_объект")
    got = out.set_index("ид_объект")
    assert got.loc[exp.index, "маска"].isna().all()                  # в журнале только сутки без маски
    assert set(got.index[got["маска"].isna()]) == set(exp.index)
    got = got.loc[exp.index]
    for t in scorer.targets + scorer.freq_targets:
        err = np.abs(got[f"вероятность_{t}"].to_numpy() - exp[f"вероятность_{t}"].to_numpy()).max()
        assert err <= 1e-9, (t, err)
    for t in scorer.targets:
        assert (got[f"уровень_{t}"] == exp[f"уровень_{t}"]).all(), t
        assert (got[f"причины_{t}"] == exp[f"причины_{t}"]).all(), t
    assert (got["каналы_отказ"] == exp["каналы_отказ"]).all() and (got["пикеты_отказ"] == exp["пикеты_отказ"]).all()
    print(f"\n{day}: {summary['с_прогнозом']} объектов с прогнозом, {summary['секунд']} с")
