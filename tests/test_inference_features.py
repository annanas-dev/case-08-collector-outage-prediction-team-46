"""Путь сервиса: признаки на одни сутки по данным, обрезанным на полночь, совпадают с матрицей обучения.

Для нескольких суток разных выборок: источники из кэшей `datasets/processed` → `cut(S, D)` → `build_all` на
сетке «все объекты × {D}» → сравнение со строками `features.parquet` (все признаки матрицы, atol 1e-6).
Долгий тест (`make test-slow`), нужны данные, построенные ноутбуками 01-04."""
import time

import numpy as np
import pandas as pd
import pytest

from annanas import aggregates as agg
from annanas.inference import features as fi
from annanas.paths import PROCESSED_DIR

pytestmark = pytest.mark.slow

DAYS = ["2021-09-01", "2024-07-15", "2025-06-10", "2025-11-19", "2026-03-02", "2026-06-30"]


@pytest.fixture(scope="module")
def data():
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        pytest.skip("нет datasets/processed/features.parquet")
    feats = pd.read_parquet(path)
    feats["день"] = pd.to_datetime(feats["день"])
    catalog = pd.read_parquet(PROCESSED_DIR / "features_catalog.parquet")
    cols = catalog.loc[catalog["решение"] == "оставить", "признак"].tolist()
    channels = agg.read_vitrine(PROCESSED_DIR / "channels.parquet")
    ctx = fi.load_context(PROCESSED_DIR / "features_context.json", channels)
    S = fi.read_sources()
    S["mask"] = feats[fi.KEY + ["маска"]]
    return feats, cols, ctx, S


@pytest.mark.parametrize("day", DAYS)
def test_features_for_one_day_match_matrix(data, day):
    feats, cols, ctx, S = data
    D = pd.Timestamp(day)
    rows = feats[feats["день"] == D].reset_index(drop=True)
    grid = rows[fi.KEY + ["период_смву", "эпоха_флага"]]
    t0 = time.time()
    X = fi.build_all(fi.cut(S, D), grid, ctx)
    seconds = time.time() - t0
    print(f"\n{day}: {len(grid)} объектов, {len(cols)} признаков, cut + build_all {seconds:.1f} с")
    assert X[fi.KEY].equals(grid[fi.KEY])
    got = X[cols].to_numpy(dtype="float32").astype(float)
    exp = rows[cols].to_numpy(dtype=float)
    bad = ~np.isclose(got, exp, rtol=0, atol=1e-6, equal_nan=True)
    assert not bad.any(), {c: int(bad[:, j].sum()) for j, c in enumerate(cols) if bad[:, j].any()}
