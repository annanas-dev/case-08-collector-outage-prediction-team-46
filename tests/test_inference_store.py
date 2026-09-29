"""Store сервиса на реальных кэшах: маски = целевая, признаки на сутки = матрица обучения (make test-slow)."""
import time

import numpy as np
import pandas as pd
import pytest

from annanas.inference.store import Store
from annanas.paths import PROCESSED_DIR

pytestmark = pytest.mark.slow
N_DAYS = 30


@pytest.fixture(scope="module")
def store():
    if not (PROCESSED_DIR / "features.parquet").exists():
        pytest.skip("нет datasets/processed/features.parquet")
    return Store(PROCESSED_DIR)


@pytest.fixture(scope="module")
def feats():
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    f["день"] = pd.to_datetime(f["день"])
    return f


def test_mask_history_equals_target(store):
    target = pd.read_parquet(PROCESSED_DIR / "target.parquet", columns=["ид_объект", "день", "маска"])
    target["день"] = pd.to_datetime(target["день"])
    m = target.merge(store.mask_history, on=["ид_объект", "день"], how="left", suffixes=("_цель", "_сервис"))
    assert len(m) == len(target)
    diff = m["маска_цель"].fillna("-") != m["маска_сервис"].fillna("-")
    assert not diff.any(), m[diff].head(10)


def test_ppr_windows_equal_target_windows(store):
    saved = pd.read_parquet(PROCESSED_DIR / "maintenance" / "ppr_windows.parquet")
    assert store.windows.reset_index(drop=True).equals(saved.reset_index(drop=True))


def test_features_equal_matrix_on_validation_days(store, feats):
    catalog = pd.read_parquet(PROCESSED_DIR / "features_catalog.parquet")
    cols = catalog.loc[catalog["решение"] == "оставить", "признак"].tolist()
    days = feats.loc[feats["выборка"] == "валидация", "день"].drop_duplicates().sample(N_DAYS, random_state=0)
    t0 = time.time()
    for D in sorted(days):
        X, _ = store.features(D)
        rows = feats[feats["день"] == D].reset_index(drop=True)
        assert X[["ид_объект", "день"]].equals(rows[["ид_объект", "день"]])
        assert X["маска"].fillna("-").equals(rows["маска"].fillna("-")), D
        got = X[cols].to_numpy(dtype="float32").astype(float)
        bad = ~np.isclose(got, rows[cols].to_numpy(float), rtol=0, atol=1e-6, equal_nan=True)
        assert not bad.any(), (D, {c: int(bad[:, j].sum()) for j, c in enumerate(cols) if bad[:, j].any()})
    print(f"\n{N_DAYS} суток валидации: признаки совпали, {(time.time() - t0) / N_DAYS:.1f} с на сутки")
