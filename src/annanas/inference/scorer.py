"""Прогноз на сутки D по артефактам `datasets/processed/models` (их пишет notebooks/05_model).

Для каждого объекта:

- `маска` – почему прогноза нет (объект не на связи, плановые работы АКМ, ...), иначе None;
- модельные цели (отказ, пожар, подтопление, любой инцидент): `вероятность` (среднее пяти копий LightGBM после
  калибровки Platt), `балл` (то же до калибровки), `уровень` по порогам config.json, `причины` – топ-3 признака,
  поднявших риск (modeling.top_reasons); у замаскированных объектов – пропуск и уровень «нет данных»;
- частотные цели (газ, потеря связи, неисправность охраны): сглаженная частота за 365 суток до D
  (modeling.smoothed_rate_before с силой сжатия из frequency.json) – «фоновый риск»;
- где искать отказ: топ-3 каналов и пикетов (location.rank_grid);
- окно плановых работ АКМ, в которое попадают сутки D (с запасом ±3 сут.), или None.

Результат – плоская таблица в формате журнала прогнозов `predictions.parquet` (причины, каналы и окно работ –
JSON-строки) плюс `балл_<цель>`, `маска`, `плановые_работы`."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from annanas import location as loc
from annanas import modeling as md
from annanas.inference.features import KEY

NO_DATA = "нет данных"


def level(p, thresholds) -> np.ndarray:
    """Уровень риска по порогам {"высокий": ..., "средний": ...}; пропуск вероятности – «нет данных»."""
    p = np.asarray(p, float)
    out = np.where(p >= thresholds["высокий"], "высокий", np.where(p >= thresholds["средний"], "средний", "низкий"))
    return np.where(np.isnan(p), NO_DATA, out).astype(object)


def meaning_from_catalog(catalog: pd.DataFrame) -> dict:
    """Смысл признака для причин прогноза – как в notebooks/05_model: окно дописывается, если его нет в тексте."""
    kept = catalog[catalog["решение"] == "оставить"].set_index("признак")
    return {f: (m if pd.isna(w) or "окно" in m else f"{m}, окно {int(w)} сут.")
            for f, m, w in zip(kept.index, kept["смысл"], kept["окно"])}


class Scorer:
    def __init__(self, models_dir, catalog_file):
        models_dir = Path(models_dir)
        self.config = json.loads((models_dir / "config.json").read_text(encoding="utf-8"))
        freq = json.loads((models_dir / "frequency.json").read_text(encoding="utf-8"))
        self.features = self.config["features"]
        self.targets = list(self.config["targets"])
        self.models = {t: md.SeedBag.load([models_dir / f for f in spec["model_files"]])
                       for t, spec in self.config["targets"].items()}
        self.calibrators = {t: md.Calibrator.load(models_dir / spec["calibration_file"])
                            for t, spec in self.config["targets"].items()}
        self.thresholds = {t: spec["thresholds"] for t, spec in self.config["targets"].items()}
        self.freq_targets = self.config["frequency_targets"]
        self.freq_window = int(freq["window_days"])
        self.strength = {t: freq["targets"][t]["prior_strength"] for t in self.freq_targets}
        self.meaning = meaning_from_catalog(pd.read_parquet(catalog_file))
        self.k = int(self.config["location"]["k"])

    @property
    def version(self) -> str:
        return self.config["created_at"]

    def score_models(self, X: pd.DataFrame) -> pd.DataFrame:
        """Модельные цели по строкам X без маски: вероятность, балл, уровень, причины (JSON)."""
        out = pd.DataFrame(index=X.index)
        ok = X["маска"].isna().to_numpy()
        Xf = X.loc[ok, self.features].astype("float32")   # матрица обучения хранит признаки во float32
        for t in self.targets:
            raw = np.full(len(X), np.nan)
            prob = np.full(len(X), np.nan)
            reasons = ["[]"] * len(X)
            if ok.any():
                raw[ok] = md.predict_lgbm(self.models[t], Xf)
                prob[ok] = self.calibrators[t].predict(raw[ok])
                for i, r in zip(np.flatnonzero(ok), md.top_reasons(self.models[t], Xf, self.meaning, k=3)):
                    reasons[i] = json.dumps(r, ensure_ascii=False)
            out[f"вероятность_{t}"] = prob
            out[f"балл_{t}"] = raw
            out[f"уровень_{t}"] = level(prob, self.thresholds[t])
            out[f"причины_{t}"] = reasons
        return out

    def background(self, ev: pd.DataFrame, mask_history: pd.DataFrame, D) -> pd.DataFrame:
        """Сглаженная частота частотных целей на сутки D по всем объектам (порядок – по ид_объект).
        История – сутки [D − окно, D − 1]: 1, если в сутки стал известен длинный эпизод класса, пропуск – сутки
        с маской (как `известно_<цель>` в notebooks/05_model)."""
        D = pd.Timestamp(D)
        h = mask_history[(mask_history["день"] >= D - pd.Timedelta(days=self.freq_window))
                         & (mask_history["день"] <= D)].reset_index(drop=True)
        e = ev[ev["длинный"] & ev["класс"].isin(self.freq_targets)]
        hit = e[KEY + ["класс"]].drop_duplicates()
        out = h.loc[h["день"] == D, ["ид_объект"]].reset_index(drop=True)
        for t in self.freq_targets:
            y = h[KEY].merge(hit[hit["класс"] == t][KEY].assign(_y=1.0), on=KEY, how="left")["_y"].fillna(0.0)
            h[t] = y.where(h["маска"].isna().to_numpy()).to_numpy()
            r = md.smoothed_rate_before(h, t, self.strength[t])
            out[f"вероятность_{t}"] = r[h["день"] == D].to_numpy()
        return out


def ppr_json(windows: pd.DataFrame, objects) -> list:
    """Окно работ объекта (JSON) или None; при нескольких окнах – самое позднее по началу."""
    w = windows.sort_values("начало").groupby("ид_объект").last()
    return [None if o not in w.index else json.dumps(
        {"начало": f"{w.at[o, 'начало']:%Y-%m-%d}", "конец": f"{w.at[o, 'конец']:%Y-%m-%d}",
         "источник": w.at[o, "источник"]}, ensure_ascii=False) for o in objects]


def predict_day(store, scorer: Scorer, D) -> tuple[pd.DataFrame, dict]:
    """Прогноз на сутки D для всех объектов: плоская таблица и сводка (объектов, с прогнозом, секунд,
    предупреждения)."""
    t0 = time.time()
    D = store.check_day(D)
    X, S = store.features(D)
    out = X[KEY + ["маска"]].copy()
    out = pd.concat([out, scorer.score_models(X)], axis=1)
    bg = scorer.background(S["ev"], store.mask_history, D)
    out = out.merge(bg, on="ид_объект", how="left")
    for lvl, col in [("канал", "каналы_отказ"), ("пикет", "пикеты_отказ")]:
        out[col] = [json.dumps(x, ensure_ascii=False)
                    for x in loc.rank_grid(store.eps_fault, out[KEY], k=scorer.k, level=lvl)]
    out["плановые_работы"] = ppr_json(store.active_windows(D), out["ид_объект"])
    out["день"] = out["день"].dt.date
    warnings = []
    gaps = store.gaps_before(D)
    if gaps:
        warnings.append(f"за 30 суток до {D:%Y-%m-%d} в журнале нет данных {gaps} сут.: признаки за эти окна "
                        f"занижены, прогноз менее надежен")
    masked = out["маска"].notna()
    summary = {"дата": D.date(), "объектов": len(out), "с_прогнозом": int((~masked).sum()),
               "замаскировано": {k: int(v) for k, v in out.loc[masked, "маска"].value_counts().items()},
               "секунд": round(time.time() - t0, 1), "предупреждения": warnings}
    return out, summary
