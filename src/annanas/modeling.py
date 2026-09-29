"""Модели этапа 4: выборки, метрики, обучение, калибровка, причины прогноза. Решения – в notebooks/05_model."""
import json
import os
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from annanas.episodes import FREQUENCY_CLASSES
from annanas.features import average_precision  # noqa: F401  PR-AUC без sklearn, одинаковые оценки – один порог
from annanas.features import window_sums

MODEL_TARGETS = ["цель_отказ", "цель_пожар", "цель_подтопление", "цель_любой_инцидент"]
FREQUENCY_TARGETS = [f"цель_{c}" for c in FREQUENCY_CLASSES] + ["цель_потеря_связи"]


def feature_list(catalog: pd.DataFrame) -> list[str]:
    return catalog.loc[catalog["решение"] == "оставить", "признак"].tolist()


def working_rows(fm: pd.DataFrame, split: str) -> pd.DataFrame:
    """Строки выборки без маски: только они идут в обучение и оценку."""
    return fm[(fm["выборка"] == split) & fm["маска"].isna()]


def precision_recall_at_top(frame, y_col, p_col, n) -> dict:
    """Каждые сутки берем n объектов с наибольшей вероятностью. Точность – среднее по дням доли попаданий в топе;
    полнота – среднее по дням с инцидентами доли инцидентов дня, попавших в топ; полнота пулом (recall_pooled) –
    все попадания за период к всем инцидентам периода. Среднее по дням завышает полноту: день с одним
    инцидентом весит столько же, сколько день с десятью. Для обещаний заказчику – пулом. Нет инцидентов – NaN.

    Порядок детерминирован: при равных вероятностях выше меньший ид_объект, NaN – в конце; от порядка строк
    не зависит. Если в сутках меньше n объектов, в топ идут все, и точность дня – доля попаданий среди них."""
    f = frame.sort_values(["день", p_col, "ид_объект"], ascending=[True, False, True],
                          na_position="last", kind="mergesort")
    top = f[f.groupby("день").cumcount().to_numpy() < n]
    hits = top.groupby("день")[y_col].sum()
    per_day = f.groupby("день")[y_col].sum()
    precision = (hits / top.groupby("день").size()).mean()
    has = per_day > 0
    recall = (hits[has] / per_day[has]).mean()
    total = per_day.sum()
    pooled = hits.sum() / total if total > 0 else np.nan
    return {"precision": float(precision), "recall": float(recall), "recall_pooled": float(pooled)}


def brier(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def reliability_table(y, p, bins=10) -> pd.DataFrame:
    """Корзины по прогнозу равной ширины на [0, 1]: средний прогноз против фактической доли; пустые корзины
    не выводятся."""
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"корзина": idx, "y": np.asarray(y, float), "p": p})
    return df.groupby("корзина").agg(прогноз_средний=("p", "mean"), доля_фактическая=("y", "mean"),
                                     количество=("y", "size")).reset_index()


def block_bootstrap(frame, metric, block_days=28, n=1000, seed=0):
    """Бутстреп по блокам суток: соседние дни зависимы, поэтому пересэмплируем блоки целиком. Недельные блоки
    занижают интервал (зависимость длиннее недели), по умолчанию – 28 суток.

    Возвращает (оценка на всей выборке, 2,5 %, 97,5 %). В каждой повторной выборке блоки раскладываются подряд
    и `день` перенумеровывается: блок, выпавший дважды, дает разные сутки, а не одни с удвоенным числом объектов.
    Поэтому метрика должна использовать `день` только как ключ суток. Индексы блоков считаются один раз."""
    days = pd.to_datetime(frame["день"])
    start = days.min().to_datetime64().astype("datetime64[D]")
    day = (days.to_numpy().astype("datetime64[D]") - start).astype(np.int64)
    blk = day // block_days
    order = np.argsort(blk, kind="stable")
    keys, first, sizes = np.unique(blk[order], return_index=True, return_counts=True)
    blocks = np.split(order, first[1:])
    base = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([blocks[k] for k in pick])
        shift = np.repeat((np.arange(len(keys)) - keys[pick]) * block_days, sizes[pick])
        sample = base.take(idx)
        sample["день"] = start + (day[idx] + shift).astype("timedelta64[D]")
        vals.append(metric(sample))
    lo, hi = np.nanpercentile(np.asarray(vals, float), [2.5, 97.5])
    return float(metric(frame)), float(lo), float(hi)


def cluster_bootstrap(frame, metric, cluster_col="ид_объект", n=1000, seed=0):
    """Бутстреп по объектам: строки одного объекта зависимы, пересэмплируем объекты целиком. Объект, выпавший
    k раз, дает k разных объектов (новые ид 0, 1, ... по порядку выбора), чтобы топ-N за сутки не слипался.
    Возвращает (оценка на всей выборке, 2,5 %, 97,5 %)."""
    base = frame.reset_index(drop=True)
    codes, uniq = pd.factorize(base[cluster_col])
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes, minlength=len(uniq))
    groups = np.split(order, np.cumsum(sizes)[:-1])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([groups[k] for k in pick])
        sample = base.take(idx)
        sample[cluster_col] = np.repeat(np.arange(len(pick)), sizes[pick])
        vals.append(metric(sample))
    lo, hi = np.nanpercentile(np.asarray(vals, float), [2.5, 97.5])
    return float(metric(frame)), float(lo), float(hi)


def rolling_folds(days: pd.Series, valid_years):
    """Скользящие фолды: учимся на всем до года, проверяем на году. Индексы позиционные."""
    days = pd.to_datetime(pd.Series(days)).reset_index(drop=True)
    out = []
    for year in valid_years:
        start = pd.Timestamp(f"{year}-01-01")
        out.append((np.flatnonzero(days < start), np.flatnonzero(days.dt.year == year), year))
    return out


_JSON_SPECIAL = re.compile(r'[\[\]{}",:]')


def _safe_names(names) -> list[str]:
    """Имена признаков для LightGBM: кириллицу он принимает, символы JSON ([]{}",:) – нет; заменяем на «_»,
    при совпадении добавляем номер столбца. Причины прогноза берут имена из X по позиции, так что замена
    на них не влияет."""
    out, seen = [], set()
    for i, n in enumerate(map(str, names)):
        s = _JSON_SPECIAL.sub("_", n)
        if s in seen:
            s = f"{s}_{i}"
        seen.add(s)
        out.append(s)
    return out


def fit_lgbm(X: pd.DataFrame, y, params, n_estimators, seed=0) -> lgb.Booster:
    """LightGBM на всех ядрах (num_threads = os.cpu_count(), в контейнере 11); params перекрывают базовые."""
    base = {"objective": "binary", "verbose": -1, "seed": seed, "num_threads": os.cpu_count()}
    names = _safe_names(X.columns)
    data = lgb.Dataset(np.asarray(X), label=np.asarray(y), feature_name=names)
    return lgb.train({**base, **params}, data, num_boost_round=n_estimators)


# Набор 39 случайного поиска (исследование моделей): лучший для отказа и любого инцидента на фолдах 2022-2024
PARAMS_CONFIG39 = {"num_leaves": 63, "min_data_in_leaf": 50, "learning_rate": 0.03, "feature_fraction": 0.7,
                   "bagging_fraction": 1.0, "bagging_freq": 1, "lambda_l2": 10.0, "lambda_l1": 1.0,
                   "min_gain_to_split": 0.0, "max_depth": -1, "extra_trees": True, "path_smooth": 10.0}


class SeedBag:
    """Модели LightGBM с разными сидами. predict – среднее вероятностей; с pred_contrib=True – средние вклады
    (среднее SHAP-значений в логитах; их сумма – средний логит моделей, а не логит средней вероятности, но
    для ранжирования причин этого достаточно). Интерфейс predict как у Booster, поэтому predict_lgbm и
    top_reasons работают без изменений."""

    def __init__(self, models):
        self.models = list(models)

    def predict(self, X, **kwargs):
        return np.mean([m.predict(X, **kwargs) for m in self.models], axis=0)

    def save(self, prefix) -> list:
        """Файлы `<prefix>.seed<i>.lgb.txt` (i – номер модели); возвращает пути."""
        prefix = Path(prefix)
        files = [prefix.with_name(f"{prefix.name}.seed{i}.lgb.txt") for i in range(len(self.models))]
        for m, f in zip(self.models, files):
            m.save_model(str(f))
        return files

    @classmethod
    def load(cls, files):
        return cls(lgb.Booster(model_file=str(f)) for f in files)


def fit_lgbm_seeds(X: pd.DataFrame, y, params, n_estimators, seeds=(0, 1, 2, 3, 4)) -> SeedBag:
    """fit_lgbm для каждого сида: усреднение 5 сидов устойчиво добавляет PR-AUC (исследование моделей).
    Сид задает bagging и feature_fraction; без них модели совпадут."""
    return SeedBag(fit_lgbm(X, y, params, n_estimators, seed=s) for s in seeds)


def fit_lgbm_early(X, y, X_val, y_val, params, max_rounds=2000, patience=100, seed=0):
    """LightGBM с ранней остановкой по PR-AUC (average_precision) на (X_val, y_val).
    Для подбора параметров внутри скользящего фолда. Возвращает (модель, лучшая итерация, PR-AUC на ней)."""
    base = {"objective": "binary", "verbose": -1, "seed": seed, "num_threads": os.cpu_count(),
            "metric": "average_precision"}
    names = _safe_names(X.columns)
    train = lgb.Dataset(np.asarray(X), label=np.asarray(y), feature_name=names)
    valid = lgb.Dataset(np.asarray(X_val), label=np.asarray(y_val), reference=train)
    model = lgb.train({**base, **params}, train, num_boost_round=max_rounds, valid_sets=[valid],
                      callbacks=[lgb.early_stopping(patience, verbose=False)])
    return model, int(model.best_iteration), float(model.best_score["valid_0"]["average_precision"])


def predict_lgbm(model, X, **kwargs) -> np.ndarray:
    """Прогноз по позиции столбцов (как при обучении), без сверки имен с DataFrame. model – Booster или SeedBag."""
    return model.predict(np.asarray(X), **kwargs)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """Изотоническая регрессия или Platt; сохраняется в JSON (UTF-8), чтобы сервис не зависел от pickle.

    Изотоническая хранит точки (x, y) sklearn IsotonicRegression; прогноз – линейная интерполяция между ними
    с обрезкой по краям, то же, что IsotonicRegression.predict. Platt: σ(a · logit(p) + b)."""

    METHODS = ("изотоническая", "Платт")

    def __init__(self, method, payload):
        self.method, self.payload = method, payload

    @classmethod
    def fit(cls, p, y, method="изотоническая"):
        if method not in cls.METHODS:
            raise ValueError(f"метод калибровки: {cls.METHODS}, получено {method!r}")
        p = np.asarray(p, float); y = np.asarray(y, int)
        if method == "изотоническая":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
            return cls(method, {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()})
        z = _logit(p)
        if y.min() == y.max() or np.ptp(z) == 0:   # одна метка или один прогноз: калибруем в среднюю долю
            return cls(method, {"a": 0.0, "b": float(_logit(y.mean()))})
        lr = LogisticRegression(C=1e6).fit(z[:, None], y)
        return cls(method, {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])})

    def predict(self, p):
        p = np.asarray(p, float)
        if self.method == "изотоническая":
            return np.interp(p, self.payload["x"], self.payload["y"])
        return 1 / (1 + np.exp(-(self.payload["a"] * _logit(p) + self.payload["b"])))

    def save(self, path):
        path.write_text(json.dumps({"method": self.method, **self.payload}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d.pop("method"), d)


def smoothed_rate(history: pd.DataFrame, y_col, prior_strength) -> pd.Series:
    """Доля суток с целью на объекте, сжатая к общей доле: (k + s·p0) / (n + s)."""
    p0 = history[y_col].mean()
    g = history.groupby("ид_объект")[y_col].agg(["sum", "size"])
    return (g["sum"] + prior_strength * p0) / (g["size"] + prior_strength)


def smoothed_rate_before(grid_with_y: pd.DataFrame, y_col, prior_strength, window_days=365) -> pd.Series:
    """Сглаженная доля суток с целью на объекте строго до D: по дням [D − window, D − 1],
    (k + s·p0) / (n + s), где k – суток с целью, n – суток с известной меткой (NaN в y_col не считается).

    p0 – общая доля по всем объектам за то же окно [D − window, D − 1], а не за период обучения: так она тоже
    строго в прошлом, одинаково считается на обучении, валидации и в сервисе и следует за ростом доли после
    смены разметки. Метка дня D − 1 – эпизод, начавшийся в [D − 1, D), к D 00:00 уже известна (кроме длительности
    эпизода, начавшегося незадолго до полуночи; так же устроен частотный прогноз этапа 3). Без прошлого – NaN.
    Возвращает Series с индексом grid_with_y."""
    obs = grid_with_y.loc[grid_with_y[y_col].notna(), ["ид_объект", "день", y_col]]
    daily = obs.assign(_k=obs[y_col].astype(float), _n=1.0)
    own = window_sums(grid_with_y, daily, ["_k", "_n"], [window_days])
    k, n = own[f"_k_{window_days}д"].to_numpy(), own[f"_n_{window_days}д"].to_numpy()
    total = daily.groupby("день", as_index=False)[["_k", "_n"]].sum().assign(ид_объект=0)
    glob = window_sums(grid_with_y[["день"]].assign(ид_объект=0), total, ["_k", "_n"], [window_days])
    gk, gn = glob[f"_k_{window_days}д"].to_numpy(), glob[f"_n_{window_days}д"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        p0 = np.where(gn > 0, gk / gn, np.nan)
    return pd.Series((k + prior_strength * p0) / (n + prior_strength), index=grid_with_y.index, name=y_col)


def top_reasons(model, X: pd.DataFrame, meaning: dict, k=3) -> list[list[dict]]:
    """Топ-k признаков, сильнее всего поднявших риск в этой строке: вклады в логит из pred_contrib (SHAP-значения
    деревьев без пакета shap; у SeedBag – средние по сидам). Только положительные вклады, по убыванию; при равенстве – порядок столбцов.
    Имена – исходные столбцы X (по позиции), текст – из meaning, иначе имя признака."""
    contrib = predict_lgbm(model, X, pred_contrib=True)[:, :-1]  # последний столбец – базовое значение
    cols = X.columns.to_numpy()
    out = []
    for row, vals in zip(contrib, np.asarray(X, dtype=float)):
        order = np.argsort(-row, kind="stable")[:k]
        out.append([{"признак": cols[i], "смысл": meaning.get(cols[i], cols[i]),
                     "значение": float(vals[i]), "вклад": float(row[i])} for i in order if row[i] > 0])
    return out


def threshold_report(frame, y_col, p_col, threshold) -> dict:
    """Что даст порог «p ≥ threshold»: средняя нагрузка (объектов в сутки), худший месяц по средней нагрузке,
    точность и полнота пулом за весь период. Сутки без отмеченных объектов тоже входят в среднее."""
    flag = frame[p_col].to_numpy(float) >= threshold
    y = frame[y_col].to_numpy(float)
    days = pd.to_datetime(frame["день"])
    per_day = pd.Series(flag, index=days).groupby(level=0).sum()
    monthly = per_day.groupby(per_day.index.to_period("M")).mean()
    return {"порог": float(threshold), "нагрузка_в_сутки": float(per_day.mean()),
            "нагрузка_макс_месяц": float(monthly.max()),
            "точность": float(y[flag].mean()) if flag.any() else np.nan,
            "полнота": float(y[flag].sum() / y.sum()) if y.sum() > 0 else np.nan}


def thresholds_with_cap(frame, y_col, p_col, max_per_day):
    """Самый низкий порог, при котором ни в одном месяце frame средняя нагрузка не больше max_per_day объектов
    в сутки. Порог подбирается на прошлом периоде (например, валидации) и проверяется на следующем через
    threshold_report – так обещание «не больше N объектов в сутки» честное, вне выборки.

    max_per_day – число (вернется dict threshold_report) или {уровень: потолок} (DataFrame по уровням).
    Кандидаты – сами значения p; нагрузка не растет с порогом, поэтому двоичный поиск."""
    if isinstance(max_per_day, dict):
        return pd.DataFrame({lvl: thresholds_with_cap(frame, y_col, p_col, cap)
                             for lvl, cap in max_per_day.items()}).T
    cand = np.unique(frame[p_col].dropna().to_numpy(float))
    lo, hi = 0, len(cand)           # ищем первый кандидат, где худший месяц ≤ потолка
    while lo < hi:
        mid = (lo + hi) // 2
        if threshold_report(frame, y_col, p_col, cand[mid])["нагрузка_макс_месяц"] <= max_per_day:
            hi = mid
        else:
            lo = mid + 1
    t = cand[lo] if lo < len(cand) else np.inf
    return threshold_report(frame, y_col, p_col, t)
