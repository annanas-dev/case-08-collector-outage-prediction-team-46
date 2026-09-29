"""Признаки «объект × сутки». Правило: признак дня D использует только данные строго до D 00:00.
Выбор и обоснование признаков – в notebooks/04_features.

Окна и «время с последнего» считаются по календарным дням, а не по строкам сетки: результат не зависит
от порядка строк, пропусков дней в сетке и от того, есть ли у объекта события. Сетка возвращается
в исходном порядке и с исходным индексом."""
import numpy as np
import pandas as pd

KEY = ["ид_объект", "день"]
_DAY_OFFSET = 1 << 30   # сдвиг номера дня, чтобы он был неотрицательным и в окне D − w − 1
_OBJ_STRIDE = 1 << 32   # ключ = код объекта · stride + день: сортировка по объекту, затем по дню


def daily_counts(events: pd.DataFrame, by: str, values: list, time_col="начало") -> pd.DataFrame:
    """Число событий каждого значения `by` за календарный день на объекте (широкая таблица)."""
    d = events.loc[events[by].isin(values), ["ид_объект", by, time_col]]
    d = d.assign(день=d[time_col].dt.floor("D"))
    out = d.groupby(["ид_объект", "день", by]).size().unstack(by, fill_value=0)
    out = out.reindex(columns=values, fill_value=0).reset_index()
    out.columns.name = None
    return out


def _day_number(days) -> np.ndarray:
    return pd.to_datetime(pd.Series(days)).to_numpy().astype("datetime64[D]").astype(np.int64) + _DAY_OFFSET


def _keys(grid, daily):
    """Общие ключи «объект, день» для сетки и суточного ряда (коды объектов общие)."""
    codes, _ = pd.factorize(pd.concat([grid["ид_объект"], daily["ид_объект"]], ignore_index=True))
    g_code, d_code = codes[: len(grid)].astype(np.int64), codes[len(grid):].astype(np.int64)
    return g_code, _day_number(grid["день"]), d_code * _OBJ_STRIDE + _day_number(daily["день"])


def window_sums(grid, daily, cols, windows) -> pd.DataFrame:
    """Сумма `cols` за дни [D − w, D − 1] для каждого окна w; столбцы `<col>_<w>д`.

    Префиксные суммы по отсортированным ключам «объект, день»: сумма окна = C(D − 1) − C(D − w − 1), где
    C(t) – сумма по строкам daily того же объекта с днем ≤ t. Дни daily вне сетки тоже учитываются."""
    g_code, g_day, d_key = _keys(grid, daily)
    order = np.argsort(d_key, kind="stable")
    d_key = d_key[order]
    vals = daily[cols].to_numpy(dtype=float)[order]
    vals = np.nan_to_num(vals)
    cum = np.vstack([np.zeros((1, len(cols))), np.cumsum(vals, axis=0)])
    base = g_code * _OBJ_STRIDE
    hi = np.searchsorted(d_key, base + g_day - 1, side="right")
    out = grid[KEY].copy()
    for w in windows:
        lo = np.searchsorted(d_key, base + g_day - w - 1, side="right")
        s = cum[hi] - cum[lo]
        for j, c in enumerate(cols):
            out[f"{c}_{w}д"] = s[:, j]
    return out


def days_since_last(grid, daily, cols, cap=3650) -> pd.DataFrame:
    """Дней с последнего дня, где col > 0, строго до D; если не было – cap и флаг `не_было_<col>`."""
    g_code, g_day, _ = _keys(grid, daily)
    base = g_code * _OBJ_STRIDE
    out = grid[KEY].copy()
    for c in cols:
        hit = daily[daily[c].fillna(0) > 0]
        _, _, d_key = _keys(grid, hit)
        d_key = np.sort(d_key)
        i = np.searchsorted(d_key, base + g_day - 1, side="right") - 1   # последний день ≤ D − 1
        prev = d_key[np.clip(i, 0, None)] if len(d_key) else np.zeros(len(grid), np.int64)
        found = (i >= 0) & (prev // _OBJ_STRIDE == g_code)
        days = np.where(found, g_day - prev % _OBJ_STRIDE, cap)
        out[f"дней_с_последнего_{c}"] = np.minimum(days, cap).astype(int)
        out[f"не_было_{c}"] = (~found).astype("int8")
    return out


def valid_days(grid, windows, mask_col="маска", masks=None) -> pd.DataFrame:
    """Сколько дней объекта в [D − w, D − 1] есть в сетке без маски; столбцы `дней_с_данными_<w>д`.
    Дни вне сетки (до ее начала) считаются днями без данных. Тихий день без маски – день с данными: при маске
    connection_mask это обычный день без событий, и он должен быть в знаменателе долей (см. window_shares).
    masks – таблица `ид_объект, день, <mask_col>` с историей масок, если строки grid ее не содержат (например,
    сетка из одних суток прогноза); по умолчанию маски берутся из самой grid."""
    masks = grid if masks is None else masks
    ok = masks.loc[masks[mask_col].isna(), KEY].assign(_ok=1.0)
    out = window_sums(grid, ok, ["_ok"], windows)
    return out.rename(columns={f"_ok_{w}д": f"дней_с_данными_{w}д" for w in windows})


def window_shares(grid, daily, cols, windows, mask_col="маска", masks=None) -> pd.DataFrame:
    """Доля `cols` на день за [D − w, D − 1] по дням объекта в сетке без маски; столбцы `доля_<col>_<w>д`.

    Числитель и знаменатель берутся по одним и тем же дням: день под маской (объект не подключен или не на
    связи) выпадает из обоих, тихий день без маски – ноль в числителе и день в знаменателе. Дни daily вне
    сетки не считаются. Нет ни одного дня без маски – NaN. masks – история масок, как в valid_days."""
    masks = grid if masks is None else masks
    ok = masks.loc[masks[mask_col].isna(), KEY]
    num = ok.merge(daily[KEY + list(cols)], on=KEY, how="inner")
    sums = window_sums(grid, num, list(cols), windows)
    vd = valid_days(grid, windows, mask_col, masks)
    out = grid[KEY].copy()
    for w in windows:
        den = vd[f"дней_с_данными_{w}д"].replace(0, np.nan)
        for c in cols:
            out[f"доля_{c}_{w}д"] = sums[f"{c}_{w}д"] / den
    return out


def active_at_midnight(grid, intervals, name) -> pd.DataFrame:
    """Сколько интервалов объекта открыто в D 00:00: начало < D и (конец пуст или конец > D).

    Число открытых = #(начало < D) − #(конец' ≤ D), где конец' = max(конец, начало + 1 нс): интервал
    нулевой или отрицательной длины никогда не открыт, незакрытый (конец пуст) не закрывается."""
    out = grid[KEY].copy()
    out[name] = 0
    iv = intervals.dropna(subset=["начало"])
    days = out["день"].to_numpy(dtype="datetime64[ns]")
    for obj, g in iv.groupby("ид_объект"):
        m = (out["ид_объект"] == obj).to_numpy()
        if not m.any():
            continue
        start = g["начало"].to_numpy(dtype="datetime64[ns]")
        end = g["конец"].to_numpy(dtype="datetime64[ns]")
        end = np.where(np.isnat(end), np.datetime64("NaT"), np.maximum(end, start + np.timedelta64(1, "ns")))
        end = np.sort(end[~np.isnat(end)])
        d = days[m]
        out.loc[m, name] = np.searchsorted(np.sort(start), d, side="left") - np.searchsorted(end, d, side="right")
    return out


_SEC_STRIDE = 1 << 40   # ключ = код объекта · stride + секунда от 1970 (2^40 с ≈ 35 тыс. лет)


def _class_groups(classes) -> dict:
    """Список классов → {класс: [класс]}; словарь {имя: [классы]} – как есть (например, «любой_инцидент»)."""
    return dict(classes) if isinstance(classes, dict) else {c: [c] for c in classes}


def _seconds(values) -> np.ndarray:
    return pd.to_datetime(pd.Series(values)).to_numpy().astype("datetime64[s]").astype(np.int64)


def hours_since_last(grid, events, classes, cap_h=720, by="класс", time_col="момент") -> pd.DataFrame:
    """Часов от последнего события класса строго до D 00:00 до D; не было или дальше cap_h – cap_h.
    Столбцы `часов_с_последнего_<класс>`. classes – список классов или {имя: [классы]}.
    events: `ид_объект`, `by`, `time_col` (момент, когда событие стало известно). Моменты округляются вниз
    до секунды: событие ровно в D 00:00 остается «не до D», погрешность часов < 1 с."""
    codes, _ = pd.factorize(pd.concat([grid["ид_объект"], events["ид_объект"]], ignore_index=True))
    g_code, e_code = codes[: len(grid)].astype(np.int64), codes[len(grid):].astype(np.int64)
    g_sec = _seconds(grid["день"])
    e_sec = _seconds(events[time_col])
    base = g_code * _SEC_STRIDE
    out = grid[KEY].copy()
    for name, cls in _class_groups(classes).items():
        sel = events[by].isin(cls).to_numpy() & ~pd.isna(events[time_col]).to_numpy()
        key = np.sort(e_code[sel] * _SEC_STRIDE + e_sec[sel])
        i = np.searchsorted(key, base + g_sec, side="left") - 1              # последний момент < D
        prev = key[np.clip(i, 0, None)] if len(key) else np.zeros(len(grid), np.int64)
        found = (i >= 0) & (prev // _SEC_STRIDE == g_code)
        hours = np.where(found, (g_sec - prev % _SEC_STRIDE) / 3600, cap_h)
        out[f"часов_с_последнего_{name}"] = np.minimum(hours, cap_h)
    return out


def open_age_at_midnight(grid, episodes, classes, by="класс") -> pd.DataFrame:
    """Возраст самого старого открытого в D 00:00 эпизода класса, часов: max(D − начало) по эпизодам
    с начало < D и (конец пуст или конец > D) – то же определение «открыт», что в active_at_midnight
    (закрытый ровно в D 00:00 уже закрыт). Нет открытых – 0. Столбцы `часов_открыт_<класс>_макс`.

    Эпизод разворачивается по полуночам, которые он накрывает (в пределах дней сетки); незакрытый тянется
    до последнего дня сетки."""
    out = grid[KEY].copy()
    g_first, g_last = grid["день"].min(), grid["день"].max()
    for name, cls in _class_groups(classes).items():
        e = episodes.loc[episodes[by].isin(cls) & episodes["начало"].notna(), ["ид_объект", "начало", "конец"]]
        start = e["начало"].to_numpy(dtype="datetime64[ns]")
        end = e["конец"].to_numpy(dtype="datetime64[ns]")
        first = np.maximum(start.astype("datetime64[D]") + np.timedelta64(1, "D"),  # первая полночь > начало
                           np.datetime64(g_first, "D"))
        last = np.where(np.isnat(end), np.datetime64(g_last, "D"),
                        (end - np.timedelta64(1, "ns")).astype("datetime64[D]"))       # последняя полночь < конец
        last = np.minimum(last, np.datetime64(g_last, "D"))
        k = np.maximum((last - first).astype(np.int64) + 1, 0)
        rep = np.repeat(np.arange(len(e)), k)
        off = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
        mid = (first[rep] + off.astype("timedelta64[D]")).astype("datetime64[ns]")
        age = pd.DataFrame({"ид_объект": e["ид_объект"].to_numpy()[rep], "день": mid,
                            "_age": (mid - start[rep]) / np.timedelta64(1, "h")})
        age = age.groupby(KEY, as_index=False)["_age"].max()
        col = f"часов_открыт_{name}_макс"
        merged = grid[KEY].merge(age, on=KEY, how="left")["_age"].fillna(0.0).to_numpy()
        out[col] = merged
    return out


def sibling_sums(feats, cols, parents) -> pd.DataFrame:
    """Сумма признака по другим объектам того же родителя в тот же день; объект без родителя – NaN."""
    f = feats[KEY + cols].assign(родитель=feats["ид_объект"].map(parents))
    tot = f.groupby(["родитель", "день"])[cols].transform("sum")   # строки без родителя → NaN
    out = f[KEY].copy()
    for c in cols:
        out[f"соседи_{c}"] = tot[c] - f[c]
    return out


def per_channel(feats, col, n_channels) -> pd.Series:
    """Счетчик на один канал объекта нужного типа; нет каналов – NaN."""
    n = feats["ид_объект"].map(n_channels).replace(0, np.nan)
    return feats[col] / n


def average_precision(y, score) -> float:
    """PR-AUC (average precision) = Σ (R_k − R_{k−1}) · P_k по порогам; строки с одинаковой оценкой – один
    порог (как sklearn.metrics.average_precision_score), поэтому порядок внутри связки не влияет.
    Строки с NaN-оценкой выбрасываются; без положительных – NaN. Без sklearn."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[s[1:] != s[:-1], True]          # последний элемент каждой группы одинаковых оценок
    tp = np.cumsum(y)[last]
    k = np.flatnonzero(last) + 1
    precision, recall = tp / k, tp / y.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def psi(expected, actual, bins=10) -> float:
    """Population stability index; > 0,25 – распределение заметно сдвинулось.

    Признак с не более чем `bins` различными значениями (флаги, коды, счетчики с малым размахом) сравнивается по
    самим значениям. Иначе границы – внутренние квантили expected, а корзины [q_i, q_{i+1}): при массе нулей
    нули получают свою корзину и не сливаются с положительными значениями."""
    e = np.asarray(expected, float)
    a = np.asarray(actual, float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(e) == 0 or len(a) == 0:
        return np.nan
    vals = np.unique(e)
    if len(vals) <= bins:
        cats = np.union1d(vals, np.unique(a))
        pe = np.searchsorted(cats, e)
        pa = np.searchsorted(cats, a)
        k = len(cats)
    else:
        inner = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1))[1:-1])
        pe = np.searchsorted(inner, e, side="right")
        pa = np.searchsorted(inner, a, side="right")
        k = len(inner) + 1
    pe = np.bincount(pe, minlength=k) / len(e)
    pa = np.bincount(pa, minlength=k) / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def signal_table(X: pd.DataFrame, y) -> pd.DataFrame:
    """Одиночный сигнал признака: PR-AUC признака как оценки (берем лучшую из двух сторон) к доле положительных
    на тех же строках, где признак не пропущен (иначе пропуски, связанные с целью, дают ложный подъем)."""
    y = np.asarray(y, dtype=float)
    rows = []
    for c in X.columns:
        v = X[c].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        base = y[ok].mean() if ok.any() else np.nan
        aps = [average_precision(y, v), average_precision(y, -v)]
        ap = np.nan if np.isnan(aps).all() else float(np.nanmax(aps))
        lift = ap / base if base > 0 else np.nan
        rows.append({"признак": c, "pr_auc": ap, "доля_положительных": base, "подъем_pr_auc": lift,
                     "доля_пропусков": 1 - ok.mean()})
    return pd.DataFrame(rows).sort_values("подъем_pr_auc", ascending=False, na_position="last")
