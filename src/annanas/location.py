"""Где на объекте искать: ранжирование каналов и пикетов по истории отказов.

Модель «канал × сутки» почти не лучше простого правила «канал, отказывавший последним» (исследование
локализации: попадание в топ-1 около 0,53 против 0,13 у случайного канала), поэтому сервис ранжирует по
правилу: меньше дней с последнего отказа канала – выше; при равенстве – больше дней с отказом за 30 суток,
затем меньший ид. Используются только эпизоды, начавшиеся строго до D 00:00."""
import numpy as np
import pandas as pd

LEVELS = {"канал": "ид_канала_данных", "пикет": "пикет"}


def rank_channels(episodes: pd.DataFrame, object_id, day, k=3, level="канал", window_days=30) -> pd.DataFrame:
    """Топ-k каналов (level="канал") или пикетов (level="пикет") объекта на сутки `day`.

    episodes: эпизоды класса (например, отказа) со столбцами `ид_объект, ид_канала_данных, пикет, начало`.
    Возвращает `<ид_канала_данных, пикет | пикет>, дней_с_последнего, дней_с_отказом_30д, ранг`:
    дней_с_последнего = D − день последнего эпизода; дней_с_отказом_30д – сутки в [D − window_days, D − 1],
    когда на канале (пикете) начинался эпизод. Каналы без эпизодов до D не ранжируются."""
    key = LEVELS[level]
    D = pd.Timestamp(day).normalize()
    e = episodes.loc[(episodes["ид_объект"] == object_id) & (episodes["начало"] < D)]
    e = e.dropna(subset=[key]).assign(_день=e["начало"].dt.floor("D"))
    cols = ["ид_канала_данных", "пикет"] if level == "канал" else ["пикет"]
    out_cols = cols + ["дней_с_последнего", "дней_с_отказом_30д", "ранг"]
    if e.empty:
        return pd.DataFrame(columns=out_cols)
    recent = e.loc[e["_день"] >= D - pd.Timedelta(days=window_days)]
    g = e.groupby(key).agg(_последний=("_день", "max"), **({"пикет": ("пикет", "last")} if level == "канал" else {}))
    g["дней_с_последнего"] = (D - g["_последний"]).dt.days
    g["дней_с_отказом_30д"] = recent.groupby(key)["_день"].nunique().reindex(g.index).fillna(0).astype(int)
    g = (g.reset_index()
         .sort_values(["дней_с_последнего", "дней_с_отказом_30д", key], ascending=[True, False, True],
                      kind="mergesort")
         .head(k))
    g["ранг"] = range(1, len(g) + 1)
    return g[out_cols].reset_index(drop=True)


def rank_grid(episodes: pd.DataFrame, keys: pd.DataFrame, k=3, level="канал", window_days=30) -> pd.Series:
    """То же правило, что rank_channels, сразу для многих пар `ид_объект, день` (журнал прогнозов, оценка
    попаданий): по объекту строится матрица «канал (пикет) × сутки» с началами эпизодов, давность и число суток
    с отказом за окно берутся из накопленных сумм. Возвращает Series с индексом keys: список до k словарей
    `{<ид_канала_данных, пикет | пикет>, дней_с_последнего, дней_с_отказом_30д}` по рангу (пустой – истории нет)."""
    key = LEVELS[level]
    e = episodes.dropna(subset=[key])
    e = e.assign(_день=e["начало"].dt.floor("D"))
    out = pd.Series([[] for _ in range(len(keys))], index=keys.index, dtype=object)
    for obj, kd in keys.groupby("ид_объект", sort=False):
        eo = e.loc[e["ид_объект"] == obj]
        if eo.empty:
            continue
        items = np.sort(eo[key].unique())
        if level == "канал":
            picket = eo.groupby(key)["пикет"].last().reindex(items).to_numpy()
        t0 = eo["_день"].min()
        target = (pd.to_datetime(kd["день"]).dt.normalize() - t0).dt.days.to_numpy()   # D в сутках от t0
        T = int(max(target.max(), 1))                  # D ≤ T: столбцы матрицы – сутки 0..T
        ind = np.zeros((len(items), T + 1), dtype=np.int32)
        ep_day = (eo["_день"] - t0).dt.days.to_numpy()
        ok = ep_day < T + 1
        ind[np.searchsorted(items, eo[key].to_numpy()[ok]), ep_day[ok]] = 1
        # до_дня[:, x] – суток с отказом среди [0, x); последний[:, x] – последний такой день < x (−1 – нет)
        cum = np.concatenate([np.zeros((len(items), 1), np.int32), np.cumsum(ind, axis=1)], axis=1)
        last = np.maximum.accumulate(np.where(ind > 0, np.arange(T + 1), -1), axis=1)
        last = np.concatenate([np.full((len(items), 1), -1), last], axis=1)
        for row, D in zip(kd.index, target):
            if D <= 0:
                continue
            lst = last[:, D]
            has = lst >= 0
            if not has.any():
                continue
            since = D - lst
            cnt = cum[:, D] - cum[:, max(D - window_days, 0)]
            idx = np.flatnonzero(has)
            order = idx[np.lexsort((idx, -cnt[idx], since[idx]))][:k]
            if level == "канал":
                out.at[row] = [{"ид_канала_данных": items[i].item(), "пикет": float(picket[i]),
                                "дней_с_последнего": int(since[i]), "дней_с_отказом_30д": int(cnt[i])} for i in order]
            else:
                out.at[row] = [{"пикет": float(items[i]), "дней_с_последнего": int(since[i]),
                                "дней_с_отказом_30д": int(cnt[i])} for i in order]
    return out
