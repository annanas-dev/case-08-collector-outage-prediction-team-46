"""Признаки «объект × сутки» для модели и сервиса. Какие признаки, почему и как проверены – notebooks/04_features.

Правило: признак суток D вычисляется только по данным, известным до D 00:00. Один и тот же код считает всю
матрицу обучения (сетка «все объекты × все сутки») и прогноз на одни сутки (сетка «все объекты × {D}» по
источникам, обрезанным `cut(S, D)`); результат для суток D совпадает.

Источники S (dict pandas DataFrame, время московское):

- `eps` – эпизоды тревог (`ид_канала_данных, тип_датчика, ид_объект, начало, конец, длительность_с, класс`);
- `cl` – эпизоды потери связи (`ид_объект, начало, последняя_минута, длительность_мин`);
- `ev` – `known_events(eps, cl, masked_keys(mask))`: эпизоды с моментом и днем известности;
- `ods` – смены состояний «объект × день × тип датчика × состояние»;
- `dc`, `mg`, `ud`, `ub`, `ht` – витрины daily_channel, meas_gaps, undefined_daily, undefined_bursts, hourly_type;
- `ppr` – окна плановых работ АКМ (`ид_объект, начало, конец, источник`, maintenance.ppr_windows);
- `mask` – история масок `ид_объект, день, маска` (null – сутки без маски) за сутки до D включительно.

grid – строки, для которых считаются признаки: `ид_объект, день (00:00), период_смву, эпоха_флага`. В сетке
должны быть все объекты (признаки «другие объекты», «соседи» считаются по объектам сетки в те же сутки).

ctx – статичные таблицы и параметры, выведенные на обучении (`make_context`): справочник каналов, типы
датчиков каждого класса, месяцы сезона подтопления, первый день истории."""
import glob
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from annanas import aggregates as agg
from annanas import calendar_ru as cal
from annanas import features as fe
from annanas.paths import PROCESSED_DIR

KEY = fe.KEY

# Момент известности эпизода
T_FLAP = pd.Timedelta(seconds=60)      # эпизод от минуты – длинный (цель), короче – дребезг
COMM_MIN = pd.Timedelta(minutes=10)    # потеря связи – от 10 минут

# История объекта по классам
HIST = ["отказ", "подтопление", "пожар", "газ", "потеря_связи", "проникновение", "неисправность_охраны"]
INCIDENT = ["отказ", "подтопление", "пожар", "газ", "потеря_связи"]
HIST_ALL = HIST + ["любой_инцидент"]
EP_CLASSES = ["отказ", "подтопление", "пожар", "газ", "проникновение", "неисправность_охраны", "откачка",
              "вывод_из_работы", "потеря_связи"]
WINDOWS, EP_WINDOWS = [1, 7, 30, 90], [1, 7, 30]
SHARE_WINDOWS = [7, 30, 90]
PER_CHANNEL = ["отказ", "подтопление", "пожар", "газ", "проникновение", "неисправность_охраны"]

# Часы с последнего эпизода и возраст открытых эпизодов
HOURS_CLASSES = {c: [c] for c in ["отказ", "пожар", "подтопление", "газ", "потеря_связи", "проникновение", "откачка",
                                  "неисправность_охраны"]}
HOURS_CLASSES["любой_инцидент"] = INCIDENT
OPEN_AGE_CLASSES = ["отказ", "пожар", "подтопление"]
HOURS_CAP = 720          # 30 суток: дальше разница в часах уже ничего не меняет

# Каналы
N_CHRONIC = 5            # с 5 длинных эпизодов за 30 суток вероятность эпизода канала выходит на плато
CHANNEL_CLASSES = ["отказ", "пожар", "подтопление"]
OPEN_CLASSES = ["отказ", "пожар", "подтопление", "вывод_из_работы"]

# Связь
BURST_MIN = 3            # пачка «Неопределен»: не меньше 3 каналов объекта в одну минуту

# Телеметрия
TEMP, GAS = "Датчик температуры", "Газовый датчик"
TEMP_RANGE = (-60, 90)            # физически возможная температура в коллекторе, °C
GAS_CODES = [2.55, -2.56, 327.68]  # служебные значения газовых каналов

# Плановые работы АКМ
PPR_CAP = 365

# Эпохи, система, постоянные свойства
AFTER_FLAG = "после смены разметки"
SMVU_CODE = {"до": 0, "переход": 1, "после": 2}
STATIC_TYPES = {"Датчик дыма": "дым", "Датчик температуры": "температура", "Газовый датчик": "газ",
                "Состояние насоса": "насос", "Датчик затопления": "затопление", "КД Дверь": "кд_дверь",
                "КД Люк": "кд_люк", "Состояние фазы": "фаза", "ИБП": "ибп", "Тепловой датчик": "тепловой",
                "Состояние вентилятора": "вентилятор", "Ручной извещатель": "ручной_извещатель"}
SYS_CLASSES = ["отказ", "пожар", "подтопление", "любой_инцидент"]


# --- Контекст ----------------------------------------------------------------------------------------

def make_context(channels, class_types, season_months=None, start=None) -> dict:
    """Статичные таблицы для строителей признаков.

    channels – справочник каналов (`ид_канала_данных, ид_объект, тип_датчика, родитель, пикет`);
    class_types – {класс: [типы датчиков, которые дают класс]} (выводится на обучении);
    season_months – месяцы сезона подтопления (выводятся на обучении);
    start – первый день истории: начало оси суток для скользящих окон и медиан."""
    ch_types = channels.groupby(["ид_объект", "тип_датчика"]).size().unstack(fill_value=0)
    return {
        "channels": channels, "class_types": class_types, "season_months": season_months,
        "start": None if start is None else pd.Timestamp(start),
        "ch_types": ch_types,
        "n_ch": {c: ch_types[class_types[c]].sum(axis=1) for c in PER_CHANNEL},
        "n_all": channels.groupby("ид_объект").size(),
        "parent": channels.groupby("ид_объект")["родитель"].first(),
    }


def save_context(ctx, path) -> None:
    """Параметры контекста, выведенные на обучении, в JSON (справочник каналов хранится отдельно)."""
    data = {"class_types": ctx["class_types"], "season_months": [int(m) for m in ctx["season_months"]],
            "start": None if ctx["start"] is None else f"{ctx['start']:%Y-%m-%d}"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_context(path, channels) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return make_context(channels, data["class_types"], data["season_months"], data["start"])


# --- Источники ---------------------------------------------------------------------------------------

EP_COLS = ["ид_канала_данных", "тип_датчика", "ид_объект", "начало", "конец", "длительность_с", "класс"]


def episodes_dir(processed_dir=PROCESSED_DIR) -> Path:
    """Кэш эпизодов тревог действующего словаря классов: все классы целей есть, лишних нет (ровно один каталог)."""
    from annanas import episodes as ep   # pyspark нужен только здесь

    def classes(path):
        return set(ds.dataset(path, format="parquet").to_table(columns=["класс"]).column("класс").unique().to_pylist())

    dirs = [d for d in sorted(glob.glob(str(Path(processed_dir) / "episodes" / "alarms_pairs_*")))
            if set(ep.TARGET_CLASSES) <= classes(d) <= set(ep.PRIORITY)]
    if len(dirs) != 1:
        raise RuntimeError(f"ожидался один каталог эпизодов действующего словаря, найдено: {dirs}")
    return Path(dirs[0])


def read_sources(processed_dir=PROCESSED_DIR) -> dict:
    """Источники признаков из кэшей `datasets/processed` (их строят notebooks/02_eda, 03_target, 04_features):
    всё, кроме `ev` и `mask` (их добавляет вызывающий: маски зависят от сетки)."""
    processed_dir = Path(processed_dir)

    def vitrine(path, day_col="день"):
        d = agg.read_vitrine(path)
        if day_col:
            d[day_col] = pd.to_datetime(d[day_col])
        return d

    eda = processed_dir / "eda"
    return {"eps": agg.read_vitrine(episodes_dir(processed_dir))[EP_COLS],
            "cl": agg.read_vitrine(processed_dir / "episodes" / "comm_loss_k10_s50.parquet"),
            "ods": vitrine(processed_dir / "features" / "object_day_states.parquet"),
            "dc": vitrine(eda / "daily_channel.parquet"), "mg": vitrine(eda / "meas_gaps.parquet"),
            "ud": vitrine(eda / "undefined_daily.parquet"), "ub": vitrine(eda / "undefined_bursts.parquet", None),
            "ht": vitrine(eda / "hourly_type.parquet", None),
            # таблицу окон пишет pandas (даты без часового пояса), поэтому read_parquet, а не read_vitrine
            "ppr": pd.read_parquet(processed_dir / "maintenance" / "ppr_windows.parquet")}


# --- События и маски ---------------------------------------------------------------------------------

def masked_keys(mask) -> pd.DataFrame:
    """Ключи замаскированных суток из истории масок."""
    return mask.loc[mask["маска"].notna(), KEY]


def drop_masked(ev, masked) -> pd.DataFrame:
    """Убирает события, чей день известности на объекте замаскирован (masked – ключи замаскированных суток)."""
    m = ev.merge(masked.assign(_m=1), on=KEY, how="left")["_m"].isna().to_numpy()
    return ev[m]


def known_events(eps, cl, masked) -> pd.DataFrame:
    """Эпизоды тревог и потери связи с моментом известности и днем, в котором они становятся известны.
    Длинный эпизод известен через T_FLAP (потеря связи – COMM_MIN) после начала, короткий – в момент конца."""
    dur = eps["длительность_с"]
    long_ = dur.isna() | (dur >= T_FLAP.total_seconds())
    a = eps.assign(длинный=long_, момент=(eps["начало"] + T_FLAP).where(long_, eps["конец"]))
    long_cl = cl["длительность_мин"] >= COMM_MIN.total_seconds() / 60
    b = pd.DataFrame({"ид_канала_данных": -1, "тип_датчика": None, "ид_объект": cl["ид_объект"],
                      "начало": cl["начало"], "конец": cl["последняя_минута"] + pd.Timedelta(minutes=1),
                      "длительность_с": cl["длительность_мин"] * 60.0, "класс": "потеря_связи", "длинный": long_cl,
                      "момент": (cl["начало"] + COMM_MIN).where(long_cl, cl["последняя_минута"] + pd.Timedelta(minutes=1))})
    ev = pd.concat([a, b], ignore_index=True)
    ev["день"] = ev["момент"].dt.floor("D")
    return drop_masked(ev, masked)


def cut(S, D) -> dict:
    """Данные в том виде, в каком они были бы доступны в D 00:00: эпизоды, начавшиеся до D (незакрытые к D
    теряют конец и длительность), витрины до суток D, маски по сутки D. Окна работ по графику известны
    заранее, окна, восстановленные по журналу, – только закончившиеся до D."""
    D = pd.Timestamp(D)
    eps = S["eps"][S["eps"]["начало"] < D].copy()
    late = eps["конец"].isna() | (eps["конец"] >= D)
    eps.loc[late, "конец"] = pd.NaT
    eps.loc[late, "длительность_с"] = np.nan
    cl = S["cl"][S["cl"]["начало"] < D].copy()
    late = cl["последняя_минута"] >= D
    cl.loc[late, "последняя_минута"] = D - pd.Timedelta(minutes=1)
    cl["длительность_мин"] = ((cl["последняя_минута"] - cl["начало"]).dt.total_seconds() // 60 + 1).astype(int)
    mask = S["mask"][S["mask"]["день"] <= D]
    return {"eps": eps, "cl": cl, "ev": known_events(eps, cl, masked_keys(mask)), "mask": mask,
            "ods": S["ods"][S["ods"]["день"] < D], "dc": S["dc"][S["dc"]["день"] < D],
            "mg": S["mg"][S["mg"]["день"] < D], "ud": S["ud"][S["ud"]["день"] < D],
            "ub": S["ub"][S["ub"]["минута"] < D], "ht": S["ht"][S["ht"]["час"] < D],
            "ppr": S["ppr"][(S["ppr"]["источник"] == "график") | (S["ppr"]["конец"] < D)]}


# --- Вспомогательные функции -------------------------------------------------------------------------

def class_days(ev) -> pd.DataFrame:
    """Сутки, когда на объекте стал известен длинный эпизод класса (0/1), и «любой инцидент»."""
    d = fe.daily_counts(ev[ev["длинный"]], "класс", HIST, time_col="момент")
    d[HIST] = (d[HIST] > 0).astype(int)
    d["любой_инцидент"] = d[INCIDENT].max(axis=1)
    return d


def prefixed(d, cols, prefix):
    return d.rename(columns={c: f"{prefix}{c}" for c in cols}), [f"{prefix}{c}" for c in cols]


class Axes(NamedTuple):
    """Ось суток и объектов для матриц «сутки × объекты» и позиция каждой строки сетки на ней."""
    days: pd.DatetimeIndex
    objects: np.ndarray
    day_idx: np.ndarray
    obj_idx: np.ndarray


def axes(grid, start=None) -> Axes:
    """Сутки – от start (или первых суток сетки) до последних суток сетки, объекты – объекты сетки."""
    days = pd.date_range(grid["день"].min() if start is None else start, grid["день"].max(), freq="D")
    objects = np.sort(grid["ид_объект"].unique())
    return Axes(days, objects, (grid["день"] - days[0]).dt.days.to_numpy(),
                np.searchsorted(objects, grid["ид_объект"].to_numpy()))


def mat_to_grid(df, ax) -> np.ndarray:
    """Таблица «сутки × объекты» → вектор в порядке строк сетки."""
    return df.reindex(index=ax.days, columns=ax.objects).to_numpy(dtype=float)[ax.day_idx, ax.obj_idx]


def channel_matrix(ev, cls, days):
    """Длинные эпизоды класса по каналам и дням известности: матрица «каналы × сутки» и объект каждого канала."""
    e = ev[ev["длинный"] & (ev["класс"] == cls) & (ev["ид_канала_данных"] >= 0)]
    di = (e["день"] - days[0]).dt.days.to_numpy()
    keep = (di >= 0) & (di < len(days))
    e, di = e[keep], di[keep]
    ch, uniq = pd.factorize(e["ид_канала_данных"].to_numpy())
    m = np.zeros((len(uniq), len(days)), np.int32)
    np.add.at(m, (ch, di), 1)
    obj = e.groupby("ид_канала_данных")["ид_объект"].first().reindex(uniq).to_numpy()
    return m, obj


def window_matrix(m, w):
    """Сумма по суткам [D − w, D − 1] для каждой строки матрицы «× сутки»."""
    c = np.zeros((m.shape[0], m.shape[1] + 1), np.int64)
    np.cumsum(m, axis=1, out=c[:, 1:])
    t = np.arange(m.shape[1])
    return c[:, t] - c[:, np.maximum(t - w, 0)]


def objects_sum(mat, obj, ax) -> np.ndarray:
    """Сумма строк матрицы «каналы × сутки» по объектам → вектор в порядке сетки (каналы объектов вне сетки
    не учитываются)."""
    keep = np.isin(obj, ax.objects)
    a = np.zeros((len(ax.objects), mat.shape[1]))
    np.add.at(a, np.searchsorted(ax.objects, obj[keep]), mat[keep])
    return a[ax.obj_idx, ax.day_idx]


def code_days(m) -> np.ndarray:
    """Флаг «в эти сутки канал писал служебный код» по минимуму и максимуму суток."""
    is_t = m["тип_датчика"] == TEMP
    t_bad = (m["минимум_значения"] < TEMP_RANGE[0]) | (m["максимум_значения"] > TEMP_RANGE[1])
    g_bad = m["минимум_значения"].round(2).isin(GAS_CODES) | m["максимум_значения"].round(2).isin(GAS_CODES)
    return np.where(is_t, t_bad, g_bad)


def work_hours(ts):
    """Будни с 8 до 18."""
    return (ts.dt.dayofweek < 5) & (ts.dt.hour >= 8) & (ts.dt.hour < 18)


def pivot_ch(d, col, days):
    return d.pivot_table(index="день", columns="ид_канала_данных", values=col, aggfunc="first").reindex(days)


def by_object(mat, ch_obj, how, ax) -> np.ndarray:
    """Сворачивает «сутки × каналы» по объектам функцией how → вектор в порядке сетки. Сумма по одним пропускам –
    пропуск: канал, который еще ни разу не писал, не должен выглядеть «нулем»."""
    g = mat.T.groupby(ch_obj.reindex(mat.columns).to_numpy())
    return mat_to_grid((g.sum(min_count=1) if how == "sum" else g.agg(how)).T, ax)


# --- Строители групп ---------------------------------------------------------------------------------

def build_history(S, grid, ctx) -> pd.DataFrame:
    """История объекта: суток с эпизодом класса и их доля среди суток без маски, сутки с последнего эпизода,
    число длинных и коротких эпизодов, эпизоды на канал, доля суток окна под маской."""
    ev = S["ev"]
    days = class_days(ev)
    d, cols = prefixed(days, HIST_ALL, "дней_")
    out = fe.window_sums(grid, d, cols, WINDOWS)
    shares = fe.window_shares(grid, d, cols, SHARE_WINDOWS, mask_col="маска", masks=S["mask"])
    vd = fe.valid_days(grid, [30], masks=S["mask"])
    out = pd.concat([out, shares.drop(columns=KEY)], axis=1)
    since = fe.days_since_last(grid, days, HIST_ALL)
    d, cols = prefixed(fe.daily_counts(ev[ev["длинный"]], "класс", EP_CLASSES, time_col="момент"), EP_CLASSES, "эпизодов_")
    long_w = fe.window_sums(grid, d, cols, EP_WINDOWS)
    d, cols = prefixed(fe.daily_counts(ev[~ev["длинный"]], "класс", EP_CLASSES, time_col="момент"), EP_CLASSES, "коротких_")
    short_w = fe.window_sums(grid, d, cols, EP_WINDOWS)
    out = pd.concat([out, since.drop(columns=KEY), long_w.drop(columns=KEY), short_w.drop(columns=KEY)], axis=1)
    for c in PER_CHANNEL:
        out[f"эпизодов_{c}_30д_на_канал"] = fe.per_channel(out, f"эпизодов_{c}_30д", ctx["n_ch"][c])
    out["доля_дней_окна_без_данных_30д"] = 1 - vd["дней_с_данными_30д"] / 30
    return out


def build_hours(S, grid, ctx) -> pd.DataFrame:
    """Часы с последнего длинного эпизода класса до полуночи и возраст самого старого открытого эпизода."""
    ev = S["ev"]
    out = fe.hours_since_last(grid, ev[ev["длинный"]], HOURS_CLASSES, cap_h=HOURS_CAP)
    age = fe.open_age_at_midnight(grid, ev[ev["ид_канала_данных"] >= 0], OPEN_AGE_CLASSES)
    return pd.concat([out, age.drop(columns=KEY)], axis=1)


def build_channels(S, grid, ctx) -> pd.DataFrame:
    """Каналы: сколько разных каналов с эпизодом, хронические каналы, доля коротких отказов, каналы в эпизоде
    на полночь."""
    ev = S["ev"]
    ax = axes(grid, ctx["start"])
    out = grid[KEY].copy()
    for c in CHANNEL_CLASSES:
        m, obj = channel_matrix(ev, c, ax.days)
        for w in [7, 30]:
            out[f"каналов_с_{c}_{w}д"] = objects_sum(window_matrix(m, w) > 0, obj, ax)
        if c in ["отказ", "пожар"]:
            out[f"хронических_каналов_{c}_30д"] = objects_sum(window_matrix(m, 30) >= N_CHRONIC, obj, ax)
    f = ev[ev["класс"] == "отказ"]
    d = fe.daily_counts(f.assign(вид=np.where(f["длинный"], "длинный", "короткий")), "вид", ["длинный", "короткий"],
                        time_col="момент")
    ws = fe.window_sums(grid, d, ["длинный", "короткий"], [30])
    out["доля_коротких_отказ_30д"] = ws["короткий_30д"] / (ws["короткий_30д"] + ws["длинный_30д"]).replace(0, np.nan)
    for c in OPEN_CLASSES:
        iv = ev[(ev["класс"] == c) & (ev["ид_канала_данных"] >= 0)]
        out[f"каналов_в_{c}_на_полночь"] = fe.active_at_midnight(grid, iv, "n")["n"].to_numpy()
    return out


def build_comm(S, grid, ctx) -> pd.DataFrame:
    """Связь и активность: пачки «Неопределен», «Неопределен» на канал, активность вчера к медиане."""
    ax = axes(grid, ctx["start"])
    channels, masked = ctx["channels"], masked_keys(S["mask"])
    out = grid[KEY].copy()
    ub = S["ub"].dropna(subset=["ид_объект"])
    ub = ub[ub["каналов_в_неопределен"] >= BURST_MIN]
    d = (ub.assign(ид_объект=ub["ид_объект"].astype("int64"), день=ub["минута"].dt.floor("D"))
         .groupby(KEY).size().rename("минут_пачек_неопределен").reset_index())
    w = fe.window_sums(grid, drop_masked(d, masked), ["минут_пачек_неопределен"], [1, 7, 30])
    out = pd.concat([out, w.drop(columns=KEY)], axis=1)
    ud = S["ud"].merge(channels[["ид_канала_данных", "ид_объект"]], on="ид_канала_данных")
    d = ud.groupby(KEY)["количество_неопределен"].sum().reset_index()
    w = fe.window_sums(grid, drop_masked(d, masked), ["количество_неопределен"], [7])
    out["неопределен_на_канал_7д"] = fe.per_channel(w, "количество_неопределен_7д", ctx["n_all"]).to_numpy()
    dc = S["dc"].merge(channels[["ид_канала_данных", "ид_объект"]], on="ид_канала_данных")
    act = dc.groupby(KEY).agg(каналов=("ид_канала_данных", "nunique"), записей=("количество_событий", "sum"))
    for col, name in [("каналов", "активных_каналов_к_медиане_1д"), ("записей", "записей_к_медиане_1д")]:
        a = act[col].unstack("ид_объект").reindex(index=ax.days, columns=ax.objects).fillna(0)
        y1 = a.shift(1)
        med = y1.rolling(28, min_periods=7).median()
        out[name] = mat_to_grid(y1 / med.replace(0, np.nan), ax)
    a = act["каналов"].unstack("ид_объект").reindex(index=ax.days, columns=ax.objects).fillna(0).shift(1)
    out["доля_активных_каналов_1д"] = mat_to_grid(a / ctx["n_all"].reindex(ax.objects).to_numpy(), ax)
    return out


def build_telemetry(S, grid, ctx) -> pd.DataFrame:
    """Газ и температура: крайние значения и отклонения вчера, служебные коды, молчание газовых каналов,
    климатические состояния температуры, флаги наличия каналов."""
    ax = axes(grid, ctx["start"])
    channels, masked = ctx["channels"], masked_keys(S["mask"])
    out = grid[KEY].copy()
    m = (S["dc"][S["dc"]["класс_записи"] == "измерение"]
         .merge(channels[["ид_канала_данных", "ид_объект", "тип_датчика"]], on="ид_канала_данных"))
    m = m[m["тип_датчика"].isin([TEMP, GAS])].copy()
    m["код"] = code_days(m)
    for c in ["среднее_значение", "минимум_значения", "максимум_значения"]:
        m.loc[m["код"], c] = np.nan
    ch_obj = channels.set_index("ид_канала_данных")["ид_объект"]
    for тип, p in [(TEMP, "температура"), (GAS, "газ")]:
        g = m[m["тип_датчика"] == тип]
        code = pivot_ch(g.assign(к=g["код"].astype(float)), "к", ax.days)
        code = code.fillna(0).where(code.notna().cumsum() > 0)   # до первого замера канала – пропуск
        c1 = by_object(code.shift(1), ch_obj, "sum", ax)
        c7 = by_object(code.rolling(7, min_periods=1).sum().shift(1), ch_obj, "sum", ax)
        out[f"{p}_каналов_с_кодом_1д"], out[f"{p}_каналов_с_кодом_7д"] = c1, c7
        mx = pivot_ch(g, "максимум_значения", ax.days)
        if тип == TEMP:
            mean = pivot_ch(g, "среднее_значение", ax.days)
            dev = mean - mean.rolling(30, min_periods=7).median().shift(1)
            out["температура_макс_1д"] = by_object(mx.shift(1), ch_obj, "max", ax)
            out["температура_мин_1д"] = by_object(pivot_ch(g, "минимум_значения", ax.days).shift(1), ch_obj, "min", ax)
            out["температура_откл_макс_1д"] = by_object(dev.shift(1), ch_obj, "max", ax)
            out["температура_откл_мин_1д"] = by_object(dev.shift(1), ch_obj, "min", ax)
        else:
            exceed = mx - mx.rolling(30, min_periods=7).median().shift(1)
            out["газ_макс_1д"] = by_object(mx.shift(1), ch_obj, "max", ax)
            out["газ_превышение_1д"] = by_object(exceed.shift(1), ch_obj, "max", ax)
    mg = S["mg"].merge(channels[["ид_канала_данных", "тип_датчика"]], on="ид_канала_данных")
    mg = mg[mg["тип_датчика"] == GAS]
    cnt = pivot_ch(mg, "количество_замеров", ax.days)
    seen = cnt.notna().cumsum() > 0                      # канал уже появлялся в данных
    cnt = cnt.fillna(0).where(seen)
    y1 = cnt.shift(1)
    med = y1.rolling(28, min_periods=7).median()
    out["газ_замеров_к_медиане_1д"] = by_object(y1 / med.replace(0, np.nan), ch_obj, "min", ax)
    silent = (y1 == 0).astype(float).where(y1.notna())
    out["доля_газовых_без_замеров_1д"] = by_object(silent, ch_obj, "mean", ax)
    interval = pivot_ch(mg, "медианный_интервал_с", ax.days).shift(1).rolling(30, min_periods=7).median()
    out["газ_доля_частого_опроса"] = by_object((interval < 600).astype(float).where(interval.notna()), ch_obj, "mean", ax)
    o = S["ods"][S["ods"]["тип_датчика"] == TEMP]
    for state, name in [("Температура ниже 3ºC", "температура_ниже_3"), ("Температура выше 40ºC", "температура_выше_40")]:
        d = o[o["состояние"] == state].groupby(KEY)["количество_записей"].sum().rename(name).reset_index()
        out = pd.concat([out, fe.window_sums(grid, drop_masked(d, masked), [name], [1, 7]).drop(columns=KEY)], axis=1)
    has = channels.groupby("ид_объект")["тип_датчика"].agg(lambda s: set(s))
    out["есть_газ"] = grid["ид_объект"].map(has.map(lambda s: GAS in s)).fillna(False).astype(int).to_numpy()
    out["есть_температура"] = grid["ид_объект"].map(has.map(lambda s: TEMP in s)).fillna(False).astype(int).to_numpy()
    return out


def build_context(S, grid, ctx) -> pd.DataFrame:
    """Контекст работ и календарь: открытия люков и дверей, доля тревог в рабочее время, день недели, выходной,
    месяц, сезон подтопления."""
    masked = masked_keys(S["mask"])
    out = grid[KEY].copy()
    o = S["ods"][S["ods"]["состояние"] == "Не замкнут"]
    for тип, name in [("КД Люк", "открытий_люков"), ("КД Дверь", "открытий_дверей")]:
        d = o[o["тип_датчика"] == тип].groupby(KEY)["количество_записей"].sum().rename(name).reset_index()
        out = pd.concat([out, fe.window_sums(grid, drop_masked(d, masked), [name], [1, 7]).drop(columns=KEY)], axis=1)
    ev = S["ev"]
    e = ev[ev["длинный"] & ev["класс"].isin(["отказ", "пожар", "подтопление", "газ"])]
    e = e.assign(вид=np.where(work_hours(e["начало"]), "рабочее", "нерабочее"))
    d = fe.daily_counts(e, "вид", ["рабочее", "нерабочее"], time_col="момент")
    ws = fe.window_sums(grid, d, ["рабочее", "нерабочее"], [30])
    out["доля_тревог_в_рабочее_время_30д"] = ws["рабочее_30д"] / (ws["рабочее_30д"] + ws["нерабочее_30д"]).replace(0, np.nan)
    out["день_недели"] = grid["день"].dt.dayofweek.to_numpy()
    out["выходной"] = (out["день_недели"] >= 5).astype(int)
    out["месяц"] = grid["день"].dt.month.to_numpy()
    out["сезон_подтопления"] = out["месяц"].isin(ctx["season_months"]).astype(int)
    return out


def build_calendar(S, grid, ctx) -> pd.DataFrame:
    """Производственный календарь РФ по самой дате D (calendar_ru.calendar_features)."""
    return pd.concat([grid[KEY], cal.calendar_features(grid["день"])], axis=1)


def build_ppr(S, grid, ctx) -> pd.DataFrame:
    """`дней_после_ппр`: суток с конца последнего окна ППР АКМ объекта, закончившегося до D (не было – PPR_CAP);
    у объектов без газовых каналов – NaN."""
    channels = ctx["channels"]
    out = grid[KEY].copy()
    w = S["ppr"]
    ends = w[["ид_объект"]].assign(день=w["конец"].dt.floor("D"), окно=1)
    since = fe.days_since_last(grid, ends, ["окно"], cap=PPR_CAP)["дней_с_последнего_окно"].astype(float)
    has_gas = grid["ид_объект"].isin(channels.loc[channels["тип_датчика"] == GAS, "ид_объект"]).to_numpy()
    out["дней_после_ппр"] = since.where(has_gas).to_numpy()
    return out


def build_pumps(S, grid, ctx) -> pd.DataFrame:
    """Насосы: переключения вчера и за 7 суток, вчера к медиане за 90 суток, сколько насосов включалось."""
    ax = axes(grid, ctx["start"])
    masked = masked_keys(S["mask"])
    out = grid[KEY].copy()
    o = S["ods"][(S["ods"]["тип_датчика"] == "Состояние насоса")]
    sw = (o[o["состояние"].isin(["Включен", "Выключен"])].groupby(KEY)["количество_записей"].sum()
          .rename("переключений_насосов").reset_index())
    out = pd.concat([out, fe.window_sums(grid, drop_masked(sw, masked), ["переключений_насосов"], [1, 7])
                     .drop(columns=KEY)], axis=1)
    a = (sw.set_index(KEY)["переключений_насосов"].unstack("ид_объект")
         .reindex(index=ax.days, columns=ax.objects).fillna(0))
    y1 = a.shift(1)
    out["переключений_насосов_к_медиане_1д"] = mat_to_grid(
        y1 / y1.rolling(90, min_periods=14).median().replace(0, np.nan), ax)
    on = o[o["состояние"] == "Включен"].groupby(KEY)["каналов"].sum().rename("насосов_включалось").reset_index()
    out["насосов_включалось_1д"] = fe.window_sums(grid, drop_masked(on, masked), ["насосов_включалось"],
                                                  [1])["насосов_включалось_1д"]
    return out


def build_system(S, grid, ctx) -> pd.DataFrame:
    """Эпохи, вся система, соседи по родителю и постоянные свойства объекта."""
    ax = axes(grid, ctx["start"])
    channels = ctx["channels"]
    out = grid[KEY].copy()
    out["период_смву_код"] = grid["период_смву"].map(SMVU_CODE).to_numpy()
    out["эпоха_флага_код"] = (grid["эпоха_флага"] == AFTER_FLAG).to_numpy().astype(int)
    ht = S["ht"].assign(день=S["ht"]["час"].dt.floor("D"))
    tot = ht.groupby("день")["количество_событий"].sum().reindex(ax.days).fillna(0)
    unk = ht[ht["тип_датчика"].isna()].groupby("день")["количество_событий"].sum().reindex(ax.days).fillna(0)
    share = (unk.rolling(7, min_periods=1).sum() / tot.rolling(7, min_periods=1).sum().replace(0, np.nan)).shift(1)
    out["доля_вне_справочника_7д"] = share.reindex(grid["день"]).to_numpy()
    days = class_days(S["ev"])
    for c in SYS_CLASSES:
        i = days.pivot_table(index="день", columns="ид_объект", values=c, aggfunc="max").reindex(
            index=ax.days, columns=ax.objects).fillna(0)
        other = (i.to_numpy().sum(axis=1, keepdims=True) - i).shift(1)
        out[f"других_объектов_{c}_1д"] = mat_to_grid(other, ax)
    d, cols = prefixed(days, SYS_CLASSES, "дней_")
    w7 = fe.window_sums(grid, d, cols, [7])
    sib = fe.sibling_sums(w7, [f"дней_{c}_7д" for c in SYS_CLASSES], ctx["parent"])
    out = pd.concat([out, sib.drop(columns=KEY)], axis=1)
    cnt = ctx["ch_types"].rename(columns=STATIC_TYPES)[list(STATIC_TYPES.values())].add_prefix("каналов_")
    cnt["каналов_всего"] = ctx["n_all"]
    cnt["число_пикетов"] = channels.groupby("ид_объект")["пикет"].nunique()
    out = pd.concat([out, grid[["ид_объект"]].merge(cnt, left_on="ид_объект", right_index=True, how="left")
                     .drop(columns="ид_объект").set_index(grid.index)], axis=1)
    return out


BUILDERS = [build_history, build_hours, build_channels, build_comm, build_telemetry, build_context, build_calendar,
            build_ppr, build_pumps, build_system]


def build_all(S, grid, ctx) -> pd.DataFrame:
    """Все построенные признаки (ключ + признаки всех групп) в порядке строк grid."""
    parts = [b(S, grid, ctx) for b in BUILDERS]
    return pd.concat([parts[0]] + [p.drop(columns=KEY) for p in parts[1:]], axis=1)
