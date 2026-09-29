"""Хранилище сервиса: источники признаков, маски и сетка на сутки D из кэшей `datasets/processed`.

Все источники маленькие (эпизоды и суточные витрины), поэтому Store держит их в памяти целиком и на каждые
сутки D обрезает через `features.cut(S, D)` – тот же путь, что проверен против матрицы обучения. Журнал
событий (`events/`) Store не читает: его дописывает и по нему пересчитывает витрины `ingest`.

Маска суток – как в целевой (notebooks/03_target): общие маски `target.GLOBAL_MASKS`, затем «объект не
подключен / не на связи» (`target.connection_mask`), затем «плановые работы АКМ» (`maintenance.ppr_mask`, окна
`maintenance.ppr_windows` – по графику и дням тестового газа). Сервис добавляет одно правило для пропусков
выгрузки: сутки, когда журнал пуст целиком (например, июль 2026 в данных заказчика), получают общую маску «нет
данных в журнале», а медиана записей для «не на связи» считается по суткам с данными
(`connection_mask(gap_days=...)`). Пропуски истории, закрытые общими масками обучения, так не обрабатываются,
поэтому маски до 2026-06-30 совпадают с целевой (проверяет tests/test_inference_store.py)."""
from functools import cached_property
from pathlib import Path

import pandas as pd

from annanas import aggregates as agg
from annanas import maintenance as mt
from annanas import smvu
from annanas import target as tg
from annanas.inference import features as fi
from annanas.paths import PROCESSED_DIR

KEY = fi.KEY
GAP_MASK = "нет данных в журнале"
T_FLAP_S = fi.T_FLAP.total_seconds()


class NoData(ValueError):
    """Для суток D нет данных (раньше начала истории или позже следующих суток после конца данных)."""


class Store:
    def __init__(self, root=PROCESSED_DIR):
        self.root = Path(root)

    def reload(self) -> None:
        """Сбросить все, что прочитано с диска (после дозагрузки событий)."""
        for name in ["S", "channels", "ctx", "objects", "daily_records", "gap_days", "windows", "eps_fault",
                     "mask_history"]:
            self.__dict__.pop(name, None)

    # --- статичное -------------------------------------------------------------------------------------

    @cached_property
    def channels(self) -> pd.DataFrame:
        return agg.read_vitrine(self.root / "channels.parquet")

    @cached_property
    def ctx(self) -> dict:
        return fi.load_context(self.root / "features_context.json", self.channels)

    def context(self) -> dict:
        return self.ctx

    @cached_property
    def objects(self) -> pd.DataFrame:
        """Справочник объектов: `ид_объект, название, родитель, название_родителя, каналов, есть_телеметрия`."""
        ch = self.channels
        g = ch.groupby("ид_объект").agg(название=("диспетчерское_название_объекта", "first"),
                                        родитель=("родитель", "first"),
                                        название_родителя=("название_родителя", "first"),
                                        каналов=("ид_канала_данных", "nunique"))
        dc = self.S["dc"]
        meas = dc.loc[dc["класс_записи"] == "измерение", "ид_канала_данных"].unique()
        g["есть_телеметрия"] = g.index.isin(ch.loc[ch["ид_канала_данных"].isin(meas), "ид_объект"].unique())
        return g.reset_index()

    # --- источники -------------------------------------------------------------------------------------

    @cached_property
    def S(self) -> dict:
        """Источники признаков за всю историю (без `ev` и `mask`: их добавляет sources(D)). Окна ППР строятся
        заново из дней тестового газа и демонтажа и графика `maintenance.SCHEDULE_FILE` – так же, как в
        03_target, поэтому правка графика действует после перезапуска сервиса."""
        S = fi.read_sources(self.root)
        maint = self.root / "maintenance"
        burst = agg.read_vitrine(maint / "gas_burst_days.parquet")
        fault = agg.read_vitrine(maint / "gas_fault_days.parquet")
        S["ppr"] = mt.ppr_windows(mt.load_schedule(), mt.extend_to_dismantling(mt.detect_windows(burst), fault))
        return S

    @cached_property
    def eps_fault(self) -> pd.DataFrame:
        """Эпизоды отказа от минуты или незакрытые, с пикетом – для «где искать» (location.rank_grid)."""
        e = agg.read_vitrine(fi.episodes_dir(self.root))
        e = e[["ид_канала_данных", "ид_объект", "пикет", "начало", "длительность_с", "класс"]]
        dur = pd.to_numeric(e["длительность_с"], errors="coerce")
        return e[(e["класс"] == "отказ") & (dur.isna() | (dur >= T_FLAP_S))].reset_index(drop=True)

    @cached_property
    def windows(self) -> pd.DataFrame:
        """Окна ППР АКМ: `ид_объект, начало, конец, источник`."""
        return self.S["ppr"]

    @cached_property
    def daily_records(self) -> pd.DataFrame:
        """Записей объекта за сутки (все каналы справочника) – для маски связи."""
        dc = self.S["dc"].merge(self.channels[["ид_канала_данных", "ид_объект"]], on="ид_канала_данных")
        return dc.groupby(KEY, as_index=False)["количество_событий"].sum().rename(
            columns={"количество_событий": "записей"})

    @cached_property
    def gap_days(self) -> pd.DatetimeIndex:
        """Сутки между первым и последним днем данных, когда в журнале нет ни одной записи, кроме суток под
        общими масками обучения (там все как в целевой: 2024-04-07/08, 2026-06-01)."""
        days = pd.DatetimeIndex(self.S["dc"]["день"].unique()).sort_values()
        gaps = pd.date_range(days[0], days[-1]).difference(days)
        known = tg.add_masks(pd.DataFrame({"день": gaps}), tg.GLOBAL_MASKS)["маска"].notna().to_numpy()
        return gaps[~known]

    def first_day(self) -> pd.Timestamp:
        return self.ctx["start"]

    def last_data_day(self) -> pd.Timestamp:
        return pd.Timestamp(self.S["dc"]["день"].max())

    def check_day(self, D) -> pd.Timestamp:
        """Сутки D допустимы: после первого дня истории и не позже суток, следующих за последним днем данных."""
        D = pd.Timestamp(D).normalize()
        last = self.last_data_day()
        if D <= self.first_day():
            raise NoData(f"дата {D:%Y-%m-%d} не позже начала истории {self.first_day():%Y-%m-%d}")
        if D > last + pd.Timedelta(days=1):
            raise NoData(f"данные есть по {last:%Y-%m-%d}: прогноз возможен на даты не позже "
                         f"{last + pd.Timedelta(days=1):%Y-%m-%d}; загрузите события через POST /events")
        return D

    # --- маски и сетка ---------------------------------------------------------------------------------

    @cached_property
    def mask_history(self) -> pd.DataFrame:
        """Маски `ид_объект, день, маска` для всех объектов с первого дня истории по сутки после последнего дня
        данных. Маска суток D опирается только на данные до D (connection_mask), поэтому одна таблица годится для
        любых D; строится на всю длину, так как connection_mask ищет первую запись объекта в пределах сетки."""
        grid = tg.build_grid(self.objects["ид_объект"], self.first_day(), self.last_data_day() + pd.Timedelta(days=1))
        gaps = [(GAP_MASK, d, d + pd.Timedelta(days=1)) for d in self.gap_days]
        grid = tg.add_masks(grid, tg.GLOBAL_MASKS + gaps)
        conn = tg.connection_mask(grid, self.daily_records, min_median=200, window=30, gap_days=self.gap_days)
        grid["маска"] = grid["маска"].where(grid["маска"].notna(), conn)
        grid["маска"] = grid["маска"].where(grid["маска"].notna(), mt.ppr_mask(grid, self.windows))
        grid["маска"] = grid["маска"].astype(object).where(grid["маска"].notna(), None)
        return grid

    def grid(self, D) -> pd.DataFrame:
        """Сетка «все объекты × {D}»: `ид_объект, день, период_смву, эпоха_флага, маска`."""
        D = pd.Timestamp(D)
        m = self.mask_history
        g = m[m["день"] == D].reset_index(drop=True)
        g["период_смву"] = smvu.period_of(g["день"])
        g["эпоха_флага"] = tg.flag_epoch(g["день"])
        return g[KEY + ["период_смву", "эпоха_флага", "маска"]]

    def sources(self, D) -> dict:
        """Источники признаков в том виде, в каком они были в D 00:00 (features.cut)."""
        D = pd.Timestamp(D)
        return fi.cut({**self.S, "mask": self.mask_history}, D)

    def features(self, D) -> tuple[pd.DataFrame, dict]:
        """Признаки всех объектов на сутки D (ключ, маска и все построенные признаки) и источники cut(S, D)."""
        D = self.check_day(D)
        grid = self.grid(D)
        S = self.sources(D)
        X = fi.build_all(S, grid, self.ctx)
        X.insert(2, "маска", grid["маска"].to_numpy())
        return X, S

    def gaps_before(self, D, days=30) -> int:
        """Сколько из `days` суток до D журнал пуст целиком (для предупреждения в ответе)."""
        D = pd.Timestamp(D)
        lo = D - pd.Timedelta(days=days)
        last = self.last_data_day()
        missing = ((self.gap_days >= lo) & (self.gap_days < D)).sum()
        return int(missing + max(0, (D - pd.Timedelta(days=1) - last).days))

    def active_windows(self, D, pad_days=mt.PAD_DAYS) -> pd.DataFrame:
        """Окна ППР, в которые (с запасом ±pad_days) попадают сутки D."""
        D = pd.Timestamp(D)
        w = self.windows
        pad = pd.Timedelta(days=pad_days)
        hit = (w["начало"].dt.normalize() - pad <= D) & (D <= w["конец"].dt.normalize() + pad)
        return w[hit].reset_index(drop=True)
