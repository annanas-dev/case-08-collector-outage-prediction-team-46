"""HTTP-сервис прогноза инцидентов: загрузка событий, прогноз на сутки, чтение прогнозов.

Запуск: `uvicorn annanas.service.app:app --host 0.0.0.0 --port 8000` (сервис `inference` в docker-compose.yml).
Инструкция для клиента – docs/inference.md, контракт – docs/openapi.yaml."""
import datetime as dt
import json
import math
import os
import shutil
import threading
from functools import cached_property
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from annanas import calendar_ru as cal
from annanas import etl
from annanas.inference.scorer import Scorer, predict_day
from annanas.inference.store import NoData, Store
from annanas.paths import PROCESSED_DIR
from annanas.service import schemas as sc

MODEL_TARGETS = ["отказ", "пожар", "подтопление", "любой_инцидент"]
FREQ_TARGETS = ["газ", "потеря_связи", "неисправность_охраны"]
ERRORS = {code: {"model": sc.Error} for code in (400, 404, 409, 422)}
# Примеры ответов для OpenAPI – реальный день теста (пишет scripts/export_openapi.py --examples)
EXAMPLES_FILE = Path(__file__).with_name("examples.json")
EXAMPLES = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8")) if EXAMPLES_FILE.exists() else {}


def responses(errors=True) -> dict:
    return dict(ERRORS) if errors else {}


def add_examples(spec: dict) -> dict:
    """Примеры ответов 200 по operationId. Вставляются в готовую схему: FastAPI выгружает OpenAPI с
    exclude_none и выбросил бы из примеров null-поля (маска, вероятность, плановые_работы)."""
    for ops in spec["paths"].values():
        for op in ops.values():
            if op.get("operationId") in EXAMPLES:
                op["responses"]["200"]["content"]["application/json"]["example"] = EXAMPLES[op["operationId"]]
    return spec


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Записать во временный файл рядом и переименовать: читатель видит старый файл или новый целиком."""
    tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Service:
    """Хранилище, модели и Spark-сессия (создается при первой загрузке событий). Загрузка и прогноз идут по
    очереди под одной блокировкой: Store меняется только при загрузке."""

    def __init__(self, root):
        self.root = Path(root)
        self.store = Store(self.root)
        self.lock = threading.Lock()
        self.pred_dir = self.root / "service" / "predictions"
        shutil.rmtree(self.root / "service" / "uploads", ignore_errors=True)   # файлы загрузок, прерванных сбоем

    @cached_property
    def scorer(self) -> Scorer:
        return Scorer(self.root / "models", self.root / "features_catalog.parquet")

    @cached_property
    def spark(self):
        from annanas.spark import get_spark
        return get_spark("inference", cores=4, memory="4g")

    def cache_file(self, day: dt.date) -> Path:
        return self.pred_dir / f"{day:%Y-%m-%d}.parquet"

    def predict(self, day: dt.date) -> dict:
        with self.lock:
            out, summary = predict_day(self.store, self.scorer, day)
            if pd.Timestamp(day).year > cal.load_calendar()["дата"].max().year:
                summary["предупреждения"].append(
                    f"производственный календарь на {day.year} год не загружен: праздники считаются рабочими днями")
            self.pred_dir.mkdir(parents=True, exist_ok=True)
            write_parquet_atomic(out, self.cache_file(day))
            return summary

    def ingest(self, path: Path) -> dict:
        from annanas.inference.ingest import ingest_csv
        with self.lock:
            report = ingest_csv(self.spark, path, self.root)
            if report["пересчитано_с"]:   # новые строки или доведенный после сбоя пересчет
                self.store.reload()
                for f in self.pred_dir.glob("*.parquet"):   # прогнозы, которые могли измениться
                    if dt.date.fromisoformat(f.stem) > report["пересчитано_с"]:
                        f.unlink()
            return report


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _reasons(text):
    return [{**r, "значение": _num(r["значение"])} for r in json.loads(text)]


def to_forecast(row) -> dict:
    """Строка кэша прогнозов (формат scorer.predict_day) → ObjectForecast."""
    ppr = json.loads(row["плановые_работы"]) if row["плановые_работы"] else None
    return {
        "ид_объект": int(row["ид_объект"]), "дата": row["день"], "маска": row["маска"],
        "прогноз": {t: {"вероятность": _num(row[f"вероятность_{t}"]), "балл": _num(row[f"балл_{t}"]),
                        "уровень": row[f"уровень_{t}"], "причины": _reasons(row[f"причины_{t}"])}
                    for t in MODEL_TARGETS},
        "фоновый_риск": {t: _num(row[f"вероятность_{t}"]) for t in FREQ_TARGETS},
        "локация_отказа": {"каналы": json.loads(row["каналы_отказ"]), "пикеты": json.loads(row["пикеты_отказ"])},
        "плановые_работы": ppr,
    }


def _error(status, message, details=None):
    return JSONResponse(status_code=status, content={"ошибка": message, "подробности": details})


def create_app(root=PROCESSED_DIR) -> FastAPI:
    svc = Service(root)
    app = FastAPI(
        title="Прогноз инцидентов в коллекторах",
        version="1.0",
        generate_unique_id_function=lambda route: route.name,   # operationId = имя функции: health, predict, ...
        description="Вероятность отказа, пожара, подтопления и любого инцидента на объекте в ближайшие 24 ч "
                    "по журналу событий СМВУ. Порядок работы: POST /events → POST /predict → GET /predictions.",
    )
    app.state.service = svc
    base_openapi = app.openapi
    app.openapi = lambda: app.openapi_schema or add_examples(base_openapi())

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        details = [{"где": ".".join(map(str, e["loc"])), "что": e["msg"]} for e in exc.errors()]
        return _error(422, "неверные параметры запроса", details)

    @app.exception_handler(NoData)
    async def no_data(request, exc):
        store = svc.store
        return _error(409, str(exc), {"первый_день_данных": f"{store.first_day():%Y-%m-%d}",
                                      "последний_день_данных": f"{store.last_data_day():%Y-%m-%d}"})

    @app.get("/health", response_model=sc.Health, responses=responses(False),
             summary="Состояние сервиса и данных")
    def health():
        st = svc.store
        last = st.last_data_day()
        return {"статус": "ok", "версия_модели": svc.scorer.version, "первый_день_данных": st.first_day().date(),
                "последний_день_данных": last.date(), "прогноз_возможен_по": (last + pd.Timedelta(days=1)).date(),
                "календарь_по": dt.date(cal.load_calendar()["дата"].max().year, 12, 31),
                "объектов": len(st.objects)}

    @app.get("/objects", response_model=list[sc.ObjectInfo], responses=responses(False),
             summary="Справочник объектов")
    def objects():
        o = svc.store.objects.astype(object).where(svc.store.objects.notna(), None)
        return o.to_dict("records")

    @app.post("/events", response_model=sc.IngestReport, responses=responses(),
              summary="Загрузить события журнала",
              description="CSV формата ext-journal (`ид_события, ид_канала_данных, дата, время, тревожное, "
                          "значение_датчика`) телом запроса (`text/csv`) или файлом (`multipart/form-data`, поле "
                          "`file`). Повторная загрузка тех же строк ничего не меняет. После загрузки пересчитываются "
                          "эпизоды и суточные витрины; прогнозы на затронутые даты нужно запросить заново.",
              openapi_extra={"requestBody": {"required": True, "content": {
                  "text/csv": {"schema": {"type": "string"}},
                  "multipart/form-data": {"schema": {"type": "object", "required": ["file"], "properties": {
                      "file": {"type": "string", "format": "binary"}}}}}}})
    async def events(request: Request):
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise HTTPException(400, "в multipart нет файла в поле file")
            data = await upload.read()
        else:
            data = await request.body()
        header = data.split(b"\n", 1)[0].decode("utf-8-sig", errors="replace").replace('"', "").strip()
        if [c.strip() for c in header.split(",")] != etl.RAW_COLUMNS:
            raise HTTPException(400, {"ошибка": "неверный формат CSV", "подробности": {
                "ожидается_заголовок": ",".join(etl.RAW_COLUMNS), "получено": header[:200]}})
        up_dir = svc.root / "service" / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        path = up_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        path.write_bytes(data)
        try:
            return await run_in_threadpool(svc.ingest, path)
        finally:
            path.unlink(missing_ok=True)

    @app.post("/predict", response_model=sc.PredictSummary, responses=responses(),
              summary="Рассчитать прогноз на сутки",
              description="Прогноз на сутки `date` по данным строго до `date` 00:00 для всех объектов; результат "
                          "сохраняется и читается через GET /predictions. Повторный вызов пересчитывает.")
    async def predict(date: dt.date = Query(description="сутки прогноза D, YYYY-MM-DD")):
        return await run_in_threadpool(svc.predict, date)

    @app.get("/predictions", response_model=sc.Predictions, responses=responses(),
             summary="Прогнозы на сутки")
    def predictions(date: dt.date = Query(description="сутки прогноза D, YYYY-MM-DD"),
                    ид_объект: int | None = Query(None, description="только этот объект")):
        try:   # файл может удалить параллельная загрузка событий: прогноз устарел
            df = pd.read_parquet(svc.cache_file(date))
        except FileNotFoundError:
            raise HTTPException(404, f"прогноз на {date} не рассчитан: вызовите POST /predict?date={date}")
        if ид_объект is not None:
            df = df[df["ид_объект"] == ид_объект]
            if df.empty:
                raise HTTPException(404, f"объекта {ид_объект} нет в справочнике")
        df = df.astype(object).where(df.notna(), None)
        return {"дата": date, "версия_модели": svc.scorer.version,
                "объекты": [to_forecast(r) for r in df.to_dict("records")]}

    @app.get("/schedule", response_model=list[sc.ScheduleWindow], responses=responses(),
             summary="Окна плановых работ АКМ",
             description="Окна ППР аппаратуры контроля метана: график 2026 и окна, найденные по журналу. С `date` – "
                         "только окна, в которые сутки попадают с запасом ±3 сут. (эти объекты замаскированы).")
    def schedule(date: dt.date | None = Query(None, description="YYYY-MM-DD")):
        st = svc.store
        w = st.windows if date is None else st.active_windows(date)
        return [{"ид_объект": int(r["ид_объект"]), "начало": r["начало"].date(), "конец": r["конец"].date(),
                 "источник": r["источник"]} for r in w.to_dict("records")]

    return app


app = create_app()
