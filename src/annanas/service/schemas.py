"""Схемы ответов API. Имена полей – по-русски, как в данных; имена схем – латиницей (для генераторов клиентов)."""
import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field

Level = Literal["высокий", "средний", "низкий", "нет данных"]


class Error(BaseModel):
    ошибка: str = Field(description="что пошло не так")
    подробности: Any = Field(None, description="детали: ошибки валидации, допустимый диапазон дат и т. п.")


class Health(BaseModel):
    статус: Literal["ok"]
    версия_модели: str = Field(description="created_at из models/config.json")
    первый_день_данных: dt.date
    последний_день_данных: dt.date = Field(description="последние сутки, за которые загружены события")
    прогноз_возможен_по: dt.date = Field(description="последний день данных + 1: прогноз строится по данным до D 00:00")
    календарь_по: dt.date = Field(description="до какой даты загружен производственный календарь РФ")
    объектов: int


class ObjectInfo(BaseModel):
    ид_объект: int
    название: str | None
    родитель: int | None
    название_родителя: str | None
    каналов: int = Field(description="каналов в справочнике")
    есть_телеметрия: bool = Field(description="каналы объекта присылали замеры (температура, газ)")


class IngestReport(BaseModel):
    строк_в_файле: int
    строк_заголовка: int = Field(description="строки-заголовки внутри файла (отброшены)")
    полных_дублей: int = Field(description="полные повторы строк в файле (отброшены)")
    битых_дат: int = Field(description="строки с нераспознанной датой/временем (отброшены)")
    строк_после_очистки: int
    тревожных: int = Field(description="строк с флагом «тревожное»")
    строк_каналов_вне_справочника: int = Field(description="приняты, но в прогноз не идут: канала нет в справочнике")
    уже_загружено: int = Field(description="строки, которые уже были в хранилище (повторная загрузка)")
    добавлено: int
    первый_день: dt.date | None = Field(description="сутки новых строк; null – новых строк нет")
    последний_день: dt.date | None
    пересчитано_с: dt.date | None = Field(description="с каких суток пересчитаны витрины (первый день − 2 сут.; "
                                                     "включая пересчет, не завершенный из-за сбоя сервиса)")
    пересчитано_по: dt.date | None
    секунд: float


class PredictSummary(BaseModel):
    дата: dt.date
    объектов: int
    с_прогнозом: int = Field(description="объектов без маски – для них есть вероятности")
    замаскировано: dict[str, int] = Field(description="маска → число объектов")
    секунд: float
    предупреждения: list[str]


class Reason(BaseModel):
    признак: str
    смысл: str
    значение: float | None
    вклад: float = Field(description="вклад признака в логит модели (SHAP), только положительные")


class TargetForecast(BaseModel):
    вероятность: float | None = Field(description="вероятность начала эпизода класса в [D 00:00, D + 24 ч)")
    балл: float | None = Field(description="та же оценка модели до калибровки; для сортировки")
    уровень: Level
    причины: list[Reason] = Field(description="до 3 признаков, сильнее всего поднявших риск")


class Forecasts(BaseModel):
    отказ: TargetForecast
    пожар: TargetForecast
    подтопление: TargetForecast
    любой_инцидент: TargetForecast


class Background(BaseModel):
    газ: float | None
    потеря_связи: float | None
    неисправность_охраны: float | None


class Channel(BaseModel):
    ид_канала_данных: int
    пикет: float | None
    дней_с_последнего: int = Field(description="суток с последнего отказа канала")
    дней_с_отказом_30д: int


class Picket(BaseModel):
    пикет: float
    дней_с_последнего: int
    дней_с_отказом_30д: int


class FaultLocation(BaseModel):
    каналы: list[Channel]
    пикеты: list[Picket]


class Window(BaseModel):
    начало: dt.date
    конец: dt.date
    источник: Literal["график", "журнал"] = Field(description="график ППР 2026 или окно, найденное по журналу")


class ScheduleWindow(Window):
    ид_объект: int


class ObjectForecast(BaseModel):
    ид_объект: int
    дата: dt.date
    маска: str | None = Field(description="почему прогноза нет (объект не на связи, плановые работы АКМ, ...); "
                                          "null – прогноз есть")
    прогноз: Forecasts
    фоновый_риск: Background = Field(description="сглаженная частота за 365 суток: газ, потеря связи, "
                                                "неисправность охраны")
    локация_отказа: FaultLocation = Field(description="где искать отказ: топ-3 каналов и пикетов по истории")
    плановые_работы: Window | None = Field(description="окно ППР аппаратуры контроля метана (±3 сут.), "
                                                      "в которое попадают сутки D")


class Predictions(BaseModel):
    дата: dt.date
    версия_модели: str
    объекты: list[ObjectForecast]
