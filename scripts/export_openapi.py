"""Выгрузка контракта API в docs/openapi.yaml (make openapi).

С `--examples` сначала пересчитывает примеры ответов на реальном дне теста и пишет их в
src/annanas/service/examples.json (нужны данные datasets/processed и модели; прогноз пишется в кэш сервиса).
Пример ответа POST /events – отчет загрузки демо-дня 2026-08-01 из сквозного теста (tests/test_service_e2e.py)."""
import argparse
import json
from pathlib import Path

import yaml

from annanas.paths import PROJECT_ROOT

EXAMPLE_DAY = "2026-06-30"
OUT = PROJECT_ROOT / "docs" / "openapi.yaml"
EVENTS_EXAMPLE = {   # отчет POST /events для datasets/журнал_событий_пример.csv на копии хранилища (e2e)
    "строк_в_файле": 169993, "строк_заголовка": 0, "полных_дублей": 0, "битых_дат": 0,
    "строк_после_очистки": 169993, "тревожных": 207, "строк_каналов_вне_справочника": 0, "уже_загружено": 0,
    "добавлено": 169993, "первый_день": "2026-08-01", "последний_день": "2026-08-01",
    "пересчитано_с": "2026-07-30", "пересчитано_по": "2026-08-01", "секунд": 33.8}


def build_examples() -> dict:
    from fastapi.testclient import TestClient

    from annanas.service.app import create_app

    client = TestClient(create_app())
    ex = {"health": client.get("/health").json(), "objects": client.get("/objects").json()[:2],
          "events": EVENTS_EXAMPLE,
          "predict": client.post("/predict", params={"date": EXAMPLE_DAY}).json(),
          "schedule": client.get("/schedule", params={"date": EXAMPLE_DAY}).json()}
    objs = client.get("/predictions", params={"date": EXAMPLE_DAY}).json()
    rows = objs["объекты"]
    top = max((r for r in rows if r["маска"] is None), key=lambda r: r["прогноз"]["отказ"]["вероятность"])
    masked = [r for r in rows if r["маска"] is not None][:1]
    ex["predictions"] = {**objs, "объекты": [top] + masked}
    return ex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", action="store_true", help="пересчитать примеры на реальном дне теста")
    args = parser.parse_args()
    if args.examples:
        from annanas.service import app as service_app
        service_app.EXAMPLES_FILE.write_text(json.dumps(build_examples(), ensure_ascii=False, indent=1) + "\n",
                                             encoding="utf-8")
        print(f"примеры: {service_app.EXAMPLES_FILE.relative_to(PROJECT_ROOT)}")
    # новый процесс не нужен: create_app читает examples.json при импорте модуля, перечитываем явно
    import importlib

    from annanas.service import app as service_app
    importlib.reload(service_app)
    spec = json.loads(json.dumps(service_app.create_app().openapi()))
    OUT.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    print(f"{Path(OUT).relative_to(PROJECT_ROOT)}: {len(spec['paths'])} путей")


if __name__ == "__main__":
    main()
