# Хакатон ДЖКХ

Анализ журнала событий диспетчерской системы (2019-2026).

## Структура

```
.
├── datasets/            # сырые данные (не в git): ext-journal-YYYY.csv, справочники, пример журнала
├── notebooks/           # исследования
├── src/annanas/         # общий код, пути проекта в paths.py
├── docs/                # ТЗ, схемы данных, разработка, глоссарий
├── archive/             # старые EDA-ноутбуки, код, отчеты, оригинал ТЗ
├── AGENTS.md            # карта проекта для AI-агентов (CLAUDE.md импортирует его)
└── Dockerfile, docker-compose.yml, Makefile
```

## Документация

- [ТЗ](docs/task.md) и [глоссарий](docs/glossary.md)
- [Данные](docs/data.md): файлы и схемы
- Исследование и выводы – в ноутбуках `notebooks/` ([правила](.agents/rules/notebooks.md))
- [Разработка](docs/development.md)

## Запуск

### В Docker (рекомендуется)

Нужен только Docker. Java 17 и зависимости ставятся в образ.

```bash
make up      # сборка + запуск, Jupyter Lab: http://localhost:8888, Spark UI: http://localhost:4040
make down    # остановить
make help    # все команды
```

Проект монтируется в контейнер целиком, так что правки ноутбуков и `src/` видны сразу.
После изменения зависимостей (`uv add ...`) нужно выполнить `make up`, образ пересоберется.

Spark в ноутбуках работает в local-режиме внутри контейнера. Для `spark.driver.memory=8g`
в Docker Desktop нужно выделить хотя бы 12 GB: Settings → Resources → Memory.

### Локально

Нужны [uv](https://docs.astral.sh/uv/) и Java 17+.

```bash
make install
make lab
```

Код из `src/annanas` ставится в окружение как пакет: `from annanas.paths import DATASETS_DIR`.

Добавить зависимость: `uv add <пакет>` (для инструментов разработки: `uv add --dev <пакет>`), затем `make up`.
