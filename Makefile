COMPOSE := docker compose

.DEFAULT_GOAL := help
.PHONY: help build up down restart logs shell lab install lock test test-slow serve openapi bundle clean

help: ## Показать команды
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Собрать образ (Java 17 + зависимости из uv.lock)
	$(COMPOSE) build

up: ## Запустить Jupyter Lab в Docker: http://localhost:8888
	$(COMPOSE) up -d --build jupyter
	@echo "Jupyter Lab: http://localhost:8888  |  Spark UI: http://localhost:4040"

down: ## Остановить контейнер
	$(COMPOSE) down

restart: ## Перезапустить контейнер
	$(COMPOSE) restart

logs: ## Логи контейнера
	$(COMPOSE) logs -f

shell: ## Shell внутри контейнера
	$(COMPOSE) exec jupyter bash

install: ## Локальное окружение через uv (без Docker)
	uv sync

lab: ## Jupyter Lab локально (нужна Java 17+)
	uv run jupyter lab

lock: ## Обновить uv.lock после правки pyproject.toml, потом make up
	uv lock

test: ## Тесты (в контейнере, нужна Java)
	$(COMPOSE) exec -T jupyter pytest tests -v

test-slow: ## Долгие тесты на данных datasets/processed (признаки на сутки = features.parquet)
	$(COMPOSE) exec -T jupyter pytest tests -v -s -m slow

serve: ## Сервис прогноза в Docker: http://localhost:8000/docs (docs/inference.md)
	$(COMPOSE) up -d --build inference
	@echo "API: http://localhost:8000/docs  |  проверка: curl http://localhost:8000/health"

openapi: ## Перевыгрузить docs/openapi.yaml из FastAPI (EXAMPLES=1 – пересчитать примеры на данных)
	$(COMPOSE) exec -T jupyter python scripts/export_openapi.py $(if $(EXAMPLES),--examples)

bundle: ## Архив сервиса для бэкенда: dist/annanas-inference-<дата>.tar.gz (код HEAD + модель + данные)
	scripts/make_backend_bundle.sh

clean: ## Удалить контейнер, образ и том со Spark temp
	$(COMPOSE) down --rmi local --volumes
