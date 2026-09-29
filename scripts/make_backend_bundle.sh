#!/bin/bash
# Архив сервиса прогноза для бэкенда: код (git archive HEAD), модель и минимальные данные datasets/processed.
# scripts/make_backend_bundle.sh  →  dist/annanas-inference-<YYYYMMDD>/ и dist/annanas-inference-<YYYYMMDD>.tar.gz
# Данные берутся из DATASETS_DIR (по умолчанию datasets/), каталог эпизодов – тот, что выбирает сервис.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SRC=${DATASETS_DIR:-$ROOT/datasets}/processed
SAMPLE=${DATASETS_DIR:-$ROOT/datasets}/журнал_событий_пример.csv
NAME=annanas-inference-$(date +%Y%m%d)
OUT=$ROOT/dist/$NAME
# события: годы, нужные дозагрузке (дедупликация по месяцам новых строк, эпизоды пересчитываются по году целиком
# плюс сутки до него, замеры – 31 сутки назад); история для прогноза – в эпизодах и витринах, не в событиях
EVENT_YEARS=${EVENT_YEARS:-"2025 2026"}

clone() {   # копия APFS-клоном (без лишнего места на диске), иначе обычная
  mkdir -p "$(dirname "$2")"
  cp -cR "$1" "$2" 2>/dev/null || cp -R "$1" "$2"
}

[ -f "$SRC/models/config.json" ] || { echo "нет $SRC/models/config.json" >&2; exit 1; }
EP_DIR=$(DATASETS_DIR=$(dirname "$SRC") uv run --frozen python -c \
  "from annanas.inference.features import episodes_dir; print(episodes_dir().name)")
MODEL=$(uv run --frozen python -c "import json,sys; c=json.load(open(sys.argv[1])); \
print(c['created_at'], *c['train_period'])" "$SRC/models/config.json")
read -r VERSION TRAIN_FROM TRAIN_TO <<<"$MODEL"
SHA=$(git rev-parse --short HEAD)
[ -z "$(git status --porcelain -- src tests Dockerfile pyproject.toml uv.lock docs)" ] \
  || echo "внимание: незакоммиченные правки не попадут в архив (git archive HEAD)" >&2

rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT/code"

# --- код -------------------------------------------------------------------------------------------------
# тесты: только быстрые (помеченные slow читают полный datasets/processed и пропускаются без него)
FAST_TESTS=$(git ls-files tests | grep -v -e test_inference_features -e test_inference_scorer \
  -e test_inference_store -e test_service_e2e)
git archive HEAD src $FAST_TESTS scripts/export_openapi.py Dockerfile .dockerignore pyproject.toml uv.lock \
  .python-version README.md docs/inference.md docs/openapi.yaml docs/data.md | tar -x -C "$OUT/code"

# --- данные ----------------------------------------------------------------------------------------------
D=$OUT/data/processed
for rel in models features_catalog.parquet features_context.json channels.parquet maintenance \
  "episodes/$EP_DIR" episodes/comm_loss_k10_s50.parquet episodes/flag_hourly.parquet \
  eda/daily_channel.parquet eda/meas_gaps.parquet eda/undefined_daily.parquet eda/undefined_bursts.parquet \
  eda/hourly_type.parquet features/object_day_states.parquet; do
  clone "$SRC/$rel" "$D/$rel"
done
mkdir -p "$D/events"
for y in $EVENT_YEARS; do clone "$SRC/events/год=$y" "$D/events/год=$y"; done
find "$D" -name '*.tmp' -prune -exec rm -rf {} +   # остатки прерванных замен витрин
# контрольные суммы Hadoop (.part-*.crc, ._SUCCESS.crc) не нужны для чтения, а имена «._*» tar на macOS
# принимает за метаданные AppleDouble и при распаковке падает с «Failed to restore metadata»
find "$D" -name '*.crc' -delete
clone "$SAMPLE" "$OUT/data/samples/журнал_событий_пример.csv"

# --- docker-compose.yml и README -------------------------------------------------------------------------
cat > "$OUT/docker-compose.yml" <<'EOF'
services:
  inference:
    build: ./code   # Python 3.11 + Java 17 (Spark нужен для POST /events)
    command: ["uvicorn", "annanas.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
    ports:
      - "127.0.0.1:${ANNANAS_PORT:-8000}:8000"   # API, Swagger: http://localhost:8000/docs
    volumes:
      - ./data:/data            # хранилище: события, витрины, модель, кэш прогнозов (сервис пишет сюда)
      - spark-tmp:/spark-tmp
    environment:
      DATASETS_DIR: /data
      SPARK_TMP_DIR: /spark-tmp
      # локальное время процесса = время Spark-сессии
      TZ: Europe/Moscow
      # glibc: не больше 2 арен malloc, иначе потоки uvicorn раздувают память Python до OOM
      MALLOC_ARENA_MAX: 2
    restart: unless-stopped

volumes:
  spark-tmp:
EOF

SIZE_GB=$(du -sk "$OUT/data" | awk '{printf "%.1f", $1 / 1048576}')
cat > "$OUT/README.md" <<EOF
# Сервис прогноза инцидентов в коллекторах ($NAME)

Внутри: \`code/\` – исходники и Dockerfile сервиса; \`data/processed/\` – модель (\`models/\`), справочник
каналов, эпизоды и суточные витрины 2019 – 2026-06-30, журнал событий за ${EVENT_YEARS// /, } (нужен для
дозагрузки); \`data/samples/журнал_событий_пример.csv\` – демо-сутки 2026-08-01. Модель \`$VERSION\`
(\`версия_модели\` в ответах), обучена на $TRAIN_FROM – $TRAIN_TO, пороги уровней – на 2025. Код – коммит $SHA.

Нужно: Docker с Compose v2, у Docker не меньше 10 ГБ памяти, диск около 10 ГБ: образ 7,5 ГБ, данные
${SIZE_GB} ГБ (растут примерно на 70 МБ за месяц загруженного журнала).

\`\`\`bash
docker compose up -d --build     # первая сборка около 3 мин; другой порт: ANNANAS_PORT=8080 docker compose up -d
curl localhost:8000/health       # последний_день_данных: 2026-06-30
curl -X POST localhost:8000/events -F "file=@data/samples/журнал_событий_пример.csv"   # около 40 с
curl -X POST "localhost:8000/predict?date=2026-08-02"                                 # около 10 с
curl "localhost:8000/predictions?date=2026-08-02"
# тесты (184, около 40 с): docker compose run --rm -v ./code/tests:/app/tests -v ./code/docs:/app/docs \\
#   -v ./docker-compose.yml:/app/docker-compose.yml inference pytest tests -q
\`\`\`

Порядок работы, поля, ошибки – \`code/docs/inference.md\`, контракт – \`code/docs/openapi.yaml\`, Swagger –
http://localhost:8000/docs. В inference.md пути даны для репозитория: \`datasets/processed/\` здесь –
\`data/processed/\`, демо-журнал – \`data/samples/\`; \`make\` и ноутбуки в архив не входят.

- Сервис хранит состояние в \`data/\`: загруженные события остаются навсегда (демо-день тоже), для чистого
  старта распакуйте архив заново.
- Новая модель: заменить \`data/processed/models/\` целиком (и \`features_catalog.parquet\`, если его прислали
  вместе с моделью), затем \`docker compose restart\`.
- График ППР и производственный календарь лежат в \`code/src/annanas/data/\` и попадают в образ: после правки
  \`docker compose up -d --build\`.
EOF

# --- архив -----------------------------------------------------------------------------------------------
COPYFILE_DISABLE=1 tar -C "$ROOT/dist" --no-mac-metadata --no-xattrs --no-acls -czf "$OUT.tar.gz" "$NAME"
du -sh "$OUT" "$OUT.tar.gz"
