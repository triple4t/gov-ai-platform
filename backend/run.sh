#!/usr/bin/env bash
# Run the backend with the venv's Python so all dependencies are found.
# Usage: ./run.sh         — development (no auto-reload; stable for ZIP uploads / Neo4j sync)
#        ./run.sh reload   — dev with --reload (excludes uploads/)
#        ./run.sh prod     — production (no reload, bind 0.0.0.0)
cd "$(dirname "$0")"
if [ "${1:-}" = "prod" ]; then
  exec ./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
elif [ "${1:-}" = "reload" ]; then
  exec ./venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 \
    --reload-exclude "uploads"
else
  exec ./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
fi
