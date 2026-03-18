#!/usr/bin/env bash
# Run the backend with the venv's Python so all dependencies (e.g. deepgram-sdk) are found.
cd "$(dirname "$0")"
exec ./venv/bin/python -m uvicorn app.main:app --reload
