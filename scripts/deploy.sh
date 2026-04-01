#!/usr/bin/env bash
# One-shot deploy: git pull → backend venv → frontend build → restart services.
#
# Usage (from repo root on the VM):
#   VITE_API_URL=https://your-host.eastus.cloudapp.azure.com ./scripts/deploy.sh
#
# Or with a server config file:
#   sudo mkdir -p /etc/gov-ai
#   echo 'VITE_API_URL=https://your-host.eastus.cloudapp.azure.com' | sudo tee /etc/gov-ai/deploy.env
#   ./scripts/deploy.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f /etc/gov-ai/deploy.env ]]; then
  # shellcheck source=/dev/null
  source /etc/gov-ai/deploy.env
fi

VITE_API_URL="${VITE_API_URL:-}"
if [[ -z "${VITE_API_URL}" ]]; then
  echo "ERROR: Set VITE_API_URL to your public site URL (no trailing slash), e.g." >&2
  echo "  VITE_API_URL=https://government-ai.eastus.cloudapp.azure.com ./scripts/deploy.sh" >&2
  exit 1
fi

GIT_BRANCH="${GIT_BRANCH:-main}"
LLAMA_CUDA_REINSTALL="${LLAMA_CUDA_REINSTALL:-0}"

echo "==> Repo: $ROOT"
echo "==> Branch: $GIT_BRANCH"
echo "==> VITE_API_URL: $VITE_API_URL"

echo "==> Git pull"
git fetch origin
git pull origin "$GIT_BRANCH"

BACKEND_PY="${ROOT}/backend/.venv/bin/python"
BACKEND_PIP="${ROOT}/backend/.venv/bin/pip"
if [[ ! -x "$BACKEND_PY" ]]; then
  echo "ERROR: Backend venv not found at backend/.venv — create it first:" >&2
  echo "  cd backend && python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

echo "==> Backend: pip install -r requirements.txt"
(
  cd "${ROOT}/backend"
  "$BACKEND_PIP" install -r requirements.txt
)

if [[ "$LLAMA_CUDA_REINSTALL" == "1" ]]; then
  echo "==> Reinstalling llama-cpp-python (CUDA cu124 wheel)"
  "$BACKEND_PIP" uninstall -y llama-cpp-python 2>/dev/null || true
  "$BACKEND_PIP" install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
fi

echo "==> Frontend: npm ci && npm run build"
(
  cd "${ROOT}/frontend"
  printf '%s\n' "VITE_API_URL=${VITE_API_URL}" > .env.production
  npm ci
  npm run build
)

# Nginx must traverse home and read dist (typical Azureuser layout)
if [[ -d "${ROOT}/frontend/dist" ]]; then
  chmod 755 "$HOME" 2>/dev/null || true
  chmod -R a+rX "${ROOT}/frontend/dist"
fi

if systemctl is-active --quiet gov-ai-api.service 2>/dev/null; then
  echo "==> systemctl restart gov-ai-api.service"
  sudo systemctl restart gov-ai-api.service
else
  echo "WARN: gov-ai-api.service is not active — start it manually or create the systemd unit." >&2
fi

echo "==> nginx test + reload"
sudo nginx -t
sudo systemctl reload nginx

echo "==> Deploy finished OK."
echo "    Open: ${VITE_API_URL}/"
echo "    Ensure backend CORS_ORIGINS includes: ${VITE_API_URL}"
