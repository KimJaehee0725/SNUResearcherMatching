#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -f trained_model/model.safetensors || ! -f data/demo/ver1/full.faiss ]]; then
  echo "Missing demo artifacts. Run ./demo/download_artifacts.sh first." >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  docker compose up --build
elif command -v docker-compose >/dev/null 2>&1; then
  docker-compose up --build
else
  echo "Missing docker compose. Install Docker Compose v2 or docker-compose." >&2
  exit 1
fi
