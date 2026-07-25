#!/usr/bin/env bash
# Usage:
#   source scripts/use_mode.sh online
#   source scripts/use_mode.sh offline
#
# This script:
# 1) Copies mode template into .env.runtime
# 2) Exports variables from .env.runtime to current shell

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this script so env vars apply to current shell:"
  echo "  source scripts/use_mode.sh <online|offline>"
  exit 1
fi

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Missing mode. Use: source scripts/use_mode.sh <online|offline>"
  return 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

case "${MODE}" in
  online)
    TEMPLATE="${PROJECT_ROOT}/.env.online.example"
    ;;
  offline)
    TEMPLATE="${PROJECT_ROOT}/.env.offline.example"
    ;;
  *)
    echo "Invalid mode: ${MODE}. Use online or offline."
    return 1
    ;;
esac

RUNTIME_FILE="${PROJECT_ROOT}/.env.runtime"
cp "${TEMPLATE}" "${RUNTIME_FILE}"

set -a
# shellcheck disable=SC1090
source "${RUNTIME_FILE}"
set +a

echo "[mode-switch] applied ${MODE}"
echo "[mode-switch] runtime file: ${RUNTIME_FILE}"
echo "[mode-switch] RAG_NETWORK_MODE=${RAG_NETWORK_MODE:-unset}"
