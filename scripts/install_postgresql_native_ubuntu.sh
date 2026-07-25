#!/usr/bin/env bash
# PaperSearchAssistant2 — 在 Ubuntu/Debian 上安装本机 PostgreSQL（不用 Docker）
#
# 用法（需能 sudo）：
#   cd PaperSearchAssistant2
#   sudo bash scripts/install_postgresql_native_ubuntu.sh
#
# 若 apt update 因 nvidia.github.io 等源报错：脚本会自动注释这些行并重试 update 一次（.list 会先 .bak-psa2 备份）。
# 仍失败可跳过 update：sudo SKIP_APT_UPDATE=1 bash scripts/install_postgresql_native_ubuntu.sh
# 禁止自动改源：sudo NO_AUTO_FIX_APT=1 bash scripts/install_postgresql_native_ubuntu.sh
#
# 成功后 .env.runtime 中 DATABASE_URL 使用端口 5432，再执行：
#   python3 scripts/init_pg_schema.py

set -euo pipefail

DB_NAME="papersearch"
DB_USER="papersearch"
DB_PASS="papersearch_dev"

if [[ "${EUID:-0}" -ne 0 ]]; then
  echo "请用 root 执行: sudo bash scripts/install_postgresql_native_ubuntu.sh"
  exit 1
fi

fix_bad_nvidia_github_apt_sources() {
  local f
  shopt -s nullglob
  for f in /etc/apt/sources.list.d/*.list; do
    [[ -f "$f" ]] || continue
    grep -q 'nvidia\.github\.io' "$f" 2>/dev/null || continue
    grep -qE '^[[:space:]]*[^#[:space:]]' "$f" 2>/dev/null || continue
    cp -a -- "$f" "${f}.bak-psa2"
    sed -i '/nvidia\.github\.io/ { /^[[:space:]]*#/! s/^/# /; }' "$f"
    echo "已注释失效源（备份 ${f}.bak-psa2）: $f"
  done
  # deb822 .sources（若含 nvidia.github.io，整体停用以免 update 失败）
  for f in /etc/apt/sources.list.d/*.sources; do
    [[ -f "$f" ]] || continue
    grep -q 'nvidia\.github\.io' "$f" 2>/dev/null || continue
    mv -f -- "$f" "${f}.disabled-psa2"
    echo "已停用 deb822 源（改名 .disabled-psa2）: $f"
  done
  shopt -u nullglob
  return 0
}

run_apt_update() {
  if [[ "${SKIP_APT_UPDATE:-0}" == "1" ]]; then
    echo "跳过 apt-get update（SKIP_APT_UPDATE=1）"
    return 0
  fi
  if apt-get update -qq; then
    return 0
  fi
  echo ""
  echo "apt-get update 失败（常见：nvidia.github.io 在 Focal 上已无 Release）。"
  if [[ "${NO_AUTO_FIX_APT:-0}" != "1" ]]; then
    echo "将注释含 nvidia.github.io 的 .list 行并重试 update 一次…"
    fix_bad_nvidia_github_apt_sources || true
    if apt-get update -qq; then
      return 0
    fi
  fi
  echo "仍失败。可手动注释 /etc/apt/sources.list.d 下相关源，或："
  echo "  sudo SKIP_APT_UPDATE=1 bash $0"
  return 1
}

export DEBIAN_FRONTEND=noninteractive
run_apt_update
apt-get install -y postgresql postgresql-contrib

PG_VER=""
if [[ -d /etc/postgresql ]]; then
  PG_VER="$(ls -1 /etc/postgresql | sort -V | tail -1)"
fi
if [[ -z "${PG_VER}" ]]; then
  echo "错误: 未在 /etc/postgresql 下找到版本目录"
  exit 1
fi

CONF_DIR="/etc/postgresql/${PG_VER}/main"
PG_CONF="${CONF_DIR}/postgresql.conf"
HBA="${CONF_DIR}/pg_hba.conf"

# 监听本机（供 127.0.0.1 连接）
if grep -qE '^[[:space:]]*listen_addresses[[:space:]]*=' "${PG_CONF}"; then
  sed -i -E "s/^[[:space:]]*#?[[:space:]]*listen_addresses[[:space:]]*=.*/listen_addresses = 'localhost'/" "${PG_CONF}"
else
  echo "listen_addresses = 'localhost'" >> "${PG_CONF}"
fi

if ! grep -qE 'host[[:space:]]+all[[:space:]]+all[[:space:]]+127\.0\.0\.1/32' "${HBA}"; then
  echo "host    all             all             127.0.0.1/32            scram-sha-256" >> "${HBA}"
fi

systemctl enable --now postgresql

ROLE_SQL="DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}') THEN
    CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}';
  ELSE
    ALTER ROLE ${DB_USER} WITH PASSWORD '${DB_PASS}';
  END IF;
END \$\$;"
sudo -u postgres psql -v ON_ERROR_STOP=1 -c "${ROLE_SQL}"

if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"
fi

sudo -u postgres psql -v ON_ERROR_STOP=1 -d "${DB_NAME}" -c "GRANT ALL ON SCHEMA public TO ${DB_USER}; GRANT CREATE ON SCHEMA public TO ${DB_USER};"

systemctl restart postgresql

echo ""
echo "完成: PostgreSQL ${PG_VER}；库=${DB_NAME} 用户=${DB_USER}"
echo "DATABASE_URL=postgresql+psycopg://${DB_USER}:${DB_PASS}@127.0.0.1:5432/${DB_NAME}"
echo "下一步（勿用 root）: cd PaperSearchAssistant2 && python3 scripts/init_pg_schema.py"
echo ""
