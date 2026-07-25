import os, sys, shutil, subprocess
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else None
if MODE not in ("online", "offline"):
    raise SystemExit("Usage: python scripts/run_mode.py <online|offline> -- <command...>")

# 命令从 "--" 后开始
sep_idx = sys.argv.index("--")
cmd = sys.argv[sep_idx + 1:]
if not cmd:
    raise SystemExit("Missing command after --")

root = Path(__file__).resolve().parent.parent
template = root / (".env.online.example" if MODE == "online" else ".env.offline.example")
runtime = root / ".env.runtime"

shutil.copyfile(template, runtime)

# 读取 .env.runtime 并装配环境变量
env = os.environ.copy()
for line in runtime.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k.strip()] = v.strip()

print(f"[run_mode] mode={MODE}, running: {' '.join(cmd)}")
subprocess.run(cmd, env=env, check=True)