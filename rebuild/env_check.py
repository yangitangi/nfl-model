"""
rebuild/env_check.py
--------------------
Run FIRST in any new environment (especially a cloud session). Reports Python
version, CPU/memory/disk, which libraries import, and whether outbound network
to the data source and package index works. Writes rebuild/env_report.json.

    python rebuild/env_check.py
"""

import importlib
import json
import os
import platform
import shutil
import sys
import time
import urllib.request
from pathlib import Path

LIBS = ["pandas", "numpy", "pyarrow", "sklearn", "xgboost", "lightgbm", "scipy", "statsmodels",
        "pymc", "arviz", "cmdstanpy", "torch", "streamlit", "requests"]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

URLS = {
    "nflverse_schedules": config.NFLVERSE_SCHEDULE_URL,
    "nflverse_pbp_2025": config.NFLVERSE_PBP_URL_TEMPLATE.format(season=2025),
    "pypi": "https://pypi.org/simple/pip/",
}


def head(url: str, timeout: int = 20) -> dict:
    t0 = time.time()
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "env-check"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"ok": True, "status": r.status, "seconds": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200], "seconds": round(time.time() - t0, 2)}


def main():
    libs = {}
    for name in LIBS:
        try:
            m = importlib.import_module(name)
            libs[name] = getattr(m, "__version__", "installed")
        except Exception:  # noqa: BLE001
            libs[name] = None
    total, used, free = shutil.disk_usage(Path(__file__).resolve().parent)
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "disk_free_gb": round(free / 1e9, 1),
        "libs": libs,
        "network": {k: head(v) for k, v in URLS.items()},
    }
    out = Path(__file__).resolve().parent / "env_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    missing = [k for k, v in libs.items() if v is None and k in ("pandas", "numpy", "pyarrow", "sklearn", "xgboost")]
    if missing:
        print(f"\nCORE LIBS MISSING: {missing} -> pip install -r requirements.txt")


if __name__ == "__main__":
    main()
