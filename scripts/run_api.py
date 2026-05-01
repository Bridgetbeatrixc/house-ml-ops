from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "artifacts" / "frontend-api.log"

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

log_file = LOG_PATH.open("a", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000)
