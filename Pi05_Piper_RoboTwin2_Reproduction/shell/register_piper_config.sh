#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="/root/rivermind-data/XPolicyLab/policy/Pi_05/openpi"
cd "$OPENPI_ROOT"

python - <<'PY'
from pathlib import Path

p = Path("src/openpi/training/config.py")
s = p.read_text()
anchor = "\nif len({config.name for config in _CONFIGS}) != len(_CONFIGS):"
insert = (
    "\n# Local RoboTwin Piper single-arm config. Keep upstream configs unchanged.\n"
    "from openpi.training.piper_config import PIPER_CONFIGS as _PIPER_CONFIGS\n"
    "_CONFIGS.extend(_PIPER_CONFIGS)\n"
)

if "_PIPER_CONFIGS" in s:
    raise SystemExit("Piper config registration already exists; inspect before rerunning.")
if s.count(anchor) != 1:
    raise SystemExit("Could not find the expected _CONFIGS registration anchor; no file changed.")

p.write_text(s.replace(anchor, insert + anchor, 1))
print("Registered pi05_piper_single_lora in config.py")
PY
