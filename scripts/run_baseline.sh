#!/usr/bin/env bash
set -euo pipefail

python src/evaluate_sql.py --config configs/qwen25_05b_smoke.yaml --mode baseline
