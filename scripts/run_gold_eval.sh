#!/usr/bin/env bash
set -euo pipefail

python src/evaluate_sql.py --config configs/base.yaml --mode gold
