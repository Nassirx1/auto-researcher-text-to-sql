#!/usr/bin/env bash
set -euo pipefail

python src/generate_databases.py
python src/generate_sql_dataset.py
