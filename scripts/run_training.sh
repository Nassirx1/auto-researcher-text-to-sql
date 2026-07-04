#!/usr/bin/env bash
set -euo pipefail

python src/train_lora.py --config configs/qwen25_05b_smoke.yaml --experiment_id exp_test
