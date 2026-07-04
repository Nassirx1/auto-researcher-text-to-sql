#!/usr/bin/env bash
set -euo pipefail

python src/experiment_runner.py --config configs/qwen25_coder_3b_lora.yaml --n_experiments 5
