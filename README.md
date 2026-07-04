# autoresearch-sql-slm

This project implements an autonomous research loop that proposes controlled training configuration changes, runs LoRA fine-tuning experiments, evaluates Text-to-SQL execution accuracy, and preserves the best-performing adapter.

`autoresearch-sql-slm` is designed for business Text-to-SQL experimentation with small Hugging Face language models. It generates synthetic multi-table SQLite databases, creates train/eval examples, evaluates generated SQL by execution, fine-tunes adapters with LoRA/QLoRA, and records every experiment in a leaderboard and research log. It is not a fully self-improving AI system.

## Architecture

The pipeline is script-first:

1. Generate deterministic synthetic SQLite databases.
2. Extract compact schema context for prompts.
3. Generate Text-to-SQL train/eval JSONL examples with executable gold SQL.
4. Validate gold SQL execution on CPU.
5. Run baseline model inference.
6. Fine-tune LoRA adapters.
7. Evaluate generated SQL by executing safe `SELECT`/`WITH` queries.
8. Use a rule-based research agent to propose controlled config changes.
9. Keep leaderboard entries, research notes, and the best adapter.

## Repository Structure

```text
configs/       YAML model, LoRA, training, and search-space configs
data/          Runtime datasets and SQLite databases
src/           Python scripts and reusable modules
experiments/   Per-experiment configs, reports, predictions, and adapters
best/          Best adapter and config copied from successful experiments
notebooks/     Operational Colab runner
scripts/       Shell wrappers for common workflows
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## CPU-Safe Smoke Test

These commands run without a GPU:

```bash
python src/generate_databases.py
python src/generate_sql_dataset.py --small
python src/evaluate_sql.py --config configs/base.yaml --mode gold
python src/experiment_runner.py --config configs/base.yaml --n_experiments 1 --dry_run
```

Gold mode validates that databases and JSONL files exist, eval IDs do not overlap with train IDs, gold SQL executes, and stored expected results match fresh execution.

## Colab Workflow

Use `notebooks/colab_runner.ipynb` or run:

```bash
git clone https://github.com/YOUR_USERNAME/autoresearch-sql-slm.git
cd autoresearch-sql-slm
pip install -r requirements.txt
python src/generate_databases.py
python src/generate_sql_dataset.py
python src/evaluate_sql.py --config configs/qwen25_05b_smoke.yaml --mode baseline
python src/experiment_runner.py --config configs/qwen25_coder_3b_lora.yaml --n_experiments 5
```

## Model Configs

Model IDs are configured in YAML, not hardcoded in scripts.

- `configs/qwen25_05b_smoke.yaml`: `Qwen/Qwen2.5-0.5B-Instruct` for fast pipeline checks.
- `configs/qwen25_coder_3b_lora.yaml`: `Qwen/Qwen2.5-Coder-3B-Instruct` for main development.
- `configs/qwen35_4b_lora.yaml`: `Qwen/Qwen3.5-4B` for optional stronger final runs. If Colab memory or compatibility issues appear, use the Qwen2.5 Coder 3B config.

## Data Generation

```bash
python src/generate_databases.py
python src/generate_sql_dataset.py
```

Fast test mode:

```bash
python src/generate_sql_dataset.py --small
```

Generated databases:

- `data/databases/ecommerce.db`
- `data/databases/saas.db`
- `data/databases/banking.db`

Generated datasets:

- `data/train.jsonl`
- `data/eval.jsonl`

## Evaluation

Gold validation:

```bash
python src/evaluate_sql.py --config configs/base.yaml --mode gold
```

Baseline model evaluation:

```bash
python src/evaluate_sql.py --config configs/qwen25_05b_smoke.yaml --mode baseline
```

Adapter evaluation:

```bash
python src/evaluate_sql.py --config configs/qwen25_coder_3b_lora.yaml --mode adapter --adapter_path best/adapter
```

Reports are saved under `experiments/<experiment_id>/eval_report.json` and predictions under `experiments/<experiment_id>/predictions.jsonl`.

## LoRA Training

```bash
python src/train_lora.py --config configs/qwen25_05b_smoke.yaml --experiment_id exp_test
```

Training uses the configured LoRA rank, alpha, dropout, target modules, max steps, batch size, gradient accumulation, and max sequence length. QLoRA 4-bit loading is enabled when `model.load_in_4bit: true`.

## AutoResearch Loop

```bash
python src/experiment_runner.py --config configs/qwen25_coder_3b_lora.yaml --n_experiments 5
```

The v0.1 research agent is rule-based. It reads prior leaderboard rows, proposes one or two config changes from `configs/search_space.yaml`, saves a hypothesis, trains, evaluates, logs results, and promotes the best adapter based on execution accuracy with hard accuracy, syntax validity, and error count as tie-breakers.

## Metrics

- `execution_accuracy`: fraction of examples whose generated SQL result matches the gold result.
- `syntax_validity`: fraction of generated SQL queries that execute safely.
- `exact_result_match`: same result comparison as execution accuracy.
- `timeout_count`: queries exceeding the configured timeout.
- `error_count`: failed or rejected SQL executions.
- `easy_accuracy`, `medium_accuracy`, `hard_accuracy`: accuracy by difficulty.
- `ecommerce_accuracy`, `saas_accuracy`, `banking_accuracy`: accuracy by database.

## Example Leaderboard

| experiment_id | model_id | execution_accuracy | syntax_validity | hard_accuracy | decision |
| --- | --- | ---: | ---: | ---: | --- |
| exp_001 | Qwen/Qwen2.5-Coder-3B-Instruct | 0.58 | 0.74 | 0.31 | kept |

## Limitations

- Synthetic data is realistic but not a substitute for production data.
- Execution accuracy rewards result equivalence, not necessarily SQL style.
- The v0.1 research agent is rule-based and only changes YAML configuration values.
- Baseline and training runs require compatible GPU memory and working Hugging Face dependencies.
- SQLite timeout handling is conservative and intended for local experimentation.

## Future Work

- Add richer SQL template families and more domains.
- Add semantic query clustering for dataset diversity.
- Add an optional LLM-based research agent behind the existing proposal interface.
- Add CI smoke tests for CPU-safe commands.
- Add model-card style experiment summaries for published adapters.
