# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

GREEN-IA is a Master's thesis experiment measuring the energy consumption of LLM inference under Q4 vs Q8 quantization. Two models (llama-2-7b and qwen2.5-7b) run locally via llama-cpp-python, energy is measured with CodeCarbon, and quality is evaluated using Claude as LLM-as-a-Judge on 40 MT-Bench prompts (8 categories × 5 questions × 15 repetitions).

Hardware targets: Apple M4 Mac Mini (Metal backend) and Windows RTX 4060 (CUDA backend).

## Environment setup

```bash
# Activate the venv (already created at ./venv, Python 3.13)
source venv/bin/activate

# macOS Metal (required for GPU inference)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Windows CUDA
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Validate backend and full environment
python scripts/validate_config.py --all
python scripts/validate_config.py --check-backend
```

The venv is at `./venv` (Python 3.13). `ANTHROPIC_API_KEY` must be set for quality evaluation.

## Running the experiment

```bash
# Prepare MT-Bench subset (must run before the benchmark)
python scripts/prepare_mt_bench_subset.py --save

# Hash models for reproducibility
python scripts/hash_artifacts.py --save

# Dry run (validates without inference)
python run_experiment.py --dry-run

# Smoke test (3 conversations, saves to results/smoke_test/)
python run_experiment.py --limit-conversations 3

# Full experiment (GPU, profile from config.yaml)
python run_experiment.py

# Override hardware/device
python run_experiment.py --hardware-profile mac_m4 --execution-device gpu
```

Results from `--limit-conversations < 5` or `--repetitions < 15` go to `results/smoke_test/` to avoid contaminating production results.

## Analysis pipeline (after benchmark)

```bash
# Quality evaluation with Claude (requires ANTHROPIC_API_KEY)
python judge_with_claude.py --input results/raw/conversation_results.csv --output-dir results/judge/

# Statistical analysis → 14 CSVs in results/summary/
python analysis_summary.py

# Plots → PNG families in results/plots/
python analysis_plots.py

# Phase 2 optimization (picks winner from Phase 1 Pareto front)
python optimize_winner.py
python optimization_plots.py

# Per-category scripts (current active work)
python scripts/analisis_categoria.py     # runs energy measurement per category
python scripts/juez_calidad.py           # pairwise quality comparison per category
python scripts/regenerar_plots.py        # regenerates plots from existing CSVs
```

## Architecture

**Config flow:** `config.yaml` → `src/config_loader.py:load_config()` → `ExperimentConfig` (frozen dataclass). `ExperimentConfig` is the single source of truth for all runtime parameters. Hardware-specific values (`n_threads`, `n_batch`, `n_gpu_layers`) are derived from `hardware_profile` + `execution_device`; do not hardcode them elsewhere.

**Prompt building:** `src/prompt_builder.py:build_prompt(model_name, messages)` dispatches to the correct chat template:
- `qwen*` → ChatML (`<|im_start|>/<|im_end|>`)
- `llama-2*chat` or `llama-2*instruct` → Llama-2 Chat (`[INST]/<<SYS>>`)
- anything else → generic plain-text fallback (used for base llama-2-7b)

Using the wrong template increases energy consumption and degrades quality because the model generates extra tokens to compensate.

**Energy measurement:** CodeCarbon `EmissionsTracker` wraps only the inference call. Model loading, warmup runs, cooldowns, and baseline calibration are all outside the tracker. Baseline idle power is measured before each model config and subtracted to get `baseline_corrected_energy`. Both raw and corrected metrics are always saved.

**MT-Bench subset:** 40 questions (5 per category, 8 categories), selected deterministically (`first_n_per_category`, seed=42) from `data/mt_bench/official/question.jsonl`. The experiment aborts if this file is missing — there is no fallback dataset (principle P7).

**Results layout:**
- `results/raw/` — turn and conversation CSVs from Phase 1
- `results/judge/` — Claude quality evaluation CSVs
- `results/summary/` — statistical analysis tables
- `results/plots/` — PNG plot families
- `results/analisis_categoria/` — per-category energy results (current active work)
- `results/metadata/` — reproducibility artifacts (hashes, env report, config snapshot)

## Key methodology constraints

- `temperature = 0.0`, `seed = 42`, `vary_seed_by_repetition = false` — all 15 repetitions of the same prompt are identical. Repetitions measure **energy/latency stability**, not output diversity.
- `on_context_overflow: skip` — prompts exceeding `n_ctx − max_tokens` are saved with `status = context_overflow` and excluded from energy analysis.
- Claude evaluates Llama and Qwen responses only — never its own. This avoids self-preference bias.
- `llama-2-7b` is a **base** model; `qwen2.5-7b` is **instruct**. Quality scores between them are not directly comparable on MT-Bench (which targets instruction-tuned models).
- CodeCarbon reports TDP-based estimates, not physical measurements. Always label energy values as estimates.
- Carbon intensity factor: Paraguay 26 gCO₂eq/kWh (EMBER 2024, predominantly hydroelectric).

## Model files

GGUF files are not in the repo. Expected paths:
- `models/llama-2-7b/Q4_K_M.gguf`
- `models/llama-2-7b/llama-2-7b.Q8_0.gguf`
- `models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf`
- `models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q8_0.gguf`
