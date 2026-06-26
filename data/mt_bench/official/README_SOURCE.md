# MT-Bench Official Dataset Files

This directory holds the unmodified official MT-Bench files from the FastChat
repository. These files must be obtained manually before running the experiment.

## Source

Repository : https://github.com/lm-sys/FastChat
License    : Apache 2.0
Reference  : Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and
             Chatbot Arena. NeurIPS 2023. arXiv:2306.05685v4

## Files expected here

| File | Source path in FastChat repo |
|------|------------------------------|
| `question.jsonl` | `fastchat/llm_judge/data/mt_bench/question.jsonl` |
| `judge_prompts.jsonl` | `fastchat/llm_judge/data/mt_bench/judge_prompts.jsonl` (optional) |

## How to obtain

```bash
git clone https://github.com/lm-sys/FastChat.git

mkdir -p data/mt_bench/official
mkdir -p data/mt_bench/subset

cp FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
   data/mt_bench/official/question.jsonl
```

Optionally, copy the judge prompts file as well:

```bash
cp FastChat/fastchat/llm_judge/data/judge_prompts.jsonl \
   data/mt_bench/official/judge_prompts.jsonl
```

After copying, the FastChat clone is no longer needed and can be removed.

## Integrity

- `question.jsonl` — 80 questions, IDs 81-160, 8 categories, 2 turns each
- The experiment aborts at startup if this file is not present
- The SHA256 of this file is recorded in `results/metadata/dataset_manifest.csv`
  and `results/metadata/experiment_manifest.json` at every run

## Methodological constraints

- **P5** — Prompts are used verbatim. No modifications.
- **P6** — Prompts are used in English. No translation.
- **P7** — If this file is absent the experiment aborts. No fallback prompts,
  synthetic prompts, paraphrases, or placeholders are used.
- **P8** — The 2-turn structure is preserved per question.

The experiment can only be reported as MT-Bench-based if this exact file is
used without modifying any prompt text.
