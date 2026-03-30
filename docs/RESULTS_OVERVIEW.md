# Results Overview

This page is the fastest way to inspect the latest outputs without reading the full methodology first.

## Primary Artifacts

Offline:

- [`outputs/report.md`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/report.md)
- [`outputs/attack_search.csv`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/attack_search.csv)
- [`outputs/transferability.csv`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/transferability.csv)
- [`outputs/defense_summary.csv`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/defense_summary.csv)
- [`outputs/CASE_STUDIES.md`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/CASE_STUDIES.md)

Real-model:

- [`outputs/hf/report.md`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/hf/report.md)
- [`outputs/hf/attack_search.csv`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/hf/attack_search.csv)
- [`outputs/hf/transferability.csv`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/hf/transferability.csv)
- [`outputs/hf/defense_summary.csv`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/hf/defense_summary.csv)
- [`outputs/hf/CASE_STUDIES.md`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/hf/CASE_STUDIES.md)

## What To Look For

1. `summary.csv`
   This tells you which perturbation families move reward scores the most.

2. `attack_search.csv`
   This tells you what the search loop actually found worth exploiting.

3. `transferability.csv`
   This tells you whether a discovered exploit generalizes to other reward models.

4. `defense_summary.csv`
   This tells you whether canonicalization removes the attack gain.

5. `CASE_STUDIES.md`
   This gives you inspection-ready examples for interviews or writeups.

## Recommended Reading Order

1. README
2. `docs/THREAT_MODEL.md`
3. this file
4. `outputs/hf/report.md`
5. `outputs/hf/CASE_STUDIES.md`

## Reproduction Commands

Offline:

```bash
cd /path/to/reward-model-bias-auditor
source .venv/bin/activate
MPLCONFIGDIR=$PWD/.mplconfig python scripts/run_demo.py
```

Real-model:

```bash
cd /path/to/reward-model-bias-auditor
source .venv/bin/activate
MPLCONFIGDIR=$PWD/.mplconfig python scripts/run_hf_audit.py \
  --model OpenAssistant/reward-model-deberta-v3-base \
  --model OpenAssistant/reward-model-electra-large-discriminator \
  --pairs-per-bias 2
```

Preset-based:

```bash
cd /path/to/reward-model-bias-auditor
source .venv/bin/activate
MPLCONFIGDIR=$PWD/.mplconfig python scripts/run_hf_audit.py \
  --model-preset openassistant \
  --pairs-per-bias 1
```
