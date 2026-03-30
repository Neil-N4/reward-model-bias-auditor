# Results

## Offline Profiles

The seeded offline profiles are designed to make the benchmark non-trivial:

- stronger benchmark profiles are not automatically more robust
- exploit-search composites dominate single-cue perturbations
- sanitization removes most or all of the attack gain
- transferability is measurable rather than assumed

Representative artifacts:

- [`outputs/report.md`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/report.md)
- [`outputs/transferability.csv`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/transferability.csv)
- [`outputs/defense_summary.csv`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/defense_summary.csv)

## Real Reward Models

The HF path currently validates the same pipeline on:

- `OpenAssistant/reward-model-deberta-v3-base`
- `OpenAssistant/reward-model-electra-large-discriminator`

Representative artifacts:

- [`outputs/hf/report.md`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/hf/report.md)
- [`outputs/hf/transferability.csv`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/hf/transferability.csv)
- [`outputs/hf/defense_summary.csv`](/Users/neilnair/Documents/Playground/reward-model-bias-auditor/outputs/hf/defense_summary.csv)

## Reading The Results

Three outputs matter most:

1. `attack_search.csv`
   This shows the best discovered exploit per prompt and source model.
2. `transferability.csv`
   This measures whether attacks found on one model generalize to another.
3. `defense_summary.csv`
   This measures whether sanitization collapses the inflated reward.

Together, they answer a much stronger question than “is there bias?”:

Can reward models be manipulated systematically, do those manipulations transfer, and does a simple defense remove the gain?
