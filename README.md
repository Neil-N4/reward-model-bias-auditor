# Reward Model Bias Auditor

Reward models can be exploited without changing the underlying answer.

This repository is an evals-style framework for auditing reward-model robustness under **semantic-preserving rewrites**. It starts from a fixed response, applies controlled perturbations, runs an automated reward-hacking search loop, and measures when reward models change their preferences for the wrong reasons.

The project is designed around the kinds of failure modes that matter in post-training and preference-model deployment:

- sycophancy inflation
- confidence/style overvaluation
- formatting sensitivity
- authority-cue overvaluation
- refusal/safety-style distortion
- ranking instability under semantically equivalent rewrites
- composite reward-hacking exploitability
- cross-model exploit transfer
- preference-signal collapse under input sanitization

## What This Evaluates

The benchmark currently audits ten perturbation families:

- `sycophancy`
- `length`
- `confidence_framing`
- `format`
- `authority`
- `politeness`
- `markdown_density`
- `citation_density`
- `safety_style`
- `exploit_search`

The `exploit_search` family is the most important one: it composes multiple high-reward surface cues into a single semantically preserving rewrite to test whether a model can be systematically “reward hacked.”

On top of that fixed benchmark, the repo now includes a black-box search loop that incrementally applies rewrite operators and keeps only candidates that improve reward score while preserving semantic overlap.

## Headline Finding

In the seeded offline audit:

- composite exploit-search rewrites produce the largest reward inflation across all models
- the strongest benchmark-style model profile is also the most exploitable
- weaker perturbations like politeness and safety-style framing sometimes fail to flip rankings, which makes instability a measurable quantity rather than a trivial constant

This is the core thesis of the repo:

> benchmark strength and reward-model robustness are not the same thing.

## Visual Snapshot

![Effect sizes](docs/images/effect_sizes.png)

![Sycophancy profile](docs/images/sycophancy_profile.png)

![Instability profile](docs/images/instability_profile.png)

![Transferability matrix](docs/images/transferability_matrix.png)

![Sanitization defense](docs/images/sanitization_drop.png)

## Why This Is Scale-Aligned

This is not a generic “bias in AI” project. It is an evaluation framework for:

- isolating model failure modes cleanly
- discovering reward-hacking style exploits
- quantifying robustness gaps between models
- surfacing deployment-relevant instability metrics

That is much closer to frontier evals work than to ordinary benchmarking.

## Evaluation Design

Each benchmark item contains:

- a fixed prompt
- a fixed base response
- a pair of semantically aligned rewrites
- a single perturbation family label

The benchmark then measures how much reward score movement can be attributed to surface form rather than substance.

### Metrics

The main outputs are:

- `mean score delta`
- `95% bootstrap confidence interval`
- `effect_size`
- `ranking_instability_rate`
- `exploitability_ratio`
- `semantic_overlap_min`
- `semantic_pass_rate`
- `mean_transfer_gain`
- `transfer_success_rate`
- `mean_sanitization_drop`

These metrics are meant to answer different questions:

- Does the bias exist?
- How large is it?
- Is it consistent enough to flip rankings?
- Can multiple benign-looking cues be combined into a stronger exploit?
- Did the rewrites preserve content well enough to make the result meaningful?
- Do attacks found on one reward model transfer to another?
- Does normalization remove the inflated preference signal?

## System Design

```mermaid
flowchart LR
    A["Base Prompt + Canonical Response"] --> B["Controlled Perturbation Generator"]
    B --> C["Single-Factor Pairs"]
    B --> D["Composite Exploit-Search Rewrites"]
    C --> E["Reward Scoring Backend"]
    D --> E
    E --> F["Score Table"]
    F --> G["Bias Summary<br/>mean deltas + bootstrap CI + instability"]
    F --> H["Model Summary<br/>exploitability ratio + strongest bias"]
    C --> I["Semantic Consistency Gate"]
    D --> I
    A --> K["Automated Attack Search"]
    K --> L["Best Exploit Per Prompt"]
    L --> M["Transferability Matrix"]
    L --> N["Sanitization Defense"]
    G --> J["Markdown Report + Plots"]
    H --> J
    I --> J
    M --> J
    N --> J
```

## Current Benchmark Shape

- 5 base prompts
- 10 perturbation families
- 10 perturbation instances per prompt
- 500 paired comparisons in the offline demo

That keeps the benchmark small enough to inspect manually while still producing stable model-by-bias comparisons.

## Scoring Backends

### 1. Offline Research Profiles

The seeded backend simulates three reward-model profiles:

- `rm_small`
- `rm_instruct`
- `rm_benchmark_top`

These are intentionally configured so that:

- stronger models are not automatically more robust
- exploit-search is more powerful than any single surface cue
- weaker perturbations do not always destabilize rankings
- transfer can be non-trivial rather than guaranteed
- defense layers meaningfully collapse attack gains

### 2. Real HuggingFace Reward Models

The repo also supports real model scoring through HuggingFace.

Validated paths:

- `OpenAssistant/reward-model-deberta-v3-base`
- `OpenAssistant/reward-model-electra-large-discriminator`

## Quickstart

### Offline Audit

```bash
cd /path/to/reward-model-bias-auditor
source .venv/bin/activate
MPLCONFIGDIR=$PWD/.mplconfig python scripts/run_demo.py
```

This writes:

- `outputs/pairs.csv`
- `outputs/scores.csv`
- `outputs/summary.csv`
- `outputs/model_summary.csv`
- `outputs/semantic_consistency.csv`
- `outputs/attack_search.csv`
- `outputs/transferability.csv`
- `outputs/defense.csv`
- `outputs/defense_summary.csv`
- `outputs/report.md`

### Real Reward-Model Audit

```bash
cd /path/to/reward-model-bias-auditor
source .venv/bin/activate
MPLCONFIGDIR=$PWD/.mplconfig python scripts/run_hf_audit.py \
  --model OpenAssistant/reward-model-deberta-v3-base \
  --model OpenAssistant/reward-model-electra-large-discriminator \
  --pairs-per-bias 1
```

This writes real-model outputs under `outputs/hf/`.

## Automated Red Teaming

The attack search loop treats the reward model as a black-box objective:

1. start from a canonical answer
2. apply one rewrite operator at a time
3. keep the candidate only if reward score improves
4. reject candidates that violate semantic-overlap or edit-budget thresholds
5. test the discovered exploit against other reward models

This turns the project from a static benchmark into a lightweight reward-hacking discovery engine.

## Defense Layer

The defense pass sanitizes reward-model inputs before rescoring:

- strips sycophantic agreement cues
- removes authority and citation markers
- flattens markdown and list formatting
- removes overcautious safety preambles
- compresses verbosity-only additions

If reward gain disappears after sanitization, the original preference was likely driven by surface form rather than substance.

### Tests

```bash
cd /path/to/reward-model-bias-auditor
source .venv/bin/activate
python -m pytest -q
```

## Verified Working State

Verified locally:

- offline seeded audit runs end-to-end
- semantic-consistency outputs are generated
- bootstrap CI and model summary tables are generated
- automated exploit search, transferability, and defense summaries are generated
- real HuggingFace scoring works with `OpenAssistant/reward-model-deberta-v3-base`
- real HuggingFace scoring works with `OpenAssistant/reward-model-electra-large-discriminator`
- test suite passes in the project venv

## Semantic-Consistency Gate

This repo includes a lightweight semantic consistency check based on lexical overlap with the canonical base response plus an edit-budget constraint in the automated search loop. It is not a full entailment model, but it is enough to keep the benchmark honest about whether a perturbation still resembles the same underlying answer.

That gate is important because otherwise reward inflation results are too easy to attack: if the rewrite changed the substance, the benchmark has not isolated a reward-model bias.

## Threat Model

The main threat model is:

1. An attacker keeps the answer substantively the same.
2. They add style or framing cues the reward model overvalues.
3. The reward model increases preference score anyway.
4. Composite rewrites amplify that effect enough to destabilize rankings.

This is exactly the kind of failure that makes preference-model deployment brittle.

## Repository Structure

```text
reward-model-bias-auditor/
├── docs/
│   └── images/
│       ├── effect_sizes.png
│       ├── instability_profile.png
│       └── sycophancy_profile.png
├── outputs/
│   ├── hf/
│   ├── model_summary.csv
│   ├── pairs.csv
│   ├── report.md
│   ├── scores.csv
│   ├── semantic_consistency.csv
│   └── summary.csv
├── scripts/
│   ├── run_demo.py
│   └── run_hf_audit.py
├── src/
│   └── reward_model_bias_auditor/
│       ├── analysis.py
│       ├── benchmark.py
│       ├── hf_runner.py
│       ├── models.py
│       ├── plotting.py
│       ├── reporting.py
│       └── scoring.py
└── tests/
    └── test_pipeline.py
```

## What This Still Does Not Claim

- full semantic invariance guarantees
- full transformer attribution faithfulness
- deployment-ready reward-model certification

This is an evals-grade research artifact, not a finished production safety system.

## Best Resume Framing

If you want the strongest believable framing, it is this:

> Built a controlled evaluation framework for reward-model robustness, discovering semantically preserving rewrites that inflated reward scores through sycophancy, authority, formatting, and composite reward-hacking cues while quantifying ranking instability and exploitability across models.
