# Methodology

## Benchmark Construction

The benchmark starts from a small set of canonical prompts and base responses. For each base response, the generator creates controlled perturbation pairs across ten bias families:

- sycophancy
- length
- confidence framing
- format
- authority
- politeness
- markdown density
- citation density
- safety style
- exploit search

Each pair is intended to preserve substance while changing only surface form.

## Metrics

The evaluation pipeline computes:

- mean score delta
- bootstrap confidence intervals
- effect size
- ranking instability rate
- exploitability ratio
- transfer success rate
- sanitization drop

## Automated Attack Search

The red-teaming loop uses beam search over rewrite operators.

At each search depth:

1. expand each beam item with all unused rewrite operators
2. reject candidates that fail the semantic guard
3. score surviving candidates with the reward model
4. keep the top-k highest-scoring candidates

This is a small black-box optimization routine rather than a single handcrafted rewrite.

## Semantic Guard

The semantic filter is hybrid:

- lexical overlap
- MiniLM embedding similarity when available
- contradiction heuristic based on negation changes
- edit-budget constraint

Candidates must pass all gating thresholds before they can be kept in the beam.

## Defense Pass

The defense layer sanitizes reward-model inputs by stripping high-risk surface cues, then rescoring the normalized answer. This estimates how much of the original reward came from substance versus presentation.
