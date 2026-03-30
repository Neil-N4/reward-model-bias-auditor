# Threat Model

This project targets a specific post-training failure mode:

1. A response is substantively correct.
2. An attacker rewrites it without changing its core meaning.
3. The rewrite adds surface cues a reward model overvalues.
4. The reward model inflates the score or flips the ranking anyway.

The attack surface includes:

- sycophantic agreement
- authority framing
- confidence framing
- markdown / formatting density
- citation density
- verbosity inflation
- overcautious safety phrasing

The repository treats reward models as black-box preference functions. The goal is not to jailbreak them into unsafe content, but to quantify how easily they can be manipulated by semantically preserving rewrites.

## Threats To Validity

- Semantic preservation is approximate, not a formal guarantee.
- Open-source reward models may differ from deployment models in architecture and calibration.
- Small-sample HF runs are useful for case studies but not for strong statistical claims.

## What Counts As Success

An exploit is considered meaningful if it:

- increases reward score
- passes the semantic guard
- stays within edit-budget limits
- transfers to another model or collapses under sanitization in a diagnostically useful way
