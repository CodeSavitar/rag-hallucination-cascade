# Hallucination Triage

A tiered evaluator for RAG outputs: cheap heuristics filter first, a trained classifier handles the middle, and only genuinely ambiguous responses reach an LLM judge.

## Problem

Production RAG systems produce confident answers that are not supported by the documents they retrieved, and most teams have no systematic way to catch this. LLM-as-judge evaluation works but is too slow and expensive to run on every response, so teams sample and hope.

This project makes the check cheap enough to run on everything.

## Status

Cascade built and measured (M1–M5). On 40 held-out answers it sends about a third of answers to the
LLM judge and catches 11 of 12 hallucinations, at $2.60 instead of $7.08 per 1,000 answers. Full
results, intervals and limits: [`reports/cascade_results.md`](reports/cascade_results.md). Task
board: `STATUS.md`. Deployment and a full README come next (M6).

## Data and attribution

The contracts and clause labels come from the **Contract Understanding Atticus Dataset (CUAD) v1**,
curated by [The Atticus Project](https://www.atticusprojectai.org/cuad) and released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset itself is not included in this
repository; download it from the link above. Some files under `data/` and `reports/` quote short
contract passages and clause labels from CUAD.

> Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball. *CUAD: An Expert-Annotated NLP Dataset for
> Legal Contract Review.* NeurIPS 2021 Datasets and Benchmarks. [arXiv:2103.06268](https://arxiv.org/abs/2103.06268)

Use of CUAD is also subject to The Atticus Project's
[privacy policy](https://www.atticusprojectai.org/privacy-policy) and
[disclaimer](https://www.atticusprojectai.org/disclaimer).
