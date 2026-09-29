# Hallucination Triage

A tiered evaluator for RAG outputs: cheap heuristics filter first, a trained classifier handles the middle, and only genuinely ambiguous responses reach an LLM judge.

## Problem

Production RAG systems produce confident answers that are not supported by the documents they retrieved, and most teams have no systematic way to catch this. LLM-as-judge evaluation works but is too slow and expensive to run on every response, so teams sample and hope.

This project makes the check cheap enough to run on everything.

## Status

In development. See `STATUS.md`.
