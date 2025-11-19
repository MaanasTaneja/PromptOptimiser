# Prompt Optimization as State-Space Search

This repository contains the implementation code for the paper "Prompt Optimization as State-Space Search Problem" by Maanas Taneja (University of Minnesota, 2025).

## Overview

This project explores prompt optimization by treating it as a classical AI search problem. Instead of optimizing through demonstration generation (like DSPy), we model the prompt space as a graph where nodes represent prompt variants and edges represent deliberate transformations (operators). We then apply classical search algorithms (beam search, random walk) to discover optimized prompts.

## Paper

The full paper is available on arXiv: [arXiv:XXXX.XXXXX] (link to be added upon publication)

## Key Contributions

- Formalization of prompt optimization as a state-space search problem
- Implementation of beam search and random walk algorithms for prompt exploration
- Evaluation across 5 NLP tasks: sentiment classification, question answering, summarization, reasoning, and natural language inference
- Analysis of which prompt engineering techniques (operators) contribute most to optimization
- Empirical demonstration that conciseness-focused transformations consistently improve prompts

## Repository Structure