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
```
.
├── prompt_optimizer_v3.py    # Core optimization engine
├── rq1.py                     # Experiments for Research Question 1
├── moves.py                   # Prompt transformation operators
├── datasets.py                # Dataset generation utilities
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## Core Components

### prompt_optimizer_v3.py

Main optimization engine containing:

- `PromptNode`: Data structure representing prompts as graph nodes
- `beam_search()`: Beam search implementation over prompt space
- `random_walk_search()`: Random walk baseline implementation
- `generate_seed_prompt()`: Automatic seed prompt generation from examples
- `evaluate_metric_string_match()`: Evaluation for objective tasks (sentiment, QA)
- `evaluate_metric_critic_lm()`: Evaluation using GPT-5 as critic for subjective tasks
- `CriticLM`: Wrapper for using more capable model as evaluator

### moves.py

Prompt transformation operators:

- `VerboseMove`: Expands prompt with additional detail
- `ShortenMove`: Makes prompt more concise
- `ReorderMove`: Reorganizes prompt structure
- `AddExamplesMove`: Adds few-shot examples
- Additional operators (ChainOfThought, AddConstraints, RoleAssignment, etc.)

### rq1.py

Experimental framework for answering Research Question 1: "Can classical search algorithms meaningfully improve prompt performance under computational constraints?"

Implements:
- `run_rq1_experiments()`: Runs all optimization methods on a single task
- `run_rq1_all_tasks_and_export()`: Runs experiments across all 5 tasks
- `run_specific_rq1_all_task_and_export()`: Runs experiments for a specific task
- CSV export functionality for results

### datasets.py

Synthetic dataset generation for 5 NLP tasks using GPT-5:
- Sentiment classification
- Question answering
- Summarization
- Complex reasoning
- Natural language inference

## Installation

Requires Python 3.8+ and Poetry for dependency management.
```bash
# Install dependencies
poetry install

# Set up API key
export OPENAI_KEY="your-openai-api-key"
```

## Usage

### Interactive Mode

Run the main optimizer interactively:
```bash
python prompt_optimizer_v3.py
```

Select a task (1-5) and the system will:
1. Generate a synthetic dataset
2. Create a seed prompt automatically
3. Run beam search optimization
4. Evaluate on held-out test set

### Run RQ1 Experiments

Run experiments for a specific task:
```bash
python rq1.py sentiment output.csv
python rq1.py reasoning output.csv
python rq1.py nli output.csv
```

Available tasks: sentiment, qa, summarization, reasoning, nli

This will:
- Generate seed prompt
- Run 4 optimization methods: seed baseline, one-hop improvement, random walk, beam search
- Evaluate on both dev and test sets
- Export results to CSV

### Configuration

Key parameters in the code:

- Beam width: Default 2 (see `beam_width` parameter)
- Beam depth: Default 2 (see `depth` parameter)
- Random walk steps: Default 5 (see `steps` parameter)
- Dataset split: 25% train, 25% dev, 50% test
- Model: GPT-4o for optimization, GPT-5 for critic evaluation

## Experimental Results

Key findings from the paper:

- Beam search consistently improves dev set performance (e.g., reasoning: 0.40 to 0.80)
- Test set improvements are more modest due to overfitting (reasoning: 0.20 to 0.50)
- `ShortenMove` (make concise) is the most frequently used operator in successful paths
- `VerboseMove` never appears in successful optimization paths
- Random walk performs well when all operators are beneficial, but inconsistently on complex tasks

## Limitations

- Shallow search depth (width=2, depth=2) due to computational constraints
- Small synthetic datasets (20 examples per task)
- Critic-based evaluation introduces noise for subjective tasks
- Limited operator set (4 operators tested, 5 additional designed but not evaluated)

See Section 6.2 of the paper for detailed discussion of limitations and future work.

## Citation

If you use this code or build upon this work, please cite:
```bibtex
@article{taneja2025prompt,
  title={Prompt Optimization as a State-Space Search Problem},
  author={Taneja, Maanas},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

Maanas Taneja - University of Minnesota

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

This work was completed as an independent research project for CSCI 4511W by Professor James Moen at the University of Minnesota. Inspired by the DSPy framework from Stanford NLP and related work on automated prompt engineering (OPRO, APE, AutoPrompt).