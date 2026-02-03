# Contributing to ECLIPSE

Thank you for your interest in contributing to ECLIPSE (Evolutionary Clinical Intake and Prioritization System).

## Getting Started

1. Fork the repository and clone locally.
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest test_all_components.py -v`
4. For LLM mutation, set `OPENAI_API_KEY` and install `openai`.

## Development Workflow

- Create a branch for your changes.
- Ensure tests pass before submitting.
- Follow existing code style (type hints, docstrings).

## Reporting Issues

Open an issue describing:

- What you expected
- What actually happened
- Steps to reproduce
- Python version and environment

## Pull Requests

1. Update documentation if you change behavior.
2. Add tests for new functionality.
3. Keep PRs focused; split large changes into smaller ones.

## Code Structure

- `sim/` — Discrete-event simulator
- `mechanisms/` — Mechanism schema, validation, mutation
- `evolution/` — Evolutionary loop, LLM mutation, selection
- `eval/` — Episodes, baselines, robustness, ablation
- `baselines/` — Baseline mechanism definitions
- `scripts/` — CLI entrypoints and reporting

## Configuration

Primary config: `config/default.yaml`. Parameters are documented in `docs/CALIBRATION.md`.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
