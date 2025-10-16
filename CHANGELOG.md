# Changelog

All notable changes to ACE Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Playbook persistence with `save_to_file()` and `load_from_file()` methods
- Example demonstrating playbook save/load functionality (`examples/playbook_persistence.py`)
- py.typed file for PEP 561 type hint support
- Documentation for playbook persistence in README

### Fixed
- README incorrectly referenced non-existent docs/ directory
- Test badge URL in README (test.yml → tests.yml)

## [0.2.0] - 2025-10-15

### Added
- LangChain integration via `LangChainLiteLLMClient` for advanced workflows
- Router support for load balancing across multiple model deployments
- Comprehensive example for LangChain usage (`examples/langchain_example.py`)
- Optional installation group: `pip install ace-framework[langchain]`
- PyPI badges and Quick Links section in README
- CHANGELOG.md for version tracking

### Fixed
- Parameter filtering in LiteLLM and LangChain clients (refinement_round, max_refinement_rounds)
- GitHub Actions workflow using deprecated artifact actions v3 → v4

### Changed
- Improved README with better structure and badges
- Updated .gitignore to exclude build artifacts and development files

### Removed
- Unnecessary development files from repository

## [0.1.1] - 2025-10-15

### Fixed
- GitHub Actions workflow for PyPI publishing
- Updated artifact upload/download actions from v3 to v4

## [0.1.0] - 2025-10-15

### Added
- Initial release of ACE Framework
- Core ACE implementation based on paper (arXiv:2510.04618)
- Three-role architecture: Generator, Reflector, and Curator
- Playbook system for storing and evolving strategies
- LiteLLM integration supporting 100+ LLM providers
- Offline and Online adaptation modes
- Async and streaming support
- Example scripts for quick start
- Comprehensive test suite
- PyPI packaging and GitHub Actions CI/CD

### Features
- Self-improving agents that learn from experience
- Delta operations for incremental playbook updates
- Support for OpenAI, Anthropic, Google, and more via LiteLLM
- Type hints and modern Python practices
- MIT licensed for open source use

[0.2.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Kayba-ai/agentic-context-engine/releases/tag/v0.1.0