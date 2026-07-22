.PHONY: help install dev-install test test-unit test-integration test-slow lint typecheck format check clean knowledge-vault

help:
	@echo "curLit development commands"
	@echo ""
	@echo "  make install          Install production dependencies"
	@echo "  make dev-install      Install with dev + train + viz extras"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-slow        Run slow tests (hypothesis property tests)"
	@echo "  make lint             Run ruff linter"
	@echo "  make typecheck        Run mypy type checker"
	@echo "  make format           Auto-format with ruff"
	@echo "  make check            Run lint + typecheck + test (CI pipeline)"
	@echo "  make clean            Remove build artifacts and caches"
	@echo "  make knowledge-vault  Regenerate the offline Obsidian research vault (CL-uuy0)"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev,train,viz]"

test:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v

test-slow:
	python -m pytest tests/ -v -m "slow"

lint:
	python -m ruff check src/ tests/

typecheck:
	python -m mypy src/

format:
	python -m ruff check --fix src/ tests/
	python -m ruff format src/ tests/

check: lint typecheck test
	@echo "All checks passed"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/

knowledge-vault:
	.venv/bin/python scripts/export_knowledge_graph.py
	@echo "Obsidian research vault regenerated at knowledge/obsidian/vault (CL-uuy0) — research-only, NOT part of the live fleet"
