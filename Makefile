.PHONY: sync run test test-cov lint lint-fix format format-check typecheck security check build install uninstall clean clean-all info help

APP := face-rec

sync:
	uv sync

run:
	uv run $(APP) $(ARGS)

test:
	uv run python -m pytest

test-cov:
	uv run python -m pytest --cov=src/face_rec --cov-report=term-missing --cov-report=html

lint:
	uv run ruff check src tests

lint-fix:
	uv run ruff check --fix src tests

format:
	uv run ruff format src tests

format-check:
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

security:
	uv run bandit -q -r src

check: lint format-check typecheck security test-cov

build:
	uv build

install:
	uv tool install --python 3.12 --force .   # 3.13+ has no InsightFace/onnx wheels

uninstall:
	uv tool uninstall $(APP)

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-all: clean
	rm -rf .venv dist build *.egg-info

info:
	@echo "app=$(APP)"
	@uv run python -c "import sys; print('python', sys.version.split()[0])"

help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | sed 's/:.*//' | sort | tr '\n' ' '; echo
