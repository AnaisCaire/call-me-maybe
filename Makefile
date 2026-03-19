PY = python3
FUNC_DEF = data/input/functions_definition.json
INPUT = data/input/function_calling_tests.json
OUTPUT = data/output/function_calling_results.json
CODE_DIRS = src

.PHONY: install debug run lint lint-strict

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition $(FUNC_DEF) \
		--input $(INPUT) \
		--output $(OUTPUT)

clean:
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "--- Cleanup Complete ---"

debug:
	uv run python -m pdb -c continue -m src

lint:
	@echo "--- Running Flake8 ---"
	uv run flake8 $(CODE_DIRS)
	@echo "--- Running Mypy ---"
	PYTHONPATH=. uv run mypy $(CODE_DIRS) \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--warn-return-any \
		--warn-unused-ignores

lint-strict:
	@echo "--- Running Strict Linting ---"
	uv run flake8
	PYTHONPATH=. uv run mypy $(CODE_DIRS) --strict
