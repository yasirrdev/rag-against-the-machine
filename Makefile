.PHONY: install run debug clean lint

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	uv run --no-sync python -c "from pathlib import Path; from shutil import rmtree; caches = [*Path('src').rglob('__pycache__'), Path('.mypy_cache'), Path('.pytest_cache')]; [rmtree(path) for path in caches if path.is_dir()]"

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
