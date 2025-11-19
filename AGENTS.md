# Repository Guidelines

## Project Structure & Module Organization
 Trading logic, strategies, and shared contexts live in `core/`, while order routing and layered risk controls live in `execution/`. Operational tooling (`ops/`), cache/data adapters (`data/`, `features/`), and UI surfaces (`ui/`) keep runtime code modular. Tests mirror those packages inside `tests/` (for example `tests/test_signal_flow_e2e.py`) so failures can be traced quickly.

## Build, Test, and Development Commands
- `make install-dev` installs pytest, coverage, flake8, and black.
- `make test`, `make test-unit`, `make test-integration`, and `make test-e2e` run the pytest marker suites declared in `pytest.ini`.
- `make coverage` or `make test-all` emits HTML and XML coverage for `core`, `ops`, `data`, `ui`, and `execution` in `htmlcov/`.
- `make lint` runs flake8 (120-char lines, ignores E501/W503/E203) plus `black --check`; `make format` applies black fixes.
- `make run-dashboard` and `make run-chat` launch the Plotly dashboard and chat interface in the current shell, so ensure `.env` is loaded.
- `make backup` and `make clean` call `ops.cache_manager.CacheManager` to snapshot or purge caches, coverage files, and `__pycache__` folders.

## Coding Style & Naming Conventions
Use Python 3.10+, four-space indentation, and type hints on public functions. Modules stay snake_case, classes PascalCase, async helpers end with `_async`, and tests or fixtures belong under `tests/` or `tests/helpers/`. Run black before committing, keep imports grouped stdlib/third-party/local, and rely on flake8 to guard line length and unused symbols.

## Testing Guidelines
Pytest discovers `test_*.py`, `Test*`, and `test_*` by default; reuse fixtures for Binance clients, cache layers, and Plotly helpers rather than reaching external services. Mark suites with `unit`, `integration`, `e2e`, `slow`, or `asyncio` so coworkers can target `pytest -m unit` or `make test-integration`. Coverage must stay ≥80% (enforced via `--cov-fail-under=80`), so add assertions or synthetic fixtures whenever you introduce new modules.

## Commit & Pull Request Guidelines
Even though the exported archive lacks `.git`, upstream commits follow short imperative subjects such as `feat(core): add adaptive spread guard`. Keep subjects ≤72 characters, describe rationale plus executed commands in the body, and reference tickets or dashboards when relevant. Pull requests should summarize behavior changes, list affected modules, attach screenshots for `ui/` edits, and flag any `.env`, schema, or backup impacts.

## Security & Configuration Tips
Store keys only in `.env` and surface them through `core/system_context.py` or `config/` defaults; keep `TESTNET=true` until production sign-off. Encrypt anything copied from `backups/`, run `ops.cache_manager.CacheManager.restore_backup()` only on trusted machines, and scrub sensitive data from `logs/` before sharing bug reports.
