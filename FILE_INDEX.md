# File Index — uv Adaptation

## Canonical modified files

- `management/MASTER_PLAN.md` — governance, DoD, risk and release rules adapted to uv
- `management/STATUS.md` — current status, evidence, blocker and source registry adapted to uv
- `weekly/W01_PLAN.md` — full Week 1 plan with uv commands, locked execution and Docker path
- `management/RESOURCE_MAP.md` — current uv/Python/dependency versions and exact official resources

## Added review and execution files

- `management/UV_MIGRATION_REVIEW.md` — blocker/major/minor audit and migration acceptance criteria
- `weekly/W01_DAY1_EXECUTION.md` — canonical uv replacement for the historical Day 1 execution commands

## Added project templates

- `.python-version`
- `pyproject.toml`
- `src/llm_serving_lab/__init__.py`
- `.gitignore`
- `.dockerignore`
- `README.md`

## Deliberately absent

- `uv.lock` is not included because it could not be honestly generated in the current environment with Python 3.11.15. Generate it on the actual development machine with `uv lock`, validate with `uv lock --check` and `uv sync --locked`, then commit it.
- `requirements.txt` is not a canonical project file. Only export one from a committed lock when an external integration requires it.
